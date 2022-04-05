mod helper;
pub mod messages;
pub mod overlap;
#[cfg(test)]
mod tests;
pub mod weights;

use crate::data_store::edge::MaterializedEdge;
use crate::data_store::node::NodeRef;
use crate::messages::PoisonPill;
use crate::training::scoring::helper::ScoringHelper;
use crate::training::scoring::messages::{
    EdgeWeights, NodeDegrees, OverlapRotation, ScoringDone, ScoringHelperInstruction,
    ScoringHelperResponse, SubScores,
};
use crate::training::scoring::weights::ScoringWeights;
use crate::training::Training;
use crate::utils::itertools::LengthAble;
use crate::utils::logging::progress_bar::S2GppProgressBar;
use crate::utils::rotation_protocol::RotationProtocol;
use crate::utils::HelperProtocol;
use actix::{Addr, AsyncContext, Context, Handler, SyncArbiter};
use anyhow::Result;
use csv::WriterBuilder;
use ndarray::{concatenate, Array1, ArrayView1, Axis};
use ndarray_stats::QuantileExt;
use num_traits::Float;
use std::collections::HashMap;
use std::fs::File;
use std::ops::{Index, IndexMut};

#[derive(Default)]
pub(crate) struct Scoring {
    pub score: Option<Array1<f32>>,
    single_scores: Vec<f32>,
    /// cluster_node_id -> (subscores, first_empty?)
    subscores: HashMap<usize, (Array1<f32>, bool)>,
    first_empty: bool,
    pub node_degrees: HashMap<NodeRef, usize>, // must be sent
    edge_weight: HashMap<MaterializedEdge, usize>, // must be sent
    edges_in_time: Vec<usize>,
    node_degrees_rotation_protocol: RotationProtocol<NodeDegrees>,
    edge_weight_rotation_protocol: RotationProtocol<EdgeWeights>,
    overlap_rotation_protocol: RotationProtocol<OverlapRotation>,

    helpers: Option<Addr<ScoringHelper>>,
    score_rotation_protocol: RotationProtocol<SubScores>,
    helper_protocol: HelperProtocol,
    progress_bar: S2GppProgressBar,
    helper_buffer: HashMap<usize, ScoringHelperResponse>,
}

pub(crate) trait Scorer {
    fn init_scoring(&mut self, ctx: &mut Context<Training>);
    fn score(&mut self, ctx: &mut Context<Training>);
    fn parallel_score(&mut self, score_length: usize);
    fn finalize_parallel_score(&mut self, ctx: &mut Context<Training>);
    fn normalize_score(&mut self, score: &mut Array1<f32>);
    fn finalize_scoring(&mut self, ctx: &mut Context<Training>);
    fn output_score(&mut self, output_path: String) -> Result<()>;
}

impl Scorer for Training {
    fn init_scoring(&mut self, ctx: &mut Context<Training>) {
        self.scoring.edges_in_time = self.count_edges_in_time();
        self.scoring.edge_weight = self.calculate_edge_weight();

        if self.cluster_nodes.len() > 0 {
            self.start_node_degrees_rotation(ctx);
            self.start_edge_weight_rotation(ctx);
        } else {
            // non-distributed
            self.score(ctx);
        }
    }

    fn score(&mut self, ctx: &mut Context<Training>) {
        if self.scoring.edges_in_time.len() < (self.parameters.query_length - 1) {
            panic!("There are less edges than the given 'query_length'!");
        }

        let score_length = self.scoring.edges_in_time.len() - (self.parameters.query_length - 1);

        self.scoring.helper_protocol.n_total = self.parameters.n_threads;
        self.scoring.progress_bar = S2GppProgressBar::new_from_len("info", score_length);

        let edges = self.data_store.get_edges();
        let edges_in_time = self.scoring.edges_in_time.clone();
        let edge_weight = self.scoring.edge_weight.clone();
        let node_degrees = self.scoring.node_degrees.clone();
        let query_length = self.parameters.query_length;
        let receiver = ctx.address().recipient();

        self.scoring.helpers = Some(SyncArbiter::start(self.parameters.n_threads, move || {
            ScoringHelper {
                edges: edges.clone(),
                edges_in_time: edges_in_time.clone(),
                edge_weight: edge_weight.clone(),
                node_degrees: node_degrees.clone(),
                query_length,
                receiver: receiver.clone(),
            }
        }));

        self.parallel_score(score_length);
    }

    fn parallel_score(&mut self, score_length: usize) {
        let n_per_thread = score_length / self.parameters.n_threads;
        let n_rest = score_length % self.parameters.n_threads;

        for i in 0..self.parameters.n_threads {
            let rest = if i == self.parameters.n_threads - 1 {
                n_rest
            } else {
                0
            };

            self.scoring
                .helpers
                .as_ref()
                .unwrap()
                .do_send(ScoringHelperInstruction {
                    start: i * n_per_thread,
                    length: n_per_thread + rest,
                });

            self.scoring.helper_protocol.sent();
        }
    }

    fn finalize_parallel_score(&mut self, ctx: &mut Context<Training>) {
        self.scoring.helpers.as_ref().unwrap().do_send(PoisonPill);
        let mut scores: Array1<f32> = self.scoring.single_scores.clone().into_iter().collect();
        self.scoring.single_scores.clear();

        if self.cluster_nodes.len() > 0 {
            let own_idx = self.cluster_nodes.get_own_idx();
            self.scoring
                .score_rotation_protocol
                .start(self.cluster_nodes.len());
            self.scoring
                .score_rotation_protocol
                .resolve_buffer(ctx.address().recipient());
            self.scoring
                .subscores
                .insert(own_idx, (scores.clone(), self.scoring.first_empty));
            self.cluster_nodes
                .get_next_as("Training")
                .unwrap()
                .do_send(SubScores {
                    cluster_node_id: own_idx,
                    scores,
                    first_empty: self.scoring.first_empty,
                });
            self.scoring.score_rotation_protocol.sent();
        } else {
            self.normalize_score(&mut scores);
            self.scoring.score = Some(scores);
            self.finalize_scoring(ctx);
        }
    }

    fn normalize_score(&mut self, scores: &mut Array1<f32>) {
        let all_score_max = *scores.max().unwrap();
        let all_score_min = *scores.min().unwrap();
        *scores = scores
            .into_iter()
            .map(|x| (*x - all_score_min) / (all_score_max - all_score_min))
            .collect();
    }

    fn finalize_scoring(&mut self, ctx: &mut Context<Training>) {
        if self.scoring.score.is_none() {
            let mut scores: Vec<Array1<f32>> = vec![];
            for cluster_node_id in 0..self.parameters.n_cluster_nodes {
                let (mut sub_score, first_empty) = self
                    .scoring
                    .subscores
                    .remove(&cluster_node_id)
                    .expect("A subscore is missing!");
                if first_empty {
                    let last_score = scores
                        .last()
                        .expect("First cannot be empty if it's the first overall score point!");

                    fill_up_first_missing_points(&mut sub_score, last_score[last_score.len() - 1]);
                }
                scores.push(sub_score);
            }
            let mut cat_scores = concatenate(
                Axis(0),
                scores
                    .iter()
                    .map(|s| s.view())
                    .collect::<Vec<ArrayView1<f32>>>()
                    .as_slice(),
            )
            .expect("Could not concatenate subscores!");
            self.normalize_score(&mut cat_scores);
            self.scoring.score = Some(cat_scores);
        }

        if let Some(output_path) = self.parameters.score_output_path.clone() {
            self.output_score(output_path).unwrap();
        }

        ctx.address().do_send(ScoringDone);
    }

    fn output_score(&mut self, output_path: String) -> Result<()> {
        let score = self
            .scoring
            .score
            .as_ref()
            .expect("Please, calculate score before saving to file!");
        let file = File::create(output_path)?;
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        for s in score.iter() {
            writer.serialize(s)?;
        }
        Ok(())
    }
}

impl Handler<ScoringHelperResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: ScoringHelperResponse, ctx: &mut Self::Context) -> Self::Result {
        if msg.start != self.scoring.single_scores.len() {
            self.scoring.helper_buffer.insert(msg.start, msg);
            return;
        }
        self.scoring.helper_protocol.received();

        let mut scores = msg.scores;
        if msg.first_empty {
            match self.scoring.single_scores.last() {
                Some(last_score) => fill_up_first_missing_points(&mut scores, *last_score),
                None => {
                    self.scoring.first_empty = true;
                }
            }
        }
        self.scoring.progress_bar.inc_by(scores.len() as u64);
        self.scoring.single_scores.extend(scores);

        if !self.scoring.helper_protocol.is_running() {
            self.scoring.progress_bar.finish_and_clear();
            self.finalize_parallel_score(ctx);
        } else {
            match self
                .scoring
                .helper_buffer
                .remove(&self.scoring.single_scores.len())
            {
                None => {}
                Some(scoring_helper_response) => {
                    ctx.address().do_send(scoring_helper_response);
                }
            }
        }
    }
}

fn fill_up_first_missing_points<T: IndexMut<usize, Output = f32> + LengthAble>(
    scores: &mut T,
    initial_score: f32,
) where
    <T as Index<usize>>::Output: Float,
{
    for i in 0..scores.get_length() {
        if scores[i] == 0.0 {
            if i == 0 {
                scores[i] = initial_score;
            } else {
                scores[i] = scores[i - 1]
            }
        }
    }
}

impl Handler<SubScores> for Training {
    type Result = ();

    fn handle(&mut self, msg: SubScores, ctx: &mut Self::Context) -> Self::Result {
        if !self.scoring.score_rotation_protocol.received(&msg) {
            return;
        }

        self.scoring
            .subscores
            .insert(msg.cluster_node_id, (msg.scores.clone(), msg.first_empty));

        if self.scoring.score_rotation_protocol.is_running() {
            self.cluster_nodes
                .get_next_as("Training")
                .unwrap()
                .do_send(msg);
            self.scoring.score_rotation_protocol.sent();
        } else {
            self.finalize_scoring(ctx);
        }
    }
}
