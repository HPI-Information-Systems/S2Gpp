#[cfg(test)]
mod tests;
pub mod messages;

use ndarray::{Array1, ArrayView1, Axis, concatenate};
use crate::training::Training;
use std::collections::{HashMap, HashSet};
use crate::utils::{Edge, NodeName};
use ndarray_stats::QuantileExt;
use std::ops::{Range, Sub};
use anyhow::Result;
use std::fs::File;
use actix::{AsyncContext, Context, Handler};
use csv::WriterBuilder;
use crate::training::scoring::messages::{ScoringDone, NodeDegrees, SubScores, ScoreInitDone, EdgeWeights, OverlapRotation};
use crate::utils::rotation_protocol::RotationProtocol;

#[derive(Default)]
pub struct Scoring {
    pub score: Option<Array1<f32>>,
    subscores: HashMap<usize, Array1<f32>>,
    pub node_degrees: HashMap<NodeName, usize>,
    edge_weight: HashMap<Edge, usize>,
    edges_in_time: Vec<usize>,
    node_degrees_rotation_protocol: RotationProtocol<NodeDegrees>,
    edge_weight_rotation_protocol: RotationProtocol<EdgeWeights>,
    overlap_rotation_protocol: RotationProtocol<OverlapRotation>,
    score_rotation_protocol: RotationProtocol<SubScores>
}

pub trait Scorer {
    fn init_scoring(&mut self, ctx: &mut Context<Training>);
    fn init_done(&mut self) -> bool;
    fn count_edges_in_time(&mut self) -> Vec<usize>;
    fn calculate_edge_weight(&mut self) -> HashMap<Edge, usize>;
    fn calculate_node_degrees(&mut self) -> HashMap<NodeName, usize>;
    fn send_overlap_to_neighbor(&mut self, ctx: &mut Context<Training>);
    fn score(&mut self, ctx: &mut Context<Training>);
    fn score_p_degree(&mut self, edge_range: Range<usize>) -> (f32, usize);
    fn finalize_scoring(&mut self, ctx: &mut Context<Training>);
    fn output_score(&mut self, output_path: String) -> Result<()>;
}


impl Scorer for Training {
    fn init_scoring(&mut self, ctx: &mut Context<Training>) {
        self.scoring.edges_in_time = self.count_edges_in_time();
        self.scoring.edge_weight = self.calculate_edge_weight();

        if self.cluster_nodes.len() > 0 {
            self.scoring.node_degrees_rotation_protocol.start(self.cluster_nodes.len());
            self.scoring.node_degrees_rotation_protocol.resolve_buffer(ctx.address().recipient());
            self.cluster_nodes.get_as(&self.cluster_nodes.get_next_idx().unwrap(), "Training").unwrap()
                .do_send(NodeDegrees { degrees: self.scoring.node_degrees.clone() });
            self.scoring.node_degrees_rotation_protocol.sent();
            
            self.scoring.edge_weight_rotation_protocol.start(self.cluster_nodes.len());
            self.scoring.edge_weight_rotation_protocol.resolve_buffer(ctx.address().recipient());
            self.cluster_nodes.get_as(&self.cluster_nodes.get_next_idx().unwrap(), "Training").unwrap()
                .do_send(EdgeWeights { weights: self.scoring.edge_weight.clone() });
            self.scoring.edge_weight_rotation_protocol.sent();

        } else { // non-distributed
            self.score(ctx);
        }
    }

    fn init_done(&mut self) -> bool {
        !(
            self.scoring.node_degrees_rotation_protocol.is_running() ||
            self.scoring.edge_weight_rotation_protocol.is_running()
        )
    }

    fn count_edges_in_time(&mut self) -> Vec<usize> {
        let start_point = self.transposition.range_start_point.unwrap_or(0);
        let pseudo_edge = (0, Edge(NodeName(0, 0), NodeName(0, 0)));
        let mut edges_in_time = vec![];
        let mut last_point_id = None;
        let mut last_len: usize = 0;
        for (i, (point_id, _edge)) in self.transposition.edges.iter().chain(&[pseudo_edge]).enumerate() {
            match last_point_id {
                None => { last_point_id = Some(point_id); }
                Some(last_point_id_ref) => if point_id.ne(last_point_id_ref) {
                    while edges_in_time.len().lt(&last_point_id_ref.sub(&start_point)) {
                        edges_in_time.push(last_len);
                    }
                    last_point_id = Some(point_id);
                    last_len = i;
                    edges_in_time.push(i);
                }
            }
        }

        let result_length = self.rotation.rotated.as_ref().unwrap().shape()[0] - if self.cluster_nodes.get_own_idx().eq(&self.cluster_nodes.len()) {
            1 // -1 because the last point has no outgoing edge
        } else {
            0
        };
        while edges_in_time.len().lt(&result_length) {
            edges_in_time.push(last_len);
        }

        edges_in_time
    }

    fn calculate_edge_weight(&mut self) -> HashMap<Edge, usize> {
        let mut edge_weight = HashMap::new();
        for (_, edge) in self.transposition.edges.iter() {
            match edge_weight.get_mut(edge) {
                Some(weight) => { *weight += 1; },
                None => { edge_weight.insert(edge.clone(), 1); }
            }
        }
        edge_weight
    }

    fn calculate_node_degrees(&mut self) -> HashMap<NodeName, usize> {
        let mut node_degrees = HashMap::new();
        let mut seen_edges = HashSet::new();

        for (_, edge) in self.edge_estimation.edges.iter() {
            if seen_edges.insert(edge.clone()) {
                match node_degrees.get_mut(&edge.0) {
                    Some(degree) => { *degree += 1; }
                    None => { node_degrees.insert(edge.0.clone(), 1); }
                }

                match node_degrees.get_mut(&edge.1) {
                    Some(degree) => { *degree += 1; }
                    None => { node_degrees.insert(edge.1.clone(), 1); }
                }
            }
        }

        node_degrees
    }

    fn send_overlap_to_neighbor(&mut self, ctx: &mut Context<Training>) {
        let from_edge_idx = self.scoring.edges_in_time[0];
        let to_edge_idx = self.scoring.edges_in_time[self.parameters.query_length - 1];
        let overlap = self.transposition.edges[from_edge_idx..to_edge_idx].to_vec();

        let edges_in_time = self.scoring.edges_in_time[0..(self.parameters.query_length - 1)].to_vec();

        self.cluster_nodes.get_as(&self.cluster_nodes.get_previous_idx().unwrap(), "Training").unwrap()
                .do_send(OverlapRotation { edges: overlap, edges_in_time });
        self.scoring.overlap_rotation_protocol.sent();

        self.score(ctx);
    }

    fn score(&mut self, ctx: &mut Context<Training>) {
        // todo: parallelize
        let mut all_score = vec![];

        /*for (node, degree) in self.scoring.node_degrees.iter() {
            if node.0 == 64 {
                println!("node {} -> {}", node, degree);
            }
        }*/

        /*for (edge, weight) in self.scoring.edge_weight.iter() {
            if edge.0.0 == 64 {
                println!("edge {} -> {}", edge, weight);
            }
        }*/

        if self.scoring.edges_in_time.len() < (self.parameters.query_length - 1) {
            panic!("There are less edges than the given 'query_length'!");
        }

        let end_iteration = self.scoring.edges_in_time.len() - (self.parameters.query_length - 1);

        for i in 0..end_iteration {
            let from_edge_idx = self.scoring.edges_in_time[i];
            let to_edge_idx = self.scoring.edges_in_time[i + self.parameters.query_length - 1];

            let (score, len_score) = self.score_p_degree(from_edge_idx..to_edge_idx);
            if len_score == 0 {
                all_score.push(all_score.last().unwrap_or(&0_f32).clone());
            } else {
                all_score.push(score);
            }
        }

        let all_score: Array1<f32> = all_score.into_iter().map(|x| -x).collect();
        let all_score_max = all_score.max().unwrap().clone();
        let all_score_min = all_score.min().unwrap().clone();
        let scores = (all_score - all_score_min) / (all_score_max - all_score_min);

        if self.cluster_nodes.len() > 0 {
            let own_idx = self.cluster_nodes.get_own_idx();
            self.scoring.score_rotation_protocol.start(self.cluster_nodes.len());
            self.scoring.score_rotation_protocol.resolve_buffer(ctx.address().recipient());
            self.scoring.subscores.insert(own_idx, scores.clone());
            self.cluster_nodes.get_as(&self.cluster_nodes.get_next_idx().unwrap(), "Training").unwrap()
                .do_send(SubScores { cluster_node_id: own_idx, scores });
            self.scoring.score_rotation_protocol.sent();
        } else {
            self.scoring.score = Some(scores);
            self.finalize_scoring(ctx);
        }
    }

    fn score_p_degree(&mut self, edge_range: Range<usize>) -> (f32, usize) {
        let p_edge = &self.transposition.edges[edge_range];
        let len_score = p_edge.len();
        let alpha = 0.00000001 + (len_score as f32);
        let score: f32 = p_edge.iter().map(|(_, edge)| {
            (self.scoring.edge_weight.get(edge).unwrap() * (self.scoring.node_degrees.get(&edge.0).expect("Edge with unknown Node found!") - 1)) as f32
        }).sum();
        (score / alpha, len_score)
    }

    fn finalize_scoring(&mut self, ctx: &mut Context<Training>) {
        match &self.scoring.score {
            None => {
                let score: Vec<ArrayView1<f32>> = (0..self.parameters.n_cluster_nodes).into_iter()
                    .map(|i| self.scoring.subscores.get(&i).expect("A subscore is missing!").view()).collect();
                let score = concatenate(Axis(0),score.as_slice())
                    .expect("Could not concatenate subscores!");
                self.scoring.subscores.clear();
                self.scoring.score = Some(score);
            },
            _ => ()
        }

        if let Some(output_path) = self.parameters.score_output_path.clone() {
            self.output_score(output_path).unwrap();
        }

        ctx.address().do_send(ScoringDone);
    }

    fn output_score(&mut self, output_path: String) -> Result<()> {
        let score = self.scoring.score.as_ref().expect("Please, calculate score before saving to file!");
        let file = File::create(output_path)?;
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        for s in score.iter() {
            writer.serialize(s)?;
        }
        Ok(())
    }
}


impl Handler<NodeDegrees> for Training {
    type Result = ();

    fn handle(&mut self, msg: NodeDegrees, ctx: &mut Self::Context) -> Self::Result {
        if !self.scoring.node_degrees_rotation_protocol.received(&msg) {
            return
        }

        for (node, degree) in msg.degrees.iter() {
            match self.scoring.node_degrees.get_mut(node) {
                None => {self.scoring.node_degrees.insert(node.clone(), degree.clone());},
                Some(old_degree) => { *old_degree += degree; }
            }
        }

        if self.scoring.node_degrees_rotation_protocol.is_running() {
            self.cluster_nodes.get_as(&self.cluster_nodes.get_next_idx().unwrap(), "Training").unwrap()
                .do_send(msg);
            self.scoring.node_degrees_rotation_protocol.sent();
        } else if self.init_done() {
            ctx.address().do_send(ScoreInitDone);
        }
    }
}


impl Handler<EdgeWeights> for Training {
    type Result = ();

    fn handle(&mut self, msg: EdgeWeights, ctx: &mut Self::Context) -> Self::Result {
        if !self.scoring.edge_weight_rotation_protocol.received(&msg) {
            return
        }


        for (edge, weight) in msg.weights.iter() {
            match self.scoring.edge_weight.get_mut(edge) {
                None => {self.scoring.edge_weight.insert(edge.clone(), weight.clone());},
                Some(old_weight) => { *old_weight += weight; }
            }
        }

        if self.scoring.edge_weight_rotation_protocol.is_running() {
            self.cluster_nodes.get_as(&self.cluster_nodes.get_next_idx().unwrap(), "Training").unwrap()
                .do_send(msg);
            self.scoring.edge_weight_rotation_protocol.sent();
        } else if self.init_done() {
            ctx.address().do_send(ScoreInitDone);
        }
    }
}


impl Handler<ScoreInitDone> for Training {
    type Result = ();

    fn handle(&mut self, _msg: ScoreInitDone, ctx: &mut Self::Context) -> Self::Result {
        let own_id = self.cluster_nodes.get_own_idx();
        if own_id.eq(&self.cluster_nodes.len()) {
            self.send_overlap_to_neighbor(ctx);
        } else {
            self.scoring.overlap_rotation_protocol.start(1);
            self.scoring.overlap_rotation_protocol.resolve_buffer(ctx.address().recipient());
        }
    }
}


impl Handler<OverlapRotation> for Training {
    type Result = ();

    fn handle(&mut self, msg: OverlapRotation, ctx: &mut Self::Context) -> Self::Result {
        if !self.scoring.overlap_rotation_protocol.received(&msg) {
            return
        }

        self.transposition.edges.extend(msg.edges);
        let last_edge_count = self.scoring.edges_in_time.last().unwrap();
        let received_edges_in_time: Vec<usize> = msg.edges_in_time.into_iter().map(|x| x + last_edge_count).collect();
        self.scoring.edges_in_time.extend(received_edges_in_time);
        self.send_overlap_to_neighbor(ctx);

        if self.scoring.overlap_rotation_protocol.is_running() {
            panic!("Only one overlap should be received");
        }
    }
}


impl Handler<SubScores> for Training {
    type Result = ();

    fn handle(&mut self, msg: SubScores, ctx: &mut Self::Context) -> Self::Result {
        if !self.scoring.score_rotation_protocol.received(&msg) {
            return
        }

        self.scoring.subscores.insert(msg.cluster_node_id.clone(), msg.scores.clone());

        if self.scoring.overlap_rotation_protocol.is_running() {
            self.cluster_nodes.get_as(&self.cluster_nodes.get_next_idx().unwrap(), "Training").unwrap()
                .do_send(msg);
            self.scoring.score_rotation_protocol.sent();
        } else {
            self.finalize_scoring(ctx);
        }
    }
}
