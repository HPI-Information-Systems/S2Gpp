use crate::data_store::edge::{EdgeRef, MaterializedEdge};
use crate::data_store::materialize::Materialize;
use crate::data_store::node::NodeRef;
use crate::messages::PoisonPill;
use crate::training::scoring::messages::{ScoringHelperInstruction, ScoringHelperResponse};
use actix::prelude::*;
use std::collections::HashMap;
use std::ops::Range;

pub(crate) struct ScoringHelper {
    pub edges: Vec<EdgeRef>,
    pub edges_in_time: Vec<usize>,
    pub edge_weight: HashMap<MaterializedEdge, usize>,
    pub node_degrees: HashMap<NodeRef, usize>,
    pub query_length: usize,
    pub receiver: Recipient<ScoringHelperResponse>,
}

impl ScoringHelper {
    fn score_p_degree(&mut self, edge_range: Range<usize>) -> (f32, usize) {
        let p_edge = self.edges[edge_range].iter();
        let len_score = p_edge.len();
        let alpha = 0.00000001 + (len_score as f32);
        let score: f32 = p_edge
            .map(|edge| {
                (self.edge_weight.get(&edge.materialize()).unwrap()
                    * (self
                        .node_degrees
                        .get(&edge.get_from_node())
                        .expect("Edge with unknown Node found!")
                        - 1)) as f32
            })
            .sum();
        (score / alpha, len_score)
    }
}

impl Actor for ScoringHelper {
    type Context = SyncContext<Self>;
}

impl Handler<ScoringHelperInstruction> for ScoringHelper {
    type Result = ();

    fn handle(&mut self, msg: ScoringHelperInstruction, _ctx: &mut Self::Context) -> Self::Result {
        let mut single_scores: Vec<f32> = vec![];
        let mut first_empty = false;

        for i in msg.start..msg.start + msg.length {
            let from_edge_idx = self.edges_in_time[i];
            let to_edge_idx = self.edges_in_time[i + self.query_length - 1] + 1;

            let (score, len_score) =
                self.score_p_degree(from_edge_idx..to_edge_idx.min(self.edges.len()));

            single_scores.push(if len_score == 0 {
                match single_scores.last() {
                    Some(last) => *last,
                    None => {
                        first_empty = true;
                        0.0
                    }
                }
            } else {
                -score
            });
        }
        self.receiver
            .do_send(ScoringHelperResponse {
                start: msg.start,
                scores: single_scores,
                first_empty,
            });
    }
}

impl Handler<PoisonPill> for ScoringHelper {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
