use std::collections::{HashMap, HashSet};
use std::ops::{Deref, Sub};
use actix::prelude::*;
use crate::data_store::edge::{Edge, MaterializedEdge};
use crate::data_store::materialize::Materialize;
use crate::data_store::node::{IndependentNode, NodeRef};

use crate::Training;
use crate::training::scoring::messages::{EdgeWeights, NodeDegrees, ScoreInitDone};
use crate::training::scoring::overlap::ScoringOverlap;


pub(crate) trait ScoringWeights {
    fn count_edges_in_time(&mut self) -> Vec<usize>;
    fn calculate_edge_weight(&mut self) -> HashMap<MaterializedEdge, usize>;
    fn calculate_node_degrees(&mut self) -> HashMap<NodeRef, usize>;
    fn start_node_degrees_rotation(&mut self, ctx: &mut Context<Training>);
    fn start_edge_weight_rotation(&mut self, ctx: &mut Context<Training>);
    fn init_done(&mut self) -> bool;
}

impl ScoringWeights for Training {
    fn count_edges_in_time(&mut self) -> Vec<usize> {
        let start_point = self.transposition.range_start_point.unwrap_or(0);
        let pseudo_edge = Edge::new(IndependentNode::new(0, 0, 0).to_ref(), IndependentNode::new(0, 0, 0).to_ref()).to_ref();
        let mut edges_in_time = vec![];
        let mut last_point_id = None;
        let mut last_len: usize = 0;
        for (i, edge) in self.data_store.get_edges().iter().chain(&[pseudo_edge]).enumerate() {
            match last_point_id {
                None => { last_point_id = Some(edge.get_to_id()); }
                Some(last_point_id_ref) => if edge.get_to_id().ne(&last_point_id_ref) {
                    while edges_in_time.len().lt(&last_point_id_ref.sub(&start_point)) {
                        edges_in_time.push(last_len);
                    }
                    last_point_id = Some(edge.get_to_id());
                    last_len = i;
                    edges_in_time.push(i);
                }
            }
        }

        let result_length = self.num_rotated.expect("should have been already set") - if self.cluster_nodes.get_own_idx().eq(&self.cluster_nodes.len()) {
            1 // -1 because the last point has no outgoing edge
        } else {
            0
        };
        while edges_in_time.len().lt(&result_length) {
            edges_in_time.push(last_len);
        }

        edges_in_time
    }

    fn calculate_edge_weight(&mut self) -> HashMap<MaterializedEdge, usize> {
        let mut edge_weight = HashMap::new();
        for edge in self.data_store.get_edges() {
            let materialized = edge.materialize();
            match edge_weight.get_mut(&materialized) {
                Some(weight) => { *weight += 1; },
                None => { edge_weight.insert(materialized, 1); }
            }
        }
        edge_weight
    }

    fn calculate_node_degrees(&mut self) -> HashMap<NodeRef, usize> {
        let mut node_degrees = HashMap::new();
        let mut seen_edges = HashSet::new();

        for edge in self.data_store.get_edges() {
            if seen_edges.insert(edge.clone()) {
                match node_degrees.get_mut(&edge.get_from_node()) {
                    Some(degree) => { *degree += 1; }
                    None => { node_degrees.insert(edge.get_from_node(), 1); }
                }

                match node_degrees.get_mut(&edge.get_to_node()) {
                    Some(degree) => { *degree += 1; }
                    None => { node_degrees.insert(edge.get_to_node(), 1); }
                }
            }
        }

        node_degrees
    }

    fn start_node_degrees_rotation(&mut self, ctx: &mut Context<Training>) {
        self.scoring.node_degrees_rotation_protocol.start(self.cluster_nodes.len());
        self.scoring.node_degrees_rotation_protocol.resolve_buffer(ctx.address().recipient());
        self.cluster_nodes.get_as(&self.cluster_nodes.get_next_idx().unwrap(), "Training").unwrap()
            .do_send(NodeDegrees { degrees: self.scoring.node_degrees.iter().map(|(node, degree)| (node.deref().deref().clone(), degree.clone())).collect() });
        self.scoring.node_degrees_rotation_protocol.sent();
    }

    fn start_edge_weight_rotation(&mut self, ctx: &mut Context<Training>) {
        self.scoring.edge_weight_rotation_protocol.start(self.cluster_nodes.len());
        self.scoring.edge_weight_rotation_protocol.resolve_buffer(ctx.address().recipient());
        self.cluster_nodes.get_as(&self.cluster_nodes.get_next_idx().unwrap(), "Training").unwrap()
            .do_send(EdgeWeights { weights: self.scoring.edge_weight.clone() });
        self.scoring.edge_weight_rotation_protocol.sent();
    }

    fn init_done(&mut self) -> bool {
        !(
            self.scoring.node_degrees_rotation_protocol.is_running() ||
            self.scoring.edge_weight_rotation_protocol.is_running()
        )
    }
}


impl Handler<NodeDegrees> for Training {
    type Result = ();

    fn handle(&mut self, msg: NodeDegrees, ctx: &mut Self::Context) -> Self::Result {
        if !self.scoring.node_degrees_rotation_protocol.received(&msg) {
            return
        }

        for (node, degree) in msg.degrees.iter() {
            let node_ref = node.clone().to_ref();
            match self.scoring.node_degrees.get_mut(&node_ref) {
                None => {self.scoring.node_degrees.insert(node_ref, degree.clone());},
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
        if self.cluster_nodes.get_own_idx().eq(&self.cluster_nodes.len()) {
            self.send_overlap_to_neighbor(ctx);
        } else {
            self.scoring.overlap_rotation_protocol.start(1);
            self.scoring.overlap_rotation_protocol.resolve_buffer(ctx.address().recipient());
        }
    }
}
