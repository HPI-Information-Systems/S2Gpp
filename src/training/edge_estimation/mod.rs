mod messages;
#[cfg(test)]
mod tests;
mod edges_orderer;

use actix::prelude::*;
use std::collections::HashMap;
use crate::training::Training;
pub use crate::training::edge_estimation::messages::{EdgeEstimationDone, EdgeReductionMessage, EdgeRotationMessage};
use crate::utils::{Edge, NodeName};
use num_integer::Integer;
use crate::training::edge_estimation::edges_orderer::EdgesOrderer;
use crate::utils::logging::progress_bar::S2GppProgressBar;
use crate::training::edge_estimation::messages::PointNodeName;
use crate::utils::rotation_protocol::RotationProtocol;


#[derive(Default)]
pub struct EdgeEstimation {
    /// [(point_id, Edge)]
    pub edges: Vec<(usize, Edge)>,
    /// point id -> node
    open_edges: HashMap<usize, NodeName>,
    /// point id -> node before
    open_edges_before: HashMap<usize, NodeName>,
    /// cluster node -> [(point_id, node), ...]
    send_edges: HashMap<usize, Vec<PointNodeName>>,
    received_reduction_messages: Vec<EdgeReductionMessage>,
    rotation_protocol: RotationProtocol<EdgeRotationMessage>
}

pub trait EdgeEstimator {
    fn estimate_edges(&mut self, ctx: &mut Context<Training>);
    fn connect_nodes(&mut self);
    fn rotate_edges(&mut self, ctx: &mut Context<Training>);
    fn merging_rotated_edges(&mut self, sent_edges: HashMap<usize, Vec<PointNodeName>>) -> HashMap<usize, Vec<PointNodeName>>;
    fn reintegrate_remaining_open_edges(&mut self);
    fn reduce_to_main(&mut self, ctx: &mut Context<Training>);
    fn finalize_edge_estimation(&mut self, ctx: &mut Context<Training>);
}

impl EdgeEstimator for Training {
    fn estimate_edges(&mut self, ctx: &mut Context<Training>) {
        self.connect_nodes();
        self.edge_estimation.rotation_protocol.start(self.parameters.n_cluster_nodes - 1);
        self.edge_estimation.rotation_protocol.resolve_buffer(ctx.address().recipient());
        self.rotate_edges(ctx);
    }

    fn connect_nodes(&mut self) {
        let len_dataset = self.dataset_stats.as_ref().expect("DatasetStats should've been set by now!").n.unwrap();
        let segments_per_node = (self.parameters.rate as f32 / self.parameters.n_cluster_nodes as f32).floor() as usize;
        let has_all_segments = segments_per_node == self.parameters.rate;

        let mut previous_node: Option<NodeName> = None;

        let progress_bar = S2GppProgressBar::new_from_len("info", len_dataset);
        let next_id = self.cluster_nodes.get_next_idx();
        for point_id in 0..len_dataset {
            match self.node_estimation.nodes_by_point.get(&point_id) {
                Some(intersection_nodes) => {
                    let mut edges = EdgesOrderer::new(point_id, previous_node.is_some());
                    for current_node in intersection_nodes {
                        if current_node.0 == 49 {
                            println!("{}\t->\t{}", point_id, current_node);
                        }
                        edges.add_node(&previous_node, current_node);

                        // todo: is this always the case? some jump to far off points
                        // maybe only put to send edges if nodes are skipped
                        if current_node.0.mod_floor(&segments_per_node).eq(&(segments_per_node - 1)) &&
                            !has_all_segments { // last segment
                            previous_node = None;
                            let node_id = next_id.expect("This should not happen. Only in a distributed case, this part of the code is run.");
                            match self.edge_estimation.send_edges.get_mut(&node_id) {
                                Some(open_edges) => open_edges.push((point_id, current_node.clone())),
                                None => {
                                    self.edge_estimation.send_edges.insert(node_id, vec![(point_id, current_node.clone())]);
                                }
                            }
                        } else {
                            if current_node.0.mod_floor(&segments_per_node).eq(&0) &&
                                !has_all_segments
                            { // first segment
                                self.edge_estimation.open_edges.insert(point_id, current_node.clone());
                                if let Some(prev) = previous_node {
                                    self.edge_estimation.open_edges_before.insert(point_id, prev);
                                }
                            }
                            previous_node = Some(current_node.clone());
                        }
                    }
                    previous_node = edges.last_node.or(previous_node);
                    let edges = edges.to_vec();
                    self.edge_estimation.edges.extend(edges.into_iter());
                },
                None => ()  // transition did not cross a segment
            }
            progress_bar.inc();
        }
        progress_bar.finish_and_clear();
    }

    fn rotate_edges(&mut self, ctx: &mut Context<Training>) {
        match &self.cluster_nodes.get_next_idx() {
            Some(next_idx) => {
                let next_node = self.cluster_nodes.get_as(next_idx, "Training").unwrap();
                let open_edges = self.edge_estimation.send_edges.clone();
                self.edge_estimation.send_edges.clear();
                next_node.do_send(EdgeRotationMessage { open_edges });
                self.edge_estimation.rotation_protocol.sent();
            },
            None => self.reduce_to_main(ctx)
        }
    }

    fn merging_rotated_edges(&mut self, sent_edges: HashMap<usize, Vec<PointNodeName>>) -> HashMap<usize, Vec<PointNodeName>> {
        let own_id = self.cluster_nodes.get_own_idx();
        let mut open_edges = sent_edges;
        match open_edges.remove(&own_id) {
            None => (),
            Some(assigned_edges) => for (point_id, node) in assigned_edges {
                let n = self.dataset_stats.as_ref().unwrap().n.expect("DatasetStats should be set by now!");
                for next_point in point_id..n {
                    if let Some(next_node) = self.edge_estimation.open_edges.remove(&next_point) {
                        self.edge_estimation.open_edges_before.remove(&next_point);
                        self.edge_estimation.edges.push((point_id, Edge(node, next_node)));
                        break
                    }
                }
            }
        }
        open_edges
    }

    fn reintegrate_remaining_open_edges(&mut self) {
        let cloned_open_edges = self.edge_estimation.open_edges.clone();
        self.edge_estimation.open_edges.clear();
        for (open_idx, open_node) in cloned_open_edges.into_iter() {
            let node_before = self.edge_estimation.open_edges_before.remove(&open_idx).expect("There must be one before!");
            let edge = Edge(node_before, open_node);
            self.edge_estimation.edges.push((open_idx.clone(), edge));
        }
    }

    fn reduce_to_main(&mut self, ctx: &mut Context<Training>) {
        self.reintegrate_remaining_open_edges();
        if self.cluster_nodes.get_own_idx().eq(&0) {
            let msg = EdgeReductionMessage {
                own: true,
                ..Default::default()
            };

            ctx.address().do_send(msg);
        } else {
            let msg = EdgeReductionMessage {
                edges: self.edge_estimation.edges.clone(),
                own: false
            };

            let main_addr = self.cluster_nodes.get_as(&0, "Training").expect("There should be a main node!");
            main_addr.do_send(msg);
            ctx.address().do_send(EdgeEstimationDone);
        }
    }

    /// reduce-calculate edge_in_time for all received edges
    /// maybe send offsets too for sorting
    fn finalize_edge_estimation(&mut self, ctx: &mut Context<Training>) {
        for msg in &self.edge_estimation.received_reduction_messages {
            if !msg.own {
                self.edge_estimation.edges.extend(msg.edges.clone());
            }
        }
        self.edge_estimation.received_reduction_messages = vec![];
        self.edge_estimation.edges.sort_by(|(point_id_a, _), (point_id_b, _)| point_id_a.partial_cmp(point_id_b).unwrap());

        println!("edges {}", self.edge_estimation.edges.len());
        ctx.address().do_send(EdgeEstimationDone);
    }
}


impl Handler<EdgeReductionMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: EdgeReductionMessage, ctx: &mut Self::Context) -> Self::Result {
        self.edge_estimation.received_reduction_messages.push(msg);
        if self.edge_estimation.received_reduction_messages.len() == self.cluster_nodes.len_incl_own() {
            self.finalize_edge_estimation(ctx);
        }
    }
}

impl Handler<EdgeRotationMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: EdgeRotationMessage, ctx: &mut Self::Context) -> Self::Result {
        if !self.edge_estimation.rotation_protocol.received(&msg) {
            return
        }

        let open_edges = self.merging_rotated_edges(msg.open_edges);

        if self.edge_estimation.rotation_protocol.is_running() {
            self.edge_estimation.send_edges = open_edges;
            self.rotate_edges(ctx);
        } else {
            ctx.address().do_send(EdgeEstimationDone);
        }
    }
}
