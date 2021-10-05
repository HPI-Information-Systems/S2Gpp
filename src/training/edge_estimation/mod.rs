mod messages;
#[cfg(test)]
mod tests;
mod edges_orderer;

use actix::prelude::*;
use std::collections::HashMap;
use crate::training::Training;
pub use crate::training::edge_estimation::messages::{EdgeEstimationDone, EdgeReductionMessage};
use crate::utils::{Edge, NodeName};
use num_integer::Integer;
use crate::training::edge_estimation::edges_orderer::EdgesOrderer;
use crate::training::edge_estimation::messages::EdgeRotationMessage;


#[derive(Default)]
pub struct EdgeEstimation {
    /// [(point_id, Edge)]
    pub edges: Vec<(usize, Edge)>,
    /// point id -> node
    open_edges: HashMap<usize, NodeName>,
    /// cluster node -> [(point_id, node), ...]
    send_edges: HashMap<usize, Vec<(usize, NodeName)>>,
    received_reduction_messages: Vec<EdgeReductionMessage>
}

pub trait EdgeEstimator {
    fn estimate_edges(&mut self, ctx: &mut Context<Training>);
    fn connect_nodes(&mut self);
    fn rotate_edges(&mut self, ctx: &mut Context<Training>);
    fn reduce_to_main(&mut self, ctx: &mut Context<Training>);
    fn finalize_edge_estimation(&mut self, ctx: &mut Context<Training>);
}

impl EdgeEstimator for Training {
    fn estimate_edges(&mut self, ctx: &mut Context<Training>) {
        self.connect_nodes();
        self.rotate_edges(ctx);
    }

    fn connect_nodes(&mut self) {
        let len_dataset = self.dataset_stats.as_ref().expect("DatasetStats should've been set by now!").n.unwrap();
        let segments_per_node = (self.parameters.rate as f32 / self.cluster_nodes.len_incl_own() as f32).floor() as usize;
        let has_all_segments = segments_per_node == self.parameters.rate;

        let mut previous_node: Option<NodeName> = None;

        for point_id in 0..len_dataset {
            match self.segmentation.segment_index.get(&point_id) {
                Some(transition_id) => {
                    match self.node_estimation.nodes_by_transition.get(transition_id) {
                        Some(intersection_nodes) => {
                            let mut edges = EdgesOrderer::new(point_id, previous_node.is_some());
                            for current_node in intersection_nodes {
                                edges.add_node(&previous_node, current_node);

                                if current_node.0.mod_floor(&segments_per_node).eq(&(segments_per_node - 1)) &&
                                    !has_all_segments { // last segment
                                    previous_node = None;
                                    let node_id = current_node.0 / segments_per_node;
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
                },
                None => ()
            }
        }
    }

    fn rotate_edges(&mut self, ctx: &mut Context<Training>) {
        for (cluster_node, nodes) in self.edge_estimation.send_edges.iter() {
            let addr = self.cluster_nodes.get(cluster_node).expect(&format!("This cluster node id does not exist: {}", cluster_node));
            addr.do_send(EdgeRotationMessage { open_edges: nodes.clone() })
        }
        if self.edge_estimation.open_edges.is_empty() { self.reduce_to_main(ctx) }
    }

    fn reduce_to_main(&mut self, ctx: &mut Context<Training>) {
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

            let main_addr = self.cluster_nodes.get_main_node().expect("There should be a main node!");
            main_addr.do_send(msg);
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
        for (point_id, node) in msg.open_edges {
            let mut next_point = point_id + 1;
            loop {
                match self.edge_estimation.open_edges.remove(&next_point) {
                    None => {
                        next_point = next_point + 1;
                    },
                    Some(next_node) => {
                        self.edge_estimation.edges.push((point_id, Edge(node, next_node)));
                    }
                }
            }
        }
        if self.edge_estimation.open_edges.is_empty() {
            self.reduce_to_main(ctx);
        }
    }
}
