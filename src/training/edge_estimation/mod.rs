mod helper;
mod messages;

use actix::prelude::*;
use actix::dev::MessageResponse;
use ndarray::{Dim, Axis};
use std::collections::HashMap;

use crate::training::Training;
use crate::training::edge_estimation::helper::EdgeEstimationHelper;
use crate::training::edge_estimation::messages::EdgeEstimationHelperResponse;
use crate::training::intersection_calculation::Transition;
use crate::utils::HelperProtocol;
use crate::meanshift::DistanceMeasure;

#[derive(Clone)]
pub struct NodeName(pub usize, pub usize);
pub struct Edge(pub NodeName, pub NodeName);

#[derive(Default)]
pub struct EdgeEstimation {
    edges: Vec<Edge>,
    edge_in_time: HashMap<usize, usize>,
    nodes: Vec<NodeName>,
    helpers: Option<Addr<EdgeEstimationHelper>>,
    current_transition_id: usize,
    helper_protocol: HelperProtocol
}

pub trait EdgeEstimator {
    fn estimate_edges(&mut self, source: Recipient<EdgeEstimationHelperResponse>);
}

impl EdgeEstimator for Training {
    fn estimate_edges(&mut self, source: Recipient<EdgeEstimationHelperResponse>) {
        let current_transition = Transition(self.edge_estimation.current_transition_id, self.edge_estimation.current_transition_id + 1);
        let distance_measure = DistanceMeasure::SquaredEuclidean.call();
        let mut previous_node: Option<NodeName> = None;

        match self.intersection_calculation.intersections.get(&current_transition) {
            Some(segment_ids) => {
                for segment_id in segment_ids {
                    let distance = self.intersection_calculation.intersection_coords.get(segment_id).unwrap().get(&current_transition).unwrap();
                    let nodes = self.node_estimation.nodes.get(segment_id).unwrap();
                    let distance = distance.broadcast(Dim([nodes.shape()[0], distance.len()])).unwrap();
                    let closest_id: usize = nodes.axis_iter(Axis(0))
                        .zip(distance.axis_iter(Axis(0)))
                        .map(|(node, dist)| distance_measure(&node.to_vec(), &dist.to_vec()))
                        .enumerate()
                        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx).unwrap();
                    let current_node = NodeName(segment_id.clone(), closest_id);

                    match &previous_node {
                        None => { previous_node = Some(current_node) },
                        Some(previous) => {
                            self.edge_estimation.edges.push(Edge(previous.clone(), current_node.clone()));
                            previous_node = Some(current_node)
                        }
                    }
                    self.edge_estimation.nodes.push(previous_node.as_ref().unwrap().clone());

                }
                self.edge_estimation.edge_in_time.insert(self.edge_estimation.current_transition_id, self.edge_estimation.edges.len());
            },
            None => ()
        }

        match previous_node {
            Some(previous) => self.edge_estimation.nodes.push(previous),
            None => ()
        }
    }
}

impl Handler<EdgeEstimationHelperResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: EdgeEstimationHelperResponse, ctx: &mut Self::Context) -> Self::Result {
        todo!()
    }
}
