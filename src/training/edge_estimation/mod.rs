mod helper;
mod messages;
#[cfg(test)]
mod tests;

use actix::prelude::*;
use actix::dev::MessageResponse;
use ndarray::{Dim, Axis};
use std::collections::HashMap;

use crate::training::Training;
use crate::training::edge_estimation::helper::EdgeEstimationHelper;
pub use crate::training::edge_estimation::messages::{EdgeEstimationHelperResponse, EdgeEstimationHelperTask, EdgeEstimationDone};
use crate::training::intersection_calculation::Transition;
use crate::utils::{HelperProtocol, Edge, NodeName};
use crate::meanshift::DistanceMeasure;
use std::fmt::{Display, Formatter, Result};
use std::time::Instant;
use std::sync::Arc;
use crate::messages::PoisonPill;

#[derive(Default)]
pub struct EdgeEstimation {
    pub edges: Vec<Edge>,
    pub edge_in_time: Vec<usize>,
    pub nodes: Vec<Option<NodeName>>,
    helpers: Option<Addr<EdgeEstimationHelper>>,
    current_transition_id: usize,
    task_id: usize,
    helper_protocol: HelperProtocol,
    start_time: Option<Instant>
}

pub trait EdgeEstimator {
    fn estimate_edges(&mut self);
    fn estimate_edges_parallel(&mut self, source: Recipient<EdgeEstimationHelperResponse>);
    fn send_next_batch(&mut self);
}

impl EdgeEstimator for Training {
    fn estimate_edges(&mut self) {
        self.edge_estimation.start_time = Some(Instant::now());

        let len_dataset = self.dataset_stats.as_ref().expect("DatasetStats should've been set by now!").n.unwrap();
        println!("len dataset {}", len_dataset);
        let distance_measure = DistanceMeasure::SquaredEuclidean.call();
        let mut previous_node: Option<NodeName> = None;

        for current_transition_id in 0..len_dataset {
            let current_transition = Transition(current_transition_id, current_transition_id + 1);
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
                        self.edge_estimation.nodes.push(Some(previous_node.as_ref().unwrap().clone()));
                    }
                    // todo: that doesn't work in a distributed case, because we have different number of global edges, we can calculate this locally and than aggregate on the main node
                    self.edge_estimation.edge_in_time.push(self.edge_estimation.edges.len());
                },
                None => ()
            }
        }

        match previous_node {
            Some(previous) => self.edge_estimation.nodes.push(Some(previous)),
            None => ()
        }

        let duration = self.edge_estimation.start_time.as_ref().unwrap().elapsed();
        println!("Non-parallel time: {:?}", duration);
    }

    fn estimate_edges_parallel(&mut self, source: Recipient<EdgeEstimationHelperResponse>) {
        self.edge_estimation.start_time = Some(Instant::now());
        let len = self.intersection_calculation.intersections.len();
        self.edge_estimation.helper_protocol.n_total = len;
        self.edge_estimation.edge_in_time = vec![0; len];
        let distance_measure = DistanceMeasure::SquaredEuclidean;
        let intersections = self.intersection_calculation.intersections.clone();
        let intersection_coords = self.intersection_calculation.intersection_coords.clone();
        let nodes = self.node_estimation.nodes.clone();
        self.edge_estimation.helpers = Some(SyncArbiter::start(self.parameters.n_threads, move || {
            EdgeEstimationHelper {
                source: source.clone(),
                distance_measure: distance_measure.clone(),
                intersections: intersections.clone(),
                intersection_coords: intersection_coords.clone(),
                nodes: nodes.clone()
            }
        }));
        self.send_next_batch();
    }

    fn send_next_batch(&mut self) {
        let mut current_transition = Transition(self.edge_estimation.current_transition_id, self.edge_estimation.current_transition_id + 1);
        let helpers = self.edge_estimation.helpers.as_ref().unwrap();

        let n_send = self.edge_estimation.helper_protocol.tasks_left()
            .min(self.parameters.n_threads - self.edge_estimation.helper_protocol.n_sent);

        for _ in 0..n_send {
            match self.intersection_calculation.intersections.get(&current_transition) {
                Some(segment_ids) => {
                    helpers.do_send(EdgeEstimationHelperTask { task_id: self.edge_estimation.task_id, transition: current_transition.clone() });
                    self.edge_estimation.helper_protocol.sent();
                    self.edge_estimation.nodes.extend_from_slice(vec![None; segment_ids.len()].as_slice());
                    self.edge_estimation.task_id += segment_ids.len();
                },
                None => ()
            }
            self.edge_estimation.current_transition_id += 1;
            current_transition = Transition(self.edge_estimation.current_transition_id, self.edge_estimation.current_transition_id + 1);
        }
    }
}

impl Handler<EdgeEstimationHelperResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: EdgeEstimationHelperResponse, ctx: &mut Self::Context) -> Self::Result {
        self.edge_estimation.helper_protocol.received();

        self.edge_estimation.edge_in_time[msg.transition.0] = msg.node_names.len() - if msg.transition.0 == 0 { 1 } else { 0 };

        for (i, node_name) in msg.node_names.into_iter().enumerate() {
            let task_id = msg.task_id + i;
            self.edge_estimation.nodes[task_id] = Some(node_name.clone());

            // incoming edge
            if task_id > 0 {
                match self.edge_estimation.nodes.get(task_id - 1) {
                    Some(optional_node) => match optional_node {
                        Some(node) => {
                            self.edge_estimation.edges.push(Edge(node.clone(), node_name.clone()));
                        }
                        None => ()
                    },
                    None => ()
                }
            }

            // outgoing edge
            match self.edge_estimation.nodes.get(task_id + 1) {
                Some(optional_node) => match optional_node {
                    Some(node) => {
                        self.edge_estimation.edges.push(Edge(node_name.clone(), node.clone()));
                    }
                    None => ()
                },
                None => ()
            }
        }

        if self.edge_estimation.helper_protocol.are_tasks_left() {
            self.send_next_batch();
        }
        if !self.edge_estimation.helper_protocol.is_running() {
            self.edge_estimation.edge_in_time = self.edge_estimation.edge_in_time.iter()
            .scan(0, |acc, &x| {
                *acc = *acc + x;
                Some(*acc)
            })
            .collect();
            let duration = self.edge_estimation.start_time.as_ref().unwrap().elapsed();
            println!("Parallel time: {:?}", duration);
            self.edge_estimation.helpers.as_ref().unwrap().do_send(PoisonPill);
            ctx.address().do_send(EdgeEstimationDone);
        }
    }
}
