mod messages;
#[cfg(test)]
mod tests;

use actix::prelude::*;
use actix::dev::MessageResponse;
use ndarray::{Dim, Axis};
use std::collections::HashMap;

use crate::training::Training;
pub use crate::training::edge_estimation::messages::{EdgeEstimationDone, EdgeReductionMessage};
use crate::training::intersection_calculation::Transition;
use crate::utils::{HelperProtocol, Edge, NodeName};
use std::fmt::{Display, Formatter, Result};
use std::time::Instant;
use std::sync::Arc;
use crate::messages::PoisonPill;
use log::*;


#[derive(Default)]
pub struct EdgeEstimation {
    pub edges: Vec<Edge>,
    pub edge_in_time: Vec<usize>,
    pub nodes: Vec<NodeName>,
    received_reduction_messages: usize
}

pub trait EdgeEstimator {
    fn estimate_edges(&mut self);
    fn finalize_edge_estimation(&mut self, ctx: &mut Context<Training>);
}

impl EdgeEstimator for Training {
    fn estimate_edges(&mut self) {
        let len_dataset = self.dataset_stats.as_ref().expect("DatasetStats should've been set by now!").n.unwrap();
        println!("len dataset {}", len_dataset);
        let mut previous_node: Option<NodeName> = None;

        for point_id in 0..len_dataset {
            match self.segmentation.segment_index.get(&point_id) {
                Some(transition_id) => {
                    let intersection_node = self.node_estimation.nodes_by_transition.get(transition_id).expect("This intersection node was not calculated!");
                    let current_node = NodeName(intersection_node.segment, intersection_node.cluster_id);
                    match &previous_node {
                        None => (),
                        Some(previous) => self.edge_estimation.edges.push(Edge(previous.clone(), current_node.clone()))
                    }
                    previous_node = Some(current_node);
                    self.edge_estimation.nodes.push(previous_node.as_ref().unwrap().clone());
                    self.edge_estimation.edge_in_time.push(self.edge_estimation.edges.len());
                },
                None =>()
            }
        }

        match previous_node {
            Some(previous) => self.edge_estimation.nodes.push(previous),
            None => ()
        }

        let main_addr = self.cluster_nodes.get_main_node().expect("There should be a main node!");
        main_addr.do_send(EdgeReductionMessage {
            edges: self.edge_estimation.edges.clone(),
            edge_in_time: self.edge_estimation.edge_in_time.clone(),
            nodes: self.edge_estimation.nodes.clone()
        })
    }

    fn finalize_edge_estimation(&mut self, ctx: &mut Context<Training>) {
        self.edge_estimation.edge_in_time = self.edge_estimation.edge_in_time.iter()
            .scan(0, |acc, &x| {
                *acc = *acc + x;
                Some(*acc)
            })
            .collect();

        ctx.address().do_send(EdgeEstimationDone);
    }
}


impl Handler<EdgeReductionMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: EdgeReductionMessage, ctx: &mut Self::Context) -> Self::Result {
        todo!()
    }
}
