mod messages;
#[cfg(test)]
pub(crate) mod tests;
mod data_structures;

use std::collections::HashMap;
use crate::training::intersection_calculation::{SegmentID};
use ndarray::{ArrayView1, stack_new_axis, Axis, Array2};
use crate::training::Training;
use actix::{Addr, Handler, Actor, Recipient, AsyncContext};


use indicatif::ProgressBar;

pub use crate::training::node_estimation::messages::NodeEstimationDone;
use crate::training::node_estimation::data_structures::IntersectionNode;
use meanshift_rs::{MeanShiftResponse, MeanShiftMessage, MeanShiftActor};

#[derive(Default)]
pub struct NodeEstimation {
    pub nodes: HashMap<SegmentID, Array2<f32>>,
    pub nodes_by_transition: HashMap<usize, Vec<IntersectionNode>>,
    pub meanshift: Option<Addr<MeanShiftActor>>,
    pub(crate) last_transitions: Vec<usize>,
    pub(crate) current_segment_id: usize,
    pub(crate) progress_bar: Option<ProgressBar>
}

pub trait NodeEstimator {
    fn estimate_nodes(&mut self, source: Recipient<MeanShiftResponse>);
}

impl NodeEstimator for Training {
    fn estimate_nodes(&mut self, source: Recipient<MeanShiftResponse>) {
        match &self.node_estimation.progress_bar {
            None => {
                self.node_estimation.progress_bar = Some(ProgressBar::new(self.parameters.rate as u64));
            }
            Some(pb) => {pb.inc(1)}
        }

        let segment_id = self.node_estimation.current_segment_id;

        match self.intersection_calculation.intersection_coords_by_segment.get(&segment_id) {
            Some(segment) => {
                self.node_estimation.last_transitions = segment.keys().map(|x| x.clone()).collect();
                let intersections: Vec<ArrayView1<f32>> = segment.values().map(|x| x.view()).collect();
                let data = stack_new_axis(Axis(0), intersections.as_slice()).unwrap();
                self.node_estimation.meanshift = Some(MeanShiftActor::new(self.parameters.n_threads).start());
                self.node_estimation.meanshift.as_ref().unwrap().do_send(MeanShiftMessage { source: Some(source.clone()), data });
            },
            None => {
                source.do_send(MeanShiftResponse { cluster_centers: Default::default(), labels: vec![] }).unwrap();
            }
        }
    }
}

impl Handler<MeanShiftResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftResponse, ctx: &mut Self::Context) -> Self::Result {
        if !msg.cluster_centers.is_empty() {
            let current_segment_id = self.node_estimation.current_segment_id;
            let last_transitions = self.node_estimation.last_transitions.clone();
            self.node_estimation.last_transitions = vec![];
            self.node_estimation.nodes.insert(current_segment_id, msg.cluster_centers);

            for (transition, label) in last_transitions.into_iter().zip(msg.labels)  {
                let node = IntersectionNode { segment: current_segment_id, cluster_id: label };
                match self.node_estimation.nodes_by_transition.get_mut(&transition) {
                    Some(nodes) => nodes.push(node),
                    None => { self.node_estimation.nodes_by_transition.insert(transition.clone(), vec![node]); }
                }
            }
        }
        self.node_estimation.current_segment_id += 1;

        if self.node_estimation.current_segment_id < self.parameters.rate {
            self.estimate_nodes(ctx.address().recipient());
        } else {
            let pb = self.node_estimation.progress_bar.as_ref().unwrap();
            pb.inc(1);
            pb.finish_and_clear();
            ctx.address().do_send(NodeEstimationDone);
        }
    }
}
