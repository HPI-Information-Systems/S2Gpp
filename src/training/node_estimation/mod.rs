mod messages;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use crate::training::intersection_calculation::{SegmentID, Transition};
use ndarray::{Array1, ArrayView1, stack_new_axis, Axis, Array2};
use crate::training::Training;
use crate::meanshift::{MeanShift, MeanShiftResponse, MeanShiftMessage};
use actix::{Addr, Handler, Actor, Recipient, AsyncContext};
use actix::dev::MessageResponse;

use indicatif::ProgressBar;
use crate::utils::ConsoleLogger;
pub use crate::training::node_estimation::messages::NodeEstimationDone;

#[derive(Default)]
pub struct NodeEstimation {
    pub nodes: HashMap<SegmentID, Array2<f32>>,
    pub meanshift: Option<Addr<MeanShift>>,
    pub current_segment_id: usize,
    pub progress_bar: Option<ProgressBar>
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

        match self.intersection_calculation.intersection_coords.get(&segment_id) {
            Some(segment) => {
                let intersections: Vec<ArrayView1<f32>> = segment.values().map(|x| x.view()).collect();
                let data = stack_new_axis(Axis(0), intersections.as_slice()).unwrap();
                self.node_estimation.meanshift = Some(MeanShift::new(self.parameters.n_threads).start());
                self.node_estimation.meanshift.as_ref().unwrap().do_send(MeanShiftMessage { source: Some(source.clone()), data });
            },
            None => {
                source.do_send(MeanShiftResponse { cluster_centers: Default::default() });
            }
        }
    }
}

impl Handler<MeanShiftResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftResponse, ctx: &mut Self::Context) -> Self::Result {
        if !msg.cluster_centers.is_empty() {
            self.node_estimation.nodes.insert(self.node_estimation.current_segment_id,
                                              msg.cluster_centers);
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
