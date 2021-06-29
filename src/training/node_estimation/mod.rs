use std::collections::HashMap;
use crate::training::intersection_calculation::{SegmentID, Transition};
use ndarray::{Array1, ArrayView1, stack_new_axis, Axis, Array2};
use crate::training::Training;
use crate::meanshift::{MeanShift, MeanShiftResponse, MeanShiftMessage};
use actix::{Addr, Handler, Actor, Recipient, AsyncContext};
use actix::dev::MessageResponse;

#[derive(Default)]
pub struct NodeEstimation {
    nodes: HashMap<SegmentID, Array2<f32>>,
    meanshift: Option<Addr<MeanShift>>,
    current_segment_id: usize
}

pub trait NodeEstimator {
    fn estimate_nodes(&mut self, source: Recipient<MeanShiftResponse>);
}

impl NodeEstimator for Training {
    fn estimate_nodes(&mut self, source: Recipient<MeanShiftResponse>) {
        let segment_id = self.node_estimation.current_segment_id;

        if self.node_estimation.current_segment_id < self.parameters.rate {
            match self.intersection_calculation.intersection_coords.get(&segment_id) {
                Some(segment) => {
                    let intersections: Vec<ArrayView1<f32>> = segment.values().map(|x| x.view()).collect();
                    let data = stack_new_axis(Axis(0), intersections.as_slice()).unwrap();
                    self.node_estimation.meanshift = Some(MeanShift::new(self.parameters.n_threads).start());
                    self.node_estimation.meanshift.as_ref().unwrap().do_send(MeanShiftMessage { source: Some(source.clone()), data });
                },
                None => {
                    self.node_estimation.current_segment_id += 1;
                    self.estimate_nodes(source);
                }
            }
        }
    }
}

impl Handler<MeanShiftResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftResponse, ctx: &mut Self::Context) -> Self::Result {
        self.node_estimation.nodes.insert(self.node_estimation.current_segment_id, msg.cluster_centers).unwrap();
        self.node_estimation.current_segment_id += 1;
        self.estimate_nodes(ctx.address().recipient());
    }
}
