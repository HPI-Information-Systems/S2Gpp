mod messages;
#[cfg(test)]
pub(crate) mod tests;

use std::collections::HashMap;
use crate::training::intersection_calculation::{SegmentID};
use ndarray::{ArrayView1, stack_new_axis, Axis, Array2};
use crate::training::Training;
use actix::{Addr, Handler, Actor, Recipient, AsyncContext};

pub use crate::training::node_estimation::messages::NodeEstimationDone;
use meanshift_rs::{MeanShiftResponse, MeanShiftMessage, MeanShiftActor};
use num_integer::Integer;
use crate::utils::logging::progress_bar::S2GppProgressBar;
use crate::utils::NodeName;

#[derive(Default)]
pub struct NodeEstimation {
    pub nodes: HashMap<SegmentID, Array2<f32>>, //todo remove and build a smarter node index
    /// {point_id: \[node\]}
    pub nodes_by_point: HashMap<usize, Vec<NodeName>>,
    pub meanshift: Option<Addr<MeanShiftActor>>,
    pub(crate) last_start_points: Vec<usize>,
    pub(crate) current_segment_id: usize,
    pub(crate) progress_bar: S2GppProgressBar,
    pub(crate) source: Option<Recipient<NodeEstimationDone>>
}

pub trait NodeEstimator {
    fn estimate_nodes(&mut self, mean_shift_recipient: Recipient<MeanShiftResponse>);
}

impl NodeEstimator for Training {
    fn estimate_nodes(&mut self, mean_shift_recipient: Recipient<MeanShiftResponse>) {
        self.node_estimation.progress_bar.inc_or_set("info", self.parameters.rate.div_floor(&self.parameters.n_cluster_nodes));

        let segment_id = self.node_estimation.current_segment_id;

        match self.intersection_calculation.intersection_coords_by_segment.get(&segment_id) {
            Some(intersections_by_starting_point) => {

                self.node_estimation.last_start_points = intersections_by_starting_point.keys().map(|x| x.clone()).collect();
                let intersections: Vec<ArrayView1<f32>> = intersections_by_starting_point.values().map(|x| x.view()).collect();
                let data = stack_new_axis(Axis(0), intersections.as_slice()).unwrap();
                self.node_estimation.meanshift = Some(MeanShiftActor::new(self.parameters.n_threads).start());
                self.node_estimation.meanshift.as_ref().unwrap().do_send(MeanShiftMessage { source: Some(mean_shift_recipient.clone()), data });
            }
            None => {
                mean_shift_recipient.do_send(MeanShiftResponse { cluster_centers: Default::default(), labels: vec![] }).unwrap();
            }
        }
    }
}

impl Handler<MeanShiftResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftResponse, ctx: &mut Self::Context) -> Self::Result {
        if !msg.cluster_centers.is_empty() {
            let current_segment_id = self.node_estimation.current_segment_id;
            let last_start_points = self.node_estimation.last_start_points.clone();
            self.node_estimation.last_start_points.clear();
            self.node_estimation.nodes.insert(current_segment_id, msg.cluster_centers);

            for (last_start_point, label) in last_start_points.into_iter().zip(msg.labels)  {
                let node = NodeName(current_segment_id, label);
                match self.node_estimation.nodes_by_point.get_mut(&last_start_point) {
                    Some(nodes) => nodes.push(node),
                    None => { self.node_estimation.nodes_by_point.insert(last_start_point.clone(), vec![node]); }
                }
            }
        }
        self.node_estimation.current_segment_id += 1;

        if self.node_estimation.current_segment_id < self.parameters.rate {
            self.estimate_nodes(ctx.address().recipient());
        } else {
            self.node_estimation.progress_bar.inc();
            self.node_estimation.progress_bar.finish_and_clear();
            match &self.node_estimation.source {
                Some(source) => source.clone(),
                None => ctx.address().recipient()
            }.do_send(NodeEstimationDone).unwrap();
        }
    }
}
