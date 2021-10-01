mod helper;
mod messages;
#[cfg(test)]
mod tests;
mod data_structures;

use actix::{Addr, SyncArbiter, Handler, Recipient, AsyncContext};

use ndarray::{Array1, arr1, stack_new_axis, Axis, concatenate, Array2};
use std::collections::HashMap;

use std::f32::consts::PI;

use crate::training::intersection_calculation::helper::IntersectionCalculationHelper;
pub use crate::training::intersection_calculation::messages::{IntersectionResultMessage, IntersectionTaskMessage, IntersectionCalculationDone};

use crate::training::Training;
use crate::utils::{PolarCoords, HelperProtocol};
use crate::messages::PoisonPill;
use ndarray_stats::QuantileExt;

use num_integer::Integer;
pub use crate::training::intersection_calculation::data_structures::{Transition, IntersectionsByTransition};
use indicatif::ProgressBar;
use ndarray_linalg::Norm;
use log::*;

pub type SegmentID = usize;

#[derive(Default)]
pub struct IntersectionCalculation {
    /// Which segments are intersected by a transition?
    pub intersections: HashMap<usize, Vec<SegmentID>>,
    /// Which transitions cut a segment and where?
    pub intersection_coords_by_segment: HashMap<SegmentID, IntersectionsByTransition>,
    pub helpers: Option<Addr<IntersectionCalculationHelper>>,
    /// Collects tasks for helper actors.
    pub pairs: Vec<(usize, SegmentID, Array2<f32>, Array2<f32>)>,
    pub helper_protocol: HelperProtocol,
    pub recipient: Option<Recipient<IntersectionCalculationDone>>,
    pub(crate) progress_bar: Option<ProgressBar>
}


pub trait IntersectionCalculator {
    fn calculate_intersections(&mut self, rec: Recipient<IntersectionResultMessage>);
    fn distribute_intersection_tasks(&mut self, rec: Recipient<IntersectionResultMessage>);
}


impl IntersectionCalculator for Training {
    fn calculate_intersections(&mut self, rec: Recipient<IntersectionResultMessage>) {
        let max_value = self.segmentation.segments.iter()
            .map(|x| {
                x.from.point_with_id.coords
                    .max().unwrap()
                    .max(x.from.point_with_id.coords.min().unwrap().abs())
                    .max(x.to.point_with_id.coords
                        .max().unwrap()
                        .max(x.to.point_with_id.coords.min().unwrap().abs())
                        .abs()
                    )
            }
            )
            .fold(0_f32, |a, b| { a.max(b) });
        let radius = arr1(&[max_value, max_value]).norm();

        debug!("max value {} radius {}", max_value, radius);

        let dims = self.segmentation.segments.get(0).expect("could not generate segments").from.point_with_id.coords.len();

        let origin = arr1(vec![0_f32; dims].as_slice());
        let planes_end_points: Vec<Array1<f32>> = (0..self.parameters.rate).into_iter().map(|segment_id| {
            let polar = arr1(&[radius, (2.0 * PI * segment_id as f32) / self.parameters.rate as f32]);
            let other_dims = arr1((2..dims).into_iter().map(|_| max_value).collect::<Vec<f32>>().as_slice());
            concatenate(Axis(0), &[polar.to_cartesian().view(), other_dims.view()]).unwrap()
        }).collect();

        for (transition_id, transition) in self.segmentation.segments.iter().enumerate() {
            if transition.crosses_segments() {
                let line_points = stack_new_axis(Axis(0), &[
                    transition.from.point_with_id.coords.view(),
                    transition.to.point_with_id.coords.view()
                ]).unwrap();

                let mut segment_ids = vec![];

                let raw_diff = transition.to.segment_id as isize - transition.from.segment_id as isize;
                let raw_diff_counter = (self.parameters.rate + transition.to.segment_id) as isize - transition.from.segment_id as isize;
                let mut segment_diff = raw_diff.abs() as usize;
                let half_rate = self.parameters.rate.div_floor(&2);

                let valid_direction = (0 <= raw_diff && raw_diff <= half_rate as isize) ||
                    (raw_diff < 0 && (0 <= raw_diff_counter && raw_diff_counter <= half_rate as isize));

                if valid_direction {
                    if segment_diff > half_rate {
                        if transition.to.segment_id > half_rate {
                            segment_diff = (transition.from.segment_id as isize - (-(self.parameters.rate as isize) + transition.to.segment_id as isize)).abs() as usize;
                        } else if transition.from.segment_id > half_rate {
                            segment_diff = (transition.to.segment_id as isize - (-(self.parameters.rate as isize) + transition.from.segment_id as isize)).abs() as usize;
                        }
                    }
                    segment_diff = segment_diff.min(half_rate);

                    for segment_lag in 1..(segment_diff) + 1 {
                        let segment_id = (transition.from.segment_id + segment_lag).mod_floor(&self.parameters.rate);
                        segment_ids.push(segment_id);

                        let mut arrays = vec![origin.view()];
                        let corner_points: Vec<Array1<f32>> = (2..dims).into_iter().map(|d| {
                            let mut corner_point = planes_end_points[segment_id].clone();
                            corner_point[d] = 0.;
                            corner_point
                        }).collect();

                        arrays.extend(corner_points.iter().map(|x| { x.view() }));
                        arrays.push(planes_end_points[segment_id].view());

                        let plane_points = stack_new_axis(Axis(0), arrays.as_slice()).unwrap();
                        self.intersection_calculation.pairs.push((transition_id, segment_id, line_points.clone(), plane_points));
                    }
                }

                match self.intersection_calculation.intersections.get_mut(&transition_id) {
                    Some(internal_segment_ids) => internal_segment_ids.extend(segment_ids),
                    None => { self.intersection_calculation.intersections.insert(transition_id, segment_ids); }
                };
            }
        }
        self.intersection_calculation.helper_protocol.n_total = self.intersection_calculation.pairs.len();
        self.intersection_calculation.progress_bar = Some(ProgressBar::new(self.intersection_calculation.helper_protocol.n_total as u64));

        self.intersection_calculation.helpers = Some(SyncArbiter::start(self.parameters.n_threads, move || {IntersectionCalculationHelper {}}));

        self.distribute_intersection_tasks(rec);
    }

    fn distribute_intersection_tasks(&mut self, rec: Recipient<IntersectionResultMessage>) {
        for _ in 0..(self.parameters.n_threads - self.intersection_calculation.helper_protocol.n_sent) {
            let pair = self.intersection_calculation.pairs.pop();
            match pair {
                None => {}
                Some((transition_id, segment_id, line_points, plane_points)) => {
                    self.intersection_calculation.helpers.as_ref().unwrap().do_send(IntersectionTaskMessage {
                        transition_id,
                        segment_id,
                        line_points,
                        plane_points,
                        source: rec.clone()
                    });
                    self.intersection_calculation.helper_protocol.sent();
                }
            }
        }
    }
}


impl Handler<IntersectionResultMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: IntersectionResultMessage, ctx: &mut Self::Context) -> Self::Result {
        self.intersection_calculation.helper_protocol.received();
        let pb = self.intersection_calculation.progress_bar.as_ref().unwrap();
        pb.inc(1);

        match self.intersection_calculation.intersection_coords_by_segment.get_mut(&msg.segment_id) {
            Some(transition_coord) => { transition_coord.insert(msg.transition_id, msg.intersection); },
            None => {
                let mut transition_coord = HashMap::new();
                transition_coord.insert(msg.transition_id, msg.intersection);
                self.intersection_calculation.intersection_coords_by_segment.insert(msg.segment_id, transition_coord);
            }
        };

        if self.intersection_calculation.helper_protocol.is_running() {
            self.distribute_intersection_tasks(ctx.address().recipient());
        } else {
            pb.finish_and_clear();

            self.intersection_calculation.helpers.as_ref().unwrap().do_send(PoisonPill);
            match &self.intersection_calculation.recipient {
                Some(rec) => { rec.do_send(IntersectionCalculationDone).unwrap() },
                None => ctx.address().do_send(IntersectionCalculationDone)
            }
        }
    }
}
