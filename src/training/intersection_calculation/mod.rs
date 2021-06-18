mod helper;
mod messages;
#[cfg(test)]
mod tests;

use actix::{Addr, SyncArbiter, Handler, Recipient, AsyncContext};
use actix::dev::MessageResponse;
use ndarray::{Array1, arr1, stack_new_axis, Axis, concatenate, ArrayBase, Array2, arr2, OwnedRepr, Array, ArrayView1};
use num_traits::real::Real;
use std::collections::HashMap;
use std::collections::hash_map::RandomState;
use std::f32::consts::PI;
use std::ops::{Range};
use crate::training::intersection_calculation::helper::IntersectionCalculationHelper;
pub use crate::training::intersection_calculation::messages::{IntersectionResultMessage, IntersectionTaskMessage, IntersectionCalculationDone};
use crate::training::segmenter::{SegmentedPointWithId, PointWithId};
use crate::training::Training;
use crate::utils::PolarCoords;
use crate::messages::PoisonPill;
use ndarray_stats::QuantileExt;
use std::cmp::Ordering::Equal;
use num_integer::Integer;

#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct Transition(usize, usize);
pub type SegmentID = usize;


pub struct IntersectionCalculation {
    pub intersections: HashMap<Transition, Vec<SegmentID>>,
    pub intersection_coords: HashMap<SegmentID, HashMap<Transition, Array1<f32>>>,
    pub helpers: Option<Addr<IntersectionCalculationHelper>>,
    pub pairs: Vec<(Transition, SegmentID, Array2<f32>, Array2<f32>)>,
    pub n_total: usize,
    pub n_sent: usize,
    pub n_received: usize
}


pub trait IntersectionCalculator {
    fn calculate_intersections(&mut self, rec: Recipient<IntersectionResultMessage>);
    fn distribute_intersection_tasks(&mut self, rec: Recipient<IntersectionResultMessage>);
}


impl IntersectionCalculator for Training {
    fn calculate_intersections(&mut self, rec: Recipient<IntersectionResultMessage>) {
        let max_value = self.segmentation.own_segment.iter()
            .map(|x| x.point_with_id.coords.max().unwrap().max(x.point_with_id.coords.min().unwrap().abs()))
            .fold(0_f32, |a, b| { a.max(b) });

        let dims = self.segmentation.own_segment.get(0).unwrap().point_with_id.coords.len();

        let origin = arr1(vec![0_f32; dims].as_slice());
        let planes_end_points: Vec<Array1<f32>> = (0..self.parameters.rate).into_iter().map(|segment_id| {
            let polar = arr1(&[max_value, (2.0 * PI * segment_id as f32) / self.parameters.rate as f32]);
            let other_dims = arr1((2..dims).into_iter().map(|_| max_value).collect::<Vec<f32>>().as_slice());
            concatenate(Axis(0), &[polar.to_cartesian().view(), other_dims.view()]).unwrap()
        }).collect();

        let mut last: &SegmentedPointWithId = self.segmentation.own_segment.get(0).unwrap();
        for current in self.segmentation.own_segment.iter().skip(1) {
            if last.segment_id.ne(&current.segment_id) {
                let line_points = stack_new_axis(Axis(0), &[
                    last.point_with_id.coords.view(),
                    current.point_with_id.coords.view()
                ]).unwrap();

                let transition = Transition(last.point_with_id.id, current.point_with_id.id);
                let mut segment_ids = vec![];

                let raw_diff = current.segment_id as isize - last.segment_id as isize;
                let raw_diff_counter = (self.parameters.rate + current.segment_id) as isize - last.segment_id as isize;
                let mut segment_diff = raw_diff.abs() as usize;
                let half_rate = self.parameters.rate.div_floor(&2);

                let valid_direction = (0 <= raw_diff && raw_diff <= half_rate as isize) ||
                    (raw_diff < 0 && (0 <= raw_diff_counter && raw_diff_counter <= half_rate as isize));

                if valid_direction {
                    if segment_diff > half_rate {
                        if current.segment_id > half_rate {
                            segment_diff = (last.segment_id as isize - (-(self.parameters.rate as isize) + current.segment_id as isize)).abs() as usize;
                        } else if last.segment_id > half_rate {
                            segment_diff = (current.segment_id as isize - (-(self.parameters.rate as isize) + last.segment_id as isize)).abs() as usize;
                        }
                    }
                    segment_diff = segment_diff.min(half_rate);

                    for segment_lag in 1..(segment_diff) + 1 {
                        let segment_id = (last.segment_id + segment_lag).mod_floor(&self.parameters.rate);
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
                        self.intersection_calculation.pairs.push((transition.clone(), segment_id, line_points.clone(), plane_points));
                    }
                }

                let transition = Transition(last.point_with_id.id, current.point_with_id.id);

                match self.intersection_calculation.intersections.get_mut(&transition) {
                    Some(internal_segment_ids) => internal_segment_ids.extend(segment_ids),
                    None => { self.intersection_calculation.intersections.insert(transition.clone(), segment_ids); }
                };
            }


            last = current;
        }
        self.intersection_calculation.n_total = self.intersection_calculation.pairs.len();

        self.intersection_calculation.helpers = Some(SyncArbiter::start(self.parameters.n_threads, move || {IntersectionCalculationHelper {}}));

        self.distribute_intersection_tasks(rec);
    }

    fn distribute_intersection_tasks(&mut self, rec: Recipient<IntersectionResultMessage>) {
        for _ in 0..(self.parameters.n_threads - self.intersection_calculation.n_sent) {
            let pair = self.intersection_calculation.pairs.pop();
            match pair {
                None => {}
                Some((transition, segment_id, line_points, plane_points)) => {
                    self.intersection_calculation.helpers.as_ref().unwrap().do_send(IntersectionTaskMessage {
                        transition,
                        segment_id,
                        line_points,
                        plane_points,
                        source: rec.clone()
                    });
                    self.intersection_calculation.n_sent += 1;
                }
            }
        }
    }
}


impl Handler<IntersectionResultMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: IntersectionResultMessage, ctx: &mut Self::Context) -> Self::Result {
        self.intersection_calculation.n_sent -= 1;
        self.intersection_calculation.n_received += 1;

        match self.intersection_calculation.intersection_coords.get_mut(&msg.segment_id) {
            Some(transition_coord) => { transition_coord.insert(msg.transition, msg.intersection); },
            None => {
                let mut transition_coord = HashMap::new();
                transition_coord.insert(msg.transition, msg.intersection);
                self.intersection_calculation.intersection_coords.insert(msg.segment_id, transition_coord);
            }
        };

        if self.intersection_calculation.n_received < self.intersection_calculation.n_total {
            self.distribute_intersection_tasks(ctx.address().recipient());
        } else {
            self.intersection_calculation.helpers.as_ref().unwrap().do_send(PoisonPill);
            ctx.address().do_send(IntersectionCalculationDone );
        }
    }
}
