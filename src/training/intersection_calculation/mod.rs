mod helper;
mod messages;
#[cfg(test)]
mod tests;

use actix::{Addr, SyncArbiter, Handler, Recipient, AsyncContext};

use ndarray::{Array1, arr1, stack_new_axis, Axis, concatenate, Array2};
use std::collections::HashMap;

use std::f32::consts::PI;

use crate::training::intersection_calculation::helper::IntersectionCalculationHelper;
pub(crate) use crate::training::intersection_calculation::messages::{IntersectionResultMessage, IntersectionTaskMessage, IntersectionCalculationDone, IntersectionRotationMessage};

use crate::training::Training;
use crate::utils::{PolarCoords, HelperProtocol};
use crate::messages::PoisonPill;
use ndarray_stats::QuantileExt;

use num_integer::Integer;
use ndarray_linalg::Norm;
use crate::data_store::intersection::{Intersection, MaterializedIntersection};
use crate::data_store::materialize::Materialize;
use crate::data_store::transition::{TransitionMixin, TransitionRef};
use crate::utils::logging::progress_bar::S2GppProgressBar;
use crate::utils::rotation_protocol::RotationProtocol;

pub type SegmentID = usize;

#[derive(Default)]
pub(crate) struct IntersectionCalculation {
    /// Intersections belonging to another cluster node
    pub foreign_intersections: HashMap<SegmentID, Vec<MaterializedIntersection>>,
    pub helpers: Option<Addr<IntersectionCalculationHelper>>,
    /// Collects tasks for helper actors.
    pub pairs: Vec<(TransitionRef, SegmentID, Array2<f32>, Array2<f32>)>,
    pub helper_protocol: HelperProtocol,
    pub recipient: Option<Recipient<IntersectionCalculationDone>>,
    pub(crate) progress_bar: S2GppProgressBar,
    pub rotation_protocol: RotationProtocol<IntersectionRotationMessage>,
}


pub(crate) trait IntersectionCalculator {
    fn calculate_intersections(&mut self, rec: Recipient<IntersectionResultMessage>);
    fn parallel_intersection_tasks(&mut self, rec: Recipient<IntersectionResultMessage>);
    fn rotate_foreign_assignments(&mut self, rec: Recipient<IntersectionCalculationDone>);
    fn assign_received_intersection(&mut self, segment_id: SegmentID, transition: TransitionRef, intersection: Array1<f32>);
}


impl IntersectionCalculator for Training {
    fn calculate_intersections(&mut self, rec: Recipient<IntersectionResultMessage>) {
        let max_value = self.data_store.get_transitions().iter()
            .map(|x| {
                x.get_from_point().get_coordinates()
                    .max().unwrap()
                    .max(x.get_from_point().get_coordinates().min().unwrap().abs())
                    .max(x.get_to_point().get_coordinates()
                        .max().unwrap()
                        .max(x.get_to_point().get_coordinates().min().unwrap().abs())
                        .abs()
                    )
            }
            )
            .fold(0_f32, |a, b| { a.max(b) });
        let radius = arr1(&[max_value, max_value]).norm();

        let dims = self.data_store.get_transitions().first().expect("Could not generate Segments").get_from_point().get_coordinates().len();

        let origin = arr1(vec![0_f32; dims].as_slice());
        let planes_end_points: Vec<Array1<f32>> = (0..self.parameters.rate).into_iter().map(|segment_id| {
            let polar = arr1(&[radius, (2.0 * PI * segment_id as f32) / self.parameters.rate as f32]);
            let other_dims = arr1((2..dims).into_iter().map(|_| max_value).collect::<Vec<f32>>().as_slice());
            concatenate(Axis(0), &[polar.to_cartesian().view(), other_dims.view()]).unwrap()
        }).collect();

        for transition in self.data_store.get_transitions() {
            let line_points = stack_new_axis(Axis(0), &[
                transition.get_from_point().get_coordinates().view(),
                transition.get_to_point().get_coordinates().view()
            ]).unwrap();

            let mut segment_ids = vec![];

            let mut segment_diff = transition.segment_diff();
            let half_rate = self.parameters.rate.div_floor(&2);

            if segment_diff > half_rate {
                if transition.get_to_segment() > half_rate {
                    segment_diff = (transition.get_from_segment() as isize - (-(self.parameters.rate as isize) + transition.get_to_segment() as isize)).abs() as usize;
                } else if transition.get_from_segment() > half_rate {
                    segment_diff = (transition.get_to_segment() as isize - (-(self.parameters.rate as isize) + transition.get_from_segment() as isize)).abs() as usize;
                }
            }
            segment_diff = segment_diff.min(half_rate);

            for segment_lag in 1..(segment_diff) + 1 {
                let segment_id = (transition.get_from_segment() + segment_lag).mod_floor(&self.parameters.rate);
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
        self.intersection_calculation.helper_protocol.n_total = self.intersection_calculation.pairs.len();
        self.intersection_calculation.progress_bar = S2GppProgressBar::new_from_len("info", self.intersection_calculation.helper_protocol.n_total);

        self.intersection_calculation.helpers = Some(SyncArbiter::start(self.parameters.n_threads, move || {IntersectionCalculationHelper {}}));

        self.parallel_intersection_tasks(rec);
    }

    fn parallel_intersection_tasks(&mut self, rec: Recipient<IntersectionResultMessage>) {
        for _ in 0..(self.parameters.n_threads - self.intersection_calculation.helper_protocol.n_sent) {
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
                    self.intersection_calculation.helper_protocol.sent();
                }
            }
        }
    }

    fn rotate_foreign_assignments(&mut self, rec: Recipient<IntersectionCalculationDone>) {
        match &self.cluster_nodes.get_next_idx() {
            Some(next_idx) => {
                let next_node = self.cluster_nodes.get_as(next_idx, "Training").unwrap();
                let intersection_coords_by_segment = self.intersection_calculation.foreign_intersections.clone();
                self.intersection_calculation.foreign_intersections.clear();
                next_node.do_send(IntersectionRotationMessage { intersection_coords_by_segment });
                self.intersection_calculation.rotation_protocol.sent();
            },
            None => { rec.do_send(IntersectionCalculationDone).unwrap(); }
        }
    }

    fn assign_received_intersection(&mut self, segment_id: SegmentID, transition: TransitionRef, intersection: Array1<f32>) {
        let own_id = self.cluster_nodes.get_own_idx();
        let assigned_id = self.segment_id_to_assignment(segment_id);

        let intersection = Intersection::new(transition, intersection, segment_id);

        if own_id.eq(&assigned_id) {
            self.data_store.add_intersection(intersection)
        } else {
            match &mut self.intersection_calculation.foreign_intersections.get_mut(&segment_id) {
                Some(transition_coord) => transition_coord.push(intersection.materialize()),
                None => {
                    self.intersection_calculation.foreign_intersections.insert(segment_id, vec![intersection.materialize()]);
                }
            };
        }
    }
}


impl Handler<IntersectionResultMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: IntersectionResultMessage, ctx: &mut Self::Context) -> Self::Result {
        self.intersection_calculation.helper_protocol.received();
        self.intersection_calculation.progress_bar.inc();

        self.assign_received_intersection(msg.segment_id, msg.transition, msg.intersection);

        if self.intersection_calculation.helper_protocol.is_running() {
            self.parallel_intersection_tasks(ctx.address().recipient());
        } else {
            self.intersection_calculation.progress_bar.finish_and_clear();

            self.intersection_calculation.helpers.as_ref().unwrap().do_send(PoisonPill);
            match &self.intersection_calculation.recipient {
                Some(rec) => { rec.do_send(IntersectionCalculationDone).unwrap() },
                None => {
                    self.intersection_calculation.rotation_protocol.start(self.cluster_nodes.len());
                    self.intersection_calculation.rotation_protocol.resolve_buffer(ctx.address().recipient());
                    self.rotate_foreign_assignments(ctx.address().recipient())
                }
            }
        }
    }
}


impl Handler<IntersectionRotationMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: IntersectionRotationMessage, ctx: &mut Self::Context) -> Self::Result {
        if !self.intersection_calculation.rotation_protocol.received(&msg) {
            return
        }

        let own_id = self.cluster_nodes.get_own_idx();
        for (segment_id, intersection_by_point) in msg.intersection_coords_by_segment.into_iter() {
            if own_id.eq(&self.segment_id_to_assignment(segment_id.clone())) {
                self.data_store.add_materialized_intersections(intersection_by_point)
            } else {
                match &mut self.intersection_calculation.foreign_intersections.get_mut(&segment_id) {
                    Some(transition_coord) => { transition_coord.extend(intersection_by_point) },
                    None => {
                        self.intersection_calculation.foreign_intersections.insert(segment_id, intersection_by_point);
                    }
                };
            };
        }

        if self.intersection_calculation.rotation_protocol.is_running() {
            self.rotate_foreign_assignments(ctx.address().recipient());
        } else {
            ctx.address().do_send(IntersectionCalculationDone);
        }
    }
}
