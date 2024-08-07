mod helper;
mod messages;
#[cfg(test)]
mod tests;

use actix::{Addr, AsyncContext, Handler, Recipient, SyncArbiter};

use ndarray::{arr1, concatenate, stack, Array1, Array2, Axis};
use std::collections::HashMap;

use std::f32::consts::PI;

use crate::training::intersection_calculation::helper::IntersectionCalculationHelper;
pub(crate) use crate::training::intersection_calculation::messages::{
    IntersectionCalculationDone, IntersectionResultMessage, IntersectionRotationMessage,
    IntersectionTaskMessage,
};

use crate::messages::PoisonPill;
use crate::training::Training;
use crate::utils::{HelperProtocol, PolarCoords};

use crate::data_store::intersection::Intersection;
use crate::data_store::transition::{TransitionMixin, TransitionRef};
use crate::utils::direct_protocol::DirectProtocol;
use ndarray_linalg::Norm;
use num_integer::Integer;

use self::messages::IntersectionTask;

pub type SegmentID = usize;

#[derive(Default, Clone)]
pub(crate) struct IntersectionCalculation {
    /// Intersections belonging to another cluster node
    pub foreign_intersections: HashMap<SegmentID, Vec<Intersection>>,
    pub helpers: Option<Addr<IntersectionCalculationHelper>>,
    /// Collects tasks for helper actors.
    pub pairs: Vec<(TransitionRef, SegmentID, Array2<f32>, Array2<f32>)>,
    pub helper_protocol: HelperProtocol,
    pub recipient: Option<Recipient<IntersectionCalculationDone>>,
    pub direct_protocol: DirectProtocol<IntersectionRotationMessage>,
}

pub(crate) trait IntersectionCalculator {
    fn calculate_intersections(&mut self, rec: Recipient<IntersectionResultMessage>);
    fn parallel_intersection_tasks(&mut self, rec: Recipient<IntersectionResultMessage>);
    fn rotate_foreign_assignments(&mut self, rec: Recipient<IntersectionCalculationDone>);
    fn assign_received_intersection(
        &mut self,
        segment_id: SegmentID,
        transition: TransitionRef,
        intersection: Array1<f32>,
    );
    fn start_distribution_protocol(&mut self);
}

impl IntersectionCalculator for Training {
    fn calculate_intersections(&mut self, rec: Recipient<IntersectionResultMessage>) {
        if self.data_store.count_transitions() == 0 {
            self.start_distribution_protocol();
            return;
        }

        let max_value = self
            .data_store
            .get_transitions()
            .iter()
            .map(|x| {
                x.get_from_point()
                    .get_max_coordinate()
                    .max(x.get_from_point().get_min_coordinate().abs())
                    .max(
                        x.get_to_point()
                            .get_max_coordinate()
                            .max(x.get_to_point().get_min_coordinate().abs())
                            .abs(),
                    )
            })
            .fold(0_f32, |a, b| a.max(b));
        let radius = arr1(&[max_value, max_value]).norm();

        let dims = self
            .data_store
            .get_transitions()
            .first()
            .expect("Could not generate Segments")
            .get_from_point()
            .get_dims();

        let origin = arr1(vec![0_f32; dims].as_slice());
        let planes_end_points: Vec<Array1<f32>> = (0..self.parameters.rate)
            .into_iter()
            .map(|segment_id| {
                let polar = arr1(&[
                    radius,
                    (2.0 * PI * segment_id as f32) / self.parameters.rate as f32,
                ]);
                let other_dims = arr1(
                    (2..dims)
                        .into_iter()
                        .map(|_| max_value)
                        .collect::<Vec<f32>>()
                        .as_slice(),
                );
                concatenate(Axis(0), &[polar.to_cartesian().view(), other_dims.view()]).unwrap()
            })
            .collect();

        for transition in self.data_store.get_transitions() {
            let line_points = stack(
                Axis(0),
                &[
                    transition.get_from_point().clone_coordinates().view(),
                    transition.get_to_point().clone_coordinates().view(),
                ],
            )
            .unwrap();

            let mut segment_ids = vec![];

            let mut segment_diff = transition.segment_diff();
            let half_rate = num_integer::Integer::div_floor(&self.parameters.rate, &2);

            if segment_diff > half_rate {
                if transition.get_to_segment() > half_rate {
                    segment_diff = (transition.get_from_segment() as isize
                        - (-(self.parameters.rate as isize) + transition.get_to_segment() as isize))
                        .abs() as usize;
                } else if transition.get_from_segment() > half_rate {
                    segment_diff = (transition.get_to_segment() as isize
                        - (-(self.parameters.rate as isize)
                            + transition.get_from_segment() as isize))
                        .abs() as usize;
                }
            }
            segment_diff = segment_diff.min(half_rate);

            for segment_lag in 1..(segment_diff) + 1 {
                let segment_id =
                    (transition.get_from_segment() + segment_lag).mod_floor(&self.parameters.rate);
                segment_ids.push(segment_id);

                let mut arrays = vec![origin.view()];
                let corner_points: Vec<Array1<f32>> = (2..dims)
                    .into_iter()
                    .map(|d| {
                        let mut corner_point = planes_end_points[segment_id].clone();
                        corner_point[d] = 0.;
                        corner_point
                    })
                    .collect();

                arrays.extend(corner_points.iter().map(|x| x.view()));
                arrays.push(planes_end_points[segment_id].view());

                let plane_points = stack(Axis(0), arrays.as_slice()).unwrap();
                self.intersection_calculation.pairs.push((
                    transition.clone(),
                    segment_id,
                    line_points.clone(),
                    plane_points,
                ));
            }
        }
        self.intersection_calculation.helper_protocol.n_total = 1; //self.parameters.n_threads;

        self.intersection_calculation.helpers = Some(SyncArbiter::start(1, move || {
            IntersectionCalculationHelper {}
        }));

        self.parallel_intersection_tasks(rec);
    }

    fn start_distribution_protocol(&mut self) {
        let own_addr = self.own_addr.as_ref().unwrap().clone();
        self.intersection_calculation
            .direct_protocol
            .start(self.cluster_nodes.len());
        self.intersection_calculation
            .direct_protocol
            .resolve_buffer(own_addr.clone().recipient());
        self.rotate_foreign_assignments(own_addr.recipient());
    }

    fn parallel_intersection_tasks(&mut self, rec: Recipient<IntersectionResultMessage>) {
        if self.intersection_calculation.pairs.is_empty() {
            println!("empty pairs");
            self.start_distribution_protocol();
            return;
        }

        let chunk_size = num_integer::div_ceil(self.intersection_calculation.pairs.len(), 1);
        for _ in 0..1 {
            let mut tasks = vec![];
            for _ in 0..(self.intersection_calculation.pairs.len().min(chunk_size)) {
                if let Some((transition, segment_id, line_points, plane_points)) =
                    self.intersection_calculation.pairs.pop()
                {
                    tasks.push(IntersectionTask {
                        transition,
                        segment_id,
                        line_points,
                        plane_points,
                    });
                }
            }
            self.intersection_calculation
                .helpers
                .as_ref()
                .unwrap()
                .do_send(IntersectionTaskMessage {
                    tasks,
                    source: rec.clone(),
                });
            self.intersection_calculation.helper_protocol.sent();
        }
    }

    fn rotate_foreign_assignments(&mut self, rec: Recipient<IntersectionCalculationDone>) {
        if self.cluster_nodes.len() == 0 {
            rec.do_send(IntersectionCalculationDone);
            return;
        }

        let mut intersection_coords_by_segment =
            self.intersection_calculation.foreign_intersections.clone();
        self.intersection_calculation.foreign_intersections.clear();
        for (id, node) in self.cluster_nodes.iter() {
            let mut training_node = node.clone();
            training_node.change_id("Training".to_string());

            let mut coords_to_send = HashMap::new();
            for segment_id in 0..self.parameters.rate {
                if id.eq(&self.segment_id_to_assignment(segment_id)) {
                    if let Some(coords) = intersection_coords_by_segment.remove(&segment_id) {
                        coords_to_send.insert(segment_id, coords);
                    }
                }
            }
            training_node.do_send(IntersectionRotationMessage {
                intersection_coords_by_segment: coords_to_send,
            });
            self.intersection_calculation.direct_protocol.sent();
        }
    }

    fn assign_received_intersection(
        &mut self,
        segment_id: SegmentID,
        transition: TransitionRef,
        intersection: Array1<f32>,
    ) {
        let own_id = self.cluster_nodes.get_own_idx();
        let assigned_id = self.segment_id_to_assignment(segment_id);

        let intersection = Intersection::new(transition, intersection, segment_id);

        if own_id.eq(&assigned_id) {
            self.data_store.add_intersection(intersection)
        } else {
            match &mut self
                .intersection_calculation
                .foreign_intersections
                .get_mut(&segment_id)
            {
                Some(transition_coord) => transition_coord.push(intersection),
                None => {
                    self.intersection_calculation
                        .foreign_intersections
                        .insert(segment_id, vec![intersection]);
                }
            };
        }
    }
}

impl Handler<IntersectionResultMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: IntersectionResultMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.intersection_calculation.helper_protocol.received();

        for result in msg.results {
            self.assign_received_intersection(
                result.segment_id,
                result.transition,
                result.intersection,
            );
        }

        if !self.intersection_calculation.helper_protocol.is_running() {
            self.intersection_calculation
                .helpers
                .as_ref()
                .unwrap()
                .do_send(PoisonPill);
            match &self.intersection_calculation.recipient {
                Some(rec) => rec.do_send(IntersectionCalculationDone),
                None => {
                    self.start_distribution_protocol();
                }
            }
        }
    }
}

impl Handler<IntersectionRotationMessage> for Training {
    type Result = ();

    fn handle(
        &mut self,
        msg: IntersectionRotationMessage,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        if !self.intersection_calculation.direct_protocol.received(&msg) {
            return;
        }

        let own_id = self.cluster_nodes.get_own_idx();
        for (segment_id, intersection_by_point) in msg.intersection_coords_by_segment.into_iter() {
            if own_id.eq(&self.segment_id_to_assignment(segment_id)) {
                self.data_store.add_intersections(intersection_by_point)
            } else {
                match &mut self
                    .intersection_calculation
                    .foreign_intersections
                    .get_mut(&segment_id)
                {
                    Some(transition_coord) => transition_coord.extend(intersection_by_point),
                    None => {
                        self.intersection_calculation
                            .foreign_intersections
                            .insert(segment_id, intersection_by_point);
                    }
                };
            };
        }

        if !self.intersection_calculation.direct_protocol.is_running() {
            ctx.address().do_send(IntersectionCalculationDone);
        }
    }
}
