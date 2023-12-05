use crate::messages::PoisonPill;
use crate::training::intersection_calculation::messages::{
    IntersectionResultMessage, IntersectionTaskMessage,
};
use crate::utils::line_plane_intersection;
use actix::{Actor, ActorContext, Handler, SyncContext};
use ndarray::{arr1, Axis};
use ndarray_linalg::Norm;
use num_integer::div_floor;

use super::messages::{IntersectionResult, IntersectionTask};

pub struct IntersectionCalculationHelper {}

impl IntersectionCalculationHelper {
    fn work(&self, task: IntersectionTask) -> IntersectionResult {
        match line_plane_intersection(task.line_points.clone(), task.plane_points.clone()) {
            Ok(intersection) => {
                let shape = intersection.shape();
                let reshaped = intersection.to_shape([2, div_floor(shape[0], 2)]).unwrap();
                let distance = arr1(
                    &reshaped
                        .axis_iter(Axis(1))
                        .map(|coords| coords.norm())
                        .collect::<Vec<f32>>(),
                );
                IntersectionResult {
                    transition: task.transition,
                    segment_id: task.segment_id,
                    intersection: distance,
                }
            }
            Err(e) => panic!("intersection error {:?}", e),
        }
    }
}

impl Actor for IntersectionCalculationHelper {
    type Context = SyncContext<Self>;
}

impl Handler<IntersectionTaskMessage> for IntersectionCalculationHelper {
    type Result = ();

    fn handle(&mut self, msg: IntersectionTaskMessage, _ctx: &mut Self::Context) -> Self::Result {
        let results = msg.tasks.into_iter().map(|task| self.work(task)).collect();
        msg.source
            .do_send(IntersectionResultMessage { results });
    }
}

impl Handler<PoisonPill> for IntersectionCalculationHelper {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
