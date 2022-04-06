use crate::messages::PoisonPill;
use crate::training::intersection_calculation::messages::{
    IntersectionResultMessage, IntersectionTaskMessage,
};
use crate::utils::line_plane_intersection;
use actix::{Actor, ActorContext, Handler, SyncContext};
use ndarray::{arr1, concatenate, s, Axis};
use ndarray_linalg::Norm;

use super::messages::{IntersectionResult, IntersectionTask};

pub struct IntersectionCalculationHelper {}

impl IntersectionCalculationHelper {
    fn work(&self, task: IntersectionTask) -> IntersectionResult {
        match line_plane_intersection(task.line_points.clone(), task.plane_points.clone()) {
            Ok(intersection) => {
                let first_distance = arr1(&[intersection.slice(s![0..2]).norm()]);
                let distance = concatenate(
                    Axis(0),
                    &[first_distance.view(), intersection.slice(s![2..])],
                )
                .unwrap();
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
            .do_send(IntersectionResultMessage { results })
            .unwrap();
    }
}

impl Handler<PoisonPill> for IntersectionCalculationHelper {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
