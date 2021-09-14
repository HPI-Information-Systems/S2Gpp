use actix::{Actor, ActorContext, SyncContext, Handler};
use crate::training::intersection_calculation::messages::{IntersectionTaskMessage, IntersectionResultMessage};
use crate::utils::line_plane_intersection;
use log::*;
use crate::messages::PoisonPill;
use ndarray::{concatenate, Axis, arr1, s};
use ndarray_linalg::Norm;

pub struct IntersectionCalculationHelper {}

impl Actor for IntersectionCalculationHelper {
    type Context = SyncContext<Self>;
}

impl Handler<IntersectionTaskMessage> for IntersectionCalculationHelper {
    type Result = ();

    fn handle(&mut self, msg: IntersectionTaskMessage, _ctx: &mut Self::Context) -> Self::Result {
        match line_plane_intersection(msg.line_points, msg.plane_points) {
            Ok(intersection) => {
                let first_distance = arr1(&[intersection.slice(s![0..2]).norm()]);
                let distance = concatenate(Axis(0), &[first_distance.view(), intersection.slice(s![2..])]).unwrap();
                msg.source.do_send(
                IntersectionResultMessage {
                    transition_id: msg.transition_id,
                    segment_id: msg.segment_id,
                    intersection: distance
                }).unwrap(); },
            Err(e) => warn!("intersection error {:?}", e)
        };
    }
}

impl Handler<PoisonPill> for IntersectionCalculationHelper {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
