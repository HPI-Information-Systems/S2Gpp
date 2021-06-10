use actix::{Actor, ActorContext, SyncContext, Handler};
use crate::training::intersection_calculation::messages::{IntersectionTaskMessage, IntersectionResultMessage};
use actix::dev::MessageResponse;
use crate::utils::line_plane_intersection;
use log::*;

pub struct IntersectionCalculationHelper {}

impl Actor for IntersectionCalculationHelper {
    type Context = SyncContext<Self>;
}

impl Handler<IntersectionTaskMessage> for IntersectionCalculationHelper {
    type Result = ();

    fn handle(&mut self, msg: IntersectionTaskMessage, _ctx: &mut Self::Context) -> Self::Result {
        match line_plane_intersection(msg.line_points, msg.plane_points) {
            Ok(intersection) => { msg.source.do_send(
                IntersectionResultMessage {
                    transition: msg.transition,
                    segment_id: msg.segment_id,
                    intersection
                }).unwrap(); },
            Err(e) => warn!("intersection error {:?}", e)
        };
    }
}
