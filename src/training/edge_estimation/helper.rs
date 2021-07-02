use actix::{Actor, ActorContext, SyncContext, Handler};
use crate::training::edge_estimation::messages::EdgeEstimationHelperTask;
use actix::dev::MessageResponse;

pub struct EdgeEstimationHelper {
    
}

impl Actor for EdgeEstimationHelper {
    type Context = SyncContext<EdgeEstimationHelper>;
}

impl EdgeEstimationHelper {
    fn estimate_edge(&mut self) {

    }
}

impl Handler<EdgeEstimationHelperTask> for EdgeEstimationHelper {
    type Result = ();

    fn handle(&mut self, msg: EdgeEstimationHelperTask, ctx: &mut Self::Context) -> Self::Result {
        todo!()
    }
}
