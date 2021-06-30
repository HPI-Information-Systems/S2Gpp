use actix::prelude::*;

#[derive(Message)]
#[rtype(Result = "()")]
pub struct NodeEstimationDone;
