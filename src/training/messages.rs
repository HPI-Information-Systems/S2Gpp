use crate::utils::ClusterNodes;
use actix::prelude::*;

#[derive(Message)]
#[rtype(Result = "()")]
pub struct StartTrainingMessage {
    pub nodes: ClusterNodes,
}
