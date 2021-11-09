use actix::prelude::*;
use crate::utils::ClusterNodes;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct StartTrainingMessage {
    pub nodes: ClusterNodes
}
