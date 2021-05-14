use actix::prelude::*;
use std::collections::HashMap;
use actix_telepathy::RemoteAddr;
use crate::utils::ClusterNodes;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct StartTrainingMessage {
    pub nodes: ClusterNodes
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct DataLoadedAndProcessed {
    pub data_ref: ArcArray3<f32>,
    pub phase_space: ArcArray3<f32>
}
