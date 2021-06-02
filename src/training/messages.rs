use actix::prelude::*;
use actix_telepathy::prelude::*;


use crate::utils::ClusterNodes;
use ndarray::Array1;
use std::collections::HashMap;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct StartTrainingMessage {
    pub nodes: ClusterNodes
}


#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
pub struct SegmentMessage {
    pub segments: HashMap<usize, Vec<(usize, Array1<f32>)>>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct SegmentedMessage;
