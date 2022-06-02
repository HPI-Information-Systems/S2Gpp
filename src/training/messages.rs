use crate::utils::ClusterNodes;
use actix::prelude::*;
use ndarray::{Array1, Array2};

#[derive(Message)]
#[rtype(Result = "()")]
pub struct StartTrainingMessage {
    pub nodes: ClusterNodes,
    pub source: Option<Recipient<DetectionResponse>>,
    pub data: Option<Array2<f32>>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct DetectionResponse {
    pub anomaly_score: Array1<f32>,
}
