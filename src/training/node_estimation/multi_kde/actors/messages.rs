use std::ops::Range;
use actix::{Message, Recipient};
use ndarray::{ArcArray2, Array1, Array2};
use crate::messages::PoisonPill;


#[derive(Message)]
#[rtype(Result = "()")]
pub(in crate::training::node_estimation) struct MultiKDEMessage {
    pub data: Array2<f32>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub(in crate::training::node_estimation) struct GaussianKDEMessage {
    pub column: ArcArray2<f32>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub(in crate::training::node_estimation) struct GaussianKDEResponse {
    pub kernel_estimate: Array1<f32>,
    pub source: Recipient<PoisonPill>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub(in crate::training::node_estimation) struct EstimatorTask {
    pub data_range: Range<usize>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub(in crate::training::node_estimation) struct EstimatorResponse {
    pub estimate: Array1<f32>
}
