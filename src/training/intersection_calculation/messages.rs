use actix::prelude::*;
use ndarray::{Array2, Array1};
use crate::training::intersection_calculation::Transition;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct IntersectionTaskMessage {
    pub transition_id: usize,
    pub segment_id: usize,
    pub line_points: Array2<f32>,
    pub plane_points: Array2<f32>,
    pub source: Recipient<IntersectionResultMessage>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub struct IntersectionResultMessage {
    pub transition_id: usize,
    pub segment_id: usize,
    pub intersection: Array1<f32>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct IntersectionCalculationDone;
