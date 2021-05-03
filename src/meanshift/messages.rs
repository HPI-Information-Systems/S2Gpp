use actix::prelude::*;
use ndarray::{Array2, ArcArray2, ArcArray1, Array1};
use kdtree::KdTree;
use std::sync::Arc;
use std::ops::Range;
use crate::meanshift::RefArray;

#[derive(Message)]
#[rtype(Result = "()")]
pub struct DataMessage {
    pub source: Option<Recipient<MeanShiftHelperResponse>>,
    pub data: Array2<f32>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftHelperWorkMessage {
    pub source: Recipient<MeanShiftHelperResponse>,
    pub start_center: usize
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct MeanShiftHelperResponse {
    pub source: Recipient<MeanShiftHelperWorkMessage>,
    pub mean: Array1<f32>,
    pub points_within_len: usize,
    pub iterations: usize
}
