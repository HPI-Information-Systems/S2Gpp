use std::collections::HashMap;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use ndarray::{Array2, Array1};
use crate::training::intersection_calculation::{IntersectionsByTransition, SegmentID};


#[derive(Message)]
#[rtype(Result = "()")]
pub struct IntersectionTaskMessage {
    pub point_id: usize,
    pub segment_id: usize,
    pub line_points: Array2<f32>,
    pub plane_points: Array2<f32>,
    pub source: Recipient<IntersectionResultMessage>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub struct IntersectionResultMessage {
    pub point_id: usize,
    pub segment_id: usize,
    pub intersection: Array1<f32>
}

#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Default, Debug, Clone)]
pub struct IntersectionRotationMessage {
    #[serde_as(as = "Vec<(_, Vec<(_, _)>)>")]
    pub intersection_coords_by_segment: HashMap<SegmentID, IntersectionsByTransition>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct IntersectionCalculationDone;
