use crate::data_store::intersection::Intersection;
use crate::data_store::transition::TransitionRef;
use crate::training::intersection_calculation::SegmentID;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::HashMap;

#[derive(Message)]
#[rtype(Result = "()")]
pub(crate) struct IntersectionTaskMessage {
    pub transition: TransitionRef,
    pub segment_id: usize,
    pub line_points: Array2<f32>,
    pub plane_points: Array2<f32>,
    pub source: Recipient<IntersectionResultMessage>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub(crate) struct IntersectionResultMessage {
    pub transition: TransitionRef,
    pub segment_id: usize,
    pub intersection: Array1<f32>,
}

#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Default, Debug, Clone)]
pub(crate) struct IntersectionRotationMessage {
    #[serde_as(as = "Vec<(_, _)>")]
    pub intersection_coords_by_segment: HashMap<SegmentID, Vec<Intersection>>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct IntersectionCalculationDone;
