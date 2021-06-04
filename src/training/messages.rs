use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use serde_with::serde_as;
use crate::utils::ClusterNodes;
use ndarray::Array1;
use std::collections::HashMap;
use crate::training::segmenter::PointsForNodes;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct StartTrainingMessage {
    pub nodes: ClusterNodes
}

#[serde_as]
#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
pub struct SegmentMessage {
    #[serde_as(as = "Vec<(_, _)>")]
    pub segments: PointsForNodes
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct SegmentedMessage;
