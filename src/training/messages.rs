use actix::prelude::*;
use std::collections::HashMap;
use actix_telepathy::RemoteAddr;
use crate::utils::ClusterNodes;
use ndarray::{ArcArray, Dimension, Array3, arr3, Ix3};


#[derive(Message)]
#[rtype(Result = "()")]
pub struct StartTrainingMessage {
    pub nodes: ClusterNodes
}
