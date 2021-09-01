use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Deserialize, Serialize};
use ndarray::{Array2, Array1, ArcArray1, ArcArray2};
use crate::training::edge_estimation::{NodeName, Edge};
use crate::training::intersection_calculation::Transition;


#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
pub struct EdgeReductionMessage {
    pub edges: Vec<Edge>,
    pub edge_in_time: Vec<usize>,
    pub nodes: Vec<NodeName>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub struct EdgeEstimationDone;
