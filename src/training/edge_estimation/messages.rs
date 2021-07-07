use actix::prelude::*;
use ndarray::{Array2, Array1, ArcArray1, ArcArray2};
use crate::training::edge_estimation::{NodeName, Edge};
use crate::training::intersection_calculation::Transition;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct EdgeEstimationHelperResponse {
    pub task_id: usize,
    pub node_names: Vec<NodeName>,
    pub transition: Transition
}


#[derive(Message)]
#[rtype(Result = "()")]
pub struct EdgeEstimationHelperTask {
    pub task_id: usize,
    pub transition: Transition
}


#[derive(Message)]
#[rtype(Result = "()")]
pub struct EdgeEstimationDone;
