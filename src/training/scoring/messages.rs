use std::collections::HashMap;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use crate::data_store::edge::MaterializedEdge;
use crate::data_store::node::IndependentNode;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct ScoreInitDone;


#[derive(Message)]
#[rtype(Result = "()")]
pub(crate) struct ScoringDone;


#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub(crate) struct NodeDegrees {
    #[serde_as(as = "Vec<(_, _)>")]
    pub degrees: HashMap<IndependentNode, usize>
}


#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub(crate) struct EdgeWeights {
    #[serde_as(as = "Vec<(_, _)>")]
    pub weights: HashMap<MaterializedEdge, usize>
}

#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub(crate) struct OverlapRotation {
    pub edges: Vec<MaterializedEdge>,
    pub edges_in_time: Vec<usize>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub(crate) struct ScoringHelperInstruction {
    pub start: usize,
    pub length: usize
}


#[derive(Message)]
#[rtype(Result = "()")]
pub(crate) struct ScoringHelperResponse {
    pub start: usize,
    pub scores: Vec<f32>,
    pub first_empty: bool
}


#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub struct SubScores {
    pub cluster_node_id: usize,
    pub scores: Array1<f32>
}
