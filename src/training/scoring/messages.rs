use std::collections::HashMap;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use crate::utils::{Edge, NodeName};


#[derive(Message)]
#[rtype(Result = "()")]
pub struct ScoreInitDone;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct ScoringDone;


#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub struct NodeDegrees {
    #[serde_as(as = "Vec<(_, _)>")]
    pub degrees: HashMap<NodeName, usize>
}


#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub struct EdgeWeights {
    #[serde_as(as = "Vec<(_, _)>")]
    pub weights: HashMap<Edge, usize>
}

#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub struct OverlapRotation {
    pub edges: Vec<(usize, Edge)>,
    pub edges_in_time: Vec<usize>
}


#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub struct SubScores {
    pub cluster_node_id: usize,
    pub scores: Array1<f32>
}
