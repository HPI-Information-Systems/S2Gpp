use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Deserialize, Serialize};

use crate::training::edge_estimation::{NodeName, Edge};



#[derive(Message, RemoteMessage, Serialize, Deserialize, Default)]
#[rtype(Result = "()")]
pub struct EdgeReductionMessage {
    pub edges: Vec<(usize, Edge)>,
    pub own: bool
}


#[derive(Message, RemoteMessage, Serialize, Deserialize, Default)]
#[rtype(Result = "()")]
pub struct EdgeRotationMessage {
    pub open_edges: Vec<(usize, NodeName)>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub struct EdgeEstimationDone;
