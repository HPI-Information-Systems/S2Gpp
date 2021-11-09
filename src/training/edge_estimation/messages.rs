use std::collections::HashMap;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::training::edge_estimation::{NodeName, Edge};


#[derive(Message, RemoteMessage, Serialize, Deserialize, Default)]
#[rtype(Result = "()")]
pub struct EdgeReductionMessage {
    pub edges: Vec<(usize, Edge)>,
    pub own: bool
}


pub type PointNodeName = (usize, NodeName);

#[serde_as]
#[derive(Message, RemoteMessage, Serialize, Deserialize, Default, Debug, Clone)]
#[rtype(Result = "()")]
pub struct EdgeRotationMessage {
    #[serde_as(as = "Vec<(_, _)>")]
    pub open_edges: HashMap<usize, Vec<PointNodeName>>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub struct EdgeEstimationDone;
