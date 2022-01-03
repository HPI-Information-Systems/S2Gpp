use std::collections::HashMap;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use crate::utils::NodeName;
use serde_with::serde_as;
use crate::training::segmentation::NodeInQuestion;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct NodeEstimationDone;


#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Clone, Default)]
pub struct AskForForeignNodes {
    #[serde_as(as = "Vec<(_, _)>")]
    pub asked_nodes: HashMap<usize, Vec<NodeInQuestion>>
}


#[derive(RemoteMessage, Serialize, Deserialize, Clone, Default)]
pub struct ForeignNodesAnswer {
    /// (prev_point_id, prev_segment_id, point_id, node)
    pub foreign_nodes: Vec<(usize, usize, usize, NodeName)>
}
