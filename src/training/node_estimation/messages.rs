use crate::data_store::node::IndependentNode;
use crate::data_store::node_questions::NodeQuestions;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::HashMap;

#[derive(Message)]
#[rtype(Result = "()")]
pub struct NodeEstimationDone;

#[derive(RemoteMessage, Serialize, Deserialize, Clone, Default)]
pub(crate) struct AskForForeignNodes {
    pub asked_nodes: NodeQuestions,
}

#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Clone, Default)]
pub(crate) struct ForeignNodesAnswer {
    /// (prev_point_id, prev_segment_id, point_id, node)
    #[serde_as(as = "Vec<(_, _)>")]
    pub answers: HashMap<usize, Vec<(usize, usize, usize, IndependentNode)>>,
}
