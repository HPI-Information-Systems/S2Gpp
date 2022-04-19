use crate::data_store::edge::MaterializedEdge;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub(crate) struct TranspositionRotationMessage {
    pub assignments: Vec<MaterializedEdge>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct TranspositionDone;
