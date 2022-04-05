use crate::data_store::edge::MaterializedEdge;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use std::collections::HashMap;

#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub(crate) struct TranspositionRotationMessage {
    #[serde_as(as = "Vec<(_, _)>")]
    pub assignments: HashMap<usize, Vec<MaterializedEdge>>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct TranspositionDone;
