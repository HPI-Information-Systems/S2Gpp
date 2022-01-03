use std::collections::HashMap;
use actix::prelude::*;
use crate::utils::Edge;
use actix_telepathy::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;


#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Default, Clone)]
pub struct TranspositionRotationMessage {
    #[serde_as(as = "Vec<(_, _)>")]
    pub assignments: HashMap<usize, Vec<(usize, Edge)>>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct TranspositionDone;
