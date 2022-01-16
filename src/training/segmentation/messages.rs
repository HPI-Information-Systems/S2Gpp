use std::collections::HashMap;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use serde_with::serde_as;
use crate::data_store::point::Point;
use crate::data_store::transition::MaterializedTransition;


#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Clone, Default)]
pub struct SegmentMessage {
    #[serde_as(as = "Vec<(_, _)>")]
    pub(crate) segments: HashMap<usize, Vec<MaterializedTransition>>

}

#[derive(RemoteMessage, Serialize, Deserialize)]
pub struct SendFirstPointMessage {
    pub(crate) point: Point
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct SegmentedMessage;
