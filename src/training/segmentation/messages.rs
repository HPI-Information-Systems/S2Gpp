use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use serde_with::serde_as;
use crate::training::segmentation::{SegmentedPointWithId, TransitionsForNodes};

#[serde_as]
#[derive(Message, RemoteMessage, Serialize, Deserialize, Clone, Default)]
#[rtype(Result = "()")]
pub struct SegmentMessage {
    #[serde_as(as = "Vec<(_, _)>")]
    pub segments: TransitionsForNodes

}

#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
pub struct SendFirstPointMessage {
    pub point: SegmentedPointWithId
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct SegmentedMessage;
