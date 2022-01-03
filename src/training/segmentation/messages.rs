use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use serde_with::serde_as;
use crate::training::segmentation::{SegmentedPointWithId, TransitionsForNodes};

#[serde_as]
#[derive(RemoteMessage, Serialize, Deserialize, Clone, Default)]
pub struct SegmentMessage {
    #[serde_as(as = "Vec<(_, _)>")]
    pub segments: TransitionsForNodes

}

#[derive(RemoteMessage, Serialize, Deserialize)]
pub struct SendFirstPointMessage {
    pub point: SegmentedPointWithId
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct SegmentedMessage;
