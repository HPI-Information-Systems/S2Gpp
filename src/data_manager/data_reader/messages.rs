use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(RemoteMessage, Serialize, Deserialize)]
pub struct DataPartitionMessage {
    pub data: Vec<Vec<String>>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct LocalReadDataMessage {
    pub data: Array2<f32>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct DataReceivedMessage {
    pub data: Array2<f32>,
}
