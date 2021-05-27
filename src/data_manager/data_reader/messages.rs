use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize, Serializer};
use ndarray::Array2;
use csv::{ByteRecord, StringRecord};


#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
pub struct DataPartitionMessage {
    pub data: Vec<Vec<String>>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct DataReceivedMessage {
    pub data: Array2<f32>
}
