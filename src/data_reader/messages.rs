use actix::prelude::*;
use ndarray::Array2;
use csv::ByteRecord;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct DataPartitionMessage {
    pub data: Vec<ByteRecord>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct DataReceivedMessage {
    pub data: Array2<f32>
}
