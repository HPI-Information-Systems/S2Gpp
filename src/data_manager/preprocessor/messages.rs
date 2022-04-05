use actix::prelude::Message;
use actix::Addr;

use ndarray::Array1;

use crate::data_manager::DataManager;

#[derive(Message)]
#[rtype(Result = "()")]
pub struct PreprocessColumnMessage {
    pub column: usize,
    pub source: Addr<DataManager>,
    pub std: f32,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct ProcessedColumnMessage {
    pub column: usize,
    pub processed_column: Array1<f32>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct PreprocessingDoneMessage;
