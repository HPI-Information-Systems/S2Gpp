use actix::prelude::Message;
use actix::{Recipient, Addr};
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use ndarray::{Array1, Array};
use crate::data_manager::preprocessor::Preprocessor;


#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
#[with_source(source)]
pub struct StdNodeMessage {
    pub n: usize,
    pub mean: Array1<f32>,
    pub m2: Array1<f32>,
    pub source: RemoteAddr
}


#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
pub struct StdDoneMessage {
    pub std: Array1<f32>
}


#[derive(Message)]
#[rtype(Result = "()")]
pub struct PreprocessColumnMessage {
    pub column: usize,
    pub source: Recipient<ProcessedColumnMessage>,
    pub std: f32
}


#[derive(Message)]
#[rtype(Result = "()")]
pub struct ProcessedColumnMessage {
    pub column: usize,
    pub processed_column: Array1<f32>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct PreprocessingDoneMessage;
