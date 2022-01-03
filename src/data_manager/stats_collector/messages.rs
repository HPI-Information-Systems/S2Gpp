use ndarray::Array1;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use crate::data_manager::stats_collector::DatasetStats;

#[derive(Message)]
#[rtype(Result = "()")]
pub struct DatasetStatsMessage {
    pub dataset_stats: DatasetStats
}

#[derive(RemoteMessage, Serialize, Deserialize)]
#[with_source(source)]
pub struct StdNodeMessage {
    pub n: usize,
    pub mean: Array1<f32>,
    pub m2: Array1<f32>,
    pub source: RemoteAddr
}


#[derive(RemoteMessage, Serialize, Deserialize)]
pub struct StdDoneMessage {
    pub std: Array1<f32>,
    pub n: usize
}

#[derive(RemoteMessage, Serialize, Deserialize)]
#[with_source(source)]
pub struct MinMaxNodeMessage {
    pub min: Array1<f32>,
    pub max: Array1<f32>,
    pub source: RemoteAddr
}


#[derive(RemoteMessage, Serialize, Deserialize)]
pub struct MinMaxDoneMessage {
    pub min: Array1<f32>,
    pub max: Array1<f32>
}
