use actix::prelude::Message;

use crate::data_manager::DatasetStats;
use crate::utils::ClusterNodes;
use ndarray::{ArcArray, Ix3};

#[derive(Message)]
#[rtype(Result = "()")]
pub struct LoadDataMessage {
    pub nodes: ClusterNodes,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct DataLoadedAndProcessed {
    pub data_ref: ArcArray<f32, Ix3>,
    pub phase_space: ArcArray<f32, Ix3>,
    pub dataset_stats: DatasetStats,
}
