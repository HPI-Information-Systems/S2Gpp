use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use ndarray::{Array2, Array1, ArcArray2};
use crate::pca::PCA;
use crate::utils::ClusterNodes;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct PCAMessage {
    pub data: ArcArray2<f32>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct PCAResponse {
    pub components: Array2<f32>
}

#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
pub struct PCADecompositionMessage {
    pub r: Array2<f32>,
    pub count: usize
}

#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
pub struct PCAMeansMessage {
    pub columns_means: Array2<f32>,
    pub n: usize
}

#[derive(Message, RemoteMessage, Serialize, Deserialize, Clone)]
#[rtype(Result = "()")]
pub struct PCAComponents {
    pub components: Array2<f32>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct RotatedMessage {
    pub rotated: Array2<f32>
}
