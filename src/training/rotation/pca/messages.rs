use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use ndarray::{Array2, ArcArray2, Array3, Array1};
use crate::utils::ArcArray3;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct PCAMessage {
    pub data: ArcArray2<f32>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct PCADoneMessage;

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
    pub components: Array2<f32>,
    pub means: Array1<f32>
}
