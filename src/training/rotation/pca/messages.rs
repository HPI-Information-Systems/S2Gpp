use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use ndarray::{Array2, ArcArray2, Array1};



#[derive(Message)]
#[rtype(Result = "()")]
pub struct PCAMessage {
    pub data: ArcArray2<f32>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct PCADoneMessage;

#[derive(RemoteMessage, Serialize, Deserialize)]
pub struct PCADecompositionMessage {
    pub r: Array2<f32>,
    pub count: usize
}

#[derive(RemoteMessage, Serialize, Deserialize)]
pub struct PCAMeansMessage {
    pub columns_means: Array2<f32>,
    pub n: usize
}

#[derive(RemoteMessage, Serialize, Deserialize, Clone)]
pub struct PCAComponents {
    pub components: Array2<f32>,
    pub means: Array1<f32>
}
