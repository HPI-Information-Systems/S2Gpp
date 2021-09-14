use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Serialize, Deserialize};
use ndarray::{Array2, ArcArray2, Array3};
use crate::utils::ArcArray3;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct StartRotation {
    pub phase_space: ArcArray3<f32>,
    pub data_ref: ArcArray3<f32>
}

#[derive(Message, RemoteMessage, Serialize, Deserialize)]
#[rtype(Result = "()")]
pub struct RotationMatrixMessage {
    pub rotation_matrix: Array3<f32>
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct RotationDoneMessage;
