use crate::utils::ArcArray3;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::Array3;
use serde::{Deserialize, Serialize};

#[derive(Message)]
#[rtype(Result = "()")]
pub struct StartRotation {
    pub phase_space: ArcArray3<f32>,
    pub data_ref: ArcArray3<f32>,
}

#[derive(RemoteMessage, Serialize, Deserialize, Clone)]
pub struct RotationMatrixMessage {
    pub rotation_matrix: Array3<f32>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct RotationDoneMessage;
