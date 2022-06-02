use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::{ArcArray2, Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Message)]
#[rtype(Result = "()")]
pub struct PCAMessage {
    pub data: ArcArray2<f32>,
}

#[derive(Message)]
#[rtype(Result = "()")]
pub struct PCADoneMessage;

#[derive(Debug, Clone, Message)]
#[rtype(Result = "()")]
pub enum PCAHelperMessage {
    Setup {
        neighbors: Vec<Recipient<Self>>,
        data: ArcArray2<f32>,
    },
    Decomposition {
        r: Array2<f32>,
        count: usize,
    },
    Means {
        columns_means: Array2<f32>,
        n: usize,
    },
    #[allow(dead_code)]
    Components {
        components: Array2<f32>,
        means: Array1<f32>,
    },
    Response {
        column_means: Array1<f32>,
        n: f32,
        r: Array2<f32>,
    },
}

#[derive(RemoteMessage, Serialize, Deserialize, Clone)]
pub struct PCADecompositionMessage {
    pub r: Array2<f32>,
    pub count: usize,
}

#[derive(RemoteMessage, Serialize, Deserialize, Clone)]
pub struct PCAMeansMessage {
    pub columns_means: Array2<f32>,
    pub n: usize,
}

#[derive(RemoteMessage, Serialize, Deserialize, Clone)]
pub struct PCAComponents {
    pub components: Array2<f32>,
    pub means: Array1<f32>,
}
