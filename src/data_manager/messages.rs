use actix::prelude::Message;


use ndarray::{ArcArray, Ix3};


#[derive(Message)]
#[rtype(Result = "()")]
pub struct LoadDataMessage;

#[derive(Message)]
#[rtype(Result = "()")]
pub struct DataLoadedAndProcessed {
    pub data_ref: ArcArray<f32, Ix3>,
    pub phase_space: ArcArray<f32, Ix3>
}
