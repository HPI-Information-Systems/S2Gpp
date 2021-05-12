use actix::prelude::Message;
use crate::data_manager::DataManager;
use actix::Addr;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct LoadDataMessage;
