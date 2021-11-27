use actix::prelude::*;
use actix_telepathy::prelude::*;
use serde::{Deserialize, Serialize};


#[derive(Message)]
#[rtype(Result = "()")]
pub struct EdgeEstimationDone;
