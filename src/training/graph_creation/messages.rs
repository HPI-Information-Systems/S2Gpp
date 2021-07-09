use actix::Message;


#[derive(Message)]
#[rtype(Result = "()")]
pub struct GraphCreationDone;
