use actix::{Actor, ActorContext, Context, Handler, Recipient};
use crate::data_reader::messages::{DataPartitionMessage, DataReceivedMessage};
use actix::dev::MessageResponse;
use ndarray::{Array2, Array1};
use csv::StringRecord;
use std::str::FromStr;
use log::*;


pub struct DataReceiver {
    recipient: Option<Recipient<DataReceivedMessage>>
}

impl DataReceiver {
    pub fn new(recipient: Option<Recipient<DataReceivedMessage>>) -> Self {
        Self {
            recipient
        }
    }
}

impl Actor for DataReceiver {
    type Context = Context<Self>;
}

impl Handler<DataPartitionMessage> for DataReceiver {
    type Result = ();

    fn handle(&mut self, msg: DataPartitionMessage, ctx: &mut Self::Context) -> Self::Result {
        let n_rows = msg.data.len();
        let n_columns = msg.data[0].len();

        let flat_data: Array1<f32> = msg.data.into_iter().flat_map(|rec| {
            StringRecord::from_byte_record(rec).unwrap().iter().map(|b| {
                f32::from_str(b).unwrap()
            }).collect::<Vec<f32>>()
        }).collect();

        let data = flat_data.into_shape((n_rows, n_columns)).expect("Could not deserialize sent data");

        match &self.recipient {
            Some(recipient) => { recipient.do_send(DataReceivedMessage { data }); },
            None => ()
        }
    }
}