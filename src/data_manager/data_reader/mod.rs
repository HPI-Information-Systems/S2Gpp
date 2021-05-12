mod messages;
mod receiver;
#[cfg(test)]
mod tests;

pub use receiver::DataReceiver;
pub use messages::DataReceivedMessage;
use log::*;
use ndarray::prelude::*;
use csv::{ReaderBuilder, Trim, StringRecord};
use std::path::Path;
use std::fs::{File};
use std::str::FromStr;
use actix::{Addr, Recipient, Actor, ActorContext, Context};
pub use crate::data_manager::data_reader::messages::DataPartitionMessage;
use std::io::{BufReader, BufRead};
use num_integer::Integer;
use actix_telepathy::RemoteAddr;


pub struct DataReader {
    file_path: String,
    receivers: Vec<RemoteAddr>,
    overlap: usize,
    with_header: bool
}

impl DataReader {
    pub fn new(file_path: &str, receivers: Vec<RemoteAddr>, overlap: usize) -> Self {
        Self {
            file_path: file_path.to_string(),
            receivers,
            overlap,
            with_header: true
        }
    }

    fn read_data(&mut self) {
        let file = File::open(&self.file_path).unwrap();
        let mut count_reader = BufReader::new(file);
        let n_lines = if self.with_header {
            count_reader.lines().count() - 1
        } else {
            count_reader.lines().count()
        };

        let file = File::open(&self.file_path).unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).trim(Trim::All).from_reader(file);

        let partition_len = n_lines.div_floor(&self.receivers.len());
        let last_overlap = n_lines - (partition_len * self.receivers.len());
        let mut receiver_id = 0;
        let mut buffer = vec![];
        let mut overlap_buffer = vec![];
        for record in reader.records() {
            match record {
                Ok(r) => {
                    let strings = r.iter().map(|x| x.to_string()).collect();
                    if buffer.len() < partition_len {
                        buffer.push(strings);
                    } else if self.receivers.len() > 1 &&
                        ((overlap_buffer.len() < self.overlap && receiver_id < (self.receivers.len() - 1))
                        || (overlap_buffer.len() < last_overlap && receiver_id == (self.receivers.len() - 1)))  {
                        overlap_buffer.push(strings);
                    } else {
                        let mut data = buffer.clone();
                        data.extend(overlap_buffer.clone());
                        self.receivers[receiver_id].do_send(DataPartitionMessage { data });
                        debug!("Sent data to receiver {}", receiver_id);

                        receiver_id += 1;
                        buffer.clear();
                        buffer.extend(overlap_buffer.clone());
                        buffer.push(strings);
                        overlap_buffer.clear();
                    }
                },
                Err(e) => panic!(e)
            }
        }

        let mut data = buffer.clone();
        data.extend(overlap_buffer.clone());
        self.receivers[receiver_id].do_send(DataPartitionMessage { data });
        debug!("Sent data to receiver {}", receiver_id);
    }
}

impl Actor for DataReader {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.read_data();
        ctx.stop();
    }
}

pub fn read_data_(file_path: &str) -> Array2<f32> {
    let file = File::open(file_path).unwrap();
    let mut count_reader = BufReader::new(file);
    let n_lines = count_reader.lines().count() - 1;

    let file = File::open(file_path).unwrap();
    let mut reader = ReaderBuilder::new().has_headers(true).trim(Trim::All).from_reader(file);

    let n_rows = n_lines;
    let n_columns = reader.headers().unwrap().len();

    let flat_data: Array1<f32> = reader.records().into_iter().flat_map(|rec| {
        rec.unwrap().iter().map(|b| {
            f32::from_str(b).unwrap()
        }).collect::<Vec<f32>>()
    }).collect();

    flat_data.into_shape((n_rows, n_columns)).expect("Could not deserialize sent data")
}
