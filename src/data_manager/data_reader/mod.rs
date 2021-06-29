pub(crate) mod messages;

pub use messages::DataReceivedMessage;

use ndarray::prelude::*;
use csv::{ReaderBuilder, Trim};

use std::fs::{File};
use std::str::FromStr;
use actix::{Addr, Actor};
pub use crate::data_manager::data_reader::messages::DataPartitionMessage;
use std::io::{BufReader, BufRead};
use num_integer::Integer;
use indicatif::ProgressBar;

use crate::utils::{AnyClusterNodesIterator};
use crate::utils::ConsoleLogger;
use std::ops::Not;
use crate::data_manager::DataManager;


pub struct DataReading {
    pub with_header: bool,
    pub overlap: usize
}


pub trait DataReader {
    fn read_csv(&mut self, file_path: &str, addr: Addr<Self>) where Self: Actor;
}


impl DataReader for DataManager {
    fn read_csv(&mut self, file_path: &str, addr: Addr<Self>) {
        let file = File::open(file_path).unwrap();
        let count_reader = BufReader::new(file);
        let n_lines = if self.data_reading.as_ref().unwrap().with_header {
            count_reader.lines().count() - 1
        } else {
            count_reader.lines().count()
        };

        let mut nodes = self.nodes.clone();
        nodes.change_ids("DataManager");
        let receivers = nodes.to_any(addr);
        let file = File::open(&file_path).unwrap();
        let mut reader = ReaderBuilder::new().has_headers(true).trim(Trim::All).from_reader(file);

        let partition_len = n_lines.div_floor(&receivers.len());
        let last_overlap = n_lines - (partition_len * receivers.len());
        let mut receiver_iterator: AnyClusterNodesIterator<Self> = receivers.clone().into_iter();
        let mut buffer = vec![];
        let mut overlap_buffer = vec![];

        ConsoleLogger::new(1, 8, "Reading Data".to_string()).print();
        let bar = ProgressBar::new(n_lines as u64);
        for record in reader.records() {
            match record {
                Ok(r) => {
                    let strings = r.iter().map(|x| x.to_string()).collect();
                    if buffer.len() < partition_len {
                        buffer.push(strings);
                    } else if receivers.len() > 1 &&
                        ((overlap_buffer.len() < self.data_reading.as_ref().unwrap().overlap && receiver_iterator.last_position().not())
                        || (overlap_buffer.len() < last_overlap && receiver_iterator.last_position()))  {
                        overlap_buffer.push(strings);
                    } else {
                        let mut data = buffer.clone();
                        data.extend(overlap_buffer.clone());
                        receiver_iterator.next().unwrap().do_send(DataPartitionMessage { data });
                        println!("Sent data to receiver {}", receiver_iterator.get_position() - 1);

                        buffer.clear();
                        buffer.extend(overlap_buffer.clone());
                        buffer.push(strings);
                        overlap_buffer.clear();
                    }
                },
                Err(e) => panic!(e)
            }
            bar.inc(1);
        }

        let mut data = buffer.clone();
        data.extend(overlap_buffer.clone());
        receiver_iterator.next().unwrap().do_send(DataPartitionMessage { data });
        bar.finish_and_clear();
        println!("Sent data to receiver {}", receiver_iterator.get_position() - 1);
    }
}

pub fn read_data_(file_path: &str) -> Array2<f32> {
    let file = File::open(file_path).unwrap();
    let count_reader = BufReader::new(file);
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
