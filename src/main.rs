use actix::prelude::*;

use crate::data_reader::{DataReader, DataReceiver};
use crate::meanshift::{MeanShift, MeanShiftMessage};
use std::time::SystemTime;
use kdtree::KdTree;
use ndarray::{Axis, s};
use kdtree::distance::squared_euclidean;
use num_traits::Float;
use crate::pca::{PCA, PCAMessage};

mod data_reader;
mod meanshift;
mod pca;


fn main() {
    env_logger::init();

    let system = System::new("S2G++");

    let data_receiver = DataReceiver::new(None).start();
    let data_receiver2 = DataReceiver::new(None).start();

    let data_reader = DataReader::new("data/test.csv", vec![data_receiver.recipient(), data_receiver2.recipient()], 5).start();

    system.run();
}
