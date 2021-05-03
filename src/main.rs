use actix::prelude::*;

use crate::data::read_data;
use crate::meanshift::{MeanShift, MeanShiftMessage};
use std::time::SystemTime;
use kdtree::KdTree;
use ndarray::Axis;
use kdtree::distance::squared_euclidean;
use num_traits::Float;

mod data;
mod meanshift;

fn main() {
    env_logger::init();

    let system = System::new("S2G++");

    let dataset = read_data("data/test.csv", 1, 0);
    let meanshift = MeanShift::new(20).start();
    meanshift.do_send(MeanShiftMessage { source: None, data: dataset });

    system.run();
}
