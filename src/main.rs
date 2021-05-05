use actix::prelude::*;

use crate::data::read_data;
use crate::meanshift::{MeanShift, MeanShiftMessage};
use std::time::SystemTime;
use kdtree::KdTree;
use ndarray::{Axis, s};
use kdtree::distance::squared_euclidean;
use num_traits::Float;
use crate::pca::{PCA, PCAMessage};

mod data;
mod meanshift;
mod pca;

fn main() {
    env_logger::init();

    let system = System::new("S2G++");

    let dataset = read_data("data/test.csv", 1, 0);
    let pca1 = PCA::new(None, 0, 2).start();
    let pca2 = PCA::new(None, 1, 2).start();
    let pca3 = PCA::new(None, 2, 2).start();
    pca1.do_send(PCAMessage { data: dataset.slice(s![..50, ..]).to_shared(), cluster_nodes: vec![pca1.clone(), pca2.clone(), pca3.clone()] });
    pca2.do_send(PCAMessage { data: dataset.slice(s![50..75, ..]).to_shared(), cluster_nodes: vec![pca1.clone(), pca2.clone(), pca3.clone()] });
    pca3.do_send(PCAMessage { data: dataset.slice(s![75.., ..]).to_shared(), cluster_nodes: vec![pca1.clone(), pca2.clone(), pca3.clone()] });

    system.run();
}
