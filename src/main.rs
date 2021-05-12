use std::time::SystemTime;

use actix::prelude::*;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ndarray::{Axis, s};
use num_traits::Float;
use log::*;
use structopt::StructOpt;

use data_manager::data_reader::{DataReader, DataReceiver};
use crate::meanshift::{MeanShift, MeanShiftMessage};
use crate::pca::{PCA, PCAMessage};
use crate::parameters::{Parameters, Role};
use crate::data_manager::DataManager;
use crate::cluster_listener::ClusterMemberListener;
use actix_telepathy::Cluster;

mod meanshift;
mod pca;
mod data_manager;
mod parameters;
mod cluster_listener;


fn main() {
    env_logger::init();
    let params: Parameters = Parameters::from_args();
    debug!("Parameters: {:?}", params);

    let system = System::new("S2G++");

    let host = params.local_host;
    let seed_nodes = match &params.role {
        Role::Sub { mainhost } => vec![mainhost.clone()],
        _ => vec![]
    };

    let cluster = Cluster::new(host, seed_nodes);
    let cluster_listener = ClusterMemberListener::new(params).start();


    system.run();
}
