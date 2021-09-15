use actix::prelude::*;
use log::*;
use structopt::StructOpt;

use crate::parameters::{Parameters, Role};

use crate::cluster_listener::ClusterMemberListener;
use actix_telepathy::Cluster;
use crate::training::{Training, StartTrainingMessage};
use crate::utils::ClusterNodes;

mod data_manager;
mod parameters;
mod cluster_listener;
mod training;
mod utils;
mod messages;


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


    let training = Training::new(params.clone()).start();
    if params.n_cluster_nodes > 1 {
        let _cluster = Cluster::new(host, seed_nodes);
        let _cluster_listener = ClusterMemberListener::new(params, training).start();
    } else {
        let nodes = ClusterNodes::new();
        training.do_send(StartTrainingMessage { nodes });
    }

    system.run().unwrap();
}
