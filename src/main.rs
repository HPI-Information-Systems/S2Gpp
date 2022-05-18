use actix::prelude::*;
use log::*;
use structopt::StructOpt;

use crate::parameters::{Parameters, Role};

use crate::cluster_listener::ClusterMemberListener;
use crate::training::{StartTrainingMessage, Training};
use crate::utils::ClusterNodes;
use actix_telepathy::Cluster;
use env_logger::Env;
use std::io::Write;

mod cluster_listener;
mod data_manager;
mod data_store;
mod messages;
mod parameters;
#[cfg(test)]
mod tests;
mod training;
mod utils;

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{} [S2G++]: {}", record.level(), record.args()))
        .init();

    let params: Parameters = Parameters::from_args();
    if params.explainability && params.n_cluster_nodes > 1 {
        panic!("The explainability feature is only available in a non-distributed setting.")
    }
    debug!("Parameters: {:?}", params);

    let system = System::new();

    system.block_on(async {
        let host = params.local_host;
        let seed_nodes = match &params.role {
            Role::Sub { mainhost } => vec![*mainhost],
            _ => vec![],
        };

        let training = Training::new(params.clone()).start();
        if params.n_cluster_nodes > 1 {
            let _cluster = Cluster::new(host, seed_nodes);
            let _cluster_listener = ClusterMemberListener::new(params, training).start();
        } else {
            let nodes = ClusterNodes::new();
            training.do_send(StartTrainingMessage { nodes });
        }
    });

    system.run().unwrap();
}
