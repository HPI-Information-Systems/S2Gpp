pub mod utils;

use actix::prelude::*;
use actix_rt::System;
use actix_telepathy::Cluster;
use ndarray_linalg::close_l1;
use crate::cluster_listener::ClusterMemberListener;
use crate::data_manager::data_reader::read_data_;
use crate::parameters::{Parameters, Role};
use crate::training::{StartTrainingMessage, Training};
use crate::utils::ClusterNodes;

const ESTIMATED_SCORES_PATH: &str = "ts_0.csv.scores";
const EXPECTED_SCORES_PATH: &str = "data/ts_0.csv.scores";

#[test]
#[ignore] // takes some time
fn global_comut_not_distributed() {
    let params: Parameters = Parameters {
        role: Role::Main { data_path: "data/ts_0.csv".to_string() },
        local_host: "127.0.0.1:1992".parse().unwrap(),
        score_output_path: Some(ESTIMATED_SCORES_PATH.to_string()),
        ..Default::default()
    };

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

    let expected_scores = read_data_(EXPECTED_SCORES_PATH);
    let estimated_scores = read_data_(ESTIMATED_SCORES_PATH);

    close_l1(&estimated_scores, &expected_scores, 0.000001)
}
