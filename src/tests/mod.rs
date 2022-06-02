pub mod utils;

use crate::cluster_listener::ClusterMemberListener;
use crate::data_manager::data_reader::read_data_;
use crate::parameters::{Parameters, Role};
use crate::training::{Clustering, StartTrainingMessage, Training};
use crate::utils::ClusterNodes;
use crate::{s2gpp, SyncInterface};
use actix::prelude::*;
use actix_rt::System;
use actix_telepathy::Cluster;
use ndarray_linalg::close_l1;
use port_scanner::request_open_port;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::env::temp_dir;
use std::fs::remove_file;

const ESTIMATED_SCORES_PATH: &str = "ts_0.csv.scores";
const EXPECTED_SCORES_PATH: &str = "data/ts_0.csv.scores";

fn get_output_path() -> String {
    let mut dir = temp_dir();
    dir.push(ESTIMATED_SCORES_PATH);
    dir.to_str().unwrap().to_string()
}

#[test]
#[ignore] // takes some time
fn global_test_kde_clustering() {
    let params: Parameters = Parameters {
        role: Role::Main {
            data_path: Some("data/ts_0.csv".to_string()),
        },
        local_host: "127.0.0.1:1992".parse().unwrap(),
        score_output_path: Some(get_output_path()),
        clustering: Clustering::MultiKDE,
        ..Default::default()
    };

    run_single_global_comut(params);

    check_for_comut_score();
}

#[test]
#[ignore] // takes some time
fn global_test_kde_clustering_provided_data() {
    let params: Parameters = Parameters {
        clustering: Clustering::MultiKDE,
        ..Default::default()
    };

    let data = read_data_("data/ts_0.csv");

    let result = s2gpp(params, Some(data)).unwrap();

    assert!(result.is_some());
}

#[test]
#[ignore] // takes some time
fn global_comut_distributed_2() {
    setup_distributed_global_comut(1)
}

fn setup_distributed_global_comut(n_subhosts: usize) {
    let mainhost = format!("127.0.0.1:{}", request_open_port().unwrap_or(1992))
        .parse()
        .unwrap();

    let mut parameters = vec![Parameters {
        role: Role::Main {
            data_path: Some("data/ts_0.csv".to_string()),
        },
        local_host: mainhost,
        score_output_path: Some(get_output_path()),
        n_cluster_nodes: n_subhosts + 1,
        ..Default::default()
    }];

    for i in 0..n_subhosts {
        let subhost = format!(
            "127.0.0.1:{}",
            request_open_port().unwrap_or((1993 + i) as u16)
        )
        .parse()
        .unwrap();

        parameters.push(Parameters {
            role: Role::Sub { mainhost },
            local_host: subhost,
            n_cluster_nodes: n_subhosts + 1,
            ..Default::default()
        });
    }

    parameters
        .into_par_iter()
        .for_each(|p| run_single_global_comut(p));

    check_for_comut_score();
}

fn run_single_global_comut(params: Parameters) {
    println!("{:?}", params);

    let system = System::new();

    system.block_on(async {
        let host = params.local_host;
        let seed_nodes = match &params.role {
            Role::Sub { mainhost } => vec![mainhost.clone()],
            _ => vec![],
        };

        let training = Training::init(params.clone()).start();
        if params.n_cluster_nodes > 1 {
            let _cluster = Cluster::new(host, seed_nodes);
            let _cluster_listener = ClusterMemberListener::new(params, training).start();
        } else {
            let nodes = ClusterNodes::new();
            training.do_send(StartTrainingMessage {
                nodes,
                source: None,
                data: None,
            });
        }
    });

    system.run().unwrap();
}

fn check_for_comut_score() {
    let estimated_scores_path = get_output_path();
    let expected_scores = read_data_(EXPECTED_SCORES_PATH);
    let estimated_scores = read_data_(&estimated_scores_path);
    remove_file(&estimated_scores_path).expect("Could not delete test file!");

    close_l1(&estimated_scores, &expected_scores, 0.000001);
}
