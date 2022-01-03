use std::env::current_dir;
use actix_telepathy::Cluster;
use port_scanner::request_open_port;
use crate::parameters::{Parameters, Role};
use crate::training::{Training, StartTrainingMessage};
use crate::utils::ClusterNodes;
use actix::Actor;
use tokio::time::{Duration, sleep};
use std::path::Path;
use std::fs::remove_file;


#[actix_rt::test]
async fn show_graph_output() {
    let _cluster = Cluster::new(format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap(), vec![]);
    let graph_path = &format!("{}/data/_test_graph.dot", current_dir().unwrap().to_str().unwrap());
    let parameters = Parameters {
        role: Role::Main {
            data_path: "data/test.csv".to_string()
        },
        n_threads: 20,
        n_cluster_nodes: 1,
        pattern_length: 20,
        latent: 6,
        graph_output_path: Some(graph_path.to_string()),
        ..Default::default()
    };
    let training = Training::new(parameters).start();
    training.do_send(StartTrainingMessage {nodes: ClusterNodes::new()});

    sleep(Duration::from_millis(10000)).await;

    let path = Path::new(graph_path);
    assert!(path.exists());
    remove_file(path).expect("Could not delete test file!");
}
