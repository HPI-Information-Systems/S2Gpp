mod cluster_listener;

use ndarray::prelude::*;
use actix::prelude::*;
use crate::pca::*;
use rayon::prelude::*;

use std::sync::{Arc, Mutex};
use crate::data_manager::data_reader::read_data_;
use std::net::SocketAddr;
use port_scanner::request_open_port;
use actix_rt::time::delay_for;
use actix::clock::Duration;
use actix_telepathy::{Cluster, RemoteAddr};
use std::collections::HashMap;
use std::iter::FromIterator;
use ndarray_linalg::close_l1;
use crate::pca::tests::cluster_listener::TestClusterMemberListener;

struct PCAReceiver {
    result: Arc<Mutex<Option<Array2<f32>>>>
}

impl Actor for PCAReceiver {
    type Context = Context<Self>;

    fn stopped(&mut self, _ctx: &mut Context<Self>) {
        System::current().stop();
    }
}

impl Handler<PCAResponse> for PCAReceiver {
    type Result = ();

    fn handle(&mut self, msg: PCAResponse, ctx: &mut Self::Context) -> Self::Result {
        *(self.result.lock().unwrap()) = Some(msg.components);
        ctx.stop();
    }
}

struct TestParams {
    ip: SocketAddr,
    seeds: Vec<SocketAddr>,
    other_nodes: Vec<(usize, SocketAddr)>,
    main: bool,
    data: ArcArray2<f32>,
    expected: Array2<f32>
}

#[test]
#[ignore]
fn test_distributed_pca_2() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();


    let dataset = read_data_("data/test.csv");
    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543]
    ]);

    let arr = [
        TestParams {
            ip: ip1.clone(),
            seeds: vec![],
            other_nodes: vec![(1, ip2.clone())],
            main: true,
            data: dataset.slice(s![..50, ..]).to_shared(),
            expected: expected.clone()
        },
        TestParams {
            ip: ip2.clone(),
            seeds: vec![ip1.clone()],
            other_nodes: vec![(0, ip1.clone())],
            main: false,
            data: dataset.slice(s![50.., ..]).to_shared(),
            expected: expected
        },
    ];
    arr.into_par_iter().for_each(|p| run_single_pca_node(p.ip, p.seeds.clone(), p.other_nodes, p.main, p.data, p.expected));
}

#[test]
#[ignore]
fn test_distributed_pca_3() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();
    let ip3: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();


    let dataset = read_data_("data/test.csv");
    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543]
    ]);

    let arr = [
        TestParams {
            ip: ip1.clone(),
            seeds: vec![],
            other_nodes: vec![(1, ip2.clone()), (2, ip3.clone())],
            main: true,
            data: dataset.slice(s![..33, ..]).to_shared(),
            expected: expected.clone()
        },
        TestParams {
            ip: ip2.clone(),
            seeds: vec![ip1.clone()],
            other_nodes: vec![(0, ip1.clone()), (2, ip3.clone())],
            main: false,
            data: dataset.slice(s![33..66, ..]).to_shared(),
            expected: expected.clone()
        },
        TestParams {
            ip: ip3.clone(),
            seeds: vec![ip1.clone()],
            other_nodes: vec![(0, ip1.clone()), (1, ip2.clone())],
            main: false,
            data: dataset.slice(s![66.., ..]).to_shared(),
            expected: expected
        },
    ];
    arr.into_par_iter().for_each(|p| run_single_pca_node(p.ip, p.seeds.clone(), p.other_nodes, p.main, p.data, p.expected));
}

#[actix_rt::main]
async fn run_single_pca_node(ip_address: SocketAddr, seed_nodes: Vec<SocketAddr>, other_nodes: Vec<(usize, SocketAddr)>, main: bool, data: ArcArray2<f32>, expected: Array2<f32>) {

    let arc_cluster_nodes = Arc::new(Mutex::new(None));
    let cloned_arc_cluster_nodes = arc_cluster_nodes.clone();
    let result = Arc::new(Mutex::new(None));
    let cloned = Arc::clone(&result);


    delay_for(Duration::from_millis(200)).await;

    let _cluster = Cluster::new(ip_address.clone(), seed_nodes.clone());
    if seed_nodes.len() > 0 {
        let _cluster_listener = TestClusterMemberListener::new(main, seed_nodes[0], other_nodes.len() + 1, ip_address, cloned_arc_cluster_nodes).start();
    } else {
        let _cluster_listener = TestClusterMemberListener::new(main, ip_address, other_nodes.len() + 1, ip_address, cloned_arc_cluster_nodes).start();
    }

    delay_for(Duration::from_millis(200)).await;

    let cluster_nodes: ClusterNodes = (*arc_cluster_nodes.lock().unwrap()).as_ref().unwrap().clone();

    let id = cluster_nodes.get_own_idx();

    let pca = PCA::new(cluster_nodes, Some(PCAReceiver {result: cloned}.start().recipient()), id, 2).start();
    pca.do_send(PCAMessage { data });

    delay_for(Duration::from_millis(3000)).await;

    let received = (*result.lock().unwrap()).as_ref().expect("Not yet set!").clone();
    close_l1(&received, &expected, 0.00001);
}
