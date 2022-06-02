use actix::prelude::*;
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::data_manager::data_reader::read_data_;
use actix_telepathy::Cluster;
use port_scanner::request_open_port;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tokio::time::{sleep, Duration};

use ndarray_linalg::close_l1;

use crate::parameters::Parameters;
use crate::tests::utils::TestClusterMemberListener;
use crate::training::rotation::pca::{PCAnalyzer, PCA};
use crate::training::rotation::PCAComponents;
use crate::training::Training;
use crate::utils::ClusterNodes;
use crate::SyncInterface;
use ndarray::ArcArray2;

#[derive(Message)]
#[rtype(Result = "()")]
struct StartPCA {
    data: ArcArray2<f32>,
}

impl Handler<StartPCA> for Training {
    type Result = ();

    fn handle(&mut self, msg: StartPCA, _ctx: &mut Self::Context) -> Self::Result {
        self.pca(msg.data);
    }
}

struct ResultChecker {
    components: Arc<Mutex<Option<Array2<f32>>>>,
}

impl Actor for ResultChecker {
    type Context = Context<Self>;
}

impl Handler<PCAComponents> for ResultChecker {
    type Result = ();

    fn handle(&mut self, msg: PCAComponents, _ctx: &mut Self::Context) -> Self::Result {
        *(self.components.lock().unwrap()) = Some(msg.components);
    }
}

struct TestParams {
    ip: SocketAddr,
    seeds: Vec<SocketAddr>,
    other_nodes: Vec<(usize, SocketAddr)>,
    main: bool,
    data: ArcArray2<f32>,
    expected: Array2<f32>,
}

#[test]
#[ignore]
fn test_single_pca() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();

    let dataset = read_data_("data/test.csv");
    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543],
    ]);

    let p = TestParams {
        ip: ip1.clone(),
        seeds: vec![],
        other_nodes: vec![],
        main: true,
        data: dataset.to_shared(),
        expected: expected.clone(),
    };

    run_single_pca_node(
        p.ip,
        p.seeds.clone(),
        p.other_nodes,
        p.main,
        p.data,
        p.expected,
        Parameters::default(),
    );
}

#[test]
#[ignore]
fn test_single_pca_parallel_2() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();

    let dataset = read_data_("data/test.csv");
    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543],
    ]);

    let p = TestParams {
        ip: ip1.clone(),
        seeds: vec![],
        other_nodes: vec![],
        main: true,
        data: dataset.to_shared(),
        expected: expected.clone(),
    };

    let parameters = Parameters {
        n_threads: 2,
        ..Default::default()
    };
    run_single_pca_node(
        p.ip,
        p.seeds.clone(),
        p.other_nodes,
        p.main,
        p.data,
        p.expected,
        parameters,
    );
}

#[test]
#[ignore]
fn test_single_pca_parallel_8() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();

    let dataset = read_data_("data/test.csv");

    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543],
    ]);

    let p = TestParams {
        ip: ip1.clone(),
        seeds: vec![],
        other_nodes: vec![],
        main: true,
        data: dataset.to_shared(),
        expected: expected.clone(),
    };

    let parameters = Parameters {
        n_threads: 8,
        ..Default::default()
    };
    run_single_pca_node(
        p.ip,
        p.seeds.clone(),
        p.other_nodes,
        p.main,
        p.data,
        p.expected,
        parameters,
    );
}

#[test]
#[ignore]
fn test_single_pca_parallel_20() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();

    let dataset = read_data_("data/test.csv");

    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543],
    ]);

    let p = TestParams {
        ip: ip1.clone(),
        seeds: vec![],
        other_nodes: vec![],
        main: true,
        data: dataset.to_shared(),
        expected: expected.clone(),
    };

    let parameters = Parameters {
        n_threads: 20,
        ..Default::default()
    };
    run_single_pca_node(
        p.ip,
        p.seeds.clone(),
        p.other_nodes,
        p.main,
        p.data,
        p.expected,
        parameters,
    );
}

#[test]
#[ignore]
fn test_distributed_pca_2_parallel() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();

    let dataset = read_data_("data/test.csv");
    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543],
    ]);

    let arr = [
        TestParams {
            ip: ip1.clone(),
            seeds: vec![],
            other_nodes: vec![(1, ip2.clone())],
            main: true,
            data: dataset.slice(s![..50, ..]).to_shared(),
            expected: expected.clone(),
        },
        TestParams {
            ip: ip2.clone(),
            seeds: vec![ip1.clone()],
            other_nodes: vec![(0, ip1.clone())],
            main: false,
            data: dataset.slice(s![50.., ..]).to_shared(),
            expected: expected,
        },
    ];

    let parameters = Parameters {
        n_threads: 2,
        ..Default::default()
    };
    arr.into_par_iter().for_each(|p| {
        run_single_pca_node(
            p.ip,
            p.seeds.clone(),
            p.other_nodes,
            p.main,
            p.data,
            p.expected,
            parameters.clone(),
        )
    });
}

#[test]
#[ignore]
fn test_distributed_pca_2() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();

    let dataset = read_data_("data/test.csv");
    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543],
    ]);

    let arr = [
        TestParams {
            ip: ip1.clone(),
            seeds: vec![],
            other_nodes: vec![(1, ip2.clone())],
            main: true,
            data: dataset.slice(s![..50, ..]).to_shared(),
            expected: expected.clone(),
        },
        TestParams {
            ip: ip2.clone(),
            seeds: vec![ip1.clone()],
            other_nodes: vec![(0, ip1.clone())],
            main: false,
            data: dataset.slice(s![50.., ..]).to_shared(),
            expected: expected,
        },
    ];
    arr.into_par_iter().for_each(|p| {
        run_single_pca_node(
            p.ip,
            p.seeds.clone(),
            p.other_nodes,
            p.main,
            p.data,
            p.expected,
            Parameters::default(),
        )
    });
}

#[test]
#[ignore]
fn test_distributed_pca_3() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();
    let ip3: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();

    let dataset = read_data_("data/test.csv");
    let expected: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543],
    ]);

    let arr = [
        TestParams {
            ip: ip1.clone(),
            seeds: vec![],
            other_nodes: vec![(1, ip2.clone()), (2, ip3.clone())],
            main: true,
            data: dataset.slice(s![..33, ..]).to_shared(),
            expected: expected.clone(),
        },
        TestParams {
            ip: ip2.clone(),
            seeds: vec![ip1.clone()],
            other_nodes: vec![(0, ip1.clone()), (2, ip3.clone())],
            main: false,
            data: dataset.slice(s![33..66, ..]).to_shared(),
            expected: expected.clone(),
        },
        TestParams {
            ip: ip3.clone(),
            seeds: vec![ip1.clone()],
            other_nodes: vec![(0, ip1.clone()), (1, ip2.clone())],
            main: false,
            data: dataset.slice(s![66.., ..]).to_shared(),
            expected: expected,
        },
    ];
    arr.into_par_iter().for_each(|p| {
        run_single_pca_node(
            p.ip,
            p.seeds.clone(),
            p.other_nodes,
            p.main,
            p.data,
            p.expected,
            Parameters::default(),
        )
    });
}

#[actix_rt::main]
async fn run_single_pca_node(
    ip_address: SocketAddr,
    seed_nodes: Vec<SocketAddr>,
    other_nodes: Vec<(usize, SocketAddr)>,
    main: bool,
    data: ArcArray2<f32>,
    expected: Array2<f32>,
    parameters: Parameters,
) {
    let arc_cluster_nodes = Arc::new(Mutex::new(None));
    let cloned_arc_cluster_nodes = arc_cluster_nodes.clone();
    let result = Arc::new(Mutex::new(None));
    let cloned = Arc::clone(&result);

    sleep(Duration::from_millis(200)).await;

    let _cluster = Cluster::new(ip_address.clone(), seed_nodes.clone());

    let cluster_nodes = if other_nodes.len() == 0 && main {
        ClusterNodes::new()
    } else {
        if seed_nodes.len() > 0 {
            let _cluster_listener = TestClusterMemberListener::new(
                main,
                seed_nodes[0],
                other_nodes.len() + 1,
                ip_address,
                cloned_arc_cluster_nodes,
            )
            .start();
        } else {
            let _cluster_listener = TestClusterMemberListener::new(
                main,
                ip_address,
                other_nodes.len() + 1,
                ip_address,
                cloned_arc_cluster_nodes,
            )
            .start();
        }
        sleep(Duration::from_millis(400)).await;
        (*arc_cluster_nodes.lock().unwrap())
            .as_ref()
            .unwrap()
            .clone()
    };

    let id = cluster_nodes.get_own_idx();

    let mut training = Training::init(parameters);
    training.cluster_nodes = cluster_nodes;
    training.rotation.pca = PCA::new(id, 2);
    training.rotation.pca.recipient =
        Some(ResultChecker { components: cloned }.start().recipient());
    let training_addr = training.start();
    training_addr.do_send(StartPCA { data });

    sleep(Duration::from_millis(3000)).await;

    let received = (*result.lock().unwrap())
        .as_ref()
        .expect("Not yet set!")
        .clone();
    println!("received: {:?}", received);
    close_l1(&received, &expected, 0.00001);
}
