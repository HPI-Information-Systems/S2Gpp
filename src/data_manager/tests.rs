use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::prelude::*;

use std::sync::{Arc, Mutex};

use actix_broker::BrokerSubscribe;
use ndarray_linalg::assert::close_l1;
use std::collections::HashMap;
use std::net::SocketAddr;

use crate::data_manager::messages::DataLoadedAndProcessed;
use crate::data_manager::{DataManager, LoadDataMessage};
use crate::parameters::{Parameters, Role};
use crate::utils::ClusterNodes;
use log::*;
use ndarray::arr3;
use port_scanner::request_open_port;
use rayon::prelude::*;
use tokio::time::{sleep, Duration};

#[derive(Default)]
struct DataResult {
    pub phase_space: Option<ArcArray<f32, Ix3>>,
    pub data_ref: Option<ArcArray<f32, Ix3>>,
}

struct OwnListener {
    pub cluster_nodes: HashMap<usize, RemoteAddr>,
    pub parameters: Parameters,
    pub result: Arc<Mutex<DataResult>>,
}

impl ClusterListener for OwnListener {}
impl Supervised for OwnListener {}

impl OwnListener {
    fn load_data(&mut self, ctx: &mut Context<Self>) {
        let cluster_nodes = ClusterNodes::from(self.cluster_nodes.clone());
        let dm = DataManager::new(
            cluster_nodes.clone(),
            self.parameters.clone(),
            ctx.address().recipient(),
        )
        .start();
        dm.do_send(LoadDataMessage {
            nodes: cluster_nodes,
        });
    }
}

impl Actor for OwnListener {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Context<Self>) {
        self.subscribe_system_async::<ClusterLog>(ctx);
    }
}

impl Handler<ClusterLog> for OwnListener {
    type Result = ();

    fn handle(&mut self, msg: ClusterLog, ctx: &mut Context<Self>) -> Self::Result {
        match msg {
            ClusterLog::NewMember(node) => {
                debug!("new member {}", node.socket_addr.to_string());
                match &self.parameters.role {
                    Role::Main { .. } => self.cluster_nodes.insert(1, node.get_remote_addr("".to_string())),
                    _ => self.cluster_nodes.insert(0, node.get_remote_addr("".to_string())),
                };
                self.load_data(ctx)
            }
            ClusterLog::MemberLeft(_addr) => {}
        }
    }
}

impl Handler<DataLoadedAndProcessed> for OwnListener {
    type Result = ();

    fn handle(&mut self, msg: DataLoadedAndProcessed, _ctx: &mut Self::Context) -> Self::Result {
        let mut dataresult = self.result.lock().unwrap();
        (*dataresult).data_ref = Some(msg.data_ref);
        (*dataresult).phase_space = Some(msg.phase_space);
    }
}

struct TestParams {
    ip: SocketAddr,
    seeds: Vec<SocketAddr>,
    main: bool,
    expected_phase_space: ArcArray<f32, Ix3>,
    expected_data_ref: ArcArray<f32, Ix3>,
}

#[test]
#[ignore] //gitlab workflows don't get the timing right
fn test_data_management() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000))
        .parse()
        .unwrap();

    let expected_main_phase_space = arr3(&[
        [
            [2.18241731, 4.45332896, 2.18779886],
            [2.66874929, 3.94888408, 2.75649561],
            [2.7519442, 3.59467315, 2.90582321],
            [2.40929932, 2.88509776, 3.00519938],
            [3.19799933, 2.14067433, 3.50562114],
            [3.16218139, 2.19186071, 3.4127196],
            [3.35120425, 2.04415429, 3.52248153],
            [2.87850828, 2.52159156, 3.21316997],
            [2.05225496, 3.13280805, 3.09949073],
            [1.96246017, 3.51603486, 2.89856417],
            [1.64841517, 3.98851209, 3.0967204],
            [2.10740366, 3.59520781, 2.84686332],
            [2.40328684, 3.04233411, 2.85587264],
            [2.74209838, 2.48709708, 2.71956808],
        ],
        [
            [2.66874929, 3.94888408, 2.75649561],
            [2.7519442, 3.59467315, 2.90582321],
            [2.40929932, 2.88509776, 3.00519938],
            [3.19799933, 2.14067433, 3.50562114],
            [3.16218139, 2.19186071, 3.4127196],
            [3.35120425, 2.04415429, 3.52248153],
            [2.87850828, 2.52159156, 3.21316997],
            [2.05225496, 3.13280805, 3.09949073],
            [1.96246017, 3.51603486, 2.89856417],
            [1.64841517, 3.98851209, 3.0967204],
            [2.10740366, 3.59520781, 2.84686332],
            [2.40328684, 3.04233411, 2.85587264],
            [2.74209838, 2.48709708, 2.71956808],
            [2.59091275, 2.34457487, 2.3470922],
        ],
        [
            [2.7519442, 3.59467315, 2.90582321],
            [2.40929932, 2.88509776, 3.00519938],
            [3.19799933, 2.14067433, 3.50562114],
            [3.16218139, 2.19186071, 3.4127196],
            [3.35120425, 2.04415429, 3.52248153],
            [2.87850828, 2.52159156, 3.21316997],
            [2.05225496, 3.13280805, 3.09949073],
            [1.96246017, 3.51603486, 2.89856417],
            [1.64841517, 3.98851209, 3.0967204],
            [2.10740366, 3.59520781, 2.84686332],
            [2.40328684, 3.04233411, 2.85587264],
            [2.74209838, 2.48709708, 2.71956808],
            [2.59091275, 2.34457487, 2.3470922],
            [3.10206848, 2.36022877, 2.47890813],
        ],
    ]);
    let expected_main_data_ref = arr3(&[
        [
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
            [0.03510282, 0.23744748, 0.0171921],
        ],
        [
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
            [0.0953012, 0.29507496, 0.07676196],
        ],
        [
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
            [0.15549958, 0.35270244, 0.13633183],
        ],
    ]);
    let expected_sub_phase_space = arr3(&[
        [
            [3.8206407, 2.36796312, 2.04560368],
            [3.69942443, 2.69881685, 1.92483223],
            [3.43937846, 2.47279845, 2.82239268],
            [2.68208942, 2.62061081, 2.47343859],
            [2.7694733, 2.76932965, 3.06451969],
            [3.11964936, 2.8656878, 3.40156312],
            [3.52588479, 2.65253322, 3.51483735],
            [3.70875014, 2.34566095, 4.06043449],
            [3.81531461, 2.84629147, 3.61087337],
            [4.01906744, 3.09874158, 3.6589223],
            [3.61144769, 2.81179194, 3.08275119],
            [2.90571607, 2.68702359, 2.5996823],
            [3.065959, 2.82708009, 2.32338734],
            [2.94808153, 3.21541566, 2.10343877],
        ],
        [
            [3.69942443, 2.69881685, 1.92483223],
            [3.43937846, 2.47279845, 2.82239268],
            [2.68208942, 2.62061081, 2.47343859],
            [2.7694733, 2.76932965, 3.06451969],
            [3.11964936, 2.8656878, 3.40156312],
            [3.52588479, 2.65253322, 3.51483735],
            [3.70875014, 2.34566095, 4.06043449],
            [3.81531461, 2.84629147, 3.61087337],
            [4.01906744, 3.09874158, 3.6589223],
            [3.61144769, 2.81179194, 3.08275119],
            [2.90571607, 2.68702359, 2.5996823],
            [3.065959, 2.82708009, 2.32338734],
            [2.94808153, 3.21541566, 2.10343877],
            [3.10133045, 2.82264117, 2.03362369],
        ],
        [
            [3.43937846, 2.47279845, 2.82239268],
            [2.68208942, 2.62061081, 2.47343859],
            [2.7694733, 2.76932965, 3.06451969],
            [3.11964936, 2.8656878, 3.40156312],
            [3.52588479, 2.65253322, 3.51483735],
            [3.70875014, 2.34566095, 4.06043449],
            [3.81531461, 2.84629147, 3.61087337],
            [4.01906744, 3.09874158, 3.6589223],
            [3.61144769, 2.81179194, 3.08275119],
            [2.90571607, 2.68702359, 2.5996823],
            [3.065959, 2.82708009, 2.32338734],
            [2.94808153, 3.21541566, 2.10343877],
            [3.10133045, 2.82264117, 2.03362369],
            [3.11609803, 2.0462867, 1.60809344],
        ],
    ]);
    let expected_sub_data_ref = expected_main_data_ref.clone();

    let arr = [
        TestParams {
            ip: ip1.clone(),
            seeds: vec![],
            main: true,
            expected_phase_space: expected_main_phase_space.to_shared(),
            expected_data_ref: expected_main_data_ref.to_shared(),
        },
        TestParams {
            ip: ip2.clone(),
            seeds: vec![ip1.clone()],
            main: false,
            expected_phase_space: expected_sub_phase_space.to_shared(),
            expected_data_ref: expected_sub_data_ref.to_shared(),
        },
    ];
    arr.into_par_iter().for_each(|p| {
        start_reading(
            p.ip,
            p.seeds.clone(),
            p.main,
            p.expected_phase_space.clone(),
            p.expected_data_ref.clone(),
        )
    });
}

#[actix_rt::main]
async fn start_reading(
    ip_address: SocketAddr,
    seed_nodes: Vec<SocketAddr>,
    main: bool,
    expected_phase_space: ArcArray<f32, Ix3>,
    expected_data_ref: ArcArray<f32, Ix3>,
) {
    let result: Arc<Mutex<DataResult>> = Arc::new(Mutex::new(DataResult::default()));

    let parameters = Parameters {
        role: if main {
            Role::Main {
                data_path: Some("data/test.csv".to_string()),
            }
        } else {
            Role::Sub {
                mainhost: seed_nodes[0],
            }
        },
        local_host: ip_address,
        pattern_length: 20,
        latent: 6,
        rate: 100,
        n_threads: 1,
        n_cluster_nodes: 2,
        ..Default::default()
    };

    let _listener = OwnListener {
        cluster_nodes: HashMap::new(),
        parameters,
        result: result.clone(),
    }
    .start();

    sleep(Duration::from_millis(200)).await;

    let _cluster = Cluster::new(ip_address, seed_nodes.clone());

    sleep(Duration::from_millis(200)).await;

    let data_result = result.lock().unwrap();

    println!(
        "phase_space {:?}",
        data_result.phase_space.as_ref().unwrap().shape()
    );
    println!(
        "data_ref {:?}",
        data_result.data_ref.as_ref().unwrap().shape()
    );

    close_l1(
        &data_result
            .phase_space
            .as_ref()
            .unwrap()
            .slice(s![..3_usize, .., ..])
            .to_owned(),
        &expected_phase_space,
        0.0005,
    );
    close_l1(
        &data_result
            .data_ref
            .as_ref()
            .unwrap()
            .slice(s![..3_usize, .., ..])
            .to_owned(),
        &expected_data_ref,
        0.0005,
    );
}
