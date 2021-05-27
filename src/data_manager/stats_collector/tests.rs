use ndarray::prelude::*;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use crate::data_manager::data_reader::*;
use std::sync::{Arc, Mutex};
use ndarray_linalg::assert::close_l1;
use std::net::SocketAddr;
use actix_broker::BrokerSubscribe;
use std::collections::HashMap;
use actix::dev::MessageResponse;
use actix::clock::delay_for;
use std::time::Duration;
use port_scanner::request_open_port;
use rayon::prelude::*;
use crate::data_manager::stats_collector::{DatasetStats, StatsCollector, DatasetStatsMessage};
use ndarray::{Data, ArcArray2};
use crate::parameters::Parameters;
use crate::utils::{ClusterNodes, Stats};
use log::*;

#[derive(Default)]
struct OwnListener {
    pub data: ArcArray2<f32>,
    pub parameters: Parameters,
    pub main: bool,
    pub cluster_nodes: HashMap<usize, RemoteAddr>,
    pub result: Arc<Mutex<Option<DatasetStats>>>
}
impl ClusterListener for OwnListener {}
impl Supervised for OwnListener {}

impl OwnListener {
    fn collect_stats(&mut self, ctx: &mut Context<Self>) {
        let _stats_collector = StatsCollector::new(
            self.data.clone(),
            self.parameters.clone(),
            ClusterNodes::from(self.cluster_nodes.clone()),
            ctx.address().recipient()
        ).start();
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
            ClusterLog::NewMember(addr, remote_addr) => {
                println!("new member {}", addr.to_string());
                if self.main {
                    self.cluster_nodes.insert(1, remote_addr);
                } else {
                    self.cluster_nodes.insert(0, remote_addr);
                }
                self.collect_stats(ctx)
            },
            ClusterLog::MemberLeft(_addr) => {}
        }
    }
}

impl Handler<DatasetStatsMessage> for OwnListener {
    type Result = ();

    fn handle(&mut self, msg: DatasetStatsMessage, ctx: &mut Self::Context) -> Self::Result {
        *(self.result.lock().unwrap()) = Some(msg.dataset_stats);
    }
}

struct TestParams {
    ip: SocketAddr,
    seeds: Vec<SocketAddr>,
    main: bool,
    expected: DatasetStats,
    data: ArcArray2<f32>
}

#[test]
fn test_dataset_stats() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();

    let expected = read_data_("./data/test.csv");
    let expected_std = expected.std_axis(Axis(0), 0.0);
    let expected_min = expected.to_shared().min_axis(Axis(0));
    let expected_max = expected.to_shared().max_axis(Axis(0));

    let dataset_stats = DatasetStats::new(expected_std, expected_min, expected_max);

    let arr = [
        TestParams {ip: ip1.clone(), seeds: vec![], main: true, expected: dataset_stats.clone(), data: expected.slice(s![..55, ..]).to_shared() },
        TestParams {ip: ip2.clone(), seeds: vec![ip1.clone()], main: false, expected: dataset_stats.clone(), data: expected.slice(s![50.., ..]).to_shared() },
    ];
    arr.into_par_iter().for_each(|p| start_stats(p.ip, p.seeds.clone(), p.main, p.expected.clone(), p.data.clone()));
}

#[actix_rt::main]
async fn start_stats(ip_address: SocketAddr, seed_nodes: Vec<SocketAddr>, main: bool, expected: DatasetStats, data: ArcArray2<f32>) {
    let _cluster = Cluster::new(ip_address, seed_nodes);
    let result: Arc<Mutex<Option<DatasetStats>>> = Arc::new(Mutex::new(None));

    let mut parameters = Parameters::default();
    parameters.n_cluster_nodes = 2;
    parameters.pattern_length = 5;

    let _listener = OwnListener {
        data,
        parameters,
        main,
        cluster_nodes: Default::default(),
        result: result.clone()
    }.start();
    delay_for(Duration::from_millis(500)).await;

    let guarded_dataset_stats = (result.lock().unwrap());
    let received_dataset_stats = guarded_dataset_stats.as_ref().unwrap();

    close_l1(received_dataset_stats.std_col.as_ref().unwrap(), expected.std_col.as_ref().unwrap(), 0.005);
    close_l1(received_dataset_stats.min_col.as_ref().unwrap(), expected.min_col.as_ref().unwrap(), 0.0005);
    close_l1(received_dataset_stats.max_col.as_ref().unwrap(), expected.max_col.as_ref().unwrap(), 0.0005);
}
