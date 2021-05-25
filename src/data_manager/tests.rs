use ndarray::prelude::*;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use crate::data_manager::data_reader::*;
use std::sync::{Arc, Mutex};
use crate::data_manager::data_reader::messages::DataReceivedMessage;
use ndarray_linalg::assert::close_l1;
use std::net::SocketAddr;
use actix_broker::BrokerSubscribe;
use std::collections::HashMap;
use actix::dev::MessageResponse;
use actix::clock::delay_for;
use std::time::Duration;
use port_scanner::request_open_port;
use rayon::prelude::*;
use log::*;
use crate::utils::ClusterNodes;

#[derive(Default)]
struct OwnListener {
    pub main: bool,
    pub cluster_nodes: HashMap<usize, RemoteAddr>,
    pub result: Arc<Mutex<Option<Array2<f32>>>>
}
impl ClusterListener for OwnListener {}
impl Supervised for OwnListener {}

impl OwnListener {
    fn read_data(&mut self, ctx: &mut Context<Self>) {
        let receiver = DataReceiver::new(Some(ctx.address().recipient())).start();
        if self.main {
            let mut cluster_nodes = ClusterNodes::from(self.cluster_nodes.clone());
            cluster_nodes.change_ids("DataReceiver");
            let data_reader = DataReader::new("./data/test.csv", cluster_nodes.to_any(receiver), 20).start();
        }
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
                debug!("new member {}", addr.to_string());
                if self.main {
                    self.cluster_nodes.insert(1, remote_addr);
                } else {
                    self.cluster_nodes.insert(0, remote_addr);
                }
                self.read_data(ctx)
            },
            ClusterLog::MemberLeft(_addr) => {}
        }
    }
}

impl Handler<DataReceivedMessage> for OwnListener {
    type Result = ();

    fn handle(&mut self, msg: DataReceivedMessage, ctx: &mut Self::Context) -> Self::Result {
        *(self.result.lock().unwrap()) = Some(msg.data);
    }
}

struct TestParams {
    ip: SocketAddr,
    seeds: Vec<SocketAddr>,
    main: bool,
    expected: Array2<f32>
}

#[test]
fn test_data_distribution() {
    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();

    let expected = read_data_("./data/test.csv");

    let arr = [
        TestParams {ip: ip1.clone(), seeds: vec![], main: true, expected: expected.slice(s![..70, ..]).to_owned()},
        TestParams {ip: ip2.clone(), seeds: vec![ip1.clone()], main: false, expected: expected.slice(s![50.., ..]).to_owned()},
    ];
    arr.into_par_iter().for_each(|p| start_reading(p.ip, p.seeds.clone(), p.main, p.expected.clone()));
}

#[actix_rt::main]
async fn start_reading(ip_address: SocketAddr, seed_nodes: Vec<SocketAddr>, main: bool, expected: Array2<f32>) {
    let _cluster = Cluster::new(ip_address, seed_nodes);
    let result: Arc<Mutex<Option<Array2<f32>>>> = Arc::new(Mutex::new(None));

    let _listener = OwnListener {
        main,
        cluster_nodes: Default::default(),
        result: result.clone()
    }.start();
    delay_for(Duration::from_millis(100)).await;
    close_l1((result.lock().unwrap()).as_ref().unwrap(), &expected, 0.0005);
}
