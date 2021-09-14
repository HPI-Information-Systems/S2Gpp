use std::collections::HashMap;
use actix_telepathy::{RemoteAddr, ClusterListener, ClusterLog, Cluster};
use crate::parameters::{Parameters, Role};
use actix::{Supervised, Actor, Context, Handler, Addr, Message, Response, ActorContext, WrapFuture, ActorFuture, ContextFutureSpawner, MailboxError};
use actix_broker::BrokerSubscribe;
use crate::training::Training;
use ndarray::{Array2, ArcArray, ArrayBase, ArrayView2, Array1, arr1, concatenate, Axis, ArrayView1, s, stack};
use std::net::SocketAddr;
use port_scanner::request_open_port;
use rayon::prelude::*;
use actix::clock::delay_for;
use std::time::Duration;
use std::f32::consts::PI;
use num_integer::Integer;
use crate::utils::ClusterNodes;
use crate::utils::PolarCoords;
use actix::dev::MessageResponse;
use crate::data_manager::DatasetStats;
use crate::training::segmenter::get_segment_id;
use std::sync::{Mutex, Arc};
use std::ops::Deref;
use crate::training::rotation::RotationDoneMessage;

#[derive(Message)]
#[rtype(result = "Result<usize, ()>")]
struct CheckSegments {
    pub expected_segments: Vec<usize>
}

impl Handler<CheckSegments> for Training {
    type Result = Result<usize, ()>;

    fn handle(&mut self, msg: CheckSegments, ctx: &mut Self::Context) -> Self::Result {
        let points = self.segmentation.segments.clone();
        println!("expected received {}", &points.len());

        let mut result = true;
        for rotated in points.iter() {
            let segment_id = rotated.from.segment_id;
            if !msg.expected_segments.contains(&segment_id) {
                println!("{} not contained", &segment_id);
                result = false;
                break;
            }
        };
        if result {
            Ok(0)
        } else {
            Err(())
        }
    }
}

struct Tester {
    pub addr: Addr<Training>,
    pub succeeded: Arc<Mutex<bool>>
}

impl Actor for Tester {
    type Context = Context<Self>;
}

impl Handler<CheckSegments> for Tester {
    type Result = Result<usize, ()>;

    fn handle(&mut self, msg: CheckSegments, ctx: &mut Self::Context) -> Self::Result {
        self.addr.send(msg)
            .into_actor(self)
            .map(|res: Result<Result<usize, ()>, MailboxError>, act, ctx| match res {
                Ok(r) => match r {
                    Ok(_) => {
                        *(act.succeeded.lock().unwrap()) = true;
                    },
                    Err(_) => {
                        *(act.succeeded.lock().unwrap()) = false;
                    }
                },
                Err(_) => {
                    *(act.succeeded.lock().unwrap()) = false;
                }
            }).wait(ctx);
        Ok(0)
    }
}

#[derive(Message)]
#[rtype(Result = "()")]
struct NodesMessage {
    pub nodes: ClusterNodes
}

impl Handler<NodesMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: NodesMessage, ctx: &mut Self::Context) -> Self::Result {
        self.cluster_nodes = msg.nodes;
    }
}

fn gen_spun_ring(segments: usize, length: usize, spins: usize) -> Array2<f32> {
    let segment_size = (2.0 * PI) / segments as f32;
    let spin_size = length / spins;
    let points: Vec<Array1<f32>> = (0..length).into_iter().map(|x| {
        let theta = (2.0 * PI) * ((x % spin_size) as f32 / spin_size as f32);
        let radius = x as f32;
        arr1(&[radius * theta.cos(), radius * theta.sin()])
    }).collect();

    stack(Axis(0), &points.iter().map(|x| x.view()).collect::<Vec<ArrayView1<f32>>>()).unwrap()
}

struct OwnListener {
    pub cluster_nodes: HashMap<usize, RemoteAddr>,
    pub parameters: Parameters,
    pub training_addr: Addr<Training>,
    pub rotated: Array2<f32>
}

impl ClusterListener for OwnListener {}
impl Supervised for OwnListener {}

impl OwnListener {
    fn segment(&mut self, ctx: &mut Context<Self>) {
        self.training_addr.do_send(RotationDoneMessage);
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
                println!("new member {} own role {:?}", addr.to_string(), &self.parameters.role);
                match &self.parameters.role {
                    Role::Main {..} => self.cluster_nodes.insert(1, remote_addr),
                    _ => self.cluster_nodes.insert(0, remote_addr)
                };
                let mut nodes = ClusterNodes::from(self.cluster_nodes.clone());
                nodes.change_ids("Training");
                self.training_addr.do_send(NodesMessage { nodes });
                self.segment(ctx)
            },
            ClusterLog::MemberLeft(_addr) => {}
        }
    }
}

struct TestParams {
    ip: SocketAddr,
    seeds: Vec<SocketAddr>,
    main: bool,
    data: Array2<f32>,
    expected_segments: Vec<usize>
}


#[test]
#[ignore] //gitlab workflows don't get the timing right
fn test_segmenting() {
    env_logger::init();

    let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();
    let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();

    let timeseries = gen_spun_ring(100, 1000, 10);

    let arr = [
        TestParams {ip: ip1.clone(), seeds: vec![], main: true,
            data: timeseries.slice(s![..70, ..]).to_owned(),
            expected_segments: (0..50).into_iter().collect()
        },
        TestParams {ip: ip2.clone(), seeds: vec![ip1.clone()], main: false,
            data: timeseries.slice(s![50.., ..]).to_owned(),
            expected_segments: (50..100).into_iter().collect()
        },
    ];
    arr.into_par_iter().for_each(|p| start_segmentation(p.ip, p.seeds.clone(), p.main, p.data.clone(), p.expected_segments.clone()));
}

#[actix_rt::main]
async fn start_segmentation(ip_address: SocketAddr, seed_nodes: Vec<SocketAddr>, main: bool, data: Array2<f32>, expected_segments: Vec<usize>) {
    let parameters = Parameters {
        role: if main {
            Role::Main { data_path: "data/test.csv".to_string() }
        } else {
            Role::Sub { mainhost: seed_nodes[0] }
        },
        local_host: ip_address,
        pattern_length: 20,
        latent: 6,
        rate: 100,
        n_threads: 1,
        n_cluster_nodes: 2
    };

    let mut training = Training::new(parameters.clone());
    training.dataset_stats = Some(DatasetStats::new(arr1(&[1.0]), arr1(&[1.0]), arr1(&[1.0]), 100));
    let t_addr = training.start();
    let _listener = OwnListener {
        cluster_nodes: HashMap::new(),
        parameters,
        training_addr: t_addr.clone(),
        rotated: data
    }.start();

    let result = Arc::new(Mutex::new(false));
    let tester = Tester { addr: t_addr, succeeded: result.clone() }.start();

    delay_for(Duration::from_millis(200)).await;

    let _cluster = Cluster::new(ip_address, seed_nodes.clone());

    delay_for(Duration::from_millis(200)).await;

    tester.do_send(CheckSegments { expected_segments });

    delay_for(Duration::from_millis(1000)).await;

    assert!(*(result.lock().unwrap()))
}
