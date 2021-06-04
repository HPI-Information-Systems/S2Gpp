use crate::training::Training;
use crate::utils::PolarCoords;
use std::collections::{HashMap, HashSet};
use ndarray::{Array1, Axis, arr1};
use num_integer::Integer;
use std::f32::consts::PI;
use actix_telepathy::RemoteAddr;
use num_traits::real::Real;
use crate::training::messages::{SegmentMessage, SegmentedMessage};
use actix::prelude::*;
use actix::dev::MessageResponse;
use serde::{Serialize, Serializer, Deserialize, Deserializer};


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointWithId {
    pub id: usize,
    pub coords: Array1<f32>
}

pub type PointsForNodes = HashMap<usize, Vec<PointWithId>>;

pub struct Segmentation {
    pub rate: usize,
    /// list of lists with (data ID, data point) as elements and inner list at position segment ID
    pub segments: Vec<Vec<PointWithId>>,
    /// list with (data ID, data point) as elements
    pub own_segment: Vec<PointWithId>,
    pub n_received: usize
}

pub trait Segmenter {
    fn segment(&mut self);
    fn assign_segments(&mut self);
}

impl Segmenter for Training {
    fn segment(&mut self) {
        for _ in 0..self.segmentation.rate {
            self.segmentation.segments.push(vec![]);
        }

        let partition_size = (self.dataset_stats.as_ref().unwrap().n.as_ref().unwrap().clone() as f32 / self.nodes.len_incl_own() as f32).floor() as usize;
        println!("partition_size {}", partition_size);
        let mut id = self.nodes.get_own_idx() * partition_size;
        for x in self.rotated.as_ref().unwrap().axis_iter(Axis(0)) {
            let polar = x.to_polar();
            let segment_id = get_segment_id(polar[1], self.segmentation.rate);
            self.segmentation.segments.get_mut(segment_id).unwrap().push(PointWithId { id, coords: x.iter().map(|x| x.clone()).collect() });
            id += 1;
        }
    }

    fn assign_segments(&mut self) {
        let own_id = self.nodes.get_own_idx();
        let next_id = (own_id + 1) % (&self.nodes.len_incl_own());

        let mut node_segments = PointsForNodes::new();

        for (i, _) in self.nodes.iter() {
            node_segments.insert(i.clone(), vec![]);
        }

        let segments_per_node = (self.segmentation.rate as f32 / self.nodes.len_incl_own() as f32).floor() as usize;
        let _rest = self.segmentation.rate % (&self.nodes.len_incl_own());

        for i in 0..self.segmentation.rate {
            let node_id = i / segments_per_node;
            let points = self.segmentation.segments.get_mut(i).unwrap();
            let mut points_cloned = vec![];

            let points_to_add = if i.mod_floor(&segments_per_node) == 0 {
                points_cloned = points.clone();

                let prev_node = ((node_id as isize - 1).rem_euclid(self.nodes.len_incl_own() as isize)) as usize;
                match node_segments.get_mut(&prev_node) {
                    Some(prev_node_segments) => prev_node_segments.append(points),
                    None => self.segmentation.own_segment.append(points)
                }

                points_cloned.as_mut()
            } else {
                points
            };

            match node_segments.get_mut(&node_id) {
                Some(this_node_segments) => this_node_segments.append(points_to_add),
                None => self.segmentation.own_segment.append(points_to_add)
            };
        }

        self.nodes.get(&next_id).unwrap().do_send(SegmentMessage { segments: node_segments });
    }
}

fn get_segment_id(angle: f32, n_segments: usize) -> usize {
    let positive_angle = (2.0 * PI) + angle;
    let segment_size = (2.0 * PI) / (n_segments as f32);
    (positive_angle / segment_size).floor() as usize % n_segments
}

impl Handler<SegmentMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SegmentMessage, ctx: &mut Self::Context) -> Self::Result {
        self.segmentation.n_received += 1;
        let own_id = self.nodes.get_own_idx();
        let next_id = (own_id + 1) % (&self.nodes.len_incl_own());
        let mut segments = msg.segments;
        let mut own_points = segments.remove(&own_id).unwrap();

        self.segmentation.own_segment.append(&mut own_points);
        if self.segmentation.n_received < self.nodes.len() {
            self.nodes.get(&next_id).unwrap().do_send(SegmentMessage { segments });
        } else {
            self.segmentation.own_segment.sort_by_key(|r| r.id.clone());
            self.segmentation.own_segment.dedup_by_key(|r| r.id.clone());
            ctx.address().do_send(SegmentedMessage);
            self.segmentation.segments.clear();
        }
    }
}

#[cfg(test)]
mod tests {
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
    use crate::pca::RotatedMessage;
    use crate::data_manager::DatasetStats;
    use crate::training::segmenter::get_segment_id;
    use std::sync::{Mutex, Arc};
    use std::ops::Deref;

    #[derive(Message)]
    #[rtype(result = "Result<usize, ()>")]
    struct CheckSegments {
        pub expected_segments: Vec<usize>
    }

    impl Handler<CheckSegments> for Training {
        type Result = Result<usize, ()>;

        fn handle(&mut self, msg: CheckSegments, ctx: &mut Self::Context) -> Self::Result {
            let points = self.segmentation.own_segment.clone();
            println!("expected received {}", points.len());
            let segment_size = (2.0 * PI) / 100.;

            let mut result = true;
            for rotated in points.iter() {
                let polar: Array1<f32> = rotated.coords.to_polar();
                let segment_id = get_segment_id(polar[1], 100);
                //println!("segment_id: {}", segment_id);
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
            self.nodes = msg.nodes;
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
            self.training_addr.do_send(RotatedMessage { rotated: self.rotated.clone() });
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
        let ip1: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();
        let ip2: SocketAddr = format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap();

        let timeseries = gen_spun_ring(100, 1000, 10);
        let mut expected_2nd: Vec<usize> = (50..100).into_iter().collect();
        expected_2nd.push(0);

        let arr = [
            TestParams {ip: ip1.clone(), seeds: vec![], main: true,
                data: timeseries.slice(s![..70, ..]).to_owned(),
                expected_segments: (0..51).into_iter().collect()
            },
            TestParams {ip: ip2.clone(), seeds: vec![ip1.clone()], main: false,
                data: timeseries.slice(s![50.., ..]).to_owned(),
                expected_segments: expected_2nd
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
}
