use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::{Array1, Axis, ArcArray2};
use crate::data_manager::stats_collector::messages::{StdNodeMessage, StdDoneMessage};
use crate::parameters::{Parameters, Role};

#[derive(Default, RemoteActor)]
#[remote_messages(StdNodeMessage, StdDoneMessage)]
pub struct StdCalculator {
    data: ArcArray2<f32>,
    parameters: Parameters,
    main_node: Option<RemoteAddr>,
    std_nodes: Vec<RemoteAddr>,
    n: Option<usize>,
    mean: Option<Array1<f32>>,
    m2: Option<Array1<f32>>,
    std: Option<Array1<f32>>,
    receiver: Option<Recipient<StdDoneMessage>>
}

impl StdCalculator {
    pub fn new(data: ArcArray2<f32>, main_node: Option<RemoteAddr>, parameters: Parameters, receiver: Recipient<StdDoneMessage>) -> Self{
        Self {
            data,
            parameters,
            main_node,
            std_nodes: vec![],
            n: None,
            mean: None,
            m2: None,
            std: None,
            receiver: Some(receiver)
        }
    }

    fn set_intermediate(&mut self,
                n: usize,
                mean: Array1<f32>,
                m2: Array1<f32>) {
        self.n = Some(n);
        self.mean = Some(mean);
        self.m2 = Some(m2);
    }

    fn calculate_var(&mut self, addr: Addr<Self>) {
        let data = self.data.view();
        let n = data.nrows();
        let mean = data.mean_axis(Axis(0)).unwrap();
        let delta = data.to_owned() - mean.broadcast((self.data.nrows(), mean.len())).unwrap().to_owned();
        let delta_n = delta.clone() / (n as f32);
        let m2 = (delta * delta_n * (n as f32)).sum_axis(Axis(0));

        let main = match self.main_node.as_ref() {
            None => AnyAddr::Local(addr.clone()),
            Some(any_addr) => { AnyAddr::Remote(any_addr.clone()) }
        };
        main.do_send(StdNodeMessage { n, mean, m2, source: RemoteAddr::new_from_id(self.parameters.local_host, "StdCalculator") });
    }
}

impl Actor for StdCalculator {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.register(ctx.address().recipient(), "StdCalculator".to_string());
        self.calculate_var(ctx.address());
    }
}

impl Handler<StdNodeMessage> for StdCalculator {
    type Result = ();

    fn handle(&mut self, msg: StdNodeMessage, ctx: &mut Self::Context) -> Self::Result {
        self.std_nodes.push(msg.source);

        if self.std_nodes.len() < self.parameters.n_cluster_nodes {
            match (&self.n, &self.mean, &self.m2) {
                (Some(n), Some(mean), Some(m2)) => {
                    let global_n = n + msg.n;
                    let delta: Array1<f32> = msg.mean.clone() - mean;
                    self.m2 = Some(msg.m2 + m2 + delta.clone() * delta * ((n + msg.n) as f32 / global_n as f32));
                    self.mean = Some((mean * n.clone() as f32 + msg.mean * msg.n as f32) / global_n as f32);
                    self.n = Some(global_n);
                },
                _ => {
                    self.set_intermediate(msg.n, msg.mean, msg.m2);
                }
            }
        } else {
            match (&self.n, &self.mean, &self.m2) {
                (Some(n), Some(mean), Some(m2)) => {
                    let global_n = n + msg.n;
                    let delta: Array1<f32> = msg.mean.clone() - mean;
                    let m2 = msg.m2 + m2 + delta.clone() * delta * ((n + msg.n) as f32 / global_n as f32);
                    self.std = Some((m2 / (global_n - 1) as f32).iter().map(|x| x.sqrt()).collect());
                },
                _ => panic!("Some value for variance should have been set by now!")
            }

            for node in &self.std_nodes {
                let receiving_node = match &node.network_interface {
                    Some(_) => AnyAddr::Remote(node.clone()),
                    None => AnyAddr::Local(ctx.address())
                };
                receiving_node.do_send(StdDoneMessage { std: self.std.as_ref().unwrap().clone() });
            }
        }
    }
}

impl Handler<StdDoneMessage> for StdCalculator {
    type Result = ();

    fn handle(&mut self, msg: StdDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        match &self.receiver {
            //todo also send `n`
            Some(addr) => { addr.do_send(msg); },
            _ => ()
        };
    }
}
