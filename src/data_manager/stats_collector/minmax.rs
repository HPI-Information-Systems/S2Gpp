use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::{Array1, Axis, ArcArray2};
use crate::data_manager::stats_collector::messages::{StdNodeMessage, StdDoneMessage, MinMaxDoneMessage, MinMaxNodeMessage};
use crate::utils::Stats;
use crate::parameters::Parameters;

#[derive(RemoteActor)]
#[remote_messages(MinMaxNodeMessage, MinMaxDoneMessage)]
pub struct MinMaxCalculator {
    data: ArcArray2<f32>,
    main_node: Option<RemoteAddr>,
    parameters: Parameters,
    nodes: Vec<RemoteAddr>,
    receiver: Recipient<MinMaxDoneMessage>,
    min: Option<Array1<f32>>,
    max: Option<Array1<f32>>
}

impl MinMaxCalculator {
    pub fn new(data: ArcArray2<f32>, main_node: Option<RemoteAddr>, parameters: Parameters, receiver: Recipient<MinMaxDoneMessage>) -> Self{
        Self {
            data,
            main_node,
            parameters,
            nodes: vec![],
            receiver,
            min: None,
            max: None
        }
    }

    fn set_intermediate(&mut self, min: Array1<f32>, max: Array1<f32>) {
        self.min = Some(min);
        self.max = Some(max);
    }

    fn calculate_minmax(&mut self, addr: Addr<Self>) {
        let n = self.data.nrows();
        let min = self.data.min_axis(Axis(0));
        let max = self.data.max_axis(Axis(0));

        let main = match self.main_node.as_ref() {
            None => AnyAddr::Local(addr.clone()),
            Some(any_addr) => AnyAddr::Remote(any_addr.clone())
        };
        main.do_send(MinMaxNodeMessage { min, max, source: RemoteAddr::new_from_id(self.parameters.local_host, "MinMaxCalculator") });
    }
}

impl Actor for MinMaxCalculator {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.register(ctx.address().recipient(), "MinMaxCalculator".to_string());
        self.calculate_minmax(ctx.address());
    }
}

impl Handler<MinMaxNodeMessage> for MinMaxCalculator {
    type Result = ();

    fn handle(&mut self, msg: MinMaxNodeMessage, ctx: &mut Self::Context) -> Self::Result {
        self.nodes.push(msg.source);

        match (&self.min, &self.max) {
            (Some(min), Some(max)) => {
                let new_min: Array1<f32> = msg.min.iter().zip(min.iter()).map(|(sent, local)| {
                    sent.min(local.clone())
                }).collect();
                let new_max: Array1<f32> = msg.max.iter().zip(max.iter()).map(|(sent, local)| {
                    sent.max(local.clone())
                }).collect();

                self.min = Some(new_min);
                self.max = Some(new_max);
            },
            _ => {
                self.set_intermediate(msg.min, msg.max);
            }
        }

        if self.nodes.len() == self.parameters.n_cluster_nodes {
            for node in &self.nodes {
                let receiving_node = match &node.network_interface {
                    Some(_) => AnyAddr::Remote(node.clone()),
                    None => AnyAddr::Local(ctx.address())
                };
                receiving_node.do_send(MinMaxDoneMessage { min: self.min.as_ref().unwrap().clone(), max: self.max.as_ref().unwrap().clone() });
            }
        }
    }
}

impl Handler<MinMaxDoneMessage> for MinMaxCalculator {
    type Result = ();

    fn handle(&mut self, msg: MinMaxDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        self.receiver.do_send(msg);
    }
}
