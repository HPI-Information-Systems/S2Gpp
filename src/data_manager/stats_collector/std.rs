use crate::data_manager::stats_collector::messages::{StdDoneMessage, StdNodeMessage};
use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::{s, Array1, Axis};

use crate::data_manager::DataManager;

pub struct StdCalculation {
    pub nodes: Vec<RemoteAddr>,
    pub n: Option<usize>,
    pub mean: Option<Array1<f32>>,
    pub m2: Option<Array1<f32>>,
}

pub trait StdCalculator {
    fn set_intermediate_std(&mut self, n: usize, mean: Array1<f32>, m2: Array1<f32>);
    fn calculate_std(&mut self, addr: Addr<Self>)
    where
        Self: actix::Actor;
}

impl StdCalculator for DataManager {
    fn set_intermediate_std(&mut self, n: usize, mean: Array1<f32>, m2: Array1<f32>) {
        self.std_calculation.as_mut().unwrap().n = Some(n);
        self.std_calculation.as_mut().unwrap().mean = Some(mean);
        self.std_calculation.as_mut().unwrap().m2 = Some(m2);
    }

    fn calculate_std(&mut self, addr: Addr<Self>) {
        let is_last_node = self.cluster_nodes.get_own_idx() == self.cluster_nodes.len();
        let cutoff = if is_last_node {
            0
        } else {
            self.parameters.pattern_length - 1
        };
        let end_slice = self.data.as_ref().unwrap().nrows() - cutoff;
        let data = self.data.as_ref().unwrap().slice(s![0..end_slice, ..]);
        let n = data.nrows();
        let mean = data.mean_axis(Axis(0)).unwrap();
        let delta = data.to_owned()
            - mean
                .broadcast((data.nrows(), mean.len()))
                .unwrap()
                .to_owned();
        let delta_n = delta.clone() / (n as f32);
        let m2 = (delta * delta_n * (n as f32)).sum_axis(Axis(0));

        let main = match self.cluster_nodes.get_main_node() {
            None => AnyAddr::Local(addr),
            Some(remote_addr) => {
                let mut remote_addr = remote_addr.clone();
                remote_addr.change_id("DataManager".to_string());
                AnyAddr::Remote(remote_addr)
            }
        };
        main.do_send(StdNodeMessage {
            n,
            mean,
            m2,
            source: RemoteAddr::new_from_id(self.parameters.local_host, "DataManager"),
        });
    }
}

impl Handler<StdNodeMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: StdNodeMessage, ctx: &mut Self::Context) -> Self::Result {
        let std_calcuation = self.std_calculation.as_mut().unwrap();
        std_calcuation.nodes.push(msg.source);

        if std_calcuation.nodes.len() < self.parameters.n_cluster_nodes {
            match (&std_calcuation.n, &std_calcuation.mean, &std_calcuation.m2) {
                (Some(n), Some(mean), Some(m2)) => {
                    let global_n = n + msg.n;
                    let delta: Array1<f32> = msg.mean.clone() - mean;
                    std_calcuation.m2 = Some(
                        msg.m2
                            + m2
                            + delta.clone() * delta * ((n + msg.n) as f32 / global_n as f32),
                    );

                    std_calcuation.mean =
                        Some((mean * *n as f32 + msg.mean * msg.n as f32) / global_n as f32);
                    std_calcuation.n = Some(global_n);
                }
                _ => {
                    self.set_intermediate_std(msg.n, msg.mean, msg.m2);
                }
            }
        } else {
            let std: Array1<f32> =
                match (&std_calcuation.n, &std_calcuation.mean, &std_calcuation.m2) {
                    (Some(n), Some(mean), Some(m2)) => {
                        let global_n = n + msg.n;
                        let delta: Array1<f32> = msg.mean.clone() - mean;
                        let m2 = msg.m2
                            + m2
                            + delta.clone() * delta * ((n + msg.n) as f32 / global_n as f32);
                        std_calcuation.n = Some(global_n);
                        (m2 / (global_n - 1) as f32)
                            .iter()
                            .map(|x| x.sqrt())
                            .collect()
                    }
                    _ => {
                        // for single node case or if local message is faster than remote message
                        let std = (msg.m2.clone() / (msg.n - 1) as f32)
                            .iter()
                            .map(|x| x.sqrt())
                            .collect();
                        self.set_intermediate_std(msg.n, msg.mean, msg.m2);
                        std
                    }
                };

            for node in self.std_calculation.as_ref().unwrap().nodes.iter() {
                let receiving_node = match &node.network_interface {
                    Some(_) => AnyAddr::Remote(node.clone()),
                    None => AnyAddr::Local(ctx.address()),
                };
                receiving_node.do_send(StdDoneMessage {
                    std: std.clone(),
                    n: *self.std_calculation.as_ref().unwrap().n.as_ref().unwrap(),
                });
            }
        }
    }
}
