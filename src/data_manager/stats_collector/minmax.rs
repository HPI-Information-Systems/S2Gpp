use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::{Array1, Axis};
use crate::data_manager::stats_collector::messages::{MinMaxDoneMessage, MinMaxNodeMessage};
use crate::utils::Stats;

use crate::data_manager::DataManager;


pub struct MinMaxCalculation {
    pub nodes: Vec<RemoteAddr>,
    pub min: Option<Array1<f32>>,
    pub max: Option<Array1<f32>>
}

pub trait MinMaxCalculator {
    fn set_intermediate_minmax(&mut self, min: Array1<f32>, max: Array1<f32>);
    fn calculate_minmax(&mut self, addr: Addr<Self>) where Self: actix::Actor;
}

impl MinMaxCalculator for DataManager {
    fn set_intermediate_minmax(&mut self, min: Array1<f32>, max: Array1<f32>) {
        self.minmax_calculation.as_mut().unwrap().min = Some(min);
        self.minmax_calculation.as_mut().unwrap().max = Some(max);
    }

    fn calculate_minmax(&mut self, addr: Addr<Self>) {
        let min = self.data.as_ref().unwrap().to_shared().min_axis(Axis(0));
        let max = self.data.as_ref().unwrap().to_shared().max_axis(Axis(0));

        let main = match self.nodes.get_main_node() {
            None => AnyAddr::Local(addr.clone()),
            Some(any_addr) => AnyAddr::Remote(any_addr.clone())
        };
        main.do_send(MinMaxNodeMessage { min, max, source: RemoteAddr::new_from_id(self.parameters.local_host, "DataManager") });
    }
}


impl Handler<MinMaxNodeMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: MinMaxNodeMessage, ctx: &mut Self::Context) -> Self::Result {
        let minmax_calculation = self.minmax_calculation.as_mut().unwrap();
        minmax_calculation.nodes.push(msg.source);
        match (&minmax_calculation.min, &minmax_calculation.max) {
            (Some(min), Some(max)) => {
                let new_min: Array1<f32> = msg.min.iter().zip(min.iter()).map(|(sent, local)| {
                    sent.min(local.clone())
                }).collect();
                let new_max: Array1<f32> = msg.max.iter().zip(max.iter()).map(|(sent, local)| {
                    sent.max(local.clone())
                }).collect();

                minmax_calculation.min = Some(new_min);
                minmax_calculation.max = Some(new_max);
            },
            _ => {
                self.set_intermediate_minmax(msg.min, msg.max);
            }
        }

        if self.minmax_calculation.as_ref().unwrap().nodes.len() == self.parameters.n_cluster_nodes {
            for node in self.minmax_calculation.as_ref().unwrap().nodes.iter() {
                let receiving_node = match &node.network_interface {
                    Some(_) => AnyAddr::Remote(node.clone()),
                    None => AnyAddr::Local(ctx.address())
                };
                receiving_node.do_send(MinMaxDoneMessage {
                    min: self.minmax_calculation.as_ref().unwrap().min.as_ref().unwrap().clone(),
                    max: self.minmax_calculation.as_ref().unwrap().max.as_ref().unwrap().clone() });
            }
        }
    }
}
