mod std;
mod messages;
mod minmax;
#[cfg(test)]
mod tests;

use ndarray::{Array1, ArcArray2, Array2, s};
use crate::utils::ClusterNodes;
use actix::prelude::*;
use crate::data_manager::stats_collector::std::StdCalculator;
use crate::data_manager::stats_collector::messages::{StdDoneMessage, MinMaxDoneMessage};
pub use crate::data_manager::stats_collector::messages::DatasetStatsMessage;
use crate::data_manager::stats_collector::minmax::MinMaxCalculator;
use crate::parameters::Parameters;

#[derive(Default, Clone)]
pub struct DatasetStats {
    pub min_col: Option<Array1<f32>>,
    pub max_col: Option<Array1<f32>>,
    pub std_col: Option<Array1<f32>>
}

impl DatasetStats {
    pub fn new(std_col: Array1<f32>, min_col: Array1<f32>, max_col: Array1<f32>) -> Self {
        Self {
            min_col: Some(min_col),
            max_col: Some(max_col),
            std_col: Some(std_col)
        }
    }
}

pub struct StatsCollector {
    data: ArcArray2<f32>,
    parameters: Parameters,
    cluster_nodes: ClusterNodes,
    dataset_stats: DatasetStats,
    receiver: Recipient<DatasetStatsMessage>
}

impl StatsCollector {
    pub fn new(data: ArcArray2<f32>, parameters: Parameters, cluster_nodes: ClusterNodes, receiver: Recipient<DatasetStatsMessage>) -> Self {
        let cut_data = if cluster_nodes.get_own_idx() < cluster_nodes.len() { // if not last
            let n_data_rows = data.shape()[0];
            let overlap_size = parameters.pattern_length;
            data.slice(s![..(n_data_rows - overlap_size), ..]).to_shared()
        } else {
            data
        };

        Self {
            data: cut_data,
            parameters,
            cluster_nodes,
            dataset_stats: DatasetStats::default(),
            receiver
        }
    }

    fn calculate_std(&mut self, rec: Recipient<StdDoneMessage>) {
        let main_node = match self.cluster_nodes.get_main_node() {
            Some(addr) => {
                let mut remote_addr = addr.clone();
                remote_addr.change_id("StdCalculator".to_string());
                Some(remote_addr)
            },
            None => None
        };

        let _std_calc = StdCalculator::new(
            self.data.clone(),
            main_node,
            self.parameters.clone(),
            rec
        ).start();
    }

    fn calculate_min_max(&mut self, rec: Recipient<MinMaxDoneMessage>) {
        let main_node = match self.cluster_nodes.get_main_node() {
            Some(addr) => {
                let mut remote_addr = addr.clone();
                remote_addr.change_id("MinMaxCalculator".to_string());
                Some(remote_addr)
            },
            None => None
        };

        MinMaxCalculator::new(
            self.data.clone(),
            main_node,
            self.parameters.clone(),
            rec
        ).start();
    }

    fn check_done(&mut self) {
        match (&self.dataset_stats.std_col, &self.dataset_stats.min_col, &self.dataset_stats.max_col) {
            (Some(_), Some(_), Some(_)) => { self.receiver.do_send(DatasetStatsMessage { dataset_stats: self.dataset_stats.clone() }); },
            _ => ()
        };
    }
}

impl Actor for StatsCollector {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.calculate_std(ctx.address().recipient());
        self.calculate_min_max(ctx.address().recipient());
    }
}

impl Handler<StdDoneMessage> for StatsCollector {
    type Result = ();

    fn handle(&mut self, msg: StdDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        self.dataset_stats.std_col = Some(msg.std);
        self.check_done();
    }
}

impl Handler<MinMaxDoneMessage> for StatsCollector {
    type Result = ();

    fn handle(&mut self, msg: MinMaxDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        self.dataset_stats.min_col = Some(msg.min);
        self.dataset_stats.max_col = Some(msg.max);
        self.check_done();
    }
}
