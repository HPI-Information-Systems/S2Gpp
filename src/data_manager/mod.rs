use actix::{Actor, ActorContext, Context, Handler, Recipient, Addr, AsyncContext};
use actix::dev::MessageResponse;
use ndarray::{ArcArray2, Array2, Array3};

pub use crate::data_manager::messages::{LoadDataMessage, DataLoadedAndProcessed};
use crate::data_manager::data_reader::{DataReceivedMessage, DataReader, DataPartitionMessage, DataReceiver};
use crate::data_manager::preprocessor::{Preprocessor, PreprocessingDoneMessage};
use crate::parameters::{Parameters, Role};
use actix_telepathy::{RemoteAddr, AnyAddr};
use std::borrow::Borrow;
use std::collections::HashMap;
use crate::utils::ClusterNodes;
use crate::data_manager::reference_dataset_builder::ReferenceDatasetBuilder;
use crate::data_manager::phase_spacer::PhaseSpacer;
use crate::messages::PoisonPill;
use crate::data_manager::stats_collector::{DatasetStatsMessage, DatasetStats, StatsCollector};

#[cfg(test)]
mod tests;
mod messages;
pub mod data_reader;
mod preprocessor;
mod reference_dataset_builder;
mod phase_spacer;
mod stats_collector;

pub struct DataManager {
    data: Option<Array2<f32>>,
    nodes: ClusterNodes,
    parameters: Parameters,
    receiver: Recipient<DataLoadedAndProcessed>,
    dataset_stats: DatasetStats,
    data_receiver: Option<Addr<DataReceiver>>,
    reference_dataset: Option<Array3<f32>>,
    phase_space: Option<Array3<f32>>
}

impl DataManager {
    pub fn new(nodes: ClusterNodes, parameters: Parameters, receiver: Recipient<DataLoadedAndProcessed>) -> Self {
        Self {
            data: None,
            nodes,
            parameters,
            receiver,
            dataset_stats: DatasetStats::default(),
            data_receiver: None,
            reference_dataset: None,
            phase_space: None
        }
    }

    fn load_data(&mut self, data_path: &str) {
        let mut nodes = self.nodes.clone();
        nodes.change_ids("DataReceiver");

        DataReader::new(data_path,
                        nodes.to_any(self.data_receiver.as_ref().unwrap().clone()),
                        self.parameters.pattern_length - 1).start();
    }

    fn collect_statistics(&mut self, rec: Recipient<DatasetStatsMessage>) {
        let data = self.data.as_ref().expect("Data should be received by now!").to_shared();
        StatsCollector::new(data,
                            self.parameters.clone(),
                            self.nodes.clone(),
                            rec).start();
    }

    fn preprocess(&mut self, rec: Recipient<PreprocessingDoneMessage>, dataset_stats: DatasetStats) {
        match &self.data {
            Some(data) => {
                let pp = Preprocessor::new(
                    data.to_shared(),
                    self.parameters.clone(),
                    rec,
                    dataset_stats
                ).start();
            },
            None => panic!("Data should be set by now!")
        }
    }

    fn build_reference_dataset(&mut self) {
        let reference_dataset = ReferenceDatasetBuilder::new(
            self.dataset_stats.clone(),
            self.parameters.clone()
        ).build();
        self.reference_dataset = Some(reference_dataset);
    }

    fn build_phase_space(&mut self) {
        let phase_space = PhaseSpacer::new(
            self.data.as_ref().unwrap().to_shared(),
            self.parameters.clone()
        ).build();
        self.phase_space = Some(phase_space);
    }

    fn finalize(&mut self) {
        self.receiver.do_send(DataLoadedAndProcessed {
            data_ref: self.reference_dataset.as_ref().unwrap().to_shared(),
            phase_space: self.phase_space.as_ref().unwrap().to_shared() });
    }
}

impl Actor for DataManager {
    type Context = Context<Self>;
}

impl Handler<LoadDataMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, _msg: LoadDataMessage, ctx: &mut Self::Context) -> Self::Result {
        self.data_receiver = Some(DataReceiver::new(Some(ctx.address().recipient())).start());

        let role = self.parameters.role.clone();
        match role {
            Role::Main {data_path} => self.load_data(&data_path),
            _ => ()
        }
    }
}

impl Handler<DataReceivedMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: DataReceivedMessage, ctx: &mut Self::Context) -> Self::Result {
        self.data = Some(msg.data);
        self.collect_statistics(ctx.address().recipient());
    }
}

impl Handler<DatasetStatsMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: DatasetStatsMessage, ctx: &mut Self::Context) -> Self::Result {
        self.dataset_stats = msg.dataset_stats;
        self.preprocess(ctx.address().recipient(), self.dataset_stats.clone());
    }
}

impl Handler<PreprocessingDoneMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: PreprocessingDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        self.build_reference_dataset();
        self.build_phase_space();
        self.finalize();
    }
}

impl Handler<PoisonPill> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
