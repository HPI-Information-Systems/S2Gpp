use actix::{Actor, ActorContext, Context, Handler, Recipient, Addr, AsyncContext};
use actix::dev::MessageResponse;
use ndarray::{ArcArray2, Array2, Array3};

pub use crate::data_manager::messages::{LoadDataMessage};
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

#[cfg(test)]
mod tests;
mod messages;
pub mod data_reader;
mod preprocessor;
mod reference_dataset_builder;
mod phase_spacer;

pub struct DataManager {
    data: Option<Array2<f32>>,
    nodes: ClusterNodes,
    parameters: Parameters,
    data_receiver: Option<Addr<DataReceiver>>,
    reference_dataset: Option<Array3<f32>>,
    phase_space: Option<Array3<f32>>
}

impl DataManager {
    pub fn new(nodes: ClusterNodes, parameters: Parameters) -> Self {
        Self {
            data: None,
            nodes,
            parameters,
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
                        self.parameters.pattern_length).start();
    }

    fn preprocess(&mut self, rec: Recipient<PreprocessingDoneMessage>) {
        let main_node = match self.nodes.get(&0) {
            None => None,
            Some(remote) => {
                let mut remote = remote.clone();
                remote.change_id("Preprocessor".to_string());
                Some(AnyAddr::Remote(remote))
            }
        };

        match &self.data {
            Some(data) => {
                Preprocessor::new(
                    data.to_shared(),
                    self.parameters.clone(),
                    main_node,
                    rec
                ).start();
            },
            None => panic!("Data should be set by now!")
        }
    }

    fn build_reference_dataset(&mut self) {
        let reference_dataset = ReferenceDatasetBuilder::new(
            self.data.as_ref().unwrap().to_shared(),
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
        self.preprocess(ctx.address().recipient());
    }
}

impl Handler<PreprocessingDoneMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: PreprocessingDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        self.build_reference_dataset();
        self.build_phase_space();
    }
}

impl Handler<PoisonPill> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
