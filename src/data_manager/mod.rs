use actix::{Actor, ActorContext, Context, Handler, Recipient, Addr, AsyncContext};

use ndarray::{Array2, Array3, Array1, Dim};

pub use crate::data_manager::messages::{LoadDataMessage, DataLoadedAndProcessed};
use crate::data_manager::data_reader::{DataReader, DataPartitionMessage, DataReading};
use crate::data_manager::preprocessor::{Preprocessor, PreprocessingDoneMessage, Preprocessing};
use crate::parameters::{Parameters, Role};
use actix_telepathy::prelude::*;


use crate::utils::ClusterNodes;
use crate::data_manager::reference_dataset_builder::ReferenceDatasetBuilder;
use crate::data_manager::phase_spacer::PhaseSpacer;
use crate::messages::PoisonPill;
pub use crate::data_manager::stats_collector::DatasetStats;
use crate::data_manager::stats_collector::{MinMaxNodeMessage, MinMaxDoneMessage, StdNodeMessage, StdDoneMessage, MinMaxCalculation, MinMaxCalculator, StdCalculator, StdCalculation};
use std::str::FromStr;

#[cfg(test)]
mod tests;
mod messages;
pub mod data_reader;
mod preprocessor;
mod reference_dataset_builder;
mod phase_spacer;
mod stats_collector;


#[derive(RemoteActor)]
#[remote_messages(DataPartitionMessage, StdNodeMessage, StdDoneMessage, MinMaxNodeMessage, MinMaxDoneMessage)]
pub struct DataManager {
    data: Option<Array2<f32>>,
    nodes: ClusterNodes,
    parameters: Parameters,
    data_reading: Option<DataReading>,
    minmax_calculation: Option<MinMaxCalculation>,
    std_calculation: Option<StdCalculation>,
    preprocessing: Option<Preprocessing>,
    receiver: Recipient<DataLoadedAndProcessed>,
    dataset_stats: DatasetStats,
    reference_dataset: Option<Array3<f32>>,
    phase_space: Option<Array3<f32>>
}

impl DataManager {
    pub fn new(mut nodes: ClusterNodes, parameters: Parameters, receiver: Recipient<DataLoadedAndProcessed>) -> Self {
        nodes.change_ids("DataManager");

        Self {
            data: None,
            nodes,
            parameters,
            data_reading: None,
            minmax_calculation: None,
            std_calculation: None,
            preprocessing: None,
            receiver,
            dataset_stats: DatasetStats::default(),
            reference_dataset: None,
            phase_space: None
        }
    }

    fn calculate_datastats(&mut self, addr: Addr<Self>) {
        self.minmax_calculation = Some(MinMaxCalculation { nodes: vec![], min: None, max: None });
        self.std_calculation = Some(StdCalculation {
            nodes: vec![],
            n: None,
            mean: None,
            m2: None
        });


        self.calculate_minmax(addr.clone());
        self.calculate_std(addr);
    }

    fn datastats_finished(&mut self, addr: Addr<Self>) {
        if self.dataset_stats.is_done() {
            self.preprocess(addr);
        }
    }

    fn preprocess(&mut self, addr: Addr<Self>) {
        match &self.data {
            Some(data) => {
                self.preprocessing = Some(Preprocessing::new(data.to_shared(), self.parameters.n_threads, self.parameters.pattern_length));
                self.distribute_work(addr);
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
            phase_space: self.phase_space.as_ref().unwrap().to_shared(),
            dataset_stats: self.dataset_stats.clone()
        }).unwrap();
    }
}

impl Actor for DataManager {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.register(ctx.address().recipient(), "DataManager".to_string());
    }
}

impl Handler<LoadDataMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, _msg: LoadDataMessage, ctx: &mut Self::Context) -> Self::Result {
        self.data_reading = Some(DataReading { with_header: true, overlap: self.parameters.pattern_length - 1 });

        let role = self.parameters.role.clone();
        match role {
            Role::Main {data_path} => self.read_csv(&data_path, ctx.address()),
            _ => ()
        }
    }
}

impl Handler<DataPartitionMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: DataPartitionMessage, ctx: &mut Self::Context) -> Self::Result {
        let n_rows = msg.data.len();
        let n_columns = msg.data[0].len();

        let flat_data: Array1<f32> = msg.data.into_iter().flat_map(|rec| {
            rec.iter().map(|b| {
                f32::from_str(b).unwrap()
            }).collect::<Vec<f32>>()
        }).collect();

        self.data = Some(flat_data.into_shape(Dim([n_rows, n_columns])).expect("Could not deserialize sent data"));

        self.calculate_datastats(ctx.address());
    }
}

impl Handler<MinMaxDoneMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: MinMaxDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        self.dataset_stats.min_col = Some(msg.min);
        self.dataset_stats.max_col = Some(msg.max);
        self.datastats_finished(ctx.address());
    }
}

impl Handler<StdDoneMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: StdDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        self.dataset_stats.std_col = Some(msg.std);
        self.dataset_stats.n = Some(msg.n);
        self.datastats_finished(ctx.address());
    }
}

impl Handler<PreprocessingDoneMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, _msg: PreprocessingDoneMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.build_reference_dataset();
        self.build_phase_space();
        self.finalize();
    }
}

impl Handler<PoisonPill> for DataManager {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
