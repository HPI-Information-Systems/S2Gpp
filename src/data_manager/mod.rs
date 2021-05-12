use actix::{Actor, ActorContext, Context, Handler, Recipient, Addr, AsyncContext};
use actix::dev::MessageResponse;
use ndarray::{ArcArray2, Array2};

pub use crate::data_manager::messages::{LoadDataMessage};
use crate::data_manager::data_reader::{DataReceivedMessage, DataReader, DataPartitionMessage};
use crate::data_manager::preprocessor::{Preprocessor, PreprocessingDoneMessage};
use crate::parameters::{Parameters, Role};
use actix_telepathy::{RemoteAddr, AnyAddr};
use std::borrow::Borrow;

mod messages;
pub mod data_reader;
mod preprocessor;

pub struct DataManager {
    data: Option<Array2<f32>>,
    main_node: Option<RemoteAddr>,
    nodes: Vec<RemoteAddr>,
    parameters: Parameters,
    preprocessor: Option<Addr<Preprocessor>>
}

impl DataManager {
    pub fn new(main_node: Option<RemoteAddr>, nodes: Vec<RemoteAddr>, parameters: Parameters) -> Self {
        Self {
            data: None,
            main_node,
            nodes,
            parameters,
            preprocessor: None
        }
    }

    fn load_data(&mut self, data_path: &str) {
        let nodes = self.nodes.clone().into_iter()
            .map(|mut x| { x.change_id("DataReceiver".to_string()); x }).collect();

        DataReader::new(data_path,
                        nodes,
                        self.parameters.pattern_length).start();
    }

    fn preprocess(&mut self, rec: Recipient<PreprocessingDoneMessage>) {
        let main_node = match self.main_node.as_ref() {
            None => None,
            Some(remote) => {
                let mut remote = remote.clone();
                remote.change_id("Preprocessor".to_string());
                Some(AnyAddr::Remote(remote))
            }
        };

        match &self.data {
            Some(data) => {
                self.preprocessor = Some(Preprocessor::new(
                    data.to_shared(),
                    self.parameters.clone(),
                    main_node,
                    rec
                ).start());
            },
            None => panic!("Data should be set by now!")
        }
    }

    fn build_reference_dataset(&mut self) {

    }

    fn build_phase_space(&mut self) {

    }
}

impl Actor for DataManager {
    type Context = Context<Self>;
}

impl Handler<LoadDataMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, msg: LoadDataMessage, ctx: &mut Self::Context) -> Self::Result {
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

    }
}
