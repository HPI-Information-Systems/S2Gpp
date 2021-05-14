mod messages;

use actix::prelude::*;
use crate::parameters::Parameters;
pub use crate::training::messages::StartTrainingMessage;
use actix::dev::MessageResponse;
use std::collections::HashMap;
use actix_telepathy::RemoteAddr;
use crate::data_manager::{DataManager, LoadDataMessage};
use crate::utils::ClusterNodes;
use crate::training::messages::DataLoadedAndProcessed;
use ndarray::{ArcArray, Dimension, Array, Array2};
use crate::messages::PoisonPill;
use crate::pca::PCAResponse;


pub struct Training {
    parameters: Parameters,
    nodes: ClusterNodes,
    data_manager: Option<Addr<DataManager>>,
    rotated: Option<Array2<f32>>
}

impl Training {
    pub fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            nodes: ClusterNodes::default(),
            data_manager: None,
            rotated: None
        }
    }

    fn data_management(&mut self) {
        let data_manager = DataManager::new(
            self.nodes.clone(),
            self.parameters.clone()
        ).start();

        data_manager.do_send(LoadDataMessage);
        self.data_manager = Some(data_manager);
    }

    fn pca(&mut self, phase_space: ArcArray3<f32>, data_ref: ArcArray3<f32>) {

    }

    fn rotate(&mut self, components: Array2<f32>) {

    }
}

impl Actor for Training {
    type Context = Context<Self>;
}

impl Handler<StartTrainingMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: StartTrainingMessage, ctx: &mut Self::Context) -> Self::Result {
        self.nodes = msg.nodes;
        self.data_management();
    }
}

impl Handler<DataLoadedAndProcessed> for Training {
    type Result = ();

    fn handle(&mut self, msg: DataLoadedAndProcessed, _ctx: &mut Self::Context) -> Self::Result {
        pca(msg.phase_space, msg.data_ref);
        self.data_manager.unwrap().do_send(PoisonPill);
    }
}

impl Handler<PCAResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: PCAResponse, ctx: &mut _) -> Self::Result {
        self.rotate(msg.components);

    }
}
