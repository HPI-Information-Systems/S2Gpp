mod messages;

use actix::prelude::*;
use crate::parameters::Parameters;
pub use crate::training::messages::StartTrainingMessage;



use crate::data_manager::{DataManager, LoadDataMessage, DataLoadedAndProcessed};
use crate::utils::ClusterNodes;
use ndarray::{ArcArray, Dimension, Array, Array2, Ix3};
use crate::messages::PoisonPill;
use crate::pca::{RotatedMessage};



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

    fn data_management(&mut self, rec: Recipient<DataLoadedAndProcessed>) {
        let data_manager = DataManager::new(
            self.nodes.clone(),
            self.parameters.clone(),
            rec
        ).start();

        data_manager.do_send(LoadDataMessage);
        self.data_manager = Some(data_manager);
    }
    /*
    fn rotate(&mut self, phase_space: ArcArray<f32, Ix3>, data_ref: ArcArray<f32, Ix3>, rec: Recipient<RotatedMessage>) {
        let rotator = Rotator::new(
            self.parameters.clone(),
            self.nodes.clone(),
            rec,
            phase_space,
            data_ref).start();
    }

    fn segment(&mut self, rec: Recipient<SegmentMessage>) {

    }*/
}

impl Actor for Training {
    type Context = Context<Self>;
}

impl Handler<StartTrainingMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: StartTrainingMessage, ctx: &mut Self::Context) -> Self::Result {
        self.nodes = msg.nodes;
        self.data_management(ctx.address().recipient());
    }
}

impl Handler<DataLoadedAndProcessed> for Training {
    type Result = ();

    fn handle(&mut self, _msg: DataLoadedAndProcessed, _ctx: &mut Self::Context) -> Self::Result {
        /*self.rotate(msg.phase_space, msg.data_ref, ctx.address().recipient());*/
    }
}

impl Handler<RotatedMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: RotatedMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.rotated = Some(msg.rotated);
        self.data_manager.as_ref().unwrap().do_send(PoisonPill);
    }
}
