mod messages;
mod segmenter;
mod intersection_calculation;

use actix::prelude::*;
use actix_telepathy::prelude::*;
use crate::parameters::Parameters;
pub use crate::training::messages::StartTrainingMessage;

use crate::data_manager::{DataManager, LoadDataMessage, DataLoadedAndProcessed, DatasetStats};
use crate::utils::ClusterNodes;
use ndarray::{Array2, Array1};
use crate::messages::PoisonPill;
use crate::pca::{RotatedMessage, Rotator, StartRotation};
use crate::training::segmenter::{Segmenter, Segmentation};
use std::collections::HashMap;
use crate::training::messages::{SegmentedMessage, SegmentMessage};
use actix::dev::MessageResponse;
use crate::training::intersection_calculation::{IntersectionCalculation, IntersectionCalculator, IntersectionCalculationDone};


#[derive(RemoteActor)]
#[remote_messages(SegmentMessage)]
pub struct Training {
    parameters: Parameters,
    nodes: ClusterNodes,
    data_manager: Option<Addr<DataManager>>,
    dataset_stats: Option<DatasetStats>,
    rotator: Option<Addr<Rotator>>,
    rotated: Option<Array2<f32>>,
    segmentation: Segmentation,
    intersection_calculation: IntersectionCalculation
}

impl Training {
    pub fn new(parameters: Parameters) -> Self {
        Self {
            parameters,
            nodes: ClusterNodes::default(),
            data_manager: None,
            dataset_stats: None,
            rotator: None,
            rotated: None,
            segmentation: Segmentation { segments: vec![], own_segment: vec![], n_received: 0 },
            intersection_calculation: IntersectionCalculation {
                intersections: HashMap::new(),
                intersection_coords: HashMap::new(),
                helpers: None,
                pairs: vec![],
                n_total: 0,
                n_sent: 0,
                n_received: 0 }
        }
    }
}

impl Actor for Training {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.register(ctx.address().recipient(), "Training".to_string());

        self.data_manager = Some(DataManager::new(
            self.nodes.clone(),
            self.parameters.clone(),
            ctx.address().recipient()
        ).start());

        self.rotator = Some(Rotator::new(
            self.nodes.clone(),
            self.parameters.clone(),
            ctx.address().recipient()
        ).start());
    }
}

impl Handler<StartTrainingMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: StartTrainingMessage, ctx: &mut Self::Context) -> Self::Result {
        self.nodes = msg.nodes;
        self.data_manager.as_ref().unwrap().do_send(LoadDataMessage);
    }
}

impl Handler<DataLoadedAndProcessed> for Training {
    type Result = ();

    fn handle(&mut self, msg: DataLoadedAndProcessed, _ctx: &mut Self::Context) -> Self::Result {
        self.rotator.as_ref().unwrap().do_send(StartRotation {
            phase_space: msg.phase_space,
            data_ref: msg.data_ref });
    }
}

impl Handler<RotatedMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: RotatedMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.rotated = Some(msg.rotated);
        self.data_manager.as_ref().unwrap().do_send(PoisonPill);
        self.segment();
        self.assign_segments();
    }
}

impl Handler<SegmentedMessage> for Training {
    type Result = ();

    fn handle(&mut self, _msg: SegmentedMessage, ctx: &mut Self::Context) -> Self::Result {
        self.calculate_intersections(ctx.address().recipient());
    }
}

impl Handler<IntersectionCalculationDone> for Training {
    type Result = ();

    fn handle(&mut self, _msg: IntersectionCalculationDone, _ctx: &mut Self::Context) -> Self::Result {
        //todo node calculation
    }
}
