mod messages;
mod segmenter;
mod intersection_calculation;
mod node_estimation;
mod edge_estimation;

use actix::prelude::*;
use actix_telepathy::prelude::*;
use crate::parameters::Parameters;
pub use crate::training::messages::StartTrainingMessage;

use crate::data_manager::{DataManager, LoadDataMessage, DataLoadedAndProcessed, DatasetStats};
use crate::utils::{ClusterNodes, ConsoleLogger};
use ndarray::{Array2, Array1};
use crate::messages::PoisonPill;
use crate::pca::{RotatedMessage, Rotator, StartRotation};
use crate::training::segmenter::{Segmenter, Segmentation};
use std::collections::HashMap;
use crate::training::messages::{SegmentedMessage, SegmentMessage};
use actix::dev::MessageResponse;
use crate::training::intersection_calculation::{IntersectionCalculation, IntersectionCalculator, IntersectionCalculationDone};
use crate::training::node_estimation::{NodeEstimation, NodeEstimator, NodeEstimationDone};
use crate::training::edge_estimation::{EdgeEstimation, EdgeEstimator, EdgeEstimationDone};


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
    intersection_calculation: IntersectionCalculation,
    node_estimation: NodeEstimation,
    edge_estimation: EdgeEstimation
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
            segmentation: Segmentation::default(),
            intersection_calculation: IntersectionCalculation::default(),
            node_estimation: NodeEstimation::default(),
            edge_estimation: EdgeEstimation::default()
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
        ConsoleLogger::new(6, 12, "Rotating Data".to_string()).print();
        self.rotator.as_ref().unwrap().do_send(StartRotation {
            phase_space: msg.phase_space,
            data_ref: msg.data_ref });
    }
}

impl Handler<RotatedMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: RotatedMessage, _ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(7, 12, "Segmenting Data".to_string()).print();
        self.rotated = Some(msg.rotated);
        self.data_manager.as_ref().unwrap().do_send(PoisonPill);
        self.segment();
        self.assign_segments();
    }
}

impl Handler<SegmentedMessage> for Training {
    type Result = ();

    fn handle(&mut self, _msg: SegmentedMessage, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(8, 12, "Calculating Intersections".to_string()).print();
        self.calculate_intersections(ctx.address().recipient());
    }
}

impl Handler<IntersectionCalculationDone> for Training {
    type Result = ();

    fn handle(&mut self, _msg: IntersectionCalculationDone, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(9, 12, "Estimating Nodes".to_string()).print();
        self.estimate_nodes(ctx.address().recipient());
    }
}

impl Handler<NodeEstimationDone> for Training {
    type Result = ();

    fn handle(&mut self, _msg: NodeEstimationDone, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(10, 12, "Estimating Edges".to_string()).print();
        self.estimate_edges_parallel(ctx.address().recipient());
    }
}

impl Handler<EdgeEstimationDone> for Training {
    type Result = ();

    fn handle(&mut self, _msg: EdgeEstimationDone, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(11, 12, "Building Graph".to_string()).print();
        // TODO: graph building
    }
}

/*
impl Handler<GraphBuildingDone> for Training {

    type Result = ();

    fn handle(&mut self, _msg: GraphBuildingDone, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(12, 12, "Scoring".to_string()).print();
        // TODO: Scoring
    }
}

impl Handler<ScoringDone> for Training {

    type Result = ();

    fn handle(&mut self, _msg: GraphBuildingDone, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(12, 12, "Writing Results".to_string()).print();
        // TODO: Write Result
    }
}
*/
