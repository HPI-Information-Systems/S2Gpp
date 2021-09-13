use std::collections::HashMap;

use actix::dev::MessageResponse;
use actix::prelude::*;
use actix_telepathy::prelude::*;
use ndarray::{Array1, Array2};

use crate::data_manager::{DataLoadedAndProcessed, DataManager, DatasetStats, LoadDataMessage};
use crate::messages::PoisonPill;
use crate::parameters::Parameters;
use crate::training::edge_estimation::{EdgeEstimation, EdgeEstimationDone, EdgeEstimator, EdgeReductionMessage};
use crate::training::graph_creation::{GraphCreation, GraphCreationDone, GraphCreator};
use crate::training::intersection_calculation::{IntersectionCalculation, IntersectionCalculationDone, IntersectionCalculator};
use crate::training::messages::{SegmentedMessage, SegmentMessage};
pub use crate::training::messages::StartTrainingMessage;
use crate::training::node_estimation::{NodeEstimation, NodeEstimationDone, NodeEstimator};
use crate::training::rotation::{Rotation, Rotator, RotationDoneMessage, RotationMatrixMessage, PCADoneMessage, PCAComponents, PCAMeansMessage, PCADecompositionMessage};
use crate::training::segmenter::{Segmentation, Segmenter};
use crate::utils::{ClusterNodes, ConsoleLogger};

mod messages;
mod segmenter;
mod intersection_calculation;
mod node_estimation;
mod edge_estimation;
mod graph_creation;
mod rotation;

#[derive(RemoteActor)]
#[remote_messages(SegmentMessage, EdgeReductionMessage, PCAMeansMessage, PCADecompositionMessage, PCAComponents, RotationMatrixMessage)]
pub struct Training {
    own_addr: Option<Addr<Self>>,
    parameters: Parameters,
    cluster_nodes: ClusterNodes,
    data_manager: Option<Addr<DataManager>>,
    dataset_stats: Option<DatasetStats>,
    rotation: Rotation,
    segmentation: Segmentation,
    intersection_calculation: IntersectionCalculation,
    node_estimation: NodeEstimation,
    edge_estimation: EdgeEstimation,
    graph_creation: GraphCreation
}

impl Training {
    pub fn new(parameters: Parameters) -> Self {
        Self {
            own_addr: None,
            parameters,
            cluster_nodes: ClusterNodes::default(),
            data_manager: None,
            dataset_stats: None,
            rotation: Rotation::default(),
            segmentation: Segmentation::default(),
            intersection_calculation: IntersectionCalculation::default(),
            node_estimation: NodeEstimation::default(),
            edge_estimation: EdgeEstimation::default(),
            graph_creation: GraphCreation::default()
        }
    }
}

impl Actor for Training {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.register(ctx.address().recipient(), "Training".to_string());
        self.own_addr = Some(ctx.address());

        self.data_manager = Some(DataManager::new(
            self.cluster_nodes.clone(),
            self.parameters.clone(),
            ctx.address().recipient()
        ).start());
    }
}

impl Handler<StartTrainingMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: StartTrainingMessage, ctx: &mut Self::Context) -> Self::Result {
        self.cluster_nodes = msg.nodes;
        self.data_manager.as_ref().unwrap().do_send(LoadDataMessage);
    }
}

impl Handler<DataLoadedAndProcessed> for Training {
    type Result = ();

    fn handle(&mut self, msg: DataLoadedAndProcessed, _ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(6, 12, "Rotating Data".to_string()).print();
        self.dataset_stats = Some(msg.dataset_stats);
        self.rotate(msg.phase_space, msg.data_ref);
    }
}

impl Handler<RotationDoneMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: RotationDoneMessage, _ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(7, 12, "Segmenting Data".to_string()).print();
        self.data_manager.as_ref().unwrap().do_send(PoisonPill);
        self.segment();
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
        self.estimate_edges(ctx);
    }
}

impl Handler<EdgeEstimationDone> for Training {
    type Result = ();

    fn handle(&mut self, _msg: EdgeEstimationDone, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(11, 12, "Building Graph".to_string()).print();
        self.create_graph();
    }
}

impl Handler<GraphCreationDone> for Training {

    type Result = ();

    fn handle(&mut self, _msg: GraphCreationDone, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(12, 12, "Scoring".to_string()).print();
        // TODO: Scoring
    }
}

/*
impl Handler<ScoringDone> for Training {

    type Result = ();

    fn handle(&mut self, _msg: GraphBuildingDone, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(12, 12, "Writing Results".to_string()).print();
        // TODO: Write Result
    }
}
*/
