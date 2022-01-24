use actix::prelude::*;
use actix_telepathy::prelude::*;
use std::ops::Sub;
use log::*;
use ndarray::arr1;
use num_integer::Integer;

use crate::data_manager::{DataLoadedAndProcessed, DataManager, DatasetStats, LoadDataMessage};
use crate::data_store::{DataStore};
use crate::messages::PoisonPill;
use crate::parameters::{Parameters};
use crate::training::edge_estimation::{EdgeEstimationDone, EdgeEstimator};
use crate::training::intersection_calculation::{IntersectionCalculation, IntersectionCalculationDone, IntersectionCalculator, SegmentID, IntersectionRotationMessage};
pub use crate::training::messages::StartTrainingMessage;
use crate::training::node_estimation::{NodeEstimation, NodeEstimationDone, NodeEstimator, AskForForeignNodes, ForeignNodesAnswer};
use crate::training::rotation::{Rotation, Rotator, RotationDoneMessage, RotationMatrixMessage, PCAComponents, PCAMeansMessage, PCADecompositionMessage};
use crate::training::segmentation::{Segmentation, SegmentedMessage, Segmenter};
use crate::training::segmentation::messages::{SegmentMessage, SendFirstPointMessage};
use crate::utils::{ClusterNodes, ConsoleLogger};
use crate::training::scoring::{Scoring, Scorer};
use crate::training::scoring::messages::{NodeDegrees, SubScores, EdgeWeights, OverlapRotation, ScoringDone};
use crate::training::scoring::weights::ScoringWeights;
use crate::training::transposition::{Transposition, Transposer, TranspositionDone, TranspositionRotationMessage};

mod messages;
mod segmentation;
mod intersection_calculation;
mod node_estimation;
mod edge_estimation;
mod rotation;
mod scoring;
mod transposition;


#[derive(RemoteActor)]
#[remote_messages(ForeignNodesAnswer, AskForForeignNodes, NodeDegrees, SubScores, EdgeWeights, OverlapRotation, TranspositionRotationMessage, IntersectionRotationMessage, SegmentMessage, SendFirstPointMessage, PCAMeansMessage, PCADecompositionMessage, PCAComponents, RotationMatrixMessage)]
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
    transposition: Transposition,
    scoring: Scoring,
    data_store: DataStore,
    num_rotated: Option<usize>
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
            transposition: Transposition::default(),
            scoring: Scoring::default(),
            data_store: DataStore::default(),
            num_rotated: None
        }
    }

    fn segment_id_to_assignment(&self, segment_id: SegmentID) -> usize {
        self.parameters.segment_id_to_assignment(segment_id)
    }

    fn cluster_node_diff(&self, from_node_id: usize, to_node_id: usize) -> usize {
        (to_node_id as isize)
            .sub(&(from_node_id as isize))
            .mod_floor(&(self.parameters.n_cluster_nodes as isize)) as usize
    }
}

impl Actor for Training {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.register(ctx.address().recipient());
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

    fn handle(&mut self, msg: StartTrainingMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.cluster_nodes = msg.nodes;
        self.data_manager.as_ref().unwrap().do_send(LoadDataMessage { nodes: self.cluster_nodes.clone() });
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

    fn handle(&mut self, _msg: RotationDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(7, 12, "Segmenting Data".to_string()).print();
        self.data_manager.as_ref().unwrap().do_send(PoisonPill);
        self.data_manager = None;

        self.segment(ctx);
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

        self.node_estimation.current_segment_id = self.parameters.rate.div_floor(&self.cluster_nodes.len_incl_own()) * self.cluster_nodes.get_own_idx();
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
        ConsoleLogger::new(11, 12, "Transpose Distributed Data".to_string()).print();

        self.scoring.node_degrees = self.calculate_node_degrees();

        if self.cluster_nodes.len() > 0 {
            self.transpose(ctx.address().recipient());
        } else {
            ctx.address().do_send(TranspositionDone);
        }
    }
}

impl Handler<TranspositionDone> for Training {
    type Result = ();

    fn handle(&mut self, _msg: TranspositionDone, ctx: &mut Self::Context) -> Self::Result {
        ConsoleLogger::new(12, 12, "Scoring".to_string()).print();

        self.init_scoring(ctx);
    }
}

impl Handler<ScoringDone> for Training {
    type Result = ();

    fn handle(&mut self, _msg: ScoringDone, ctx: &mut Self::Context) -> Self::Result {
        debug!("score {}", self.scoring.score.as_ref().unwrap_or(&arr1(&[])));

        ctx.stop();
        System::current().stop();
    }
}
