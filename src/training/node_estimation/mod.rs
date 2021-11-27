mod messages;
#[cfg(test)]
pub(crate) mod tests;

use std::collections::HashMap;
use crate::training::intersection_calculation::{SegmentID};
use ndarray::{ArrayView1, stack_new_axis, Axis, Array2};
use crate::training::Training;
use actix::{Addr, Handler, Actor, Recipient, AsyncContext, Context};

pub use crate::training::node_estimation::messages::{NodeEstimationDone, AskForForeignNodes, ForeignNodesAnswer};
use meanshift_rs::{MeanShiftResponse, MeanShiftMessage, MeanShiftActor};
use num_integer::Integer;
use crate::utils::logging::progress_bar::S2GppProgressBar;
use crate::utils::NodeName;
use crate::utils::rotation_protocol::RotationProtocol;

#[derive(Default)]
pub struct NodeEstimation {
    pub nodes: HashMap<SegmentID, Array2<f32>>, //todo remove and build a smarter node index
    /// {point_id: \[node\]}
    pub nodes_by_point: HashMap<usize, Vec<NodeName>>,
    pub gap_after_node: Vec<(usize, NodeName)>, // todo hashset?
    pub next_foreign_node: HashMap<(usize, usize), (usize, NodeName)>,
    pub meanshift: Option<Addr<MeanShiftActor>>,
    pub(crate) last_start_points: Vec<usize>,
    pub(crate) current_segment_id: usize,
    pub(crate) progress_bar: S2GppProgressBar,
    pub(crate) source: Option<Recipient<NodeEstimationDone>>,
    asking_rotation_protocol: RotationProtocol<AskForForeignNodes>, // no rotation actually
    answering_rotation_protocol: RotationProtocol<ForeignNodesAnswer> // no rotation actually
}

pub trait NodeEstimator {
    fn estimate_nodes(&mut self, mean_shift_recipient: Recipient<MeanShiftResponse>);
    fn ask_for_foreign_nodes(&mut self, ctx: &mut Context<Training>);
    fn search_for_asked_nodes(&mut self, msg: AskForForeignNodes);
    fn finalize_node_estimation(&mut self, ctx: &mut Context<Training>);
    fn node_question_conversation_done(&mut self) -> bool;
}

impl NodeEstimator for Training {
    fn estimate_nodes(&mut self, mean_shift_recipient: Recipient<MeanShiftResponse>) {
        self.node_estimation.progress_bar.inc_or_set("info", self.parameters.rate.div_floor(&self.parameters.n_cluster_nodes));

        let segment_id = self.node_estimation.current_segment_id;

        match self.intersection_calculation.intersection_coords_by_segment.get(&segment_id) {
            Some(intersections_by_starting_point) => {

                self.node_estimation.last_start_points = intersections_by_starting_point.keys().map(|x| x.clone()).collect();
                let intersections: Vec<ArrayView1<f32>> = intersections_by_starting_point.values().map(|x| x.view()).collect();
                let data = stack_new_axis(Axis(0), intersections.as_slice()).unwrap();
                self.node_estimation.meanshift = Some(MeanShiftActor::new(self.parameters.n_threads).start());
                self.node_estimation.meanshift.as_ref().unwrap().do_send(MeanShiftMessage { source: Some(mean_shift_recipient.clone()), data });
            }
            None => {
                mean_shift_recipient.do_send(MeanShiftResponse { cluster_centers: Default::default(), labels: vec![] }).unwrap();
            }
        }
    }

    fn ask_for_foreign_nodes(&mut self, ctx: &mut Context<Training>) {
        if self.cluster_nodes.len() > 0 {
            self.node_estimation.answering_rotation_protocol.start(self.parameters.n_cluster_nodes);
            self.node_estimation.answering_rotation_protocol.resolve_buffer(ctx.address().recipient());
            self.node_estimation.asking_rotation_protocol.start(self.parameters.n_cluster_nodes);
            self.node_estimation.asking_rotation_protocol.resolve_buffer(ctx.address().recipient());
            for (answering_node, remote_addr) in self.cluster_nodes.to_any_as(ctx.address(), "Training").into_iter().enumerate() {
                let asked_nodes =
                    match self.segmentation.node_questions.get(&answering_node) {
                        Some(questions) => questions.clone(),
                        None => HashMap::new()
                    };
                remote_addr.do_send(AskForForeignNodes { asked_nodes });
                self.node_estimation.asking_rotation_protocol.sent();
            }
            self.segmentation.node_questions.clear();
        } else {
            self.finalize_node_estimation(ctx);
        }
    }

    fn search_for_asked_nodes(&mut self, msg: AskForForeignNodes) {
        let mut asked_nodes = msg.asked_nodes;
        for (asking_node, remote_addr) in self.cluster_nodes.iter() {
            let mut specific_addr = remote_addr.clone();
            specific_addr.change_id("Training".to_string());

            let answers = match asked_nodes.remove(asking_node) {
                Some(questions) => questions.into_iter().map(|(prev_point_id, prev_segment_id, point_id, segment_id)|
                    match self.node_estimation.nodes_by_point.get(&point_id) {
                        Some(nodes) => nodes.iter().find_map(|node| node.0.eq(&segment_id).then(|| (prev_point_id, prev_segment_id, point_id, node.clone()))).expect(&format!("There is no answer here: no segment_id: {} {}", &point_id, &segment_id)),
                        None => {
                            panic!("There is no answer here!: no point_id: {}", point_id)
                        }
                    }
                ).collect(),
                None => vec![]
            };

            specific_addr.do_send(ForeignNodesAnswer { foreign_nodes: answers });
            self.node_estimation.answering_rotation_protocol.sent();
        }
    }

    fn finalize_node_estimation(&mut self, ctx: &mut Context<Training>) {
        match &self.node_estimation.source {
            Some(source) => source.clone(),
            None => ctx.address().recipient()
        }.do_send(NodeEstimationDone).unwrap();
    }

    fn node_question_conversation_done(&mut self) -> bool {
        !(self.node_estimation.answering_rotation_protocol.is_running() ||
            self.node_estimation.asking_rotation_protocol.is_running())
    }
}

impl Handler<MeanShiftResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftResponse, ctx: &mut Self::Context) -> Self::Result {
        if !msg.cluster_centers.is_empty() {
            let current_segment_id = self.node_estimation.current_segment_id;
            let last_start_points = self.node_estimation.last_start_points.clone();
            self.node_estimation.last_start_points.clear();
            self.node_estimation.nodes.insert(current_segment_id, msg.cluster_centers);

            for (last_start_point, label) in last_start_points.into_iter().zip(msg.labels)  {
                let node = NodeName(current_segment_id, label);
                match self.node_estimation.nodes_by_point.get_mut(&last_start_point) {
                    Some(nodes) => nodes.push(node),
                    None => { self.node_estimation.nodes_by_point.insert(last_start_point.clone(), vec![node]); }
                }
            }
        }
        self.node_estimation.current_segment_id += 1;

        if self.node_estimation.current_segment_id < self.parameters.rate {
            self.estimate_nodes(ctx.address().recipient());
        } else {
            self.node_estimation.progress_bar.inc();
            self.node_estimation.progress_bar.finish_and_clear();
            self.ask_for_foreign_nodes(ctx);
        }
    }
}


impl Handler<AskForForeignNodes> for Training {
    type Result = ();

    fn handle(&mut self, msg: AskForForeignNodes, ctx: &mut Self::Context) -> Self::Result {
        if !self.node_estimation.asking_rotation_protocol.received(&msg) {
            return
        }

        self.search_for_asked_nodes(msg);

        if self.node_question_conversation_done() {
            self.finalize_node_estimation(ctx);
        }
    }
}


impl Handler<ForeignNodesAnswer> for Training {
    type Result = ();

    fn handle(&mut self, msg: ForeignNodesAnswer, ctx: &mut Self::Context) -> Self::Result {
        if !self.node_estimation.answering_rotation_protocol.received(&msg) {
            return
        }

        for (prev_point_id, prev_segment_id, point_id, node) in msg.foreign_nodes {
            self.node_estimation.next_foreign_node.insert((prev_point_id, prev_segment_id), (point_id, node));
        }

        if self.node_question_conversation_done() {
            self.finalize_node_estimation(ctx);
        }
    }
}
