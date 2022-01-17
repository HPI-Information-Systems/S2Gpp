#[cfg(test)]
mod tests;
pub(crate) mod messages;

use crate::training::Training;
use std::collections::HashMap;
use std::iter::FromIterator;
use std::ops::{Deref, Mul};
use actix::prelude::*;
use num_integer::Integer;
use crate::data_store::point::{Point, PointRef};
use crate::data_store::transition::{MaterializedTransition, Transition, TransitionMixin};
use crate::data_store::materialize::Materialize;
pub use crate::training::segmentation::messages::{SegmentedMessage, SegmentMessage, SendFirstPointMessage};
use crate::utils::rotation_protocol::RotationProtocol;


pub(crate) type TransitionsForNodes = HashMap<usize, Vec<MaterializedTransition>>;
/// (prev_point_id, prev_point_segment_id, point_id, segment_id)
pub(crate) type NodeInQuestion = (usize, usize, usize, usize);

#[derive(Default)]
pub(crate) struct Segmentation {
    /// list of points that are endpoints to transitions from the previous cluster node
    pub send_point: Option<Point>,
    pub last_point: Option<Point>,
    rotation_protocol: RotationProtocol<SegmentMessage>,
    /// {cluster node id (answering): {cluster node id (asking): \[NodeInQuestion\]}}
    pub node_questions: HashMap<usize, HashMap<usize, Vec<NodeInQuestion>>>,
    pub transitions_for_nodes: TransitionsForNodes
}

pub(crate) trait Segmenter {
    fn segment(&mut self, ctx: &mut Context<Training>);
    fn build_segments(&mut self) -> TransitionsForNodes;
    fn try_send_inter_node_points(&mut self) -> bool;
    fn distribute_segments(&mut self, foreign_data: TransitionsForNodes);
    fn build_node_questions(&self, node_questions: &mut HashMap<usize, HashMap<usize, Vec<NodeInQuestion>>>, transition: &Transition, asking_node: &usize, prev_transition: Option<Transition>, within_transition: bool);
}

impl Segmenter for Training {
    fn segment(&mut self, ctx: &mut Context<Training>) {
        let node_transitions = self.build_segments();
        let wait_for_points = self.try_send_inter_node_points();
        if wait_for_points {
            self.segmentation.transitions_for_nodes = node_transitions;
        } else { // if no other cluster node exists (i.e. local only)
            self.segmentation.rotation_protocol.start(self.parameters.n_cluster_nodes - 1);
            self.segmentation.rotation_protocol.resolve_buffer(ctx.address().recipient());
            self.distribute_segments(node_transitions);
        }
    }

    fn build_segments(&mut self) -> TransitionsForNodes{
        let own_id = self.cluster_nodes.get_own_idx();
        let is_not_first = own_id.ne(&0);
        let mut foreign_data: TransitionsForNodes = HashMap::new();
        let segments_per_node = (self.parameters.rate as f32 / self.parameters.n_cluster_nodes as f32).floor() as usize;
        let mut last_point: Option<PointRef> = None;
        let mut last_to_node_id = None;
        let mut node_questions: HashMap<usize, HashMap<usize, Vec<NodeInQuestion>>> = HashMap::new();
        let mut last_transition = None;
        for point in self.data_store.get_points() {
            match last_point {
                Some(last_point) => {
                    let from_node_id = last_point.get_segment() / segments_per_node;
                    let to_node_id = point.get_segment() / segments_per_node;
                    let transition = Transition::new(last_point, point.clone());

                    if transition.crosses_segments() && transition.has_valid_direction(self.parameters.rate as isize) { // valid transition
                        if from_node_id == own_id { // normal transition
                            self.data_store.add_transition(transition.clone());
                        }

                        if let Some(last_to_node_id) = last_to_node_id {
                            if from_node_id != last_to_node_id {  // found split between two transitions
                                self.build_node_questions(&mut node_questions, &transition, &last_to_node_id, last_transition.clone(), false);
                            }
                        }

                        if from_node_id != to_node_id { // found split within transition
                            self.build_node_questions(&mut node_questions, &transition, &from_node_id, last_transition, true);
                        }

                        match foreign_data.get_mut(&from_node_id) {
                            Some(foreign_data) => foreign_data.push(transition.materialize()),
                            None => { foreign_data.insert(from_node_id, vec![transition.materialize()]); }
                        }
                        last_to_node_id = Some(to_node_id.clone());
                        last_transition = Some(transition);
                    }
                },
                None => if is_not_first {
                    self.segmentation.send_point = Some(point.deref().clone());
                }
            }

            last_point = Some(point.clone());
        }
        self.segmentation.last_point = last_point.map(|x| x.deref().clone());
        self.segmentation.node_questions = node_questions;
        foreign_data
    }

    fn try_send_inter_node_points(&mut self) -> bool {
        let point = self.segmentation.send_point.take();
        match self.cluster_nodes.get_previous_idx() {
            Some(prev_idx) => {
                match point {
                    Some(point) => {
                        self.cluster_nodes.get_as(&prev_idx, "Training").unwrap().do_send(SendFirstPointMessage { point });
                        self.segmentation.rotation_protocol.sent();
                    },
                    None => () // first node does not send a SendFirstPointMessage
                }

                let own_id = self.cluster_nodes.get_own_idx();
                own_id.ne(&self.cluster_nodes.len()) // last cluster node does not receive a SendFirstPointMessage
            },
            None => { // local only case
                assert!(self.segmentation.send_point.is_none(), "This should be empty, because there are no other cluster nodes!");
                false
            }
        }
    }

    fn distribute_segments(&mut self, foreign_data: TransitionsForNodes) {
        match self.cluster_nodes.get_next_idx() {
            Some(next_id) => {
                self.cluster_nodes.get_as(&next_id, "Training").unwrap().do_send(SegmentMessage { segments: foreign_data });
                self.segmentation.rotation_protocol.sent();
            },
            None => {
                self.own_addr.as_ref().expect("Should be set by now").do_send(SegmentedMessage);
            }
        }
    }

    fn build_node_questions(&self, node_questions: &mut HashMap<usize, HashMap<usize, Vec<NodeInQuestion>>>, transition: &Transition, asking_node: &usize, prev_transition: Option<Transition>, within_transition: bool) {

        let segments_per_node = &self.parameters.segments_per_node();
        let point_id = transition.get_from_id();
        let node_in_question = if let Some(prev) = prev_transition {
            if within_transition {
                let wanted_segment = transition.get_from_segment().div_ceil(segments_per_node).mul(segments_per_node);
                let segment_before_wanted = (wanted_segment as isize - 1).mod_floor(&(self.parameters.rate as isize)) as usize;

                let (prev_point_id, prev_segment_id) = if wanted_segment.mod_floor(&self.parameters.rate) == transition.get_first_intersection_segment(&self.parameters.rate) {
                    (prev.get_from_id(), prev.get_to_segment())
                } else {
                    (point_id, segment_before_wanted)
                };

                (
                    prev_point_id,
                    prev_segment_id,
                    point_id,
                    wanted_segment.mod_floor(&self.parameters.rate)
                )
            } else {
                (
                    prev.get_from_id(),
                    prev.get_to_segment(),
                    point_id,
                    transition.get_first_intersection_segment(&self.parameters.rate)
                )
            }
        } else {
            if within_transition {
                let wanted_segment = transition.get_from_segment().div_ceil(segments_per_node).mul(segments_per_node);
                let segment_before_wanted = (wanted_segment as isize - 1).mod_floor(&(self.parameters.rate as isize)) as usize;

                (
                    point_id,
                    segment_before_wanted,
                    point_id,
                    wanted_segment.mod_floor(&self.parameters.rate)
                )
            } else {
                panic!("It does happen, please solve!")
            }
        };

        let answering_node = node_in_question.3.div_floor(segments_per_node);

        // todo also add node before that is foot of edge to node_in_question, in edge estimation, ask for next node
        match node_questions.get_mut(&answering_node) {
            Some(questions) => match questions.get_mut(asking_node) {
                Some(nodes) => nodes.push(node_in_question),
                None => { questions.insert(asking_node.clone(), vec![node_in_question]); }
            },
            None => {
                let questions = HashMap::from_iter([(asking_node.clone(), vec![node_in_question])]);
                node_questions.insert(answering_node.clone(), questions);
            }
        }
    }
}


impl Handler<SendFirstPointMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SendFirstPointMessage, ctx: &mut Self::Context) -> Self::Result {
        let last_transition = MaterializedTransition::new(self.segmentation.last_point.take().unwrap(), msg.point);

        let mut node_transitions = self.segmentation.transitions_for_nodes.clone();
        let segments_per_node = (self.parameters.rate as f32 / self.cluster_nodes.len_incl_own() as f32).floor() as usize;
        let node_id = last_transition.get_from_segment() / segments_per_node;
        match (&mut node_transitions).get_mut(&node_id) {
            None => { node_transitions.insert(node_id, vec![last_transition]); },
            Some(transitions) => transitions.push(last_transition)
        }
        self.segmentation.transitions_for_nodes.clear();
        self.segmentation.rotation_protocol.start(self.parameters.n_cluster_nodes - 1);
        self.segmentation.rotation_protocol.resolve_buffer(ctx.address().recipient());
        self.distribute_segments(node_transitions);
    }
}

impl Handler<SegmentMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SegmentMessage, ctx: &mut Self::Context) -> Self::Result {
        if !self.segmentation.rotation_protocol.received(&msg) {
            return
        }

        let own_id = self.cluster_nodes.get_own_idx();
        let next_id = (own_id + 1) % (&self.cluster_nodes.len_incl_own());
        let mut segments = msg.segments;
        let own_transitions = segments.remove(&own_id).unwrap();

        self.data_store.add_materialized_transitions(own_transitions);

        if self.segmentation.rotation_protocol.is_running() {
            self.cluster_nodes.get_as(&next_id, "Training").unwrap().do_send(SegmentMessage { segments });
            self.segmentation.rotation_protocol.sent();
        } else {
            ctx.address().do_send(SegmentedMessage);
        }
    }
}
