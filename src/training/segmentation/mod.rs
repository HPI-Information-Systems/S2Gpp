#[cfg(test)]
mod tests;
mod data_structures;
pub(crate) mod messages;

use crate::training::Training;
use crate::utils::PolarCoords;
use std::collections::HashMap;
use ndarray::{Array1, Axis};
use std::f32::consts::PI;
use std::iter::FromIterator;
use std::ops::Mul;
use actix::prelude::*;
use num_integer::Integer;
use serde::{Serialize, Deserialize};
pub use crate::training::segmentation::data_structures::SegmentedTransition;
pub use crate::training::segmentation::messages::{SegmentedMessage, SegmentMessage, SendFirstPointMessage};
use crate::utils::rotation_protocol::RotationProtocol;


#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PointWithId {
    pub id: usize,
    pub coords: Array1<f32>
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentedPointWithId {
    pub segment_id: usize,
    pub point_with_id: PointWithId
}

pub trait ToSegmented {
    fn to_segmented(&self, segment_id: usize) -> Vec<SegmentedPointWithId>;
}

impl ToSegmented for Vec<PointWithId> {
    fn to_segmented(&self, segment_id: usize) -> Vec<SegmentedPointWithId> {
        self.iter()
            .map(|p| SegmentedPointWithId { segment_id, point_with_id: p.clone() })
            .collect()
    }
}

pub trait FromPointsWithId {
    fn append_points_with_id(&mut self, points: &Vec<PointWithId>, segment_id: usize);
}

impl FromPointsWithId for Vec<SegmentedPointWithId> {
    fn append_points_with_id(&mut self, points: &Vec<PointWithId>, segment_id: usize) {
        let mut segmented_points = points.to_segmented(segment_id);
        self.append(&mut segmented_points);
    }
}

pub type TransitionsForNodes = HashMap<usize, Vec<SegmentedTransition>>;
/// (prev_point_id, prev_point_segment_id, point_id, segment_id)
pub type NodeInQuestion = (usize, usize, usize, usize);

#[derive(Default)]
pub struct Segmentation {
    /// list of transitions of segmented points
    pub segments: Vec<SegmentedTransition>,
    /// list of points that are endpoints to transitions from the previous cluster node
    pub send_point: Option<SegmentedPointWithId>,
    /// {point_id: transition_id}
    pub segment_index: HashMap<usize, usize>,
    pub n_received: usize,
    pub node_transitions: HashMap<usize, Vec<SegmentedTransition>>,
    pub last_point: Option<SegmentedPointWithId>,
    rotation_protocol: RotationProtocol<SegmentMessage>,
    /// {cluster node id (answering): {cluster node id (asking): \[NodeInQuestion\]}}
    pub node_questions: HashMap<usize, HashMap<usize, Vec<NodeInQuestion>>>
}

pub trait Segmenter {
    fn segment(&mut self, ctx: &mut Context<Training>);
    fn build_segments(&mut self) -> HashMap<usize, Vec<SegmentedTransition>>;
    fn try_send_inter_node_points(&mut self) -> bool;
    fn distribute_segments(&mut self, node_transitions: HashMap<usize, Vec<SegmentedTransition>>);
    fn build_segment_index(&mut self);
    fn build_node_questions(&self, node_questions: &mut HashMap<usize, HashMap<usize, Vec<NodeInQuestion>>>, transition: &SegmentedTransition, asking_node: &usize, prev_transition: Option<SegmentedTransition>, within_transition: bool);
}

impl Segmenter for Training {
    fn segment(&mut self, ctx: &mut Context<Training>) {
        let node_transitions = self.build_segments();
        let wait_for_points = self.try_send_inter_node_points();
        if wait_for_points {
            self.segmentation.node_transitions = node_transitions;
        } else { // if no other cluster node exists (i.e. local only)
            self.segmentation.rotation_protocol.start(self.parameters.n_cluster_nodes - 1);
            self.segmentation.rotation_protocol.resolve_buffer(ctx.address().recipient());
            self.distribute_segments(node_transitions);
        }
    }

    fn build_segments(&mut self) -> HashMap<usize, Vec<SegmentedTransition>>{
        let own_id = self.cluster_nodes.get_own_idx();
        let is_not_first = own_id.ne(&0);
        let mut node_transitions = TransitionsForNodes::new();
        let segments_per_node = (self.parameters.rate as f32 / self.parameters.n_cluster_nodes as f32).floor() as usize;
        let mut last_point: Option<SegmentedPointWithId> = None;
        let points_per_node = self.dataset_stats.as_ref().expect("DatasetStats should've been set by now!").n.expect("DatasetStats.n should've been set by now!") / self.cluster_nodes.len_incl_own();
        let starting_id = own_id * points_per_node;
        let mut last_to_node_id = None;
        let mut node_questions: HashMap<usize, HashMap<usize, Vec<NodeInQuestion>>> = HashMap::new();
        let mut last_transition = None;
        for (id, x) in self.rotation.rotated.as_ref().unwrap().axis_iter(Axis(0)).enumerate() {
            let polar = x.to_polar();
            let segment_id = get_segment_id(polar[1], self.parameters.rate);
            let point = SegmentedPointWithId { segment_id, point_with_id: PointWithId { id: id + starting_id, coords: x.iter().map(|x| x.clone()).collect() } };
            match last_point {
                Some(last_point) => {
                    let from_node_id = last_point.segment_id / segments_per_node;
                    let to_node_id = segment_id / segments_per_node;
                    let transition = SegmentedTransition::new(last_point, point.clone());


                    if transition.crosses_segments() && transition.has_valid_direction(self.parameters.rate) { // valid transition
                        if from_node_id == own_id { // normal transition
                            self.segmentation.segments.push(transition.clone());
                        }

                        if let Some(last_to_node_id) = last_to_node_id {
                            if from_node_id != last_to_node_id {  // found split between two transitions
                                self.build_node_questions(&mut node_questions, &transition, &last_to_node_id, last_transition.clone(), false);
                            }
                        }

                        if from_node_id != to_node_id { // found split within transition
                            self.build_node_questions(&mut node_questions, &transition, &from_node_id, last_transition, true);
                        }

                        match node_transitions.get_mut(&from_node_id) {
                            Some(node_transitions) => node_transitions.push(transition.clone()),
                            None => { node_transitions.insert(from_node_id, vec![transition.clone()]); }
                        }
                        last_to_node_id = Some(to_node_id.clone());
                        last_transition = Some(transition);
                    }
                },
                None => if is_not_first {
                    self.segmentation.send_point = Some(point.clone());
                }
            }

            last_point = Some(point);
        }
        self.segmentation.last_point = last_point;
        self.segmentation.node_questions = node_questions;
        node_transitions
    }

    fn try_send_inter_node_points(&mut self) -> bool {
        let point = self.segmentation.send_point.clone();
        self.segmentation.send_point = None;
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

    fn distribute_segments(&mut self, node_transitions: HashMap<usize, Vec<SegmentedTransition>>) {
        match self.cluster_nodes.get_next_idx() {
            Some(next_id) => {
                self.cluster_nodes.get_as(&next_id, "Training").unwrap().do_send(SegmentMessage { segments: node_transitions });
                self.segmentation.rotation_protocol.sent();
            },
            None => {
                self.build_segment_index();
                self.own_addr.as_ref().expect("Should be set by now").do_send(SegmentedMessage);
            }
        }
    }

    fn build_segment_index(&mut self) {
        let index = self.segmentation.segments.iter().enumerate().map(|(id, transition)| {
            (transition.from.point_with_id.id, id)
        }).collect();
        self.segmentation.segment_index = index;
    }

    fn build_node_questions(&self, node_questions: &mut HashMap<usize, HashMap<usize, Vec<NodeInQuestion>>>, transition: &SegmentedTransition, asking_node: &usize, prev_transition: Option<SegmentedTransition>, within_transition: bool) {

        let segments_per_node = &self.parameters.segments_per_node();
        let point_id = transition.get_from_id();
        let node_in_question = if let Some(prev) = prev_transition {
            if within_transition {
                let wanted_segment = transition.get_from_segment().div_ceil(segments_per_node).mul(segments_per_node);
                let segment_before_wanted = (wanted_segment - 1).mod_floor(&self.parameters.rate);


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
                let segment_before_wanted = (wanted_segment - 1).mod_floor(&self.parameters.rate);

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

fn get_segment_id(angle: f32, n_segments: usize) -> usize {
    let positive_angle = (2.0 * PI) + angle;
    let segment_size = (2.0 * PI) / (n_segments as f32);
    (positive_angle / segment_size).floor() as usize % n_segments
}

impl Handler<SendFirstPointMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SendFirstPointMessage, ctx: &mut Self::Context) -> Self::Result {
        let last_transition = SegmentedTransition::new(self.segmentation.last_point.as_ref().unwrap().clone(), msg.point);
        self.segmentation.last_point = None;

        let mut node_transitions = self.segmentation.node_transitions.clone();
        let segments_per_node = (self.parameters.rate as f32 / self.cluster_nodes.len_incl_own() as f32).floor() as usize;
        let node_id = last_transition.from.segment_id / segments_per_node;
        match (&mut node_transitions).get_mut(&node_id) {
            None => { node_transitions.insert(node_id, vec![last_transition]); },
            Some(transitions) => transitions.push(last_transition)
        }
        self.segmentation.node_transitions.clear();
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
        let mut own_points = segments.remove(&own_id).unwrap();

        self.segmentation.segments.append(&mut own_points);

        if self.segmentation.rotation_protocol.is_running() {
            self.cluster_nodes.get_as(&next_id, "Training").unwrap().do_send(SegmentMessage { segments });
            self.segmentation.rotation_protocol.sent();
        } else {
            self.build_segment_index();
            ctx.address().do_send(SegmentedMessage);
        }
    }
}
