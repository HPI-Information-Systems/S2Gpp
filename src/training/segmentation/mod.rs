#[cfg(test)]
mod tests;
pub(crate) mod messages;

use crate::training::Training;
use std::collections::HashMap;
use std::ops::{Deref};
use actix::prelude::*;
use crate::data_store::point::{Point, PointRef};
use crate::data_store::transition::{MaterializedTransition, Transition, TransitionMixin};
use crate::data_store::materialize::Materialize;
use crate::data_store::node_questions::NodeQuestions;
pub use crate::training::segmentation::messages::{SegmentedMessage, SegmentMessage, SendFirstPointMessage};
use crate::utils::rotation_protocol::RotationProtocol;


pub(crate) type TransitionsForNodes = HashMap<usize, Vec<MaterializedTransition>>;
/// (prev_point_id, prev_point_segment_id, point_id, segment_id)

#[derive(Default)]
pub(crate) struct Segmentation {
    /// list of points that are endpoints to transitions from the previous cluster node
    pub send_point: Option<Point>,
    pub send_transition: Option<MaterializedTransition>,
    pub last_point: Option<Point>,
    pub last_transition: Option<Transition>,
    rotation_protocol: RotationProtocol<SegmentMessage>,
    /// {cluster node id (answering): {cluster node id (asking): \[NodeInQuestion\]}}
    pub node_questions: NodeQuestions,
    pub transitions_for_nodes: TransitionsForNodes
}

pub(crate) trait Segmenter {
    fn segment(&mut self, ctx: &mut Context<Training>);
    fn build_segments(&mut self) -> TransitionsForNodes;
    fn find_splits(&mut self, prev_transition: Option<Transition>, last_to_node_id: Option<usize>, transition: &Transition, from_node_id: usize, to_node_id: usize);
    fn search_split_between_transitions(&mut self, prev_transition: Option<Transition>, transition: &Transition, from_node_id: usize, last_to_node_id: usize);
    fn search_split_within_transition(&mut self, prev_transition: Option<Transition>, transition: &Transition, from_node_id: usize, to_node_id: usize);
    fn try_send_inter_node_points(&mut self) -> bool;
    fn distribute_segments(&mut self, foreign_data: TransitionsForNodes);
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
        let mut last_point: Option<PointRef> = None;
        let mut last_to_node_id = None;
        let mut last_transition: Option<Transition> = None;
        for point in self.data_store.get_points() {
            match last_point {
                Some(last_point) => {
                    let transition = Transition::new(last_point.clone(), point.clone());

                    if transition.crosses_segments() && transition.has_valid_direction(self.parameters.rate as isize) { // valid transition
                        let from_node_id = self.segment_id_to_assignment(last_point.get_segment());
                        let to_node_id = self.segment_id_to_assignment(point.get_segment());
                        if from_node_id == own_id { // normal transition
                            self.data_store.add_transition(transition.clone());
                        }

                        self.find_splits(last_transition, last_to_node_id, &transition, from_node_id, to_node_id);

                        // foreign and own because transition can reach in foreign segments
                        match foreign_data.get_mut(&from_node_id) {
                            Some(foreign_data) => foreign_data.push(transition.materialize()),
                            None => { foreign_data.insert(from_node_id, vec![transition.materialize()]); }
                        }
                        last_to_node_id = Some(to_node_id.clone());
                        if self.segmentation.send_transition.is_none() {
                            self.segmentation.send_transition = Some(transition.clone().materialize());
                        }
                        last_transition = Some(transition);
                    }
                },
                None => if is_not_first {
                    self.segmentation.send_point = Some(point.deref().clone());
                }
            }

            last_point = Some(point.clone());
        }

        if self.data_store.count_transitions() == 0 {
            panic!("Could not generate transitions! Try different pattern-length / latent parameter settings!")
        }

        self.segmentation.last_point = last_point.map(|x| x.deref().clone());
        self.segmentation.last_transition = last_transition;
        foreign_data
    }

    fn find_splits(&mut self, prev_transition: Option<Transition>, last_to_node_id: Option<usize>, transition: &Transition, from_node_id: usize, to_node_id: usize) {
        if let Some(last_to_node_id) = last_to_node_id {
            self.search_split_between_transitions(prev_transition.clone(), transition, from_node_id, last_to_node_id);
        }

        self.search_split_within_transition(prev_transition, transition, from_node_id, to_node_id);
    }

    fn search_split_between_transitions(&mut self, prev_transition: Option<Transition>, transition: &Transition, from_node_id: usize, last_to_node_id: usize) {
        if from_node_id != last_to_node_id {  // found split between two transitions
            self.segmentation.node_questions.ask(transition, prev_transition.clone(), false, 1, self.parameters.clone());
        }
    }

    fn search_split_within_transition(&mut self, prev_transition: Option<Transition>, transition: &Transition, from_node_id: usize, to_node_id: usize) {
        if from_node_id != to_node_id { // found split within transition
            self.segmentation.node_questions.ask(transition, prev_transition, true, self.cluster_node_diff(from_node_id, to_node_id), self.parameters.clone());
        }
    }

    fn try_send_inter_node_points(&mut self) -> bool {
        let point = self.segmentation.send_point.take();
        let transition = self.segmentation.send_transition.take();
        match self.cluster_nodes.get_previous_idx() {
            Some(prev_idx) => {
                match (point, transition) {
                    (Some(point), Some(transition)) => {
                        self.cluster_nodes.get_as(&prev_idx, "Training").unwrap().do_send(
                            SendFirstPointMessage {
                                point,
                                transition
                            }
                        );
                        self.segmentation.rotation_protocol.sent();
                    },
                    _ => () // first node does not send a SendFirstPointMessage
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
}


impl Handler<SendFirstPointMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SendFirstPointMessage, ctx: &mut Self::Context) -> Self::Result {
        let spanning_transition = Transition::new(self.segmentation.last_point.take().unwrap().to_ref(), msg.point.to_ref());
        let last_transition = self.segmentation.last_transition.as_ref().unwrap().clone();

        if spanning_transition.crosses_segments() & spanning_transition.has_valid_direction(self.parameters.rate as isize) { // valid transition
            let from_node_id = self.segment_id_to_assignment(spanning_transition.get_from_segment());
            let to_node_id = self.segment_id_to_assignment(spanning_transition.get_to_segment());
            let last_transition_to_cluster_id = self.segment_id_to_assignment(last_transition.get_to_segment());

            if from_node_id == self.cluster_nodes.get_own_idx() { // normal transition
                self.data_store.add_transition(spanning_transition.clone());
            }

            self.find_splits(Some(last_transition), Some(last_transition_to_cluster_id), &spanning_transition, from_node_id, to_node_id);

            match self.segmentation.transitions_for_nodes.get_mut(&from_node_id) {
                Some(foreign_data) => foreign_data.push(spanning_transition.materialize()),
                None => { self.segmentation.transitions_for_nodes.insert(from_node_id, vec![spanning_transition.materialize()]); }
            }
        } else {
            let transition = msg.transition.to_transition();
            let transition_from_cluster_id = self.segment_id_to_assignment(transition.get_from_segment());
            let last_transition_to_cluster_id = self.segment_id_to_assignment(last_transition.get_to_segment());

            self.search_split_between_transitions(Some(last_transition), &transition, transition_from_cluster_id, last_transition_to_cluster_id);
        }

        let node_transitions = self.segmentation.transitions_for_nodes.clone();
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
