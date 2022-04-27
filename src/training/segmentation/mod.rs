pub(crate) mod messages;
#[cfg(test)]
mod tests;

use crate::data_store::materialize::Materialize;
use crate::data_store::node_questions::NodeQuestions;
use crate::data_store::point::{Point, PointRef};
use crate::data_store::transition::{MaterializedTransition, Transition, TransitionMixin};
pub use crate::training::segmentation::messages::{
    SegmentMessage, SegmentedMessage, SendFirstPointMessage,
};
use crate::training::Training;
use crate::utils::direct_protocol::DirectProtocol;
use crate::utils::rotation_protocol::RotationProtocol;
use actix::prelude::*;
use log::*;
use std::collections::HashMap;

use self::messages::TransitionCountMessage;

pub(crate) type TransitionsForNodes = HashMap<usize, Vec<MaterializedTransition>>;
/// (prev_point_id, prev_point_segment_id, point_id, segment_id)

#[derive(Default)]
pub(crate) struct Segmentation {
    /// list of points that are endpoints to transitions from the previous cluster node
    pub send_point: Option<Point>,
    pub send_transition: Option<MaterializedTransition>,
    pub last_point: Option<Point>,
    pub last_transition: Option<Transition>,
    direct_protocol: DirectProtocol<SegmentMessage>,
    /// {cluster node id (answering): {cluster node id (asking): \[NodeInQuestion\]}}
    pub node_questions: NodeQuestions,
    pub transitions_for_nodes: TransitionsForNodes,
    pub transition_count_protocol: RotationProtocol<TransitionCountMessage>,
    pub global_transition_count: usize,
}

pub(crate) trait Segmenter {
    fn segment(&mut self, ctx: &mut Context<Training>);
    fn distribute_or_wait_for_segments(
        &mut self,
        node_transitions: TransitionsForNodes,
        ctx: &mut Context<Training>,
    );
    fn build_segments(&mut self) -> TransitionsForNodes;
    fn find_splits(
        &mut self,
        prev_transition: Option<Transition>,
        last_to_node_id: Option<usize>,
        transition: &Transition,
        from_node_id: usize,
        to_node_id: usize,
    );
    fn search_split_between_transitions(
        &mut self,
        prev_transition: Option<Transition>,
        transition: &Transition,
        from_node_id: usize,
        last_to_node_id: usize,
    );
    fn search_split_within_transition(
        &mut self,
        prev_transition: Option<Transition>,
        transition: &Transition,
        from_node_id: usize,
        to_node_id: usize,
    );
    fn try_send_inter_node_points(&mut self) -> bool;
    fn distribute_segments(&mut self, foreign_data: TransitionsForNodes);
    fn self_correction(
        &mut self,
        node_transitions: TransitionsForNodes,
        ctx: &mut Context<Training>,
    );
    fn try_self_correction(&mut self, ctx: &mut Context<Training>);
    fn finish_self_correction(&mut self, ctx: &mut Context<Training>);
    fn clear_segmentation(&mut self);
}

impl Segmenter for Training {
    fn segment(&mut self, ctx: &mut Context<Training>) {
        let node_transitions = self.build_segments();
        if self.parameters.self_correction {
            self.self_correction(node_transitions, ctx);
        } else {
            self.distribute_or_wait_for_segments(node_transitions, ctx);
        }
    }

    fn distribute_or_wait_for_segments(
        &mut self,
        node_transitions: TransitionsForNodes,
        ctx: &mut Context<Training>,
    ) {
        let wait_for_points = self.try_send_inter_node_points();
        if wait_for_points {
            self.segmentation.transitions_for_nodes = node_transitions;
        } else {
            // if no other cluster node exists (i.e. local only)
            self.segmentation
                .direct_protocol
                .start(self.parameters.n_cluster_nodes - 1);
            self.segmentation
                .direct_protocol
                .resolve_buffer(ctx.address().recipient());
            self.distribute_segments(node_transitions);
        }
    }

    fn build_segments(&mut self) -> TransitionsForNodes {
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

                    if transition.crosses_segments()
                        && (transition.has_valid_direction(self.parameters.rate as isize))
                    {
                        // valid transition
                        let from_node_id = self.segment_id_to_assignment(last_point.get_segment());
                        let to_node_id = self.segment_id_to_assignment(point.get_segment());
                        if from_node_id == own_id {
                            // normal transition
                            self.data_store.add_transition(transition.clone());
                        }

                        self.find_splits(
                            last_transition,
                            last_to_node_id,
                            &transition,
                            from_node_id,
                            to_node_id,
                        );

                        // foreign and own because transition can reach in foreign segments
                        match foreign_data.get_mut(&from_node_id) {
                            Some(foreign_data) => foreign_data.push(transition.materialize()),
                            None => {
                                foreign_data.insert(from_node_id, vec![transition.materialize()]);
                            }
                        }
                        last_to_node_id = Some(to_node_id);
                        if self.segmentation.send_transition.is_none() {
                            self.segmentation.send_transition =
                                Some(transition.clone().materialize());
                        }
                        last_transition = Some(transition);
                    }
                }
                None => {
                    if is_not_first {
                        self.segmentation.send_point = Some(point.deref_clone());
                    }
                }
            }

            last_point = Some(point.clone());
        }

        if self.data_store.count_transitions() == 0 {
            warn!("Could not generate transitions! Try different pattern-length / latent parameter settings!")
        }

        self.segmentation.last_point = last_point.map(|x| x.deref_clone());
        self.segmentation.last_transition = last_transition;

        debug!("#transitions: {}", self.data_store.count_transitions());

        foreign_data
    }

    fn find_splits(
        &mut self,
        prev_transition: Option<Transition>,
        last_to_node_id: Option<usize>,
        transition: &Transition,
        from_node_id: usize,
        to_node_id: usize,
    ) {
        if let Some(last_to_node_id) = last_to_node_id {
            self.search_split_between_transitions(
                prev_transition.clone(),
                transition,
                from_node_id,
                last_to_node_id,
            );
        }

        self.search_split_within_transition(prev_transition, transition, from_node_id, to_node_id);
    }

    fn search_split_between_transitions(
        &mut self,
        prev_transition: Option<Transition>,
        transition: &Transition,
        from_node_id: usize,
        last_to_node_id: usize,
    ) {
        if from_node_id != last_to_node_id {
            // found split between two transitions
            self.segmentation.node_questions.ask(
                transition,
                prev_transition,
                false,
                1,
                self.parameters.clone(),
            );
        }
    }

    fn search_split_within_transition(
        &mut self,
        prev_transition: Option<Transition>,
        transition: &Transition,
        from_node_id: usize,
        to_node_id: usize,
    ) {
        if from_node_id != to_node_id {
            // found split within transition
            self.segmentation.node_questions.ask(
                transition,
                prev_transition,
                true,
                self.cluster_node_diff(from_node_id, to_node_id),
                self.parameters.clone(),
            );
        }
    }

    fn try_send_inter_node_points(&mut self) -> bool {
        let point = self.segmentation.send_point.take();
        let transition = self.segmentation.send_transition.take();
        match self.cluster_nodes.get_previous_idx() {
            Some(prev_idx) => {
                if let (Some(point), Some(transition)) = (point, transition) {
                    self.cluster_nodes
                        .get_as(&prev_idx, "Training")
                        .unwrap()
                        .do_send(SendFirstPointMessage { point, transition });
                }

                let own_id = self.cluster_nodes.get_own_idx();
                own_id.ne(&self.cluster_nodes.len()) // last cluster node does not receive a SendFirstPointMessage
            }
            None => {
                // local only case
                assert!(
                    self.segmentation.send_point.is_none(),
                    "This should be empty, because there are no other cluster nodes!"
                );
                false
            }
        }
    }

    fn distribute_segments(&mut self, mut foreign_data: TransitionsForNodes) {
        if self.cluster_nodes.len() == 0 {
            self.own_addr
                .as_ref()
                .expect("Should be set by now")
                .do_send(SegmentedMessage);
        }

        for (id, node) in self.cluster_nodes.iter() {
            let mut training_node = node.clone();
            training_node.change_id("Training".to_string());
            match foreign_data.remove(id) {
                Some(segments) => training_node.do_send(SegmentMessage { segments }),
                None => training_node.do_send(SegmentMessage { segments: vec![] }),
            }
            self.segmentation.direct_protocol.sent();
        }
    }

    fn self_correction(
        &mut self,
        node_transitions: TransitionsForNodes,
        ctx: &mut Context<Training>,
    ) {
        match self.cluster_nodes.get_next_as("Training") {
            Some(next_node) => {
                self.segmentation
                    .transition_count_protocol
                    .start(self.cluster_nodes.len());
                self.segmentation
                    .transition_count_protocol
                    .resolve_buffer(ctx.address().recipient());
                let transition_count = self.data_store.count_transitions()
                    + node_transitions
                        .iter()
                        .map(|(_segment, transitions)| transitions.len())
                        .sum::<usize>();
                next_node.do_send(TransitionCountMessage {
                    count: transition_count,
                });
                self.segmentation.transition_count_protocol.sent();
                self.segmentation.global_transition_count += transition_count;
                self.segmentation.transitions_for_nodes = node_transitions;
            }
            None => {
                self.segmentation.global_transition_count = self.data_store.count_transitions();
                self.try_self_correction(ctx);
            }
        }
    }

    fn try_self_correction(&mut self, ctx: &mut Context<Training>) {
        debug!(
            "{} < {}",
            self.segmentation.global_transition_count,
            num_integer::div_floor(*self.dataset_stats.as_ref().unwrap().n.as_ref().unwrap(), 2)
        );
        if self
            .segmentation
            .global_transition_count
            .lt(&num_integer::div_floor(
                *self.dataset_stats.as_ref().unwrap().n.as_ref().unwrap(),
                2,
            ))
        {
            self.clear_segmentation();
            self.data_store.mirror_points(self.parameters.rate);
            let node_transitions = self.build_segments();
            self.distribute_or_wait_for_segments(node_transitions, ctx);
        } else {
            self.finish_self_correction(ctx);
        }
    }

    fn finish_self_correction(&mut self, ctx: &mut Context<Training>) {
        self.distribute_or_wait_for_segments(self.segmentation.transitions_for_nodes.clone(), ctx)
    }

    fn clear_segmentation(&mut self) {
        self.data_store.clear_transitions();
        self.segmentation.node_questions.clear();
        self.segmentation.send_point.take();
        self.segmentation.last_point.take();
        self.segmentation.send_transition.take();
        self.segmentation.last_transition.take();
    }
}

impl Handler<TransitionCountMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: TransitionCountMessage, ctx: &mut Self::Context) -> Self::Result {
        if !self.segmentation.transition_count_protocol.received(&msg) {
            return;
        }

        self.segmentation.global_transition_count += msg.count;

        if !self.segmentation.transition_count_protocol.is_running() {
            self.try_self_correction(ctx);
        } else {
            match self.cluster_nodes.get_next_as("Training") {
                Some(next_node) => next_node.do_send(msg),
                None => panic!("There is suddenly no more next node."),
            }
            self.segmentation.transition_count_protocol.sent();
        }
    }
}

impl Handler<SendFirstPointMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SendFirstPointMessage, ctx: &mut Self::Context) -> Self::Result {
        let spanning_transition = Transition::new(
            self.segmentation.last_point.take().unwrap().into_ref(),
            msg.point.into_ref(),
        );
        let last_transition = self.segmentation.last_transition.as_ref().unwrap().clone();

        if spanning_transition.crosses_segments()
            & spanning_transition.has_valid_direction(self.parameters.rate as isize)
        {
            // valid transition
            let from_node_id =
                self.segment_id_to_assignment(spanning_transition.get_from_segment());
            let to_node_id = self.segment_id_to_assignment(spanning_transition.get_to_segment());
            let last_transition_to_cluster_id =
                self.segment_id_to_assignment(last_transition.get_to_segment());

            if from_node_id == self.cluster_nodes.get_own_idx() {
                // normal transition
                self.data_store.add_transition(spanning_transition.clone());
            }

            self.find_splits(
                Some(last_transition),
                Some(last_transition_to_cluster_id),
                &spanning_transition,
                from_node_id,
                to_node_id,
            );

            match self
                .segmentation
                .transitions_for_nodes
                .get_mut(&from_node_id)
            {
                Some(foreign_data) => foreign_data.push(spanning_transition.materialize()),
                None => {
                    self.segmentation
                        .transitions_for_nodes
                        .insert(from_node_id, vec![spanning_transition.materialize()]);
                }
            }
        } else {
            let transition = msg.transition.into_transition();
            let transition_from_cluster_id =
                self.segment_id_to_assignment(transition.get_from_segment());
            let last_transition_to_cluster_id =
                self.segment_id_to_assignment(last_transition.get_to_segment());

            self.search_split_between_transitions(
                Some(last_transition),
                &transition,
                transition_from_cluster_id,
                last_transition_to_cluster_id,
            );
        }

        let node_transitions = self.segmentation.transitions_for_nodes.clone();
        self.segmentation.transitions_for_nodes.clear();
        self.segmentation
            .direct_protocol
            .start(self.parameters.n_cluster_nodes - 1);
        self.segmentation
            .direct_protocol
            .resolve_buffer(ctx.address().recipient());
        self.distribute_segments(node_transitions);
    }
}

impl Handler<SegmentMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SegmentMessage, ctx: &mut Self::Context) -> Self::Result {
        if !self.segmentation.direct_protocol.received(&msg) {
            return;
        }

        self.data_store.add_materialized_transitions(msg.segments);

        if !self.segmentation.direct_protocol.is_running() {
            ctx.address().do_send(SegmentedMessage);
        }
    }
}
