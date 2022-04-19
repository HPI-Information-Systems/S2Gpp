mod messages;
mod multi_kde;

use crate::training::Training;
use actix::{
    Actor, ActorFutureExt, AsyncContext, Context, ContextFutureSpawner, Handler, Recipient,
    WrapFuture,
};
use actix_telepathy::AnyAddr;
use meanshift_rs::{ClusteringResponse, MeanShiftActor, MeanShiftMessage};
use ndarray::{stack_new_axis, ArrayView1, Axis};
use std::collections::HashMap;
use std::ops::Deref;
use std::str::FromStr;

use crate::data_store::intersection::IntersectionRef;
use crate::data_store::node::{IndependentNode, Node};
use crate::data_store::node_questions::node_in_question::NodeInQuestion;
use crate::data_store::node_questions::NodeQuestions;
pub(crate) use crate::training::node_estimation::messages::{
    AskForForeignNodes, ForeignNodesAnswer, NodeEstimationDone,
};
use crate::training::node_estimation::multi_kde::actors::messages::MultiKDEMessage;
use crate::training::node_estimation::multi_kde::actors::MultiKDEActor;
use crate::utils::direct_protocol::DirectProtocol;

#[derive(Default)]
pub(crate) struct NodeEstimation {
    pub next_foreign_node: HashMap<(usize, usize), (usize, IndependentNode)>,
    pub(crate) current_intersections: Vec<IntersectionRef>,
    pub(crate) current_segment_id: usize,
    pub(crate) source: Option<Recipient<NodeEstimationDone>>,
    asking_direct_protocol: DirectProtocol<AskForForeignNodes>,
    answering_direct_protocol: DirectProtocol<ForeignNodesAnswer>,
    answers: HashMap<usize, Vec<(usize, usize, usize, IndependentNode)>>,
}

pub(crate) trait NodeEstimator {
    fn estimate_nodes(&mut self, clustering_recipient: Recipient<ClusteringResponse<f32>>);
    fn ask_for_foreign_nodes(&mut self, ctx: &mut Context<Training>);
    fn ask_next(&mut self);
    fn search_for_asked_nodes(&mut self, node_questions: HashMap<usize, Vec<NodeInQuestion>>);
    fn start_anwering(&mut self, ctx: &mut Context<Training>);
    fn answer_next(&mut self, ctx: &mut Context<Training>);
    fn take_in_answers(&mut self, answers: Vec<(usize, usize, usize, IndependentNode)>);
    fn finalize_node_estimation(&mut self, ctx: &mut Context<Training>);
}

impl NodeEstimator for Training {
    fn estimate_nodes(&mut self, clustering_recipient: Recipient<ClusteringResponse<f32>>) {
        let segment_id = self.node_estimation.current_segment_id;

        match self.data_store.get_intersections_from_segment(segment_id) {
            Some(intersections) => {
                self.node_estimation.current_intersections = intersections.to_vec();
                let coordinates: Vec<ArrayView1<f32>> =
                    intersections.iter().map(|x| x.get_coordinates()).collect();
                let data = stack_new_axis(Axis(0), coordinates.as_slice()).unwrap();
                match &self.parameters.clustering {
                    Clustering::MeanShift => {
                        let cluster_addr = MeanShiftActor::new(self.parameters.n_threads).start();
                        cluster_addr.do_send(MeanShiftMessage {
                            source: Some(clustering_recipient),
                            data,
                        });
                    }
                    Clustering::MultiKDE => {
                        let cluster_addr =
                            MultiKDEActor::new(clustering_recipient, self.parameters.n_threads)
                                .start();
                        cluster_addr.do_send(MultiKDEMessage { data });
                    }
                }
            }
            None => {
                clustering_recipient
                    .do_send(ClusteringResponse {
                        cluster_centers: Default::default(),
                        labels: vec![],
                    })
                    .unwrap();
            }
        }
    }

    fn ask_for_foreign_nodes(&mut self, ctx: &mut Context<Training>) {
        if self.cluster_nodes.len() == 0 {
            self.finalize_node_estimation(ctx);
            return;
        }
        self.node_estimation
            .asking_direct_protocol
            .start(self.cluster_nodes.len_incl_own());
        self.node_estimation
            .asking_direct_protocol
            .resolve_buffer(ctx.address().recipient());

        self.ask_next();
    }

    fn ask_next(&mut self) {
        for (id, node) in self
            .cluster_nodes
            .iter_any_as(self.own_addr.as_ref().unwrap().clone(), "Training")
            .enumerate()
        {
            let msg = match self.segmentation.node_questions.remove(&id) {
                Some(questions) => AskForForeignNodes {
                    asked_nodes: NodeQuestions::from_hashmap_with_value(id, questions),
                },
                None => AskForForeignNodes {
                    asked_nodes: NodeQuestions::default(),
                },
            };
            node.do_send(msg);
            self.node_estimation.asking_direct_protocol.sent();
        }
    }

    fn search_for_asked_nodes(&mut self, mut node_questions: HashMap<usize, Vec<NodeInQuestion>>) {
        for (asking_node, _remote_addr) in self.cluster_nodes.iter() {
            let answers = match node_questions.remove(asking_node) {
                Some(questions) => questions
                    .into_iter()
                    .map(
                        |niq| match self.data_store.get_nodes_by_point_id(niq.get_point_id()) {
                            Some(nodes) => nodes
                                .iter()
                                .find_map(|node| {
                                    node.get_segment_id().eq(&niq.get_segment()).then(|| {
                                        (
                                            niq.get_prev_id(),
                                            niq.get_prev_segment(),
                                            niq.get_point_id(),
                                            node.deref().clone(),
                                        )
                                    })
                                })
                                .unwrap_or_else(|| {
                                    panic!(
                                        "There is no answer here: no segment_id: {} {}",
                                        &niq.get_point_id(),
                                        &niq.get_segment()
                                    )
                                }),
                            None => {
                                panic!(
                                    "There is no answer here!: no point_id: {}",
                                    niq.get_point_id()
                                )
                            }
                        },
                    )
                    .collect(),
                None => vec![],
            };
            match self.node_estimation.answers.get_mut(asking_node) {
                Some(node_answers) => node_answers.extend(answers),
                None => {
                    self.node_estimation.answers.insert(*asking_node, answers);
                }
            }
        }
    }

    fn start_anwering(&mut self, ctx: &mut Context<Training>) {
        self.node_estimation
            .answering_direct_protocol
            .start(self.cluster_nodes.len_incl_own());
        self.node_estimation
            .answering_direct_protocol
            .resolve_buffer(ctx.address().recipient());
        self.answer_next(ctx);
    }

    fn answer_next(&mut self, ctx: &mut Context<Training>) {
        for (id, node) in self
            .cluster_nodes
            .iter_any_as(self.own_addr.as_ref().unwrap().clone(), "Training")
            .enumerate()
        {
            let msg = match self.node_estimation.answers.remove(&id) {
                Some(answers) => {
                    let mut directed_answers = HashMap::new();
                    directed_answers.insert(id, answers);
                    ForeignNodesAnswer {
                        answers: directed_answers,
                    }
                }
                None => ForeignNodesAnswer {
                    answers: HashMap::default(),
                },
            };

            match &node {
                AnyAddr::Local(addr) => addr.do_send(msg),
                AnyAddr::Remote(addr) => addr
                    .wait_send(msg)
                    .into_actor(self)
                    .map(|res, _, _| if res.is_ok() {})
                    .wait(ctx),
            }
            self.node_estimation.answering_direct_protocol.sent();
        }
    }

    fn take_in_answers(&mut self, answers: Vec<(usize, usize, usize, IndependentNode)>) {
        for (prev_point_id, prev_segment_id, point_id, node) in answers {
            if (prev_point_id == 3310) & (prev_segment_id == 99) & (point_id == 3310) {
                println!("There you go! {}\n", self.cluster_nodes.get_own_idx())
            }
            self.node_estimation
                .next_foreign_node
                .insert((prev_point_id, prev_segment_id), (point_id, node));
        }
    }

    fn finalize_node_estimation(&mut self, ctx: &mut Context<Training>) {
        match &self.node_estimation.source {
            Some(source) => source.clone(),
            None => ctx.address().recipient(),
        }
        .do_send(NodeEstimationDone)
        .unwrap();
    }
}

impl Handler<ClusteringResponse<f32>> for Training {
    type Result = ();

    fn handle(&mut self, msg: ClusteringResponse<f32>, ctx: &mut Self::Context) -> Self::Result {
        if !msg.labels.is_empty() {
            let current_intersections = self.node_estimation.current_intersections.clone();
            self.node_estimation.current_intersections.clear();

            for (intersection, label) in current_intersections.into_iter().zip(msg.labels) {
                let node = Node::new(intersection.clone(), label);
                self.data_store.add_node(node);
            }
        }
        self.node_estimation.current_segment_id += 1;

        if self.node_estimation.current_segment_id < self.parameters.rate {
            self.estimate_nodes(ctx.address().recipient());
        } else {
            self.ask_for_foreign_nodes(ctx);
        }
    }
}

impl Handler<AskForForeignNodes> for Training {
    type Result = ();

    fn handle(&mut self, msg: AskForForeignNodes, ctx: &mut Self::Context) -> Self::Result {
        if !self.node_estimation.asking_direct_protocol.received(&msg) {
            return;
        }

        let mut asked_nodes = msg.asked_nodes;
        if let Some(questions) = asked_nodes.remove(&self.cluster_nodes.get_own_idx()) {
            self.search_for_asked_nodes(questions);
        }

        if !self.node_estimation.asking_direct_protocol.is_running() {
            self.start_anwering(ctx);
        }
    }
}

impl Handler<ForeignNodesAnswer> for Training {
    type Result = ();

    fn handle(&mut self, msg: ForeignNodesAnswer, ctx: &mut Self::Context) -> Self::Result {
        if !self
            .node_estimation
            .answering_direct_protocol
            .received(&msg)
        {
            return;
        }

        let mut answers = msg.answers;
        if let Some(own_answers) = answers.remove(&self.cluster_nodes.get_own_idx()) {
            self.take_in_answers(own_answers);
        }

        if !self.node_estimation.answering_direct_protocol.is_running() {
            self.finalize_node_estimation(ctx);
        }
    }
}

#[derive(Debug, Clone)]
pub enum Clustering {
    MeanShift,
    MultiKDE,
}

impl FromStr for Clustering {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq("meanshift") {
            Ok(Clustering::MeanShift)
        } else if s.eq("kde") {
            Ok(Clustering::MultiKDE)
        } else {
            Err(format!(
                "{} is not a valid clustering method! Allowed values are: 'meanshift' and 'kde'",
                s
            ))
        }
    }
}
