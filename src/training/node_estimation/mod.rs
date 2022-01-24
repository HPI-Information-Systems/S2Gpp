mod messages;

use std::collections::HashMap;
use std::ops::Deref;
use ndarray::{ArrayView1, stack_new_axis, Axis,};
use crate::training::Training;
use actix::{Addr, Handler, Actor, Recipient, AsyncContext, Context, WrapFuture, ActorFutureExt, ContextFutureSpawner};
use meanshift_rs::{MeanShiftActor, MeanShiftMessage, MeanShiftResponse};

pub(crate) use crate::training::node_estimation::messages::{NodeEstimationDone, AskForForeignNodes, ForeignNodesAnswer};
use num_integer::Integer;
use crate::data_store::intersection::IntersectionRef;
use crate::data_store::node::{IndependentNode, Node};
use crate::data_store::node_questions::node_in_question::NodeInQuestion;
use crate::data_store::node_questions::NodeQuestions;
use crate::utils::logging::progress_bar::S2GppProgressBar;
use crate::utils::rotation_protocol::RotationProtocol;


#[derive(Default)]
pub(crate) struct NodeEstimation {
    pub next_foreign_node: HashMap<(usize, usize), (usize, IndependentNode)>,
    pub meanshift: Option<Addr<MeanShiftActor>>,
    pub(crate) current_intersections: Vec<IntersectionRef>,
    pub(crate) current_segment_id: usize,
    pub(crate) progress_bar: S2GppProgressBar,
    pub(crate) source: Option<Recipient<NodeEstimationDone>>,
    asking_rotation_protocol: RotationProtocol<AskForForeignNodes>,
    answering_rotation_protocol: RotationProtocol<ForeignNodesAnswer>,
    answers: HashMap<usize, Vec<(usize, usize, usize, IndependentNode)>>
}

pub(crate) trait NodeEstimator {
    fn estimate_nodes(&mut self, mean_shift_recipient: Recipient<MeanShiftResponse>);
    fn ask_for_foreign_nodes(&mut self, ctx: &mut Context<Training>);
    fn ask_next(&mut self, asked_nodes: NodeQuestions);
    fn search_for_asked_nodes(&mut self, node_questions: HashMap<usize, Vec<NodeInQuestion>>);
    fn start_anwering(&mut self, ctx: &mut Context<Training>);
    fn answer_next(&mut self, answers: HashMap<usize, Vec<(usize, usize, usize, IndependentNode)>>, ctx: &mut Context<Training>);
    fn take_in_answers(&mut self, answers: Vec<(usize, usize, usize, IndependentNode)>);
    fn finalize_node_estimation(&mut self, ctx: &mut Context<Training>);
}

impl NodeEstimator for Training {
    fn estimate_nodes(&mut self, mean_shift_recipient: Recipient<MeanShiftResponse>) {
        self.node_estimation.progress_bar.inc_or_set("info", self.parameters.rate.div_floor(&self.parameters.n_cluster_nodes));

        let segment_id = self.node_estimation.current_segment_id;

        match self.data_store.get_intersections_from_segment(segment_id) {
            Some(intersections) => {

                self.node_estimation.current_intersections = intersections.iter().map(|x| x.clone()).collect();
                let coordinates: Vec<ArrayView1<f32>> = intersections.iter().map(|x| x.get_coordinates()).collect();
                let data = stack_new_axis(Axis(0), coordinates.as_slice()).unwrap();
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
            self.node_estimation.asking_rotation_protocol.start(self.parameters.n_cluster_nodes);
            self.node_estimation.asking_rotation_protocol.resolve_buffer(ctx.address().recipient());

            self.ask_next(self.segmentation.node_questions.clone());
            self.segmentation.node_questions.clear();
        } else {
            self.finalize_node_estimation(ctx);
        }
    }

    fn ask_next(&mut self, asked_nodes: NodeQuestions) {
        self.cluster_nodes.get_next_as("Training").unwrap().do_send(AskForForeignNodes { asked_nodes });
        self.node_estimation.asking_rotation_protocol.sent();
    }

    fn search_for_asked_nodes(&mut self, mut node_questions: HashMap<usize, Vec<NodeInQuestion>>) {
        for (asking_node, _remote_addr) in self.cluster_nodes.iter() {
            let answers = match node_questions.remove(asking_node) {
                Some(questions) => questions.into_iter().map(|niq|
                    match self.data_store.get_nodes_by_point_id(niq.get_point_id()) {
                        Some(nodes) => nodes.iter().find_map(|node| node.get_segment_id().eq(&niq.get_segment()).then(|| (niq.get_prev_id(), niq.get_prev_segment(), niq.get_point_id(), node.deref().clone()))).expect(&format!("There is no answer here: no segment_id: {} {}", &niq.get_point_id(), &niq.get_segment())),
                        None => {
                            panic!("There is no answer here!: no point_id: {}", niq.get_point_id())
                        }
                    }
                ).collect(),
                None => vec![]
            };
            match self.node_estimation.answers.get_mut(&asking_node) {
                Some(node_answers) => node_answers.extend(answers),
                None => { self.node_estimation.answers.insert(asking_node.clone(), answers); }
            }
        }
    }

    fn start_anwering(&mut self, ctx: &mut Context<Training>) {
        self.node_estimation.answering_rotation_protocol.start(self.parameters.n_cluster_nodes);
        self.node_estimation.answering_rotation_protocol.resolve_buffer(ctx.address().recipient());
        self.answer_next(self.node_estimation.answers.clone(), ctx);
        self.node_estimation.answers.clear();
    }

    fn answer_next(&mut self, answers: HashMap<usize, Vec<(usize, usize, usize, IndependentNode)>>, ctx: &mut Context<Training>) {
        self.cluster_nodes.get_next_as("Training").unwrap()
            .wait_send(ForeignNodesAnswer { answers })
            .into_actor(self)
            .map(|res, _act, _ctx| match res {
                Ok(_) => (),
                Err(_) => ()
            })
            .wait(ctx);
        self.node_estimation.answering_rotation_protocol.sent();
    }

    fn take_in_answers(&mut self, answers: Vec<(usize, usize, usize, IndependentNode)>) {
        for (prev_point_id, prev_segment_id, point_id, node) in answers {
            if (prev_point_id == 3310) & (prev_segment_id == 99) & (point_id == 3310) {
                println!("There you go! {}\n", self.cluster_nodes.get_own_idx())
            }
            self.node_estimation.next_foreign_node.insert((prev_point_id, prev_segment_id), (point_id, node));
        }
    }

    fn finalize_node_estimation(&mut self, ctx: &mut Context<Training>) {
        match &self.node_estimation.source {
            Some(source) => source.clone(),
            None => ctx.address().recipient()
        }.do_send(NodeEstimationDone).unwrap();
    }
}

impl Handler<MeanShiftResponse> for Training {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftResponse, ctx: &mut Self::Context) -> Self::Result {
        if !msg.cluster_centers.is_empty() {
            let current_intersections = self.node_estimation.current_intersections.clone();
            self.node_estimation.current_intersections.clear();

            for (intersection, label) in current_intersections.into_iter().zip(msg.labels)  {
                let node =  Node::new(intersection.clone(), label);
                self.data_store.add_node(node);
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

        let mut asked_nodes = msg.asked_nodes;
        self.search_for_asked_nodes(asked_nodes.remove(&self.cluster_nodes.get_own_idx()).unwrap());

        if self.node_estimation.asking_rotation_protocol.is_running() {
            self.ask_next(asked_nodes);
        } else {
            self.start_anwering(ctx);
        }
    }
}


impl Handler<ForeignNodesAnswer> for Training {
    type Result = ();

    fn handle(&mut self, msg: ForeignNodesAnswer, ctx: &mut Self::Context) -> Self::Result {
        if !self.node_estimation.answering_rotation_protocol.received(&msg) {
            return
        }

        let mut answers = msg.answers;

        match answers.remove(&self.cluster_nodes.get_own_idx()) {
            None => (),
            Some(own_answers) => self.take_in_answers(own_answers)
        }

        if self.node_estimation.answering_rotation_protocol.is_running() {
            self.answer_next(answers, ctx);
        } else {
            self.finalize_node_estimation(ctx);
        }
    }
}
