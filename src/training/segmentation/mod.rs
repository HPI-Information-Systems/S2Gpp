#[cfg(test)]
mod tests;
mod data_structures;
pub(crate) mod messages;

use crate::training::Training;
use crate::utils::PolarCoords;
use std::collections::HashMap;
use ndarray::{Array1, Axis};
use std::f32::consts::PI;
use actix::prelude::*;
use serde::{Serialize, Deserialize};
pub use crate::training::segmentation::data_structures::SegmentedTransition;
pub use crate::training::segmentation::messages::{SegmentedMessage, SegmentMessage, SendFirstPointMessage};


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
    pub last_point: Option<SegmentedPointWithId>
}

pub trait Segmenter {
    fn segment(&mut self);
    fn build_segments(&mut self) -> HashMap<usize, Vec<SegmentedTransition>>;
    fn try_send_inter_node_points(&mut self) -> bool;
    fn distribute_segments(&mut self, node_transitions: HashMap<usize, Vec<SegmentedTransition>>);
    fn build_segment_index(&mut self);
}

impl Segmenter for Training {
    fn segment(&mut self) {
        let node_transitions = self.build_segments();
        let wait_for_points = self.try_send_inter_node_points();
        if wait_for_points {
            self.segmentation.node_transitions = node_transitions;
        } else { // if no other cluster node exists (i.e. local only)
            self.distribute_segments(node_transitions);
        }
    }

    fn build_segments(&mut self) -> HashMap<usize, Vec<SegmentedTransition>>{
        let own_id = self.cluster_nodes.get_own_idx();
        let is_not_first = own_id.ne(&0);
        let mut node_transitions = TransitionsForNodes::new();
        let segments_per_node = (self.parameters.rate as f32 / self.cluster_nodes.len_incl_own() as f32).floor() as usize;
        let mut last_point: Option<SegmentedPointWithId> = None;
        let points_per_node = self.dataset_stats.as_ref().expect("DatasetStats should've been set by now!").n.expect("DatasetStats.n should've been set by now!") / self.cluster_nodes.len_incl_own();
        let starting_id = own_id * points_per_node;
        for (id, x) in self.rotation.rotated.as_ref().unwrap().axis_iter(Axis(0)).enumerate() {
            let polar = x.to_polar();
            let segment_id = get_segment_id(polar[1], self.parameters.rate);
            let point = SegmentedPointWithId { segment_id, point_with_id: PointWithId { id: id + starting_id, coords: x.iter().map(|x| x.clone()).collect() } };
            match last_point {
                Some(last_point) => {
                    let node_id = last_point.segment_id / segments_per_node;
                    let transition = SegmentedTransition::new(last_point, point.clone());
                    if transition.crosses_segments() {
                        if node_id == own_id {
                            self.segmentation.segments.push(transition);
                        } else {
                            match node_transitions.get_mut(&node_id) {
                                Some(node_transitions) => node_transitions.push(transition),
                                None => { node_transitions.insert(node_id, vec![transition]); }
                            }
                        }
                    }
                },
                None => if is_not_first {
                    self.segmentation.send_point = Some(point.clone());
                }
            }

            last_point = Some(point);
        }
        self.segmentation.last_point = last_point;
        node_transitions
    }

    fn try_send_inter_node_points(&mut self) -> bool {
        let point = self.segmentation.send_point.clone();
        self.segmentation.send_point = None;
        match self.cluster_nodes.get_previous_idx() {
            Some(prev_idx) => {
                match point {
                    Some(point) => self.cluster_nodes.get_as(&prev_idx, "Training").unwrap().do_send(SendFirstPointMessage { point }),
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
            Some(next_id) => self.cluster_nodes.get_as(&next_id, "Training").unwrap().do_send(SegmentMessage { segments: node_transitions }),
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
        println!("segments {}", self.segmentation.segments.len());
    }
}

fn get_segment_id(angle: f32, n_segments: usize) -> usize {
    let positive_angle = (2.0 * PI) + angle;
    let segment_size = (2.0 * PI) / (n_segments as f32);
    (positive_angle / segment_size).floor() as usize % n_segments
}

impl Handler<SendFirstPointMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SendFirstPointMessage, _ctx: &mut Self::Context) -> Self::Result {
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
        self.distribute_segments(node_transitions);
    }
}

impl Handler<SegmentMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SegmentMessage, ctx: &mut Self::Context) -> Self::Result {
        self.segmentation.n_received += 1;
        let own_id = self.cluster_nodes.get_own_idx();
        let next_id = (own_id + 1) % (&self.cluster_nodes.len_incl_own());
        let mut segments = msg.segments;
        let mut own_points = segments.remove(&own_id).unwrap();

        self.segmentation.segments.append(&mut own_points);
        if self.segmentation.n_received < self.cluster_nodes.len() {
            self.cluster_nodes.get_as(&next_id, "Training").unwrap().do_send(SegmentMessage { segments });
        } else {
            self.build_segment_index();
            ctx.address().do_send(SegmentedMessage);
        }
    }
}
