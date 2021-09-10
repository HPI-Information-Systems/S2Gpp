#[cfg(test)]
mod tests;
mod data_structures;

use crate::training::Training;
use crate::utils::PolarCoords;
use std::collections::{HashMap, HashSet};
use ndarray::{Array1, Axis, arr1};
use num_integer::Integer;
use std::f32::consts::PI;
use actix_telepathy::RemoteAddr;
use num_traits::real::Real;
use crate::training::messages::{SegmentMessage, SegmentedMessage};
use actix::prelude::*;
use actix::dev::MessageResponse;
use serde::{Serialize, Serializer, Deserialize, Deserializer};
use std::hash::Hash;
use log::*;
pub use crate::training::segmenter::data_structures::SegmentedTransition;


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

pub type PointsForNodes = HashMap<usize, Vec<SegmentedPointWithId>>;
pub type TransitionsForNodes = HashMap<usize, Vec<SegmentedTransition>>;

#[derive(Default)]
pub struct Segmentation {
    /// list of transitions of segmented points
    pub segments: Vec<SegmentedTransition>,
    pub segment_index: HashMap<usize, usize>,
    pub n_received: usize
}

pub trait Segmenter {
    fn segment(&mut self);
    fn build_segment_index(&mut self);
}

impl Segmenter for Training {
    fn segment(&mut self) {
        let own_id = self.cluster_nodes.get_own_idx();
        let next_id = (own_id + 1) % (&self.cluster_nodes.len_incl_own());
        let mut node_transitions = TransitionsForNodes::new();
        let segments_per_node = (self.parameters.rate as f32 / self.cluster_nodes.len_incl_own() as f32).floor() as usize;
        let mut last_point: Option<SegmentedPointWithId> = None;
        for (id, x) in self.rotation.rotated.as_ref().unwrap().axis_iter(Axis(0)).enumerate() {
            let polar = x.to_polar();
            let segment_id = get_segment_id(polar[1], self.parameters.rate);
            let point = SegmentedPointWithId { segment_id, point_with_id: PointWithId { id, coords: x.iter().map(|x| x.clone()).collect() } };
            match last_point {
                Some(last_point) => {
                    let node_id = last_point.segment_id / segments_per_node;
                    let transition = SegmentedTransition::new(last_point, point.clone());
                    if node_id == own_id {
                        self.segmentation.segments.push(transition);
                    } else {
                        match node_transitions.get_mut(&node_id) {
                            Some(node_transitions) => node_transitions.push(transition),
                            None => { node_transitions.insert(node_id, vec![transition]); }
                        }
                    }
                },
                None => ()
            }

            last_point = Some(point);
        }

        self.cluster_nodes.get(&next_id).unwrap().do_send(SegmentMessage { segments: node_transitions });
    }

    fn build_segment_index(&mut self) {
        let index = self.segmentation.segments.iter().enumerate().map(|(id, transition)| {
            (transition.from.point_with_id.id, id)
        }).collect();
        self.segmentation.segment_index = index;
    }
}

fn get_segment_id(angle: f32, n_segments: usize) -> usize {
    let positive_angle = (2.0 * PI) + angle;
    let segment_size = (2.0 * PI) / (n_segments as f32);
    (positive_angle / segment_size).floor() as usize % n_segments
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
            self.cluster_nodes.get(&next_id).unwrap().do_send(SegmentMessage { segments });
        } else {
            self.build_segment_index();
            ctx.address().do_send(SegmentedMessage);
        }
    }
}
