use crate::training::Training;
use crate::utils::PolarCoords;
use std::collections::{HashMap, HashSet};
use ndarray::{Array1, Axis};
use num_integer::Integer;
use std::f32::consts::PI;
use actix_telepathy::RemoteAddr;
use num_traits::real::Real;
use crate::training::messages::{SegmentMessage, SegmentedMessage};
use actix::{Handler, AsyncContext};
use actix::dev::MessageResponse;

pub struct Segmentation {
    pub rate: usize,
    /// list of lists with (data ID, data point) as elements and inner list at position segment ID
    pub segments: Vec<Vec<(usize, Array1<f32>)>>,
    /// list with (data ID, data point) as elements
    pub own_segment: Vec<(usize, Array1<f32>)>,
    pub n_received: usize
}

pub trait Segmenter {
    fn segment(&mut self);
    fn assign_segments(&mut self);
}

impl Segmenter for Training {
    fn segment(&mut self) {
        for _ in ..self.segmentation.rate {
            self.segmentation.segments.push(vec![]);
        }

        let segment_size = (2.0 * PI) / self.segmentation.rate as f32;

        let partition_size = (self.dataset_stats.as_ref().unwrap().n.as_ref().unwrap() as f32 / self.nodes.len_incl_own() as f32).floor() as usize;
        let mut id = self.nodes.get_own_idx() * partition_size;
        for x in self.rotated.as_ref().unwrap().axis_iter(Axis(1)) {
            let polar = x.to_polar();
            let segment_id = (polar[1] / segment_size).floor() as usize;
            self.segmentation.segments.get_mut(segment_id).unwrap().push((id, x.clone));
            id += 1;
        }
    }

    fn assign_segments(&mut self) {
        let own_id = self.nodes.get_own_idx();
        let next_id = (own_id + 1).mod_floor(&self.nodes.len_incl_own());
        let mut node_segments: HashMap<usize, Vec<(usize, Array1<f32>)>> = HashMap::new();

        for (i, _) in self.nodes.iter() {
            node_segments.insert(i.clone(), vec![]);
        }

        let segments_per_node = (self.segmentation.rate as f32 / self.nodes.len_incl_own() as f32).floor() as usize;
        let _rest = self.segmentation.rate.mod_floor(&self.nodes.len_incl_own());

        for i in ..self.segmentation.rate {
            let node_id = (i / segments_per_node).floor() as usize;
            let points = self.segmentation.segments.get_mut(i).unwrap();
            match node_segments.get_mut(&node_id) {
                Some(this_node_segments) => this_node_segments.append(points),
                None => self.segmentation.own_segment.append(points)
            };

            if i.mod_floor(segments_per_node) == 0 {
                let prev_node = node_id.mod_floor(&self.nodes.len_incl_own());
                match node_segments.get_mut(&prev_node) {
                    Some(prev_node_segments) => prev_node_segments.append(points),
                    None => self.segmentation.own_segment.append(points)
                }
            }
        }

        self.nodes.get(&next_id).unwrap().do_send(SegmentMessage { segments: node_segments });
    }
}

impl Handler<SegmentMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: SegmentMessage, ctx: &mut Self::Context) -> Self::Result {
        self.segmentation.n_received += 1;
        if self.segmentation.n_received < self.nodes.len() {
            let own_id = self.nodes.get_own_idx();
            let next_id = (own_id + 1).mod_floor(&self.nodes.len_incl_own());
            let mut segments = msg.segments;
            let mut own_points = segments.remove(&own_id).unwrap();

            self.segmentation.own_segment.append(&mut own_points);
            self.nodes.get(&next_id).unwrap().do_send(SegmentMessage { segments });
        } else {
            ctx.address().do_send(SegmentedMessage);
            self.segmentation.segments.clear();
        }
    }
}

#[cfg(test)]
mod tests {

}
