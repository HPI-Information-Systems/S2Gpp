use ndarray::Array1;
use crate::training::intersection_calculation::{Transition, SegmentID};

pub struct IntersectionNode {
    pub segment: SegmentID,
    pub cluster_id: usize
}
