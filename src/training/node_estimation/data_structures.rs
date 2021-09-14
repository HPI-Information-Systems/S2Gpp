
use crate::training::intersection_calculation::{SegmentID};

pub struct IntersectionNode {
    pub segment: SegmentID,
    pub cluster_id: usize
}
