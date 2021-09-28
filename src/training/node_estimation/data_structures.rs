use crate::training::intersection_calculation::{SegmentID};
use std::fmt::{Display, Formatter, Result};


pub struct IntersectionNode {
    pub segment: SegmentID,
    pub cluster_id: usize
}

impl Display for IntersectionNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}_{}", self.segment, self.cluster_id)
    }
}
