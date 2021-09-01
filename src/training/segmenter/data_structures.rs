use crate::training::segmenter::{PointWithId, SegmentedPointWithId};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegmentedTransition {
    pub from: SegmentedPointWithId,
    pub to: SegmentedPointWithId
}

impl SegmentedTransition {
    pub fn new(from: SegmentedPointWithId, to: SegmentedPointWithId) -> Self {
        Self {
            from,
            to
        }
    }

    pub fn crosses_segments(&self) -> bool {
        self.from.segment_id.ne(&self.to.segment_id)
    }
}
