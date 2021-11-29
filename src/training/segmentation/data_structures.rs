use num_integer::Integer;
use crate::training::segmentation::SegmentedPointWithId;
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

    pub fn has_valid_direction(&self, rate: usize) -> bool {
        let raw_diff_counter = (rate + self.to.segment_id) as isize - self.from.segment_id as isize;
        let half_rate = rate.div_floor(&2);
        let raw_diff = self.raw_diff();

        (0 <= raw_diff && raw_diff <= half_rate as isize) ||
            (raw_diff < 0 && (0 <= raw_diff_counter && raw_diff_counter <= half_rate as isize))
    }

    pub fn segment_diff(&self) -> usize {
        self.raw_diff().abs() as usize
    }

    fn raw_diff(&self) -> isize {
        self.to.segment_id as isize - self.from.segment_id as isize
    }

    pub fn get_from_id(&self) -> usize {
        self.from.point_with_id.id
    }

    pub fn get_from_segment(&self) -> usize {
        self.from.segment_id
    }

    pub fn get_to_segment(&self) -> usize {
        self.to.segment_id
    }

    pub fn get_first_intersection_segment(&self, n_segments: &usize) -> usize {
        (self.get_from_segment() + 1).mod_floor(n_segments)
    }
}
