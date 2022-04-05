mod materialized;

use num_integer::Integer;
use std::sync::Arc;

use crate::data_store::point::PointRef;
pub(crate) use materialized::MaterializedTransition;

#[derive(Clone, Debug)]
pub(crate) struct Transition {
    from_point: PointRef,
    to_point: PointRef,
}

impl Transition {
    pub fn new(from_point: PointRef, to_point: PointRef) -> Self {
        Self {
            from_point,
            to_point,
        }
    }
}

pub(crate) trait TransitionMixin {
    fn get_from_point(&self) -> PointRef;
    fn get_to_point(&self) -> PointRef;

    fn get_points(&self) -> (PointRef, PointRef) {
        (self.get_from_point(), self.get_to_point())
    }

    fn crosses_segments(&self) -> bool {
        self.get_from_point()
            .get_segment()
            .ne(&self.get_to_point().get_segment())
    }

    fn has_valid_direction(&self, rate: isize) -> bool {
        let from_segment = self.get_from_point().get_segment() as isize;
        let to_segment = self.get_to_point().get_segment() as isize;

        let raw_diff_counter = (rate + to_segment) - from_segment;
        let half_rate = num_integer::Integer::div_floor(&rate, &2);
        let raw_diff = self.raw_diff();

        (0 <= raw_diff && raw_diff <= half_rate)
            || (raw_diff < 0 && (0 <= raw_diff_counter && raw_diff_counter <= half_rate))
    }

    fn segment_diff(&self) -> usize {
        self.raw_diff().abs() as usize
    }

    fn raw_diff(&self) -> isize {
        self.get_to_segment() as isize - self.get_from_segment() as isize
    }

    fn get_from_id(&self) -> usize {
        self.get_from_point().get_id()
    }

    fn get_from_segment(&self) -> usize {
        self.get_from_point().get_segment()
    }

    fn get_to_segment(&self) -> usize {
        self.get_to_point().get_segment()
    }

    fn get_first_intersection_segment(&self, n_segments: &usize) -> usize {
        (self.get_from_segment() + 1).mod_floor(n_segments)
    }
}

impl TransitionMixin for Transition {
    fn get_from_point(&self) -> PointRef {
        self.from_point.clone()
    }

    fn get_to_point(&self) -> PointRef {
        self.to_point.clone()
    }
}

pub(crate) type TransitionRef = Arc<Transition>;
