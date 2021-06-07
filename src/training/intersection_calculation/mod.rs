use crate::training::Training;
use num_traits::real::Real;
use crate::utils::PolarCoords;
use std::ops::{Range};
use std::collections::HashMap;
use ndarray::Array1;
use crate::training::segmenter::SegmentedPointWithId;

pub struct IntersectionCalculation {
    pub intersections: HashMap<usize, Vec<Array1<f32>>>, // segment radial -> Intersections
}


pub trait IntersectionCalculator {
    fn calculate_intersections(&self);
}


impl IntersectionCalculator for Training {
    fn calculate_intersections(&self) {
        let max_radius = self.segmentation.own_segment.iter()
            .map(|x| x.point_with_id.coords.to_polar()[0])
            .max_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap();
        let dims = self.segmentation.own_segment.get(0).unwrap().point_with_id.coords.len();

        let mut pairs = vec![];
        let mut first: Option<&SegmentedPointWithId> = None;
        for segmented_point in self.segmentation.own_segment.iter() {
            match first {
                Some(f) => if f.segment_id.ne(&segmented_point.segment_id) &&
                    f.point_with_id.id.eq(&(segmented_point.point_with_id.id - 1)) {
                    pairs.push((f, segmented_point))
                },
                None => { first = Some(segmented_point) }
            }
        }
    }
}
