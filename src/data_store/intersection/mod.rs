mod materialized;

use std::sync::Arc;
use ndarray::{Array1, ArrayView1};
use crate::data_store::transition::{TransitionRef};
pub(crate) use materialized::MaterializedIntersection;


#[derive(Clone, Debug)]
pub(crate) struct Intersection {
    transition: TransitionRef,
    coordinates: Array1<f32>,
    segment_id: usize
}

impl Intersection {
    pub fn new(transition: TransitionRef, coordinates: Array1<f32>, segment_id: usize) -> Self {
        Self {
            transition,
            coordinates,
            segment_id
        }
    }
}

pub(crate) trait IntersectionMixin<T> {
    fn get_coordinates(&self) -> ArrayView1<f32>;

    fn get_segment_id(&self) -> usize;

    fn get_transition(&self) -> T;
}

impl IntersectionMixin<TransitionRef> for Intersection {
    fn get_coordinates(&self) -> ArrayView1<f32> {
        self.coordinates.view()
    }

    fn get_segment_id(&self) -> usize {
        self.segment_id
    }

    fn get_transition(&self) -> TransitionRef {
        self.transition.clone()
    }
}


pub(crate) type IntersectionRef = Arc<Intersection>;
