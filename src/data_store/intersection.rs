use crate::data_store::transition::{TransitionMixin, TransitionRef};
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Intersection {
    from_point_id: usize,
    coordinates: Array1<f32>,
    segment_id: usize,
}

impl Intersection {
    pub fn new(transition: TransitionRef, coordinates: Array1<f32>, segment_id: usize) -> Self {
        Self {
            from_point_id: transition.get_from_id(),
            coordinates,
            segment_id,
        }
    }

    pub fn get_coordinates(&self) -> ArrayView1<f32> {
        self.coordinates.view()
    }

    pub fn get_segment_id(&self) -> usize {
        self.segment_id
    }

    pub fn get_from_id(&self) -> usize {
        self.from_point_id
    }

    pub fn into_ref(self) -> IntersectionRef {
        IntersectionRef::new(self)
    }
}

pub(crate) type IntersectionRef = Arc<Intersection>;
