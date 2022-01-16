use ndarray::{Array1, ArrayView1};
use crate::data_store::materialize::Materialize;
use crate::data_store::transition::{MaterializedTransition};
use serde::{Serialize, Deserialize};
use crate::data_store::intersection::{Intersection, IntersectionMixin};


#[derive(Clone, Serialize, Deserialize, Debug)]
pub(crate) struct MaterializedIntersection {
    transition: MaterializedTransition,
    coordinates: Array1<f32>,
    segment_id: usize
}


impl IntersectionMixin<MaterializedTransition> for MaterializedIntersection {
    fn get_coordinates(&self) -> ArrayView1<f32> {
        self.coordinates.view()
    }

    fn get_segment_id(&self) -> usize {
        self.segment_id
    }

    fn get_transition(&self) -> MaterializedTransition {
        self.transition.clone()
    }
}


impl Materialize<MaterializedIntersection> for Intersection {
    fn materialize(&self) -> MaterializedIntersection {
        MaterializedIntersection {
            transition: self.transition.materialize(),
            coordinates: self.coordinates.clone(),
            segment_id: self.get_segment_id()
        }
    }
}
