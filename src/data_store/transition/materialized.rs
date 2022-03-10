use std::ops::Deref;
use crate::data_store::materialize::Materialize;
use crate::data_store::point::{Point, PointRef};
use crate::data_store::transition::{Transition, TransitionMixin};
use serde::{Serialize, Deserialize};


#[derive(Clone, Serialize, Deserialize, Debug)]
pub(crate) struct MaterializedTransition {
    from_point: Point,
    to_point: Point
}

impl MaterializedTransition {
    pub fn into_transition(self) -> Transition {
        Transition::new(self.from_point.into_ref(), self.to_point.into_ref())
    }
}

impl TransitionMixin for MaterializedTransition {
    fn get_from_point(&self) -> PointRef {
        PointRef::new(self.from_point.clone()) // watch out, no real reference
    }

    fn get_to_point(&self) -> PointRef {
        PointRef::new(self.to_point.clone())
    }
}


impl Materialize<MaterializedTransition> for Transition {
    fn materialize(&self) -> MaterializedTransition {
        MaterializedTransition {
            from_point: self.from_point.deref().clone(),
            to_point: self.to_point.deref().clone()
        }
    }
}
