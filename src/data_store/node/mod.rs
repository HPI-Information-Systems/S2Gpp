mod independent;

use std::fmt::{Display, Formatter};
use std::sync::Arc;
use crate::data_store::intersection::{IntersectionMixin, IntersectionRef};
pub(crate) use crate::data_store::node::independent::IndependentNode;
use crate::data_store::transition::TransitionMixin;


#[derive(Clone)]
pub(crate) struct Node {
    // todo: is this reference necessary?
    intersection: IntersectionRef,
    cluster: usize
}

impl Node {
    pub fn new(intersection: IntersectionRef, cluster: usize) -> Self {
        Self {
            intersection,
            cluster
        }
    }

    pub fn to_independent(&self) -> IndependentNode {
        IndependentNode::new(self.get_segment_id(), self.get_cluster(), self.get_intersection().get_transition().get_from_id())
    }

    pub fn get_segment_id(&self) -> usize {
        self.intersection.get_segment_id()
    }

    pub fn get_intersection(&self) -> IntersectionRef {
        self.intersection.clone()
    }

    pub fn get_cluster(&self) -> usize {
        self.cluster
    }
}

pub(crate) type NodeRef = Arc<IndependentNode>;


impl Display for Node {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}_{}", self.get_segment_id(), self.get_cluster())
    }
}
