use std::hash::{Hash};
use std::sync::Arc;
use crate::data_store::node::NodeRef;
pub(crate) use materialized::MaterializedEdge;

mod materialized;


#[derive(Clone, Debug, Hash)]
pub(crate) struct Edge {
    from_node: NodeRef,
    to_node: NodeRef
}

impl Edge {
    pub fn new(from_node: NodeRef, to_node: NodeRef) -> Self {
        Self {
            from_node,
            to_node
        }
    }

    pub fn get_from_node(&self) -> NodeRef {
        self.from_node.clone()
    }

    pub fn get_to_node(&self) -> NodeRef {
        self.to_node.clone()
    }

    pub fn get_to_id(&self) -> usize {
        self.to_node.get_from_id()
    }

    pub fn to_ref(self) -> EdgeRef {
        EdgeRef::new(self)
    }
}

impl PartialEq<Self> for Edge {
    fn eq(&self, other: &Self) -> bool {
        self.from_node.eq(&other.from_node) & self.to_node.eq(&other.to_node)
    }
}

impl Eq for Edge {}


pub(crate) type EdgeRef = Arc<Edge>;
