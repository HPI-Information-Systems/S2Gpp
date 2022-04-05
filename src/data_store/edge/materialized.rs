use crate::data_store::edge::Edge;
use crate::data_store::materialize::Materialize;
use crate::data_store::node::IndependentNode;
use serde::{Deserialize, Serialize};
use std::ops::Deref;

#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq, Hash)]
pub(crate) struct MaterializedEdge {
    from_node: IndependentNode,
    to_node: IndependentNode,
}

impl MaterializedEdge {
    pub fn get_from_node(&self) -> IndependentNode {
        self.from_node.clone()
    }

    pub fn get_to_node(&self) -> IndependentNode {
        self.to_node.clone()
    }

    pub fn get_to_id(&self) -> usize {
        self.to_node.get_from_id()
    }
}

impl Materialize<MaterializedEdge> for Edge {
    fn materialize(&self) -> MaterializedEdge {
        MaterializedEdge {
            from_node: self.from_node.deref().clone(),
            to_node: self.to_node.deref().clone(),
        }
    }
}
