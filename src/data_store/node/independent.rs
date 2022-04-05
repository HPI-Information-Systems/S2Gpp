use crate::data_store::node::NodeRef;
use serde::{Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub(crate) struct IndependentNode {
    segment: usize,
    cluster: usize,
    from_point_id: usize,
}

impl IndependentNode {
    pub fn new(segment: usize, cluster: usize, from_point_id: usize) -> Self {
        Self {
            segment,
            cluster,
            from_point_id,
        }
    }

    pub fn get_segment_id(&self) -> usize {
        self.segment
    }

    #[cfg(test)]
    pub fn get_cluster(&self) -> usize {
        self.cluster
    }

    pub fn get_from_id(&self) -> usize {
        self.from_point_id
    }

    pub fn into_ref(self) -> NodeRef {
        NodeRef::new(self)
    }
}

impl PartialEq<Self> for IndependentNode {
    fn eq(&self, other: &Self) -> bool {
        self.segment.eq(&other.segment) & self.cluster.eq(&other.cluster)
    }
}

impl Hash for IndependentNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.segment.hash(state);
        self.cluster.hash(state);
    }
}

impl Eq for IndependentNode {}
