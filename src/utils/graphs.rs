
use std::fmt::{Display, Formatter, Result};
use ndarray_linalg::Scalar;
use num_integer::Integer;
use serde::{Serialize, Deserialize};


//todo: add specific NodeName with point_id reference
#[derive(Clone, Default, Debug, Copy, Hash, Ord, PartialOrd, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeName(pub usize, pub usize);

/// using cantor pairing function
impl NodeName {
    #[allow(dead_code)]
    pub fn inv_cantor(x: u32) -> Self {
        let z = x as f32;
        let w = (((8.0*z+1.0).sqrt() - 1.0) / 2.0).floor();
        let t = (w.pow(2.0) + w) / 2.0;
        let b = z - t;
        let a = w - b;

        Self(a as usize, b as usize)
    }

    pub fn cantor_index(&self) -> u32 {
        let a = self.0 as u32;
        let b = self.1 as u32;

        return (((a + b) * (a + b + 1)) / 2) + b
    }

    pub fn local_segment_id(&self, segments_per_node: &usize) -> usize {
        self.0.mod_floor(segments_per_node)
    }

    pub fn is_locally_last(&self, segments_per_node: &usize) -> bool {
        self.local_segment_id(segments_per_node).eq(&(segments_per_node - 1))
    }

    pub fn is_locally_first(&self, segments_per_node: &usize) -> bool {
        self.local_segment_id(segments_per_node).eq(&0)
    }
}

impl Display for NodeName {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}_{}", self.0, self.1)
    }
}

#[derive(Clone, Serialize, Deserialize, Hash, PartialEq, Eq, Debug)]
pub struct Edge(pub NodeName, pub NodeName);

impl Edge {
    pub fn to_index_tuple(&self) -> (u32, u32) {
        (self.0.cantor_index(), self.1.cantor_index())
    }
}

impl Display for Edge {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "[{}, {}]", self.0, self.1)
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::NodeName;

    #[test]
    fn test_cantor_pairing() {
        let node = NodeName::inv_cantor(0);
        assert_eq!(node.0, 0);
        assert_eq!(node.1, 0);
        let node = NodeName(1, 2);
        let idx = node.cantor_index();
        assert_eq!(idx, 8);
    }
}
