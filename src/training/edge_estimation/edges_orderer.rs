use crate::data_store::edge::Edge;
use crate::data_store::node::NodeRef;

/// A transition of one point to the next can span multiple segments and, therefore, creates
/// multiple nodes and edges. The segments for one transition are sorted. This bears a problem,
/// because a transition can span beyond the last segment to the first again. There, we end up with
/// a wrongly sorted list of nodes and three wrong edges; the first, in the middle, and the last.
///
/// The *EdgesOrderer* struct corrects the ordering and creation of edges. For each added node, the
/// struct checks whether the last node of the same transition is in the previous segment. If it's
/// true, it will add an edge. If it's false, it will remove the first created edge for the
/// transition and remembers this first node. From now on, every new edge for that transition will
/// be stored in a second bucket. When the last edge is created the second and the first edges
/// bucket will be merged.
///
/// 0   1   2   3   98  99
/// ^-middle    ^-last
#[derive(Default)]
pub(crate) struct EdgesOrderer {
    pub previous_transition_node: Option<NodeRef>,
    pub edges_small: Vec<Edge>,
    pub edges_large: Vec<Edge>,
    pub first_node: Option<NodeRef>,
    pub middle_node: Option<NodeRef>,
    pub last_node: Option<NodeRef>,
}

impl EdgesOrderer {
    pub fn new(previous_node: Option<NodeRef>) -> Self {
        Self {
            previous_transition_node: previous_node,
            ..Default::default()
        }
    }

    fn push(&mut self, value: Edge) {
        match self.last_node {
            Some(_) => self.edges_large.push(value),
            None => self.edges_small.push(value),
        }
    }

    fn check_for_gap(&mut self, previous_node: &NodeRef, current_node: &NodeRef) -> bool {
        let this_segment = current_node.get_segment_id();
        let last_segment = previous_node.get_segment_id();
        if this_segment.ne(&(last_segment + 1)) {
            self.last_node = Some(previous_node.clone());
            self.middle_node = Some(match self.edges_small.get(0) {
                Some(first_edge) => first_edge.get_from_node(),
                None => previous_node.clone(),
            });
            if let Some(previous_transition_node) = &self.previous_transition_node {
                self.edges_large.push(Edge::new(
                    previous_transition_node.clone(),
                    current_node.clone(),
                ));
            }
            true
        } else {
            false
        }
    }

    /// We only add a node if both previous_node and current_node are from the same transition,
    /// otherwise we will skip this step. The previous node from the previous transition will be
    /// involved in the edge vec in the end, after the sorting is done!
    /// _Example_: \[96 (previous transition node), 0 (last node), 97, 98, 99\]
    /// First this struct will sort \[97, 98, 99, 0\] and then prepend _96_.
    pub fn add_node(&mut self, previous_node: &Option<NodeRef>, current_node: &NodeRef) {
        // in case of previous transition node is same as first node, second node becomes first node...
        if self.previous_transition_node.eq(previous_node)
            && self.edges_small.is_empty()
            && self.first_node.is_none()
        {
            self.first_node = Some(current_node.clone());
            return;
        }
        if let Some(previous_node) = previous_node {
            if !self.check_for_gap(previous_node, current_node) {
                self.push(Edge::new(previous_node.clone(), current_node.clone()));
            }
        }
    }

    pub fn into_vec(mut self) -> Vec<Edge> {
        match &self.middle_node {
            Some(middle_node) => {
                let previous_node = self
                    .edges_large
                    .last()
                    .as_ref()
                    .expect("Should be filled, we have a middle_node!")
                    .get_to_node();
                let connecting_edge = Edge::new(previous_node, middle_node.clone());
                self.push(connecting_edge);

                let edges_small = self.edges_small.clone();
                let edges_large = self.edges_large.clone();

                edges_large
                    .into_iter()
                    .chain(edges_small.into_iter())
                    .collect()
            }
            None => {
                if self.previous_transition_node.is_some() {
                    self.edges_small.insert(
                        0,
                        Edge::new(
                            self.previous_transition_node.as_ref().unwrap().clone(),
                            self.first_node.as_ref().unwrap().clone(),
                        ),
                    );
                }
                self.edges_small.clone()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::data_store::edge::Edge;
    use crate::data_store::node::{IndependentNode, NodeRef};
    use crate::training::edge_estimation::edges_orderer::EdgesOrderer;

    #[test]
    fn orders_edges_without_gap() {
        let mut edges = EdgesOrderer::new(None);

        let nodes = vec![
            IndependentNode::new(0, 0, 0).into_ref(),
            IndependentNode::new(1, 0, 0).into_ref(),
            IndependentNode::new(2, 0, 0).into_ref(),
        ];

        let expected_edges = vec![
            Edge::new(
                IndependentNode::new(0, 0, 14).into_ref(),
                IndependentNode::new(1, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(1, 0, 14).into_ref(),
                IndependentNode::new(2, 0, 0).into_ref(),
            ),
        ];

        let mut previous_node = None;
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.into_vec())
    }

    #[test]
    fn orders_edges_without_gap_one_node() {
        let nodes = vec![IndependentNode::new(14, 0, 0).into_ref()];

        let expected_edges = vec![Edge::new(
            IndependentNode::new(13, 2, 9897).into_ref(),
            IndependentNode::new(14, 0, 0).into_ref(),
        )];

        let mut previous_node = Some(IndependentNode::new(13, 2, 0).into_ref());
        let mut edges = EdgesOrderer::new(previous_node.clone());
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.into_vec())
    }

    #[test]
    fn orders_edges_without_gap_with_previous() {
        let nodes = vec![
            IndependentNode::new(14, 0, 0).into_ref(),
            IndependentNode::new(15, 0, 0).into_ref(),
            IndependentNode::new(16, 0, 0).into_ref(),
        ];

        let expected_edges = vec![
            Edge::new(
                IndependentNode::new(13, 2, 9916).into_ref(),
                IndependentNode::new(14, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(14, 0, 9916).into_ref(),
                IndependentNode::new(15, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(15, 0, 9916).into_ref(),
                IndependentNode::new(16, 0, 0).into_ref(),
            ),
        ];

        let mut previous_node = Some(IndependentNode::new(13, 2, 0).into_ref());
        let mut edges = EdgesOrderer::new(previous_node.clone());
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.into_vec())
    }

    #[test]
    fn orders_edges_without_gap_same_segment_previous() {
        let nodes = vec![
            IndependentNode::new(14, 0, 0).into_ref(),
            IndependentNode::new(15, 0, 0).into_ref(),
            IndependentNode::new(16, 0, 0).into_ref(),
        ];

        let expected_edges = vec![
            Edge::new(
                IndependentNode::new(14, 2, 9916).into_ref(),
                IndependentNode::new(14, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(14, 0, 9916).into_ref(),
                IndependentNode::new(15, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(15, 0, 9916).into_ref(),
                IndependentNode::new(16, 0, 0).into_ref(),
            ),
        ];

        let mut previous_node = Some(IndependentNode::new(14, 2, 0).into_ref());
        let mut edges = EdgesOrderer::new(previous_node.clone());
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.into_vec())
    }

    #[test]
    fn orders_edges_with_gap_without_predecessor() {
        let mut edges = EdgesOrderer::new(None);

        let nodes = vec![
            IndependentNode::new(0, 0, 0).into_ref(),
            IndependentNode::new(1, 0, 0).into_ref(),
            IndependentNode::new(2, 0, 0).into_ref(),
            IndependentNode::new(98, 0, 0).into_ref(),
            IndependentNode::new(99, 0, 0).into_ref(),
        ];

        let expected_edges = vec![
            Edge::new(
                IndependentNode::new(98, 0, 14).into_ref(),
                IndependentNode::new(99, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(99, 0, 14).into_ref(),
                IndependentNode::new(0, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(0, 0, 14).into_ref(),
                IndependentNode::new(1, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(1, 0, 14).into_ref(),
                IndependentNode::new(2, 0, 0).into_ref(),
            ),
        ];

        let mut previous_node = None;
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.into_vec())
    }

    #[test]
    fn orders_edges_with_gap_with_predecessor() {
        let mut edges = EdgesOrderer::new(Some(IndependentNode::new(97, 0, 0).into_ref()));

        let nodes = vec![
            IndependentNode::new(0, 0, 0).into_ref(),
            IndependentNode::new(1, 0, 0).into_ref(),
            IndependentNode::new(2, 0, 0).into_ref(),
            IndependentNode::new(98, 0, 0).into_ref(),
            IndependentNode::new(99, 0, 0).into_ref(),
        ];

        let expected_edges = vec![
            Edge::new(
                IndependentNode::new(97, 0, 14).into_ref(),
                IndependentNode::new(98, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(98, 0, 14).into_ref(),
                IndependentNode::new(99, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(99, 0, 14).into_ref(),
                IndependentNode::new(0, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(0, 0, 14).into_ref(),
                IndependentNode::new(1, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(1, 0, 14).into_ref(),
                IndependentNode::new(2, 0, 0).into_ref(),
            ),
        ];

        let mut previous_node = Some(IndependentNode::new(97, 0, 0).into_ref());
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.into_vec())
    }

    #[test]
    fn orders_edges_with_small_gap_with_predecessor() {
        let mut edges = EdgesOrderer::new(Some(IndependentNode::new(97, 0, 0).into_ref()));

        let nodes = vec![
            IndependentNode::new(0, 0, 0).into_ref(),
            IndependentNode::new(98, 0, 0).into_ref(),
            IndependentNode::new(99, 0, 0).into_ref(),
        ];

        let expected_edges = vec![
            Edge::new(
                IndependentNode::new(97, 0, 14).into_ref(),
                IndependentNode::new(98, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(98, 0, 14).into_ref(),
                IndependentNode::new(99, 0, 0).into_ref(),
            ),
            Edge::new(
                IndependentNode::new(99, 0, 14).into_ref(),
                IndependentNode::new(0, 0, 0).into_ref(),
            ),
        ];

        let mut previous_node = Some(IndependentNode::new(97, 0, 0).into_ref());
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.into_vec())
    }

    #[test]
    fn orders_edges_with_gap_with_predecessor2() {
        let mut edges = EdgesOrderer::new(Some(IndependentNode::new(64, 0, 0).into_ref()));

        let mut nodes: Vec<NodeRef> = (65..100)
            .into_iter()
            .map(|x| IndependentNode::new(x, 0, 0).into_ref())
            .collect();
        nodes.insert(0, IndependentNode::new(0, 0, 0).into_ref());

        let mut expected_edges: Vec<Edge> = (64..99)
            .into_iter()
            .map(|x| {
                Edge::new(
                    IndependentNode::new(x, 0, 14).into_ref(),
                    IndependentNode::new(x + 1, 0, 14).into_ref(),
                )
            })
            .collect();

        expected_edges.push(Edge::new(
            IndependentNode::new(99, 0, 14).into_ref(),
            IndependentNode::new(0, 0, 0).into_ref(),
        ));

        let mut previous_node = Some(IndependentNode::new(64, 0, 0).into_ref());
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.into_vec())
    }
}
