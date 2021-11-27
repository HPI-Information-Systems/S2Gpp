use crate::utils::{Edge, NodeName};

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
pub struct EdgesOrderer {
    pub point_id: usize,
    pub previous_transition_node: Option<NodeName>,
    pub edges_small: Vec<(usize, Edge)>,
    pub edges_large: Vec<(usize, Edge)>,
    pub first_node: Option<NodeName>,
    pub middle_node: Option<NodeName>,
    pub last_node: Option<NodeName>
}

impl EdgesOrderer {
    pub fn new(point_id: usize, previous_node: Option<NodeName>) -> Self {
        Self {
            point_id,
            previous_transition_node: previous_node,
            ..Default::default()
        }
    }

    fn push(&mut self, value: (usize, Edge)) {
        match self.last_node {
            Some(_) => self.edges_large.push(value),
            None => self.edges_small.push(value)
        }
    }

    fn check_for_gap(&mut self, previous_node: &NodeName, current_node: &NodeName) -> bool {
        let this_segment = current_node.0;
        let last_segment = previous_node.0;
        if this_segment.ne(&(last_segment + 1)){
            self.last_node = Some(previous_node.clone());
            self.middle_node = Some(match self.edges_small.get(0).as_ref() {
                Some((_, first_edge)) => first_edge.0,
                None => *previous_node
            }.clone());
            if let Some(previous_transition_node) = self.previous_transition_node {
                self.edges_large.push((self.point_id, Edge(previous_transition_node, current_node.clone())));
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
    pub fn add_node(&mut self, previous_node: &Option<NodeName>, current_node: &NodeName) {
        // todo: in case of previous transition node is same as first node, second node becomes first node...
        if self.previous_transition_node.eq(previous_node) && self.edges_small.is_empty() && self.first_node.is_none() {
            self.first_node = Some(current_node.clone());
            return
        }
        if let Some(previous_node) = previous_node {
            if !self.check_for_gap(previous_node, current_node) {
                self.push((self.point_id, Edge(previous_node.clone(), current_node.clone())))
            }
        }
    }

    pub fn to_vec(&mut self) -> Vec<(usize, Edge)> {
        match self.middle_node {
            Some(middle_node) => {
                let previous_node = self.edges_large.last().as_ref().expect("Should be filled, we have a middle_node!").1.1;
                let connecting_edge = Edge(previous_node.clone(), middle_node.clone());
                self.push((self.point_id, connecting_edge));

                let edges_small = self.edges_small.clone();
                let edges_large = self.edges_large.clone();

                edges_large.into_iter().chain(edges_small.into_iter()).collect()
            },
            None => {
                if self.previous_transition_node.is_some(){
                    self.edges_small.insert(0, (self.point_id, Edge(self.previous_transition_node.as_ref().unwrap().clone(),
                                                   self.first_node.as_ref().unwrap().clone())));
                }
                self.edges_small.clone()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::training::edge_estimation::edges_orderer::EdgesOrderer;
    use crate::utils::{Edge, NodeName};

    #[test]
    fn orders_edges_without_gap() {
        let mut edges = EdgesOrderer::new(14, None);

        let nodes = vec![
            NodeName(0, 0),
            NodeName(1, 0),
            NodeName(2, 0)
        ];

        let expected_edges = vec![
            (14, Edge(NodeName(0, 0), NodeName(1, 0))),
            (14, Edge(NodeName(1, 0), NodeName(2, 0))),
        ];

        let mut previous_node = None;
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.to_vec())
    }

    #[test]
    fn orders_edges_without_gap_one_node() {

        let nodes = vec![
            NodeName(14, 0)
        ];

        let expected_edges = vec![
            (9897, Edge(NodeName(13, 2), NodeName(14, 0))),
        ];

        let mut previous_node = Some(NodeName(13, 2));
        let mut edges = EdgesOrderer::new(9897, previous_node);
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.to_vec())
    }

    #[test]
    fn orders_edges_without_gap_with_previous() {

        let nodes = vec![
            NodeName(14, 0),
            NodeName(15, 0),
            NodeName(16, 0),
        ];

        let expected_edges = vec![
            (9916, Edge(NodeName(13, 2), NodeName(14, 0))),
            (9916, Edge(NodeName(14, 0), NodeName(15, 0))),
            (9916, Edge(NodeName(15, 0), NodeName(16, 0))),
        ];

        let mut previous_node = Some(NodeName(13, 2));
        let mut edges = EdgesOrderer::new(9916, previous_node);
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.to_vec())
    }

    #[test]
    fn orders_edges_without_gap_same_segment_previous() {

        let nodes = vec![
            NodeName(14, 0),
            NodeName(15, 0),
            NodeName(16, 0),
        ];

        let expected_edges = vec![
            (9916, Edge(NodeName(14, 2), NodeName(14, 0))),
            (9916, Edge(NodeName(14, 0), NodeName(15, 0))),
            (9916, Edge(NodeName(15, 0), NodeName(16, 0))),
        ];

        let mut previous_node = Some(NodeName(14, 2));
        let mut edges = EdgesOrderer::new(9916, previous_node);
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.to_vec())
    }

    #[test]
    fn orders_edges_with_gap_without_predecessor() {
        let mut edges = EdgesOrderer::new(14, None);

        let nodes = vec![
            NodeName(0, 0),
            NodeName(1, 0),
            NodeName(2, 0),
            NodeName(98, 0),
            NodeName(99, 0)
        ];

        let expected_edges = vec![
            (14, Edge(NodeName(98, 0), NodeName(99, 0))),
            (14, Edge(NodeName(99, 0), NodeName(0, 0))),
            (14, Edge(NodeName(0, 0), NodeName(1, 0))),
            (14, Edge(NodeName(1, 0), NodeName(2, 0))),
        ];

        let mut previous_node = None;
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.to_vec())
    }

    #[test]
    fn orders_edges_with_gap_with_predecessor() {
        let mut edges = EdgesOrderer::new(14, Some(NodeName(97, 0)));

        let nodes = vec![
            NodeName(0, 0),
            NodeName(1, 0),
            NodeName(2, 0),
            NodeName(98, 0),
            NodeName(99, 0)
        ];

        let expected_edges = vec![
            (14, Edge(NodeName(97, 0), NodeName(98, 0))),
            (14, Edge(NodeName(98, 0), NodeName(99, 0))),
            (14, Edge(NodeName(99, 0), NodeName(0, 0))),
            (14, Edge(NodeName(0, 0), NodeName(1, 0))),
            (14, Edge(NodeName(1, 0), NodeName(2, 0))),
        ];

        let mut previous_node = Some(NodeName(97, 0));
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.to_vec())
    }

    #[test]
    fn orders_edges_with_small_gap_with_predecessor() {
        let mut edges = EdgesOrderer::new(14, Some(NodeName(97, 0)));

        let nodes = vec![
            NodeName(0, 0),
            NodeName(98, 0),
            NodeName(99, 0)
        ];

        let expected_edges = vec![
            (14, Edge(NodeName(97, 0), NodeName(98, 0))),
            (14, Edge(NodeName(98, 0), NodeName(99, 0))),
            (14, Edge(NodeName(99, 0), NodeName(0, 0))),
        ];

        let mut previous_node = Some(NodeName(97, 0));
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.to_vec())
    }

    #[test]
    fn orders_edges_with_gap_with_predecessor2() {
        let mut edges = EdgesOrderer::new(14, Some(NodeName(64, 0)));

        let mut nodes: Vec<NodeName> = (65..100).into_iter().map(|x| NodeName(x, 0)).collect();
        nodes.insert(0, NodeName(0, 0));

        let mut expected_edges: Vec<(usize, Edge)> = (64..99).into_iter().map(|x| (14, Edge(NodeName(x, 0), NodeName(x+1, 0)))).collect();

        expected_edges.push((14, Edge(NodeName(99, 0), NodeName(0, 0))));

        let mut previous_node = Some(NodeName(64, 0));
        for node in nodes.iter() {
            edges.add_node(&previous_node, node);
            previous_node = Some(node.clone());
        }

        assert_eq!(expected_edges, edges.to_vec())
    }
}
