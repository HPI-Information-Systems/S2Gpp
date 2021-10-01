use crate::utils::{Edge, NodeName};

/// A transition of one point to the next can span multiple segments and, therefore, creates
/// multiple nodes and edges. The segments for one transition are sorted. This bears a problem,
/// because a transition can span beyond the last segment to the first again. There, we end up with
/// a wrongly sorted list of nodes and three wrong edges; the first, in the middle, and the last.
///
/// The *EdgesForPoint* struct corrects the ordering and creation of edges. For each added node, the
/// class checks whether the last node of the same transition is in the previous segment. If it's
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
    pub previous_transition: bool,
    pub edges_small: Vec<(usize, Edge)>,
    pub edges_large: Vec<(usize, Edge)>,
    pub middle_node: Option<NodeName>,
    pub last_node: Option<NodeName>
}

impl EdgesOrderer {
    pub fn new(point_id: usize, previous_transition: bool) -> Self {
        Self {
            point_id,
            previous_transition,
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
        if this_segment.ne(&(last_segment + 1)) && !self.edges_small.is_empty() {
            self.last_node = Some(previous_node.clone());
            if self.previous_transition {
                let removed_edge = self.edges_small.remove(0).1;
                let incoming_node = removed_edge.0;
                self.middle_node = Some(removed_edge.1);

                let starting_edge = Edge(incoming_node, current_node.clone());
                self.push((self.point_id, starting_edge));
            } else {
                self.middle_node = Some(self.edges_small.get(0).as_ref().unwrap().1.0.clone());
            }
            true
        } else {
            false
        }
    }

    pub fn add_node(&mut self, previous_node: &Option<NodeName>, current_node: &NodeName) {
        match previous_node {
            None => (),
            Some(previous) => {
                if !self.check_for_gap(previous, current_node) {
                    self.push((self.point_id, Edge(previous.clone(), current_node.clone())))
                }
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
            None => self.edges_small.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::training::edge_estimation::edges_orderer::EdgesOrderer;
    use crate::utils::{Edge, NodeName};

    #[test]
    fn orders_edges_without_gap() {
        let mut edges = EdgesOrderer::new(14, false);

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
    fn orders_edges_with_gap_without_predecessor() {
        let mut edges = EdgesOrderer::new(14, false);

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
        let mut edges = EdgesOrderer::new(14, true);

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
}
