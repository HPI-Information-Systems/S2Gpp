use std::ops::{Deref, Range};
use std::slice::Iter;
use ndarray::{Array1};
use crate::data_store::edge::{Edge, EdgeRef, MaterializedEdge};
use crate::data_store::index::DataStoreIndex;
use crate::data_store::intersection::{Intersection, IntersectionRef};
use crate::data_store::materialize::Materialize;
use crate::data_store::node::{IndependentNode, Node, NodeRef};
use crate::data_store::point::{Point, PointRef};
use crate::data_store::transition::{MaterializedTransition, Transition, TransitionMixin, TransitionRef};

pub(crate) mod point;
pub(crate) mod intersection;
pub(crate) mod transition;
pub(crate) mod node;
pub(crate) mod edge;
pub(crate) mod materialize;
mod tests;
mod utils;
mod index;


#[derive(Default, Clone)]
pub(crate) struct DataStore {
    points: Vec<PointRef>,
    transitions: Vec<TransitionRef>,
    intersections: Vec<IntersectionRef>,
    nodes: Vec<NodeRef>,
    edges: Vec<EdgeRef>,
    index: DataStoreIndex
}

impl DataStore {
    // --- Points

    pub fn add_point(&mut self, point: Point) {
        let point = PointRef::new(point);
        self.points.push(point.clone());
        self.index.add_point(point);
    }

    pub fn add_point_return(&mut self, point: Point) -> PointRef {
        let point = PointRef::new(point);
        self.points.push(point.clone());
        self.index.add_point_return(point)
    }

    #[cfg(test)]
    pub fn add_points(&mut self, points: Vec<Array1<f32>>, n_segments: usize) {
        self.add_points_with_offset(points, 0, n_segments)
    }

    pub fn add_points_with_offset(&mut self, points: Vec<Array1<f32>>, offset: usize, n_segments: usize) {
        let mut next_id = offset;

        for point in points.into_iter() {
            self.add_point(Point::new_calculate_segment(next_id,point, n_segments));
            next_id += 1;
        }
    }

    pub fn get_points(&self) -> Vec<PointRef> {
        self.points.clone()
    }

    // --- Transitions

    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions.push(TransitionRef::new(transition))
    }

    #[cfg(test)]
    pub fn add_transitions(&mut self, transitions: Vec<Transition>) {
        for transition in transitions.into_iter() {
            self.add_transition(transition)
        }
    }

    pub fn add_materialized_transition(&mut self, transition: MaterializedTransition) {
        let (from_point, to_point) = transition.get_points();
        let from_point = self.add_point_return(from_point.deref().clone());
        let to_point = self.add_point_return(to_point.deref().clone());
        self.add_transition(Transition::new(from_point, to_point));
    }

    pub fn add_materialized_transitions(&mut self, transitions: Vec<MaterializedTransition>) {
        for transition in transitions {
            self.add_materialized_transition(transition);
        }
    }

    pub fn get_transitions(&self) -> Vec<TransitionRef> {
        self.transitions.iter().map(|transition| transition.clone()).collect()
    }

    // --- Intersections

    pub fn add_intersection(&mut self, intersection: Intersection) {
        let intersection_ref = intersection.to_ref();
        self.intersections.push(intersection_ref.clone());
        self.index.add_intersection(intersection_ref);
    }

    pub fn add_intersections(&mut self, intersections: Vec<Intersection>) {
        for intersection in intersections {
            self.add_intersection(intersection);
        }
    }

    pub fn get_intersections_from_segment(&self, segment: usize) -> Option<&Vec<IntersectionRef>> {
        self.index.get_intersections(segment)
    }

    // --- Nodes

    pub fn add_node(&mut self, node: Node) {
        self.add_independent_node(node.to_independent())
    }

    pub fn add_independent_node(&mut self, node: IndependentNode) {
        self.add_node_ref(node.to_ref());
    }

    pub fn add_node_ref(&mut self, node_ref: NodeRef) {
        self.nodes.push(node_ref.clone());
        self.index.add_node(node_ref);
    }

    pub fn get_nodes_by_point_id(&self, point_id: usize) -> Option<&Vec<NodeRef>> {
        self.index.get_nodes(point_id)
    }

    // --- Edges

    pub fn add_edge(&mut self, edge: Edge) {
        self.edges.push(edge.to_ref())
    }

    pub fn add_edges(&mut self, edges: Vec<Edge>) {
        for edge in edges {
            self.add_edge(edge)
        }
    }

    pub fn add_materialized_edges(&mut self, edges: Vec<MaterializedEdge>) {
        for edge in edges {
            let from_node = edge.get_from_node().to_ref();
            let to_node = edge.get_to_node().to_ref();
            self.add_node_ref(from_node.clone());
            self.add_node_ref(to_node.clone());
            self.add_edge(Edge::new(from_node, to_node));
        }
    }

    #[cfg(test)]
    pub fn get_edge(&self, index: usize) -> Option<&EdgeRef> {
        self.edges.get(index)
    }

    pub fn get_edges(&self) -> Vec<EdgeRef> {
        self.edges.iter().map(|edge| edge.clone()).collect()
    }

    pub fn slice_edges(&self, range: Range<usize>) -> Iter<'_, EdgeRef> {
        self.edges[range].iter()
    }

    pub fn sort_edges(&mut self) {
        self.edges.sort_by(|edge_a, edge_b| edge_a.get_to_id().partial_cmp(&edge_b.get_to_id()).unwrap());
    }

    // --- Misc

    pub fn wipe_graph(&mut self) -> Vec<MaterializedEdge> {
        let materialized = self.edges.iter().map(|e| e.materialize()).collect();
        self.edges.clear();
        self.nodes.clear();
        self.index.clear_nodes();

        materialized
    }
}
