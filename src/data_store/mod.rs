use std::collections::HashMap;
use std::ops::{Deref, Range};
use std::slice::Iter;
use indexmap::IndexMap;
use ndarray::{Array1};
use crate::data_store::edge::{Edge, EdgeRef, MaterializedEdge};
use crate::data_store::intersection::{Intersection, IntersectionMixin, IntersectionRef, MaterializedIntersection};
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


// todo: build index for certain things:
// - Point_id -> PointRef
// - Segment_id -> Vec<IntersectionRef>
// - PointRef -> Vec<Node>
#[derive(Default, Clone)]
pub(crate) struct DataStore {
    points: IndexMap<usize, PointRef>,
    transitions: Vec<TransitionRef>,
    /// HashMap: Key=SegmentId, Value=Vec<Intersections in segment SegmentID>
    intersections: HashMap<usize, Vec<IntersectionRef>>,
    /// HashMap: Key=PointId, Value=Vec<Nodes>
    nodes: HashMap<usize, Vec<NodeRef>>,
    edges: Vec<EdgeRef>
}

impl DataStore {
    pub fn add_point(&mut self, point: Point) -> bool {
        let point = PointRef::new(point);
        if self.points.contains_key(&point.get_id()) {
            false
        } else {
            self.points.insert(point.get_id(), point);
            true
        }
    }

    pub fn add_point_return(&mut self, point: Point) -> PointRef {
        let id = point.get_id();
        self.add_point(point);
        self.points.get(&id).unwrap().clone()
    }

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
        self.points.values().map(|p| p.clone()).collect()
    }

    pub fn add_transition(&mut self, transition: Transition) {
        self.transitions.push(TransitionRef::new(transition))
    }

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

    pub fn add_intersection(&mut self, intersection: Intersection) {
        match &mut self.intersections.get_mut(&intersection.get_segment_id()) {
            Some(intersections) => intersections.push(IntersectionRef::new(intersection)),
            None => { self.intersections.insert(intersection.get_segment_id(), vec![IntersectionRef::new(intersection)]); }
        }
    }

    pub fn add_materialized_intersection(&mut self, intersection: MaterializedIntersection) {
        // todo: check if points and transitions are necessary to add
        self.add_materialized_transition(intersection.get_transition());
        let transition = self.transitions.last().unwrap().clone();
        self.add_intersection(Intersection::new(transition, intersection.get_coordinates().to_owned().clone(), intersection.get_segment_id()));
    }

    pub fn add_materialized_intersections(&mut self, intersections: Vec<MaterializedIntersection>) {
        for intersection in intersections {
            self.add_materialized_intersection(intersection);
        }
    }

    pub fn get_intersections_from_segment(&self, segment: usize) -> Option<&Vec<IntersectionRef>> {
        self.intersections.get(&segment)
    }

    pub fn add_node(&mut self, node: Node) {
        self.add_independent_node(node.to_independent())
    }

    pub fn add_independent_node(&mut self, node: IndependentNode) {
        self.add_node_ref(node.to_ref());
    }

    pub fn add_node_ref(&mut self, node_ref: NodeRef) {
        let start_point = node_ref.get_from_id();
        match self.nodes.get_mut(&start_point) {
            Some(nodes) => nodes.push(node_ref),
            None => { self.nodes.insert(start_point, vec![node_ref]); }
        }
    }

    pub fn get_nodes_by_point_id(&self, point_id: usize) -> Option<&Vec<NodeRef>> {
        self.nodes.get(&point_id)
    }

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

    pub fn wipe_graph(&mut self) -> Vec<MaterializedEdge> {
        let materialized = self.edges.iter().map(|e| e.materialize()).collect();
        self.nodes.clear();
        self.edges.clear();

        materialized
    }
}
