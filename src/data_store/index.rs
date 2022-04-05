use crate::data_store::intersection::IntersectionRef;
use crate::data_store::node::NodeRef;
use crate::data_store::point::PointRef;
use indexmap::IndexMap;
use std::collections::HashMap;

#[derive(Default, Clone)]
pub(crate) struct DataStoreIndex {
    /// IndexMap: Key=PointId, Value=PointRef
    points: IndexMap<usize, PointRef>,
    /// HashMap: Key=SegmentId, Value=Vec<Intersections in segment SegmentID>
    intersections: HashMap<usize, Vec<IntersectionRef>>,
    /// HashMap: Key=PointId, Value=Vec<NodeRef from starting point PointID>
    nodes: HashMap<usize, Vec<NodeRef>>,
}

impl DataStoreIndex {
    // --- Points

    pub fn add_point(&mut self, point: PointRef) {
        self.points.insert(point.get_id(), point);
    }

    pub fn add_point_return(&mut self, point: PointRef) -> PointRef {
        let point_id = point.get_id();
        self.add_point(point);
        self.get_point_by_id(point_id).unwrap().clone()
    }

    fn get_point_by_id(&self, id: usize) -> Option<&PointRef> {
        self.points.get(&id)
    }

    // --- Intersections

    pub fn add_intersection(&mut self, intersection: IntersectionRef) {
        match &mut self.intersections.get_mut(&intersection.get_segment_id()) {
            Some(intersections) => intersections.push(intersection),
            None => {
                self.intersections
                    .insert(intersection.get_segment_id(), vec![intersection]);
            }
        }
    }

    pub fn get_intersections(&self, segment: usize) -> Option<&Vec<IntersectionRef>> {
        self.intersections.get(&segment)
    }

    // --- Nodes

    pub fn add_node(&mut self, node_ref: NodeRef) {
        let start_point = node_ref.get_from_id();
        match self.nodes.get_mut(&start_point) {
            Some(nodes) => nodes.push(node_ref),
            None => {
                self.nodes.insert(start_point, vec![node_ref]);
            }
        }
    }

    pub fn get_nodes(&self, point_id: usize) -> Option<&Vec<NodeRef>> {
        self.nodes.get(&point_id)
    }

    // --- Misc

    pub fn clear_nodes(&mut self) {
        self.nodes.clear();
    }
}
