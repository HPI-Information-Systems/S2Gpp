use actix::{Actor, ActorContext, SyncContext, Handler, Recipient};
use crate::training::edge_estimation::messages::{EdgeEstimationHelperTask, EdgeEstimationHelperResponse};
use actix::dev::MessageResponse;
use ndarray::{Dim, Axis, Array1, Array2, ArcArray1, ArcArray2};
use crate::training::edge_estimation::{NodeName, Edge};
use crate::meanshift::DistanceMeasure;
use crate::training::intersection_calculation::{Transition, SegmentID};
use std::collections::HashMap;
use crate::messages::PoisonPill;


pub struct EdgeEstimationHelper {
    pub source: Recipient<EdgeEstimationHelperResponse>,
    pub distance_measure: DistanceMeasure,
    pub intersections: HashMap<Transition, Vec<SegmentID>>,
    pub intersection_coords: HashMap<SegmentID, HashMap<Transition, Array1<f32>>>,
    pub nodes: HashMap<SegmentID, Array2<f32>>
}

impl Actor for EdgeEstimationHelper {
    type Context = SyncContext<EdgeEstimationHelper>;
}

impl EdgeEstimationHelper {
    fn estimate_edges(&mut self, task_id: usize, transition: Transition) {
        match self.intersections.get(&transition) {
            Some(segment_ids) => {
                let mut node_names = vec![];
                for segment_id in segment_ids.clone() {
                    let distance = (&self.intersection_coords).get(&segment_id).unwrap().get(&transition).unwrap().to_shared();
                    let nodes = (&self.nodes).get(&segment_id).unwrap().to_shared();
                    node_names.push(self.estimate_closest_node(segment_id.clone(), distance, nodes))
                }
                self.source.do_send(EdgeEstimationHelperResponse { task_id, node_names, transition});
            },
            None => ()
        }

    }

    fn estimate_closest_node(&mut self, segment_id: usize, distance: ArcArray1<f32>, nodes: ArcArray2<f32>) -> NodeName {
        let distance = distance.broadcast(Dim([nodes.shape()[0], distance.len()])).unwrap();
        let closest_id: usize = nodes.axis_iter(Axis(0))
            .zip(distance.axis_iter(Axis(0)))
            .map(|(node, dist)| self.distance_measure.call()(&node.to_vec(), &dist.to_vec()))
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx).unwrap();
        let current_node = NodeName(segment_id.clone(), closest_id);
        return current_node
    }
}

impl Handler<EdgeEstimationHelperTask> for EdgeEstimationHelper {
    type Result = ();

    fn handle(&mut self, msg: EdgeEstimationHelperTask, ctx: &mut Self::Context) -> Self::Result {
        self.estimate_edges(msg.task_id, msg.transition);
    }
}

impl Handler<PoisonPill> for EdgeEstimationHelper {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
