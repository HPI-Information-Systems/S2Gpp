mod messages;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use actix::prelude::*;
use crate::training::Training;
pub use crate::training::transposition::messages::{TranspositionDone, TranspositionRotationMessage};
use crate::utils::Edge;
use crate::utils::rotation_protocol::RotationProtocol;

#[derive(Default)]
pub struct Transposition {
    rotation_protocol: RotationProtocol<TranspositionRotationMessage>,
    pub(crate) range_start_point: Option<usize>,
    partition_len: Option<usize>,
    pub(crate) edges: Vec<(usize, Edge)>
}

pub trait Transposer {
    fn transpose(&mut self, rec: Recipient<TranspositionRotationMessage>);
    fn assign_edges_to_neighbours(&mut self) -> HashMap<usize, Vec<(usize, Edge)>>;
    fn transpose_rotation(&mut self, assignments: HashMap<usize, Vec<(usize, Edge)>>);
    fn transpose_finalize(&mut self, ctx: &mut Context<Training>);
}

impl Transposer for Training {
    fn transpose(&mut self, rec: Recipient<TranspositionRotationMessage>) {
        let len_dataset = self.dataset_stats.as_ref().unwrap().n.as_ref().unwrap();
        let partition_len = len_dataset / self.parameters.n_cluster_nodes;
        let own_start_point= partition_len * self.cluster_nodes.get_own_idx();

        self.transposition.range_start_point = Some(own_start_point);
        self.transposition.partition_len = Some(partition_len);

        let assignments = self.assign_edges_to_neighbours();
        self.transposition.rotation_protocol.start(self.cluster_nodes.len());
        self.transposition.rotation_protocol.resolve_buffer(rec);
        self.transpose_rotation(assignments);
    }

    fn assign_edges_to_neighbours(&mut self) -> HashMap<usize, Vec<(usize, Edge)>> {
        let mut assignments = HashMap::new();
        for cluster_id in 0..self.parameters.n_cluster_nodes {
            assignments.insert(cluster_id, vec![]);
        }

        for (point_id, edge) in self.edge_estimation.edges.iter() {
            let cluster_node_id = point_id / self.transposition.partition_len.expect("Should already be set!");
            match assignments.get_mut(&cluster_node_id) {
                None => {}
                Some(edges) => edges.push((point_id.clone(), edge.clone()))
            }
        }
        self.edge_estimation.edges.clear();

        let own_id = self.cluster_nodes.get_own_idx();
        self.transposition.edges.extend(assignments.remove(&own_id).unwrap());

        assignments
    }

    fn transpose_rotation(&mut self, assignments: HashMap<usize, Vec<(usize, Edge)>>) {
        let msg = TranspositionRotationMessage { assignments };
        let next = self.cluster_nodes.get_as(&self.cluster_nodes.get_next_idx().expect("No Transposition without other cluster nodes!"), "Training").unwrap();
        next.do_send(msg);
        self.transposition.rotation_protocol.sent();
    }

    fn transpose_finalize(&mut self, ctx: &mut Context<Training>) {
        self.transposition.edges.sort_by(|(point_id_a, _), (point_id_b, _)| point_id_a.partial_cmp(point_id_b).unwrap());
        ctx.address().do_send(TranspositionDone);
    }
}


impl Handler<TranspositionRotationMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: TranspositionRotationMessage, ctx: &mut Self::Context) -> Self::Result {
        if !self.transposition.rotation_protocol.received(&msg) {
            return
        }

        let own_id = self.cluster_nodes.get_own_idx();
        let mut assignments = msg.assignments;
        if let Some(own_edges) = assignments.remove(&own_id) {
            self.transposition.edges.extend(own_edges);
        }

        if self.transposition.rotation_protocol.is_running() {
            self.transpose_rotation(assignments);
        } else {
            self.transpose_finalize(ctx);
        }
    }
}