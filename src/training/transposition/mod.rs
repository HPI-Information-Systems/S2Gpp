mod messages;
#[cfg(test)]
mod tests;

use crate::data_store::edge::MaterializedEdge;
pub(crate) use crate::training::transposition::messages::{
    TranspositionDone, TranspositionRotationMessage,
};
use crate::training::Training;
use crate::utils::direct_protocol::DirectProtocol;
use actix::prelude::*;
use std::collections::HashMap;

#[derive(Default, Clone)]
pub(crate) struct Transposition {
    direct_protocol: DirectProtocol<TranspositionRotationMessage>,
    pub(crate) range_start_point: Option<usize>,
    partition_len: Option<usize>,
}

pub(crate) trait Transposer {
    fn transpose(&mut self, rec: Recipient<TranspositionRotationMessage>);
    fn assign_edges_to_neighbours(&mut self) -> HashMap<usize, Vec<MaterializedEdge>>;
    fn transpose_rotation(&mut self, assignments: HashMap<usize, Vec<MaterializedEdge>>);
    fn transpose_finalize(&mut self, ctx: &mut Context<Training>);
}

impl Transposer for Training {
    fn transpose(&mut self, rec: Recipient<TranspositionRotationMessage>) {
        let len_dataset = self.dataset_stats.as_ref().unwrap().n.as_ref().unwrap();
        let partition_len = len_dataset / self.parameters.n_cluster_nodes;
        let own_start_point = partition_len * self.cluster_nodes.get_own_idx();

        self.transposition.range_start_point = Some(own_start_point);
        self.transposition.partition_len = Some(partition_len);

        let assignments = self.assign_edges_to_neighbours();
        self.transposition
            .direct_protocol
            .start(self.cluster_nodes.len());
        self.transposition.direct_protocol.resolve_buffer(rec);
        self.transpose_rotation(assignments);
    }

    fn assign_edges_to_neighbours(&mut self) -> HashMap<usize, Vec<MaterializedEdge>> {
        let mut assignments = HashMap::new();
        for cluster_id in 0..self.parameters.n_cluster_nodes {
            assignments.insert(cluster_id, vec![]);
        }

        let materialized_edges = self.data_store.wipe_graph();

        for edge in materialized_edges {
            let point_id = edge.get_to_id();
            let mut cluster_node_id = point_id
                / self
                    .transposition
                    .partition_len
                    .expect("Should already be set!");
            cluster_node_id = if cluster_node_id >= self.parameters.n_cluster_nodes {
                self.parameters.n_cluster_nodes - 1
            } else {
                cluster_node_id
            };
            match assignments.get_mut(&cluster_node_id) {
                None => {}
                Some(edges) => edges.push(edge),
            }
        }

        let own_id = self.cluster_nodes.get_own_idx();
        self.data_store
            .add_materialized_edges(assignments.remove(&own_id).unwrap());

        assignments
    }

    fn transpose_rotation(&mut self, mut assignments: HashMap<usize, Vec<MaterializedEdge>>) {
        for (id, node) in self.cluster_nodes.iter() {
            let mut training_node = node.clone();
            training_node.change_id("Training".to_string());

            let msg = match assignments.remove(id) {
                Some(node_assignments) => TranspositionRotationMessage {
                    assignments: node_assignments,
                },
                None => TranspositionRotationMessage::default(),
            };

            training_node.do_send(msg);
            self.transposition.direct_protocol.sent();
        }
    }

    fn transpose_finalize(&mut self, ctx: &mut Context<Training>) {
        self.data_store.sort_edges();
        ctx.address().do_send(TranspositionDone);
    }
}

impl Handler<TranspositionRotationMessage> for Training {
    type Result = ();

    fn handle(
        &mut self,
        msg: TranspositionRotationMessage,
        ctx: &mut Self::Context,
    ) -> Self::Result {
        if !self.transposition.direct_protocol.received(&msg) {
            return;
        }

        self.data_store.add_materialized_edges(msg.assignments);

        if !self.transposition.direct_protocol.is_running() {
            self.transpose_finalize(ctx);
        }
    }
}
