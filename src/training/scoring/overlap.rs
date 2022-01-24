use actix::prelude::*;

use crate::Training;
use crate::data_store::materialize::Materialize;
use crate::training::scoring::messages::OverlapRotation;
use crate::training::scoring::Scorer;


pub(crate) trait ScoringOverlap {
    fn send_overlap_to_neighbor(&mut self, ctx: &mut Context<Training>);
}


impl ScoringOverlap for Training {
    fn send_overlap_to_neighbor(&mut self, ctx: &mut Context<Training>) {
        let to_edge_idx = self.scoring.edges_in_time[self.parameters.query_length - 1];
        let overlap = self.data_store.slice_edges(0..to_edge_idx)
            .map(|edge| edge.materialize()).collect();

        let edges_in_time = self.scoring.edges_in_time[0..(self.parameters.query_length - 1)].to_vec();

        self.cluster_nodes.get_prev_as("Training").unwrap()
            .do_send(OverlapRotation { edges: overlap, edges_in_time });
        self.scoring.overlap_rotation_protocol.sent();

        self.score(ctx);
    }
}


impl Handler<OverlapRotation> for Training {
    type Result = ();

    fn handle(&mut self, msg: OverlapRotation, ctx: &mut Self::Context) -> Self::Result {
        if !self.scoring.overlap_rotation_protocol.received(&msg) {
            return
        }

        self.data_store.add_materialized_edges(msg.edges);
        let last_edge_count = self.scoring.edges_in_time.last().unwrap();
        let received_edges_in_time: Vec<usize> = msg.edges_in_time.into_iter().map(|x| x + last_edge_count).collect();
        self.scoring.edges_in_time.extend(received_edges_in_time);
        self.send_overlap_to_neighbor(ctx);

        if self.scoring.overlap_rotation_protocol.is_running() {
            panic!("Only one overlap should be received");
        }
    }
}
