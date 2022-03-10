mod messages;
#[cfg(test)]
mod tests;
mod edges_orderer;

use actix::prelude::*;
use crate::data_store::edge::Edge;
use crate::data_store::node::{NodeRef};
use crate::training::Training;
pub use crate::training::edge_estimation::messages::{EdgeEstimationDone};
use crate::training::edge_estimation::edges_orderer::EdgesOrderer;
use crate::utils::logging::progress_bar::S2GppProgressBar;


pub trait EdgeEstimator {
    fn estimate_edges(&mut self, ctx: &mut Context<Training>);
    fn connect_nodes(&mut self);
    fn finalize_edge_estimation(&mut self, ctx: &mut Context<Training>);
}

impl EdgeEstimator for Training {
    fn estimate_edges(&mut self, ctx: &mut Context<Training>) {
        self.connect_nodes();
        self.finalize_edge_estimation(ctx);
    }

    fn connect_nodes(&mut self) {
        let len_dataset = self.dataset_stats.as_ref().expect("DatasetStats should've been set by now!").n.unwrap();

        let mut previous_node: Option<NodeRef> = None;

        let progress_bar = S2GppProgressBar::new_from_len("info", len_dataset);
        for point_id in 0..len_dataset {
            if let Some(intersection_nodes) = self.data_store.get_nodes_by_point_id(point_id) {
                let mut edges = EdgesOrderer::new(previous_node.clone());
                for current_node in intersection_nodes {
                    edges.add_node(&previous_node, current_node);
                    previous_node = Some(current_node.clone());
                }

                previous_node = edges.last_node.clone().or(previous_node);

                self.data_store.add_edges(edges.into_vec());

                if let Some(current_node) = &previous_node {
                    if let Some((_point_id, next_node)) = self.node_estimation.next_foreign_node.remove(&(point_id, current_node.get_segment_id())) {
                        self.data_store.add_edge(Edge::new(current_node.clone(), next_node.into_ref()));
                        previous_node = None;
                    }
                }
            }
            progress_bar.inc();
        }
        self.node_estimation.next_foreign_node.clear();
        progress_bar.finish_and_clear();
    }

    fn finalize_edge_estimation(&mut self, ctx: &mut Context<Training>) {
        self.data_store.sort_edges();
        ctx.address().do_send(EdgeEstimationDone);
    }
}
