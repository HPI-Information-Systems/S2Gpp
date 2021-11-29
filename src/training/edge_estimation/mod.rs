mod messages;
#[cfg(test)]
mod tests;
mod edges_orderer;

use actix::prelude::*;
use crate::training::Training;
pub use crate::training::edge_estimation::messages::{EdgeEstimationDone};
use crate::utils::{Edge, NodeName};
use crate::training::edge_estimation::edges_orderer::EdgesOrderer;
use crate::utils::logging::progress_bar::S2GppProgressBar;


#[derive(Default)]
pub struct EdgeEstimation {
    /// [(point_id, Edge)]
    pub edges: Vec<(usize, Edge)>,
}

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

        // todo: do not copy into previous_node
        let mut previous_node: Option<NodeName> = None;

        let progress_bar = S2GppProgressBar::new_from_len("info", len_dataset);
        for point_id in 0..len_dataset {
            match self.node_estimation.nodes_by_point.get(&point_id) {
                Some(intersection_nodes) => {
                    // todo: identify by previous point_id
                    let mut edges = EdgesOrderer::new(point_id, previous_node);
                    for current_node in intersection_nodes {
                        edges.add_node(&previous_node, current_node);
                        previous_node = Some(current_node.clone());
                    }

                    previous_node = edges.last_node.or(previous_node);

                    let edges = edges.to_vec();
                    self.edge_estimation.edges.extend(edges.into_iter());

                    if let Some(current_node) = previous_node {
                        if let Some((point_id, next_node)) = self.node_estimation.next_foreign_node.get(&(point_id, current_node.0)) {
                            self.edge_estimation.edges.push((point_id.clone(),
                                Edge(current_node, next_node.clone())
                            ));
                            previous_node = None;
                        }
                    }
                },
                None => ()  // transition did not cross a segment
            }
            progress_bar.inc();
        }
        progress_bar.finish_and_clear();
    }

    fn finalize_edge_estimation(&mut self, ctx: &mut Context<Training>) {
        self.edge_estimation.edges.sort_by(|(point_id_a, _), (point_id_b, _)| point_id_a.partial_cmp(point_id_b).unwrap());

        ctx.address().do_send(EdgeEstimationDone);
    }
}
