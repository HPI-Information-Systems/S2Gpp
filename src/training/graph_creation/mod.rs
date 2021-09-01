mod messages;

use log::*;
use petgraph::graph::{DiGraph};
use petgraph::dot::{Dot, Config};
use crate::training::Training;
use std::borrow::Borrow;
use crate::utils::{NodeName, Edge};
use std::fs::File;
use std::io::Write;
use std::fs;
use anyhow::{Result, Error};
pub use messages::GraphCreationDone;


#[derive(Default)]
pub struct GraphCreation {
    graph: Option<DiGraph<u32, f32>>
}


pub trait GraphCreator {
    fn create_graph(&mut self);
    fn output_graph(&mut self, output_path: String) -> Result<()>;
}


impl GraphCreator for Training {
    fn create_graph(&mut self) {
        let edges: &[Edge] = self.edge_estimation.edges.borrow();

        self.graph_creation.graph = Some(DiGraph::from_edges(edges.into_iter().map(|e| {
            e.to_index_tuple()
        }).collect::<Vec<(u32, u32)>>()));
    }

    fn output_graph(&mut self, output_path: String) -> Result<()> {
        let graph = self.graph_creation.graph.as_ref()
            .ok_or(Error::msg("No graph generated yet!"))?;
        let dot = Dot::with_config(graph, &[Config::EdgeNoLabel]);
        let dot_string = format!("{:?}", dot);

        fs::write(output_path,dot_string)?;
        Ok(())
    }
}
