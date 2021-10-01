#[cfg(test)]
mod tests;

use ndarray::Array1;
use crate::training::Training;
use std::collections::{HashMap, HashSet};
use crate::utils::{Edge, NodeName};
use ndarray_stats::QuantileExt;
use std::ops::Range;
use anyhow::Result;
use std::fs::File;
use csv::WriterBuilder;

#[derive(Default)]
pub struct Scoring {
    pub score: Option<Array1<f32>>
}

pub trait Scorer {
    fn count_edges_in_time(&mut self) -> Vec<usize>;
    fn calculate_edge_weight(&mut self) -> HashMap<Edge, usize>;
    fn calculate_node_degrees(&mut self) -> HashMap<NodeName, usize>;
    fn score(&mut self);
    fn score_p_degree(&mut self, edge_weight: &HashMap<Edge, usize>, edge_range: Range<usize>, node_degrees: &HashMap<NodeName, usize>) -> (f32, usize);
    fn output_score(&mut self, output_path: String) -> Result<()>;
}

impl Scorer for Training {
    fn count_edges_in_time(&mut self) -> Vec<usize> {
        let pseudo_edge = (0, Edge(NodeName(0, 0), NodeName(0, 0)));
        let mut edges_in_time = vec![];
        let mut last_point_id = None;
        let mut last_len: usize = 0;
        for (i, (point_id, _edge)) in self.edge_estimation.edges.iter().chain(&[pseudo_edge]).enumerate() {
            match last_point_id {
                None => { last_point_id = Some(point_id); }
                Some(last_point_id_ref) => if point_id.ne(last_point_id_ref) {
                    while edges_in_time.len().lt(last_point_id_ref) {
                        edges_in_time.push(last_len);
                    }
                    last_point_id = Some(point_id);
                    last_len = i;
                    edges_in_time.push(i);
                }
            }
        }

        while edges_in_time.len().lt(&(self.rotation.rotated.as_ref().unwrap().shape()[0] - 1)) {
            edges_in_time.push(last_len);
        }

        edges_in_time
    }

    fn calculate_edge_weight(&mut self) -> HashMap<Edge, usize> {
        let mut edge_weight = HashMap::new();
        for (_, edge) in self.edge_estimation.edges.iter() {
            match edge_weight.get_mut(edge) {
                Some(weight) => { *weight += 1; },
                None => { edge_weight.insert(edge.clone(), 1); }
            }
        }
        edge_weight
    }

    fn calculate_node_degrees(&mut self) -> HashMap<NodeName, usize> {
        let mut node_degrees = HashMap::new();
        let mut seen_edges = HashSet::new();

        for (_, edge) in self.edge_estimation.edges.iter() {
            if seen_edges.insert(edge.clone()) {
                match node_degrees.get_mut(&edge.0) {
                    Some(degree) => { *degree += 1; }
                    None => { node_degrees.insert(edge.0.clone(), 1); }
                }

                match node_degrees.get_mut(&edge.1) {
                    Some(degree) => { *degree += 1; }
                    None => { node_degrees.insert(edge.1.clone(), 1); }
                }
            }
        }

        node_degrees
    }

    fn score(&mut self) {
        let edges_in_time = self.count_edges_in_time();
        let edge_weight = self.calculate_edge_weight();
        let node_degrees = self.calculate_node_degrees();

        let mut all_score = vec![];

        if edges_in_time.len() < (self.parameters.query_length - 1) {
            panic!("There are less edges than the given 'query_length'!");
        }

        let end_iteration = edges_in_time.len() - (self.parameters.query_length - 1);

        for i in 0..end_iteration {
            let from_edge_idx = edges_in_time[i];
            let to_edge_idx = edges_in_time[i + self.parameters.query_length - 1];

            let (score, len_score) = self.score_p_degree(&edge_weight, from_edge_idx..to_edge_idx, &node_degrees);
            if len_score == 0 {
                all_score.push(all_score.last().unwrap_or(&0_f32).clone());
            } else {
                all_score.push(score);
            }
        }

        let all_score: Array1<f32> = all_score.into_iter().map(|x| -x).collect();
        let all_score_max = all_score.max().unwrap().clone();
        let all_score_min = all_score.min().unwrap().clone();
        self.scoring.score = Some((all_score - all_score_min) / (all_score_max - all_score_min));
    }

    fn score_p_degree(&mut self, edge_weight: &HashMap<Edge, usize>, edge_range: Range<usize>, node_degrees: &HashMap<NodeName, usize>) -> (f32, usize) {
        let p_edge = &self.edge_estimation.edges[edge_range];
        let len_score = p_edge.len();
        let alpha = 0.00000001 + (len_score as f32);
        let score: f32 = p_edge.iter().map(|(_, edge)| {
            (edge_weight.get(edge).unwrap() * (node_degrees.get(&edge.0).expect("Edge with unknown Node found!") - 1)) as f32
        }).sum();
        (score / alpha, len_score)
    }

    fn output_score(&mut self, output_path: String) -> Result<()> {
        let score = self.scoring.score.as_ref().expect("Please, calculate score before saving to file!");
        let file = File::create(output_path)?;
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        for s in score.iter() {
            writer.serialize(s)?;
        }
        Ok(())
    }
}
