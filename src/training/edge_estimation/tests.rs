use std::collections::HashMap;
use std::iter::FromIterator;
use ndarray::{arr1};
use crate::parameters::{Parameters};
use crate::training::{Training};
use crate::utils::{Edge, NodeName};
use crate::data_manager::DatasetStats;
use crate::training::edge_estimation::EdgeEstimator;


fn test_edge_estimation(nodes: Vec<(usize, Vec<NodeName>)>, expected_edges: Vec<Edge>) {
    let parameters = Parameters::default();
    let mut training = Training::new(parameters);

    training.dataset_stats = Some(DatasetStats::new(
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        nodes.len()
    ));

    training.node_estimation.nodes_by_point = HashMap::from_iter(nodes.clone());
    training.segmentation.segment_index = HashMap::from_iter(nodes.into_iter().map(|(transition_id, _nodes)| (transition_id, transition_id)).collect::<Vec<(usize, usize)>>());

    training.connect_nodes();

    for (i, expected_edge) in expected_edges.into_iter().enumerate() {
        assert_eq!(training.edge_estimation.edges[i].1, expected_edge);
    }
}


#[test]
fn test_edge_estimation_ordered() {
    let nodes = vec![
        (0, vec![NodeName(0, 0)]),
        (1, vec![NodeName(1, 0), NodeName(2, 1)])
    ];

    let expected_edges = vec![
        Edge(NodeName(0, 0), NodeName(1, 0)),
        Edge(NodeName(1, 0), NodeName(2, 1))
    ];

    test_edge_estimation(nodes, expected_edges);
}


#[test]
fn test_edge_estimation_transition_over_0() {
    let nodes = vec![
        (0, vec![NodeName(98, 0)]),
        (1, vec![NodeName(0, 0), NodeName(99, 1)])
    ];
    let expected_edges = vec![
        Edge(NodeName(98, 0), NodeName(99, 1)),
        Edge(NodeName(99, 1), NodeName(0, 0))
    ];

    test_edge_estimation(nodes, expected_edges);
}
