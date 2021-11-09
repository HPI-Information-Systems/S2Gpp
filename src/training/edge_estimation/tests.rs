use std::collections::HashMap;
use std::iter::FromIterator;
use actix_telepathy::RemoteAddr;
use ndarray::{arr1};
use crate::parameters::{Parameters};
use crate::training::{Training};
use crate::utils::{ClusterNodes, Edge, NodeName};
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


#[test]
fn test_edge_estimation_open_edges_full_round() {
    let parameters = Parameters {
        n_cluster_nodes: 2,
        ..Default::default()
    };

    // Training 1
    let mut training_1 = Training::new(parameters.clone());

    training_1.dataset_stats = Some(DatasetStats::new(
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        4
    ));

    let nodes= vec![
        (1, vec![NodeName(98, 0)]),
        (2, vec![NodeName(99, 0)])
    ];

    training_1.node_estimation.nodes_by_point = HashMap::from_iter(nodes.clone());
    training_1.segmentation.segment_index = HashMap::from_iter(nodes.into_iter().map(|(transition_id, _nodes)| (transition_id, transition_id)).collect::<Vec<(usize, usize)>>());

    training_1.connect_nodes();

    let mut expected = HashMap::new();
    expected.insert(1, vec![(2, NodeName(99, 0))]);
    assert_eq!(training_1.edge_estimation.send_edges, expected);

    // Training 2
    let mut training_2 = Training::new(parameters);

    training_2.dataset_stats = Some(DatasetStats::new(
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        4
    ));

    let nodes= vec![
        (2, vec![NodeName(0, 0)]),
        (3, vec![NodeName(1, 0)]),
        (4, vec![NodeName(2, 0)])
    ];

    training_2.node_estimation.nodes_by_point = HashMap::from_iter(nodes.clone());
    training_2.segmentation.segment_index = HashMap::from_iter(nodes.into_iter().map(|(transition_id, _nodes)| (transition_id, transition_id)).collect::<Vec<(usize, usize)>>());

    training_2.connect_nodes();

    let mut expected = HashMap::new();
    expected.insert(2, NodeName(0, 0));
    assert_eq!(training_2.edge_estimation.open_edges, expected);

    let cluster_nodes = HashMap::from_iter(vec![(0, RemoteAddr::new_from_id("127.0.0.1:8000".parse().unwrap(), "test"))]);
    training_2.cluster_nodes = ClusterNodes::from(cluster_nodes);

    let send_edges = training_2.merging_rotated_edges(training_1.edge_estimation.send_edges);
    assert!(send_edges.is_empty())
}


#[test]
fn test_edge_estimation_open_edges_half_round() {
    let parameters = Parameters {
        n_cluster_nodes: 2,
        ..Default::default()
    };

    // Training 1
    let mut training_1 = Training::new(parameters.clone());

    training_1.dataset_stats = Some(DatasetStats::new(
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        4
    ));

    let nodes= vec![
        (1, vec![NodeName(48, 0)]),
        (2, vec![NodeName(49, 0)])
    ];

    training_1.node_estimation.nodes_by_point = HashMap::from_iter(nodes.clone());
    training_1.segmentation.segment_index = HashMap::from_iter(nodes.into_iter().map(|(transition_id, _nodes)| (transition_id, transition_id)).collect::<Vec<(usize, usize)>>());

    training_1.connect_nodes();

    let mut expected = HashMap::new();
    expected.insert(1, vec![(2, NodeName(49, 0))]);
    assert_eq!(training_1.edge_estimation.send_edges, expected);

    // Training 2
    let mut training_2 = Training::new(parameters);

    training_2.dataset_stats = Some(DatasetStats::new(
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        4
    ));

    let nodes= vec![
        (2, vec![NodeName(50, 0)]),
        (3, vec![NodeName(51, 0)]),
        (4, vec![NodeName(52, 0)])
    ];

    training_2.node_estimation.nodes_by_point = HashMap::from_iter(nodes.clone());
    training_2.segmentation.segment_index = HashMap::from_iter(nodes.into_iter().map(|(transition_id, _nodes)| (transition_id, transition_id)).collect::<Vec<(usize, usize)>>());

    training_2.connect_nodes();

    let mut expected = HashMap::new();
    expected.insert(2, NodeName(50, 0));
    assert_eq!(training_2.edge_estimation.open_edges, expected);

    let cluster_nodes = HashMap::from_iter(vec![(0, RemoteAddr::new_from_id("127.0.0.1:8000".parse().unwrap(), "test"))]);
    training_2.cluster_nodes = ClusterNodes::from(cluster_nodes);

    let send_edges = training_2.merging_rotated_edges(training_1.edge_estimation.send_edges);
    assert!(send_edges.is_empty())
}
