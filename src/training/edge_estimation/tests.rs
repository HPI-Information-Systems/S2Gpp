use crate::data_manager::DatasetStats;
use crate::data_store::edge::Edge;
use crate::data_store::node::IndependentNode;
use crate::parameters::Parameters;
use crate::training::edge_estimation::EdgeEstimator;
use crate::training::Training;
use ndarray::arr1;
use std::ops::Deref;

fn test_edge_estimation(nodes: Vec<IndependentNode>, expected_edges: Vec<Edge>) {
    let parameters = Parameters::default();
    let mut training = Training::new(parameters);

    training.dataset_stats = Some(DatasetStats::new(
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        arr1(&[0_f32]),
        nodes.len(),
    ));

    for node in nodes {
        training.data_store.add_independent_node(node)
    }
    training.connect_nodes();

    for (i, expected_edge) in expected_edges.into_iter().enumerate() {
        assert!(training
            .data_store
            .get_edge(i)
            .unwrap()
            .deref()
            .eq(&expected_edge));
    }
}

#[test]
fn test_edge_estimation_ordered() {
    let nodes = vec![
        IndependentNode::new(0, 0, 0),
        IndependentNode::new(1, 0, 1),
        IndependentNode::new(2, 1, 1),
    ];

    let expected_edges = vec![
        Edge::new(
            IndependentNode::new(0, 0, 0).into_ref(),
            IndependentNode::new(1, 0, 1).into_ref(),
        ),
        Edge::new(
            IndependentNode::new(1, 0, 1).into_ref(),
            IndependentNode::new(2, 1, 1).into_ref(),
        ),
    ];

    test_edge_estimation(nodes, expected_edges);
}

#[test]
fn test_edge_estimation_transition_over_0() {
    let nodes = vec![
        IndependentNode::new(98, 0, 0),
        IndependentNode::new(0, 0, 1),
        IndependentNode::new(99, 1, 1),
    ];
    let expected_edges = vec![
        Edge::new(
            IndependentNode::new(98, 0, 0).into_ref(),
            IndependentNode::new(99, 1, 1).into_ref(),
        ),
        Edge::new(
            IndependentNode::new(99, 1, 1).into_ref(),
            IndependentNode::new(0, 0, 1).into_ref(),
        ),
    ];

    test_edge_estimation(nodes, expected_edges);
}
