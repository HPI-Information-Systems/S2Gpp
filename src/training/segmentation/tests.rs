use std::collections::HashMap;
use actix_telepathy::{RemoteAddr, AddrRepresentation};
use crate::parameters::{Parameters};
use crate::training::Training;
use ndarray::{arr1};
use std::iter::FromIterator;
use crate::utils::ClusterNodes;
use crate::data_manager::DatasetStats;
use crate::data_store::node_questions::node_in_question::NodeInQuestion;
use crate::data_store::transition::TransitionMixin;
use crate::training::segmentation::{Segmenter};


#[test]
fn test_segmenting() {
    let parameters = Parameters {
        n_cluster_nodes: 2,
        rate: 4,
        ..Default::default()
    };

    let mut training = Training::new(parameters.clone());
    training.dataset_stats = Some(DatasetStats::new(arr1(&[1.0]), arr1(&[1.0]), arr1(&[1.0]), 100));
    training.cluster_nodes = ClusterNodes::from(HashMap::from_iter(vec![
        (1, RemoteAddr::new("127.0.0.1:1993".parse().unwrap(), None, AddrRepresentation::Key("test1".to_string())))
    ]));
    training.data_store.add_points(vec![
        arr1(&[1.0, 1.0]),
        arr1(&[-1.0, 1.0]),
        arr1(&[-1.0, -1.0]),
        arr1(&[1.0, -1.0])
    ], parameters.rate);


    // node transitions
    let expected_node_transition = [arr1(&[-1.0, -1.0]), arr1(&[1.0, -1.0])];
    let node_transitions = training.build_segments();
    assert_eq!(node_transitions.len(), 2);
    assert_eq!(node_transitions[&1].len(), 1);
    assert_eq!(node_transitions[&1][0].get_from_segment(), 2);
    assert_eq!(node_transitions[&1][0].get_to_segment(), 3);
    assert_eq!(node_transitions[&1][0].get_from_point().get_coordinates(), expected_node_transition[0]);
    assert_eq!(node_transitions[&1][0].get_to_point().get_coordinates(), expected_node_transition[1]);

    // own transitions
    let expected_own_transition = [arr1(&[1.0, 1.0]), arr1(&[-1.0, 1.0])];
    let own_transitions = training.data_store.get_transitions();
    assert_eq!(own_transitions.len(), 2);
    assert_eq!(own_transitions[0].get_from_segment(), 0);
    assert_eq!(own_transitions[0].get_to_segment(), 1);
    assert_eq!(own_transitions[0].get_from_point().get_coordinates(), expected_own_transition[0]);
    assert_eq!(own_transitions[0].get_to_point().get_coordinates(), expected_own_transition[1]);
}

/// When segmenting the transitions that would connect 2 groups of segments,
/// this transition is not created in a distributed case.
#[test]
fn test_segment_distribution() {
    let parameters = Parameters {
        n_cluster_nodes: 2,
        rate: 4,
        ..Default::default()
    };

    let mut training = Training::new(parameters.clone());
    training.dataset_stats = Some(DatasetStats::new(arr1(&[1.0]), arr1(&[1.0]), arr1(&[1.0]), 100));
    training.cluster_nodes = ClusterNodes::from(HashMap::from_iter(vec![
        (1, RemoteAddr::new("127.0.0.1:1993".parse().unwrap(), None, AddrRepresentation::Key("test1".to_string())))
    ]));
    training.data_store.add_points(vec![
        arr1(&[1.0, 1.0]),
        arr1(&[-1.0, 1.0]),
        arr1(&[-1.0, -1.0]),
        arr1(&[1.0, -1.0])
    ], parameters.rate);

    // node transitions
    let expected_node_transition = [arr1(&[-1.0, -1.0]), arr1(&[1.0, -1.0])];
    let node_transitions = training.build_segments();

    assert_eq!(node_transitions.len(), 2);
    assert_eq!(node_transitions[&1].len(), 1);
    assert_eq!(node_transitions[&1][0].get_from_segment(), 2);
    assert_eq!(node_transitions[&1][0].get_to_segment(), 3);
    assert_eq!(node_transitions[&1][0].get_from_point().get_coordinates(), expected_node_transition[0]);
    assert_eq!(node_transitions[&1][0].get_to_point().get_coordinates(), expected_node_transition[1]);

    // own transitions
    let expected_own_transition = [arr1(&[1.0, 1.0]), arr1(&[-1.0, 1.0])];
    let own_transitions = training.data_store.get_transitions();
    assert_eq!(own_transitions.len(), 2);
    assert_eq!(own_transitions[0].get_from_segment(), 0);
    assert_eq!(own_transitions[0].get_to_segment(), 1);
    assert_eq!(own_transitions[0].get_from_point().get_coordinates(), expected_own_transition[0]);
    assert_eq!(own_transitions[0].get_to_point().get_coordinates(), expected_own_transition[1]);
}


#[test]
fn test_segment_transitions_sub() {
    let parameters = Parameters {
        n_cluster_nodes: 2,
        rate: 4,
        ..Default::default()
    };

    let mut training = Training::new(parameters.clone());
    training.dataset_stats = Some(DatasetStats::new(arr1(&[1.0]), arr1(&[1.0]), arr1(&[1.0]), 100));
    training.cluster_nodes = ClusterNodes::from(HashMap::from_iter(vec![
        (0, RemoteAddr::new("127.0.0.1:1993".parse().unwrap(), None, AddrRepresentation::Key("test1".to_string())))
    ]));
    training.data_store.add_points(vec![
        arr1(&[1.0, 1.0]),
        arr1(&[-1.0, 1.0]),
        arr1(&[-1.0, -1.0]),
        arr1(&[1.0, -1.0])
    ], parameters.rate);

    let _node_transitions = training.build_segments();

    assert_eq!(training.segmentation.send_point.unwrap().get_coordinates(), arr1(&[1.0, 1.0]));
}


#[test]
fn test_segment_transitions_main() {
    let parameters = Parameters {
        n_cluster_nodes: 2,
        rate: 4,
        ..Default::default()
    };

    let mut training = Training::new(parameters.clone());
    training.dataset_stats = Some(DatasetStats::new(arr1(&[1.0]), arr1(&[1.0]), arr1(&[1.0]), 100));
    training.cluster_nodes = ClusterNodes::from(HashMap::from_iter(vec![
        (1, RemoteAddr::new("127.0.0.1:1993".parse().unwrap(), None, AddrRepresentation::Key("test1".to_string())))
    ]));
    training.data_store.add_points(vec![
        arr1(&[1.0, 1.0]),
        arr1(&[-1.0, 1.0]),
        arr1(&[-1.0, -1.0]),
        arr1(&[1.0, -1.0])
    ], parameters.rate);

    let _node_transitions = training.build_segments();

    assert!(training.segmentation.send_point.is_none());
}


#[test]
fn test_node_questions() {
    let parameters = Parameters {
        n_cluster_nodes: 2,
        rate: 8,
        ..Default::default()
    };

    let mut training = Training::new(parameters.clone());

    training.dataset_stats = Some(DatasetStats::new(arr1(&[1.0]), arr1(&[1.0]), arr1(&[1.0]), 20));

    training.data_store.add_points(vec![
        arr1(&[-1., -2.]), // 5
        arr1(&[1., -2.]),  // 6
        arr1(&[1., 2.]),   // 1
        arr1(&[0.5, 2.5]), // 1
        arr1(&[-0.5, 2.5]),// 2
        arr1(&[2., -0.5]), // 7
        arr1(&[2., 1.]),   // 0
        arr1(&[3., 2.]),   // 0
        arr1(&[2.8, 3.]),  // 1
        arr1(&[2., -3.]),  // 6
        arr1(&[3., -2.])   // 7
    ], parameters.rate);
    training.build_segments();


    let grouped_questions = training.segmentation.node_questions.remove(&0).unwrap();
    let node_questions = grouped_questions.get(&1).unwrap();
    assert_eq!(node_questions.len(), 1);
    assert_eq!(node_questions[0], NodeInQuestion::new(1, 7, 1, 0));

    let grouped_questions = training.segmentation.node_questions.remove(&1).unwrap();
    let node_questions = grouped_questions.get(&0).unwrap();
    assert_eq!(node_questions.len(), 1);
    assert_eq!(node_questions[0], NodeInQuestion::new(7, 1, 9, 7));
}
