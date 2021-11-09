use std::collections::HashMap;
use actix_telepathy::{RemoteAddr, AddrRepresentation};
use crate::parameters::{Parameters};
use crate::training::Training;
use ndarray::{arr1, arr2};
use std::iter::FromIterator;
use crate::utils::ClusterNodes;
use crate::data_manager::DatasetStats;
use crate::training::segmentation::Segmenter;


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
    training.rotation.rotated = Some(arr2(&[
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0]
    ]));


    // node transitions
    let expected_node_transition = [arr1(&[-1.0, -1.0]), arr1(&[1.0, -1.0])];
    let node_transitions = training.build_segments();
    assert_eq!(node_transitions.len(), 1);
    assert_eq!(node_transitions[&1].len(), 1);
    assert_eq!(node_transitions[&1][0].from.segment_id, 2);
    assert_eq!(node_transitions[&1][0].to.segment_id, 3);
    assert_eq!(node_transitions[&1][0].from.point_with_id.coords, expected_node_transition[0]);
    assert_eq!(node_transitions[&1][0].to.point_with_id.coords, expected_node_transition[1]);

    // own transitions
    let expected_own_transition = [arr1(&[1.0, 1.0]), arr1(&[-1.0, 1.0])];
    let own_transitions = training.segmentation.segments.clone();
    assert_eq!(own_transitions.len(), 2);
    assert_eq!(own_transitions[0].from.segment_id, 0);
    assert_eq!(own_transitions[0].to.segment_id, 1);
    assert_eq!(own_transitions[0].from.point_with_id.coords, expected_own_transition[0]);
    assert_eq!(own_transitions[0].to.point_with_id.coords, expected_own_transition[1]);
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
    training.rotation.rotated = Some(arr2(&[
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0]
    ]));


    // node transitions
    let expected_node_transition = [arr1(&[-1.0, -1.0]), arr1(&[1.0, -1.0])];
    let node_transitions = training.build_segments();
    assert_eq!(node_transitions.len(), 1);
    assert_eq!(node_transitions[&1].len(), 1);
    assert_eq!(node_transitions[&1][0].from.segment_id, 2);
    assert_eq!(node_transitions[&1][0].to.segment_id, 3);
    assert_eq!(node_transitions[&1][0].from.point_with_id.coords, expected_node_transition[0]);
    assert_eq!(node_transitions[&1][0].to.point_with_id.coords, expected_node_transition[1]);

    // own transitions
    let expected_own_transition = [arr1(&[1.0, 1.0]), arr1(&[-1.0, 1.0])];
    let own_transitions = training.segmentation.segments.clone();
    assert_eq!(own_transitions.len(), 2);
    assert_eq!(own_transitions[0].from.segment_id, 0);
    assert_eq!(own_transitions[0].to.segment_id, 1);
    assert_eq!(own_transitions[0].from.point_with_id.coords, expected_own_transition[0]);
    assert_eq!(own_transitions[0].to.point_with_id.coords, expected_own_transition[1]);
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
    training.rotation.rotated = Some(arr2(&[
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0]
    ]));

    let _node_transitions = training.build_segments();

    assert_eq!(training.segmentation.send_point.unwrap().point_with_id.coords, arr1(&[1.0, 1.0]));
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
    training.rotation.rotated = Some(arr2(&[
        [1.0, 1.0],
        [-1.0, 1.0],
        [-1.0, -1.0],
        [1.0, -1.0]
    ]));

    let _node_transitions = training.build_segments();

    assert!(training.segmentation.send_point.is_none());
}
