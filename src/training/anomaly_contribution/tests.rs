use crate::data_store::node::IndependentNode;
use crate::training::anomaly_contribution::AnomalyContribution;
use ndarray::{arr1, arr2};

#[test]
fn test() {
    let nodes = vec![
        IndependentNode::new(0, 0, 0).into_ref(),
        IndependentNode::new(0, 1, 0).into_ref(),
        IndependentNode::new(0, 2, 0).into_ref(),
        IndependentNode::new(0, 3, 0).into_ref(),
    ];

    let cluster_centers = arr2(&[[2., 0., 0.], [0., 0., 2.], [-2., 0., 0.], [0., 0., -2.]]);

    let expected = vec![
        arr1(&[0.75_f32, 0.5_f32]),
        arr1(&[0.5, 0.75]),
        arr1(&[0.75_f32, 0.5_f32]),
        arr1(&[0.5, 0.75]),
    ];

    let mut ac = AnomalyContribution::default();
    ac.record_contributions(nodes.clone(), cluster_centers, vec![1, 1, 1, 1]);

    for (node, exp) in nodes.iter().zip(expected) {
        let contribution = ac.node_contribution.remove(node).unwrap();
        assert_eq!(contribution, exp)
    }
}

#[test]
fn test_dim_combination() {
    let cc = arr2(&[
        [0.5, 2.0, 2.0, 4.0, 4.0],
        [0.5, 2.0, 2.0, 4.0, 4.0],
        [0.5, 2.0, 2.0, 4.0, 4.0],
    ]);

    let expected = arr2(&[[0.5, 4.0, 8.0], [0.5, 4.0, 8.0], [0.5, 4.0, 8.0]]);

    let ac = AnomalyContribution::default();
    let combined = ac.combine_dimensions(cc);
    assert_eq!(combined.unwrap(), expected)
}
