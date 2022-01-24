use crate::training::Training;
use crate::parameters::Parameters;
use ndarray::arr1;
use crate::training::scoring::Scorer;
use std::path::Path;
use std::fs::remove_file;
use crate::data_store::edge::Edge;
use crate::data_store::node::IndependentNode;
use crate::training::scoring::weights::ScoringWeights;

#[test]
fn scores_are_written_to_file() {
    let mut training = Training::new(Parameters::default());
    training.scoring.score = Some(arr1(&[1.,1.,1.,1.,1.]));

    let scores_path = "data/_test_scores.csv";
    training.output_score(scores_path.to_string()).unwrap();
    let path = Path::new(scores_path);
    assert!(path.exists());
    remove_file(path).expect("Could not delete test file!");
}


#[test]
fn node_degrees_correctly_calculated() {
    let mut training = Training::new(Parameters::default());
    let edges = vec![
        Edge::new(IndependentNode::new(0, 0, 0).to_ref(), IndependentNode::new(0, 1, 1).to_ref()),
        Edge::new(IndependentNode::new(0, 1, 1).to_ref(), IndependentNode::new(1, 1, 2).to_ref()),
        Edge::new(IndependentNode::new(0, 0, 2).to_ref(), IndependentNode::new(1, 1, 3).to_ref()),
        Edge::new(IndependentNode::new(0, 0, 3).to_ref(), IndependentNode::new(0, 1, 4).to_ref()),
    ];

    let mut expected_node_degrees = vec![
        ((0, 0), 2),
        ((0, 1), 2),
        ((1, 1), 2)
    ];

    training.data_store.add_edges(edges);
    let node_degrees = training.calculate_node_degrees();

    let mut real_degrees: Vec<((usize, usize), usize)> = node_degrees.iter().map(|(node, count)| ((node.get_segment_id(), node.get_cluster()), count.clone())).collect();

    real_degrees.sort();
    expected_node_degrees.sort();

    assert_eq!(real_degrees, expected_node_degrees);
}
