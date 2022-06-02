use crate::data_store::edge::Edge;
use crate::data_store::node::IndependentNode;
use crate::parameters::Parameters;
use crate::training::transposition::Transposer;
use crate::training::Training;
use crate::SyncInterface;

#[test]
fn test_assignments() {
    let mut parameters = Parameters::default();
    parameters.n_cluster_nodes = 2;
    let mut training = Training::init(parameters);

    training.transposition.range_start_point = Some(0);
    training.transposition.partition_len = Some(2);

    training.data_store.add_edges(vec![
        Edge::new(
            IndependentNode::new(0, 0, 0).into_ref(),
            IndependentNode::new(1, 0, 0).into_ref(),
        ),
        Edge::new(
            IndependentNode::new(1, 0, 1).into_ref(),
            IndependentNode::new(2, 0, 1).into_ref(),
        ),
        Edge::new(
            IndependentNode::new(2, 0, 1).into_ref(),
            IndependentNode::new(3, 0, 1).into_ref(),
        ),
        Edge::new(
            IndependentNode::new(3, 0, 2).into_ref(),
            IndependentNode::new(4, 0, 2).into_ref(),
        ),
        Edge::new(
            IndependentNode::new(4, 0, 3).into_ref(),
            IndependentNode::new(5, 0, 3).into_ref(),
        ),
    ]);

    let assignments = training.assign_edges_to_neighbours();
    let edges = training.data_store.get_edges();
    assert_eq!(edges.len(), 3);
    assert_eq!(assignments.get(&1).unwrap().len(), 2);
}
