use crate::parameters::Parameters;
use crate::training::Training;
use crate::training::transposition::Transposer;
use crate::utils::{Edge, NodeName};

#[test]
fn test_assignments() {
    let mut parameters = Parameters::default();
    parameters.n_cluster_nodes = 2;
    let mut training = Training::new(parameters);

    training.transposition.range_start_point = Some(0);
    training.transposition.partition_len = Some(2);

    training.edge_estimation.edges = vec![
        (0, Edge(NodeName(0, 0), NodeName(1, 0))),
        (1, Edge(NodeName(1, 0), NodeName(2, 0))),
        (1, Edge(NodeName(2, 0), NodeName(3, 0))),
        (2, Edge(NodeName(3, 0), NodeName(4, 0))),
        (3, Edge(NodeName(4, 0), NodeName(5, 0)))
    ];

    let assignments = training.assign_edges_to_neighbours();
    println!("{:?}", assignments);
    assert_eq!(assignments.get(&0).unwrap().len(), 3);
    assert_eq!(assignments.get(&1).unwrap().len(), 2);
}
