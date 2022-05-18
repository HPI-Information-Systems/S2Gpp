use crate::data_store::node::NodeRef;
use actix::{Message, Recipient};
use ndarray::{Array1, Array2};

#[derive(Message)]
#[rtype(Result = "()")]
pub(crate) struct ClusterCenterMessage {
    pub cluster_centers: Array2<f32>,
    pub nodes: Vec<NodeRef>,
    pub label_counts: Vec<usize>,
}

#[derive(Message)]
#[rtype(result = "()")]
pub(crate) struct QueryClusterContribution {
    pub nodes: Vec<NodeRef>,
}

#[derive(Message)]
#[rtype(result = "()")]
pub(crate) struct QueryClustercontributionDone {
    pub receiver: Recipient<QueryClusterContributionResponse>,
}

#[derive(Message)]
#[rtype(result = "()")]
pub(crate) struct QueryClusterContributionResponse {
    pub contributions: Vec<Array1<f32>>,
}
