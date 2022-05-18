mod messages;
#[cfg(test)]
mod tests;

use crate::data_store::node::NodeRef;
pub(crate) use crate::training::anomaly_contribution::messages::{
    ClusterCenterMessage, QueryClusterContribution, QueryClusterContributionResponse,
    QueryClustercontributionDone,
};
use crate::utils::float_approx::FloatApprox;
use actix::{Actor, Context, Handler};
use anyhow::{Error, Result};
use ndarray::{concatenate, s, stack, Array1, Array2, ArrayView1, Axis, Dim};
use ndarray_stats::SummaryStatisticsExt;
use num_traits::ToPrimitive;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ops::{Div, Sub};

#[derive(Default, Debug)]
pub(crate) struct AnomalyContribution {
    node_contribution: HashMap<NodeRef, Array1<f32>>,
    query_response: Option<Vec<Array1<f32>>>,
    n_dims: Option<usize>,
}

impl AnomalyContribution {
    pub(crate) fn record_contributions(
        &mut self,
        nodes: Vec<NodeRef>,
        cluster_centers: Array2<f32>,
        label_counts: Vec<usize>,
    ) {
        let adapted_cluster_centers = cluster_centers;//self.combine_dimensions(cluster_centers).unwrap();
        let dim_scores = self.calculate_dimension_uniqueness(adapted_cluster_centers, label_counts);
        match &self.n_dims {
            None => self.n_dims = Some(dim_scores.shape()[1]),
            Some(_) => (),
        }

        for node in nodes {
            let dim_score = dim_scores.row(node.get_cluster());
            self.node_contribution.insert(node, dim_score.to_owned());
        }
    }

    fn calculate_dimension_uniqueness(
        &self,
        cluster_centers: Array2<f32>,
        label_counts: Vec<usize>,
    ) -> Array2<f32> {
        let n_intersections: f32 = label_counts.iter().map(|x| x.to_f32().unwrap()).sum();
        let result_shape = cluster_centers.shape();
        let mut result = Array2::zeros([result_shape[0], result_shape[1]]);
        for d in 0..cluster_centers.shape()[1] {
            let dim = cluster_centers.column(d);
            let mut counter = HashMap::new();
            for (i, label_count) in label_counts.iter().enumerate() {
                let dim_v = FloatApprox(*dim.get(i).unwrap());
                if let Entry::Vacant(e) = counter.entry(dim_v.clone()) {
                    e.insert(*label_count);
                } else {
                    let v = counter.get_mut(&dim_v).unwrap();
                    *v += *label_count;
                }
            }

            let vector: Vec<f32> = dim
                .iter()
                .map(|row| {
                    1_f32
                        - (counter.get(&FloatApprox(*row)).unwrap().to_f32().unwrap()
                            / n_intersections)
                })
                .collect();
            result.slice_mut(s![.., d]).assign(&Array1::from(vector));
        }
        result
    }

    #[allow(dead_code)]
    fn calculate_distances(
        &self,
        centers: Array2<f32>,
        label_counts: Vec<usize>,
    ) -> Result<Array2<f32>> {
        let label_counts_f32: Vec<f32> = label_counts.iter().filter_map(usize::to_f32).collect();
        let counts = Array1::from(label_counts_f32);
        let mean = centers
            .weighted_mean_axis(Axis(0), &counts)?
            .insert_axis(Axis(0));
        let centers_shape = Vec::from(centers.shape());
        let distances = centers
            .sub(
                mean.broadcast([centers_shape[0], centers_shape[1]])
                    .ok_or_else(|| {
                        Error::msg("Could not broadcast means to cluster center shape")
                    })?,
            )
            .mapv(f32::abs);
        Ok(distances)
    }

    /// A multidimensional time series with $d$ dimensions results in a multidimensional projection
    /// with $2d$ dimensions. The intersection coordinates have $2d-1$ dimensions, because the first
    /// dimension is the distance of the intersection in $d_1$ and $d_2$ to the origin. Thus, we must
    /// transform the remaining dimensions $2d - 2$ into a $d$-dimensional array. Thereby, we always
    /// add the coordinates together.
    #[allow(dead_code)]
    fn combine_dimensions(&self, cluster_centers: Array2<f32>) -> Result<Array2<f32>> {
        let cc_shape = cluster_centers.shape();
        let first_dim = cluster_centers.column(0).insert_axis(Axis(1));
        let remaining_dims = cluster_centers.slice(s![.., 1..cc_shape[1]]);
        let shape = remaining_dims.shape();
        let new_shape = Dim([shape[0] * shape[1].div(2), 2]);
        let reshaped_remaining = remaining_dims.to_owned().into_shape(new_shape)?;
        let remaining_dims = reshaped_remaining.sum_axis(Axis(1));
        let remaining_dims = remaining_dims
            .view()
            .into_shape(Dim([shape[0], shape[1].div(2)]))?;
        Ok(concatenate(Axis(1), &[first_dim, remaining_dims])?)
    }

    fn query_node_score(&mut self, nodes: Vec<NodeRef>) {
        let mut contributions: Vec<ArrayView1<f32>> = nodes
            .iter()
            .map(|node| {
                self.node_contribution
                    .get(node)
                    .expect("Node is not registered")
                    .view()
            })
            .collect();

        let contribution = if contributions.is_empty() {
            let n_dims = *self.n_dims.as_ref().unwrap();
            Array1::zeros([n_dims])
        } else {
            contributions = contributions
                .into_iter()
                .filter(|x| !x.sum().is_nan())
                .collect();
            stack(Axis(1), contributions.as_slice())
                .unwrap()
                .mean_axis(Axis(1))
                .unwrap()
        };

        match &mut self.query_response {
            Some(query_response) => query_response.push(contribution),
            None => {
                self.query_response = Some(vec![contribution]);
            }
        };
    }
}

impl Actor for AnomalyContribution {
    type Context = Context<Self>;
}

impl Handler<ClusterCenterMessage> for AnomalyContribution {
    type Result = ();

    fn handle(&mut self, msg: ClusterCenterMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.record_contributions(msg.nodes, msg.cluster_centers, msg.label_counts)
    }
}

impl Handler<QueryClusterContribution> for AnomalyContribution {
    type Result = ();

    fn handle(&mut self, msg: QueryClusterContribution, _ctx: &mut Self::Context) -> Self::Result {
        self.query_node_score(msg.nodes)
    }
}

impl Handler<QueryClustercontributionDone> for AnomalyContribution {
    type Result = ();

    fn handle(
        &mut self,
        msg: QueryClustercontributionDone,
        _ctx: &mut Self::Context,
    ) -> Self::Result {
        msg.receiver
            .do_send(QueryClusterContributionResponse {
                contributions: self.query_response.take().unwrap(),
            })
            .unwrap()
    }
}
