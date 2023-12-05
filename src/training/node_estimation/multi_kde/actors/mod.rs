use crate::messages::PoisonPill;
use crate::training::node_estimation::ClusteringResponse;
use crate::training::node_estimation::multi_kde::actors::gaussian_kde::GaussianKDEActor;
use crate::training::node_estimation::multi_kde::actors::messages::{
    GaussianKDEMessage, GaussianKDEResponse, MultiKDEMessage,
};
use crate::training::node_estimation::multi_kde::MultiKDEBase;
use crate::utils::pop_clear::PopClear;
use crate::utils::stack::Stack;
use actix::{Actor, ActorContext, Addr, AsyncContext, Context, Handler, Recipient};
use ndarray::{s, ArcArray2, Array1, Array2, Axis};
use ndarray_stats::QuantileExt;

mod gaussian_kde;
mod helper;
pub(crate) mod messages;
#[cfg(test)]
mod tests;

pub(in crate::training::node_estimation) struct MultiKDEActor {
    receiver: Recipient<ClusteringResponse<f32>>,
    n_threads: usize,
    multi_kde_base: MultiKDEBase,
    data: Option<Array2<f32>>,
    next_dim: usize,
    current_column: Option<ArcArray2<f32>>,
    cluster_centers: Vec<Array1<f32>>,
    gaussian_kde: Option<Addr<GaussianKDEActor>>,
}

impl MultiKDEActor {
    pub fn new(receiver: Recipient<ClusteringResponse<f32>>, n_threads: usize) -> Self {
        Self {
            receiver,
            n_threads,
            multi_kde_base: MultiKDEBase::default(),
            data: None,
            next_dim: 0,
            current_column: None,
            cluster_centers: vec![],
            gaussian_kde: None,
        }
    }

    fn cluster_next_dim(&mut self, ctx: &mut Context<Self>) {
        let data = self.data.as_ref().expect("Data is not yet received");
        let gkde = self.gaussian_kde.as_ref().unwrap();
        if self.next_dim < data.shape()[1] {
            let column = data
                .slice(s![.., self.next_dim..self.next_dim + 1])
                .to_shared();
            gkde.do_send(GaussianKDEMessage {
                column: column.clone(),
            });
            self.current_column = Some(column);
            self.next_dim += 1;
        } else {
            let cluster_centers_vec = self.cluster_centers.pop_clear();
            let cluster_centers = cluster_centers_vec.stack(Axis(1)).unwrap();
            let (labels, cluster_centers) = self
                .multi_kde_base
                .extract_labels_from_centers(cluster_centers);
            self.receiver
                .do_send(ClusteringResponse {
                    cluster_centers,
                    labels,
                });
            self.gaussian_kde.as_ref().unwrap().do_send(PoisonPill);
            ctx.stop();
        }
    }

    fn find_cluster_centers(&mut self, kernel_estimate: Array1<f32>, ctx: &mut Context<Self>) {
        let column = self
            .current_column
            .as_ref()
            .expect("You are not working on any column at the moment");
        let grid_min = *column.min().unwrap();
        let grid_max = *column.max().unwrap();
        let peaks =
            self.multi_kde_base
                .find_peak_values(kernel_estimate.view(), grid_min, grid_max);
        let assigned_peak_values = if !peaks.is_empty() {
            self.multi_kde_base
                .assign_closest_peak_values(column.view(), peaks)
        } else {
            Array1::from(vec![0.0; column.len()])
        };
        self.cluster_centers.push(assigned_peak_values);

        self.cluster_next_dim(ctx);
    }
}

impl Actor for MultiKDEActor {
    type Context = Context<Self>;
}

impl Handler<MultiKDEMessage> for MultiKDEActor {
    type Result = ();

    fn handle(&mut self, msg: MultiKDEMessage, ctx: &mut Self::Context) -> Self::Result {
        self.data = Some(msg.data);
        self.gaussian_kde =
            Some(GaussianKDEActor::new(self.n_threads, 250, ctx.address().recipient()).start());
        self.cluster_next_dim(ctx);
    }
}

impl Handler<GaussianKDEResponse> for MultiKDEActor {
    type Result = ();

    fn handle(&mut self, msg: GaussianKDEResponse, ctx: &mut Self::Context) -> Self::Result {
        self.find_cluster_centers(msg.kernel_estimate, ctx);
    }
}

impl Handler<PoisonPill> for MultiKDEActor {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
