use std::ops::{AddAssign, Mul};
use actix::{Actor, ActorContext, Addr, AsyncContext, Context, Handler, Recipient, SyncArbiter};
use ndarray::{ArcArray2, Array, Array1, Array2, ArrayView2, Axis, Dim};
use ndarray_linalg::Inverse;
use ndarray_stats::{CorrelationExt, QuantileExt};
use num_integer::Integer;
use crate::messages::PoisonPill;
use crate::training::node_estimation::multi_kde::actors::helper::EstimatorHelper;
use crate::training::node_estimation::multi_kde::actors::messages::{EstimatorResponse, EstimatorTask, GaussianKDEMessage, GaussianKDEResponse};
use crate::utils::HelperProtocol;


pub(in crate::training::node_estimation::multi_kde::actors) struct GaussianKDEActor {
    n_threads: usize,
    resolution: usize,
    data: Option<ArcArray2<f32>>,
    helper: Option<Addr<EstimatorHelper>>,
    helper_protocol: HelperProtocol,
    estimate: Option<Array1<f32>>,
    receiver: Option<Recipient<GaussianKDEResponse>>
}

impl GaussianKDEActor {
    pub fn new(n_threads: usize, resolution: usize, receiver: Recipient<GaussianKDEResponse>) -> Self {
        Self {
            n_threads,
            resolution,
            receiver: Some(receiver),
            ..Default::default()
        }
    }

    fn estimate(&mut self, data: ArcArray2<f32>, ctx: &mut Context<Self>) {
        let grid_min = data.min().unwrap().clone();
        let grid_max = data.max().unwrap().clone();
        let padding = (grid_max - grid_min).mul(0.1);
        let grid = Array::linspace(grid_min - padding, grid_max + padding, self.resolution).insert_axis(Axis(1));
        self.data = Some(data);

        let weights = self.calculate_weights();
        let scotts_factor = self.scotts_factor(weights.view());
        let precision = self.compute_covariance(scotts_factor);
        self.evaluate(grid, weights, precision, ctx);
    }

    fn evaluate(&mut self, grid: Array2<f32>, weights: Array2<f32>, precision: Array2<f32>, ctx: &mut Context<Self>) {
        let data = (*self.data.as_ref().unwrap()).clone();
        let receiver = ctx.address().recipient();
        self.helper = Some(SyncArbiter::start(self.n_threads, move || {
            EstimatorHelper::new(data.clone(), weights.clone(), grid.clone(), precision.clone(), receiver.clone())
        }));
        self.helper_protocol.n_total = self.n_threads;

        self.send_tasks();
    }

    fn send_tasks(&mut self) {
        let n = self.data.as_ref().unwrap().shape()[0];
        let chunk_size = n.div_floor(&self.n_threads);
        for t in 0..self.n_threads {
            let start = t.mul(chunk_size);
            let end = if (t + 1) < self.n_threads {
                (t+1).mul(chunk_size)
            } else {
                n
            };
            self.helper.as_ref().unwrap().do_send(EstimatorTask {
                data_range: start..end
            });
            self.helper_protocol.sent();
        }
    }

    fn compute_covariance(&self, factor: f32) -> Array2<f32> {
        let covariance = self.data.as_ref().unwrap().t().cov(1.).unwrap();
        let covariance_inv = covariance.inv().unwrap(); // todo: catch exception
        covariance_inv / factor.powi(2)
    }

    fn scotts_factor(&self, weights: ArrayView2<f32>) -> f32 {
        let d = self.data.as_ref().unwrap().shape()[1];
        let exponent = -1.0 / ((d+4) as f32);
        self.neff(weights).powf(exponent)
    }

    fn neff(&self, weights: ArrayView2<f32>) -> f32 {
        let weights_sum: f32 = weights.iter().map(|w| w.powi(2)).sum();
        1.0 / weights_sum
    }

    fn calculate_weights(&self) -> Array2<f32>{
        let n = self.data.as_ref().unwrap().shape()[0];
        Array2::ones(Dim([n, 1])) / (n as f32)
    }
}

impl Default for GaussianKDEActor {
    fn default() -> Self {
        Self {
            n_threads: 1,
            resolution: 250,
            data: None,
            helper: None,
            helper_protocol: HelperProtocol::default(),
            estimate: None,
            receiver: None
        }
    }
}

impl Actor for GaussianKDEActor {
    type Context = Context<Self>;
}

impl Handler<GaussianKDEMessage> for GaussianKDEActor {
    type Result = ();

    fn handle(&mut self, msg: GaussianKDEMessage, ctx: &mut Self::Context) -> Self::Result {
        self.estimate(msg.column, ctx);
    }
}

impl Handler<EstimatorResponse> for GaussianKDEActor {
    type Result = ();

    fn handle(&mut self, msg: EstimatorResponse, ctx: &mut Self::Context) -> Self::Result {
        self.helper_protocol.received();
        match self.estimate.as_mut() {
            Some(estimate) => estimate.add_assign(&msg.estimate),
            None => {
                self.estimate = Some(msg.estimate);
            }
        }

        if !self.helper_protocol.is_running() {
            let estimate = (*self.estimate.as_ref().unwrap()).clone();
            self.estimate = None;

            self.receiver.as_ref().unwrap().do_send(GaussianKDEResponse {
                kernel_estimate: estimate,
                source: ctx.address().recipient()
            }).unwrap();
        }
    }
}

impl Handler<PoisonPill> for GaussianKDEActor {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop()
    }
}
