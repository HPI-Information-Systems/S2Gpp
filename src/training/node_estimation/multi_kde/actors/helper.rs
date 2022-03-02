use std::f32::consts::PI;
use std::ops::{Div, Mul, Range};
use actix::{Actor, Handler, Recipient, SyncContext};
use ndarray::{ArcArray2, Array1, Array2, Dim, s};
use ndarray_linalg::Cholesky;
use ndarray_linalg::UPLO::Lower;
use crate::training::node_estimation::multi_kde::actors::messages::{EstimatorResponse, EstimatorTask};

pub(in crate::training::node_estimation::multi_kde::actors) struct EstimatorHelper {
    pub(crate) data: ArcArray2<f32>,
    grid: Array2<f32>,
    whitening_factors: Array2<f32>,
    weights: Array2<f32>,
    norm: f32,
    receiver: Recipient<EstimatorResponse>
}

impl EstimatorHelper {
    pub fn new(data: ArcArray2<f32>, weights: Array2<f32>, grid: Array2<f32>, precision: Array2<f32>, receiver: Recipient<EstimatorResponse>) -> Self {
        let d = data.shape()[1];
        assert_eq!(grid.shape()[1], d, "points and grid must have the same shape at dimension 1: {} != {}", grid.shape()[1], d);
        assert!((precision.shape()[0] == d) && (precision.shape()[1] == d), "precision matrix must match point dimensions");

        let whitening = precision.cholesky(Lower).unwrap();
        let white_grid = grid.dot(&whitening);

        let mut norm = (2. * PI).powf((d as f32).mul(-1.).div(2.));
        for i in 0..d {
            norm *= whitening[[i, i]];
        }

        Self {
            data,
            grid: white_grid,
            whitening_factors: whitening,
            weights,
            norm,
            receiver
        }
    }

    fn evaluate(&self, range: Range<usize>) -> Array1<f32> {
        let offset = range.start;
        let ranged_data = self.data.slice(s![range, ..]);
        let white_data = ranged_data.dot(&self.whitening_factors);

        let n = white_data.shape()[0];
        let d = white_data.shape()[1];
        let g = self.grid.shape()[0];
        let w = self.weights.shape()[1];

        let mut estimate = Array2::zeros(Dim([g, w]));
        for i in 0..n {
            let o = i + offset;
            for j in 0..g {
                let mut arg = 0.;
                for k in 0..d {
                    let residual: f32 = white_data[[i, k]] - self.grid[[j, k]];
                    arg += residual.powi(2);
                }
                arg = ((-arg).div(2.0).exp()) * self.norm;
                for k in 0..w {
                    estimate[[j, k]] += self.weights[[o, k]] * arg;
                }
            }
        }
        estimate.slice(s![.., 0]).into_owned()
    }
}

impl Actor for EstimatorHelper {
    type Context = SyncContext<Self>;
}

impl Handler<EstimatorTask> for EstimatorHelper {
    type Result = ();

    fn handle(&mut self, msg: EstimatorTask, _ctx: &mut Self::Context) -> Self::Result {
        let estimate = self.evaluate(msg.data_range);
        self.receiver.do_send(EstimatorResponse { estimate }).unwrap();
    }
}
