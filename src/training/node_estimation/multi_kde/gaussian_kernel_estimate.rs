use std::f32::consts::PI;
use std::ops::{Div, Mul};
use ndarray::{Array2, ArrayView2, Dim};
use ndarray_linalg::Cholesky;
use ndarray_linalg::UPLO::Lower;
use anyhow::Result;

// todo: parallelize
/// Inspired by scipy's [gaussian_kernel_estimate](https://github.com/scipy/scipy/blob/8a64c938ddf1ae4c02a08d2c5e38daeb8d061d38/scipy/stats/_stats.pyx#L693)
pub(in crate::training::node_estimation::multi_kde) fn gaussian_kernel_estimate(points: ArrayView2<f32>, weights: ArrayView2<f32>, grid: Array2<f32>, precision: ArrayView2<f32>) -> Result<Array2<f32>> {
    let n = points.shape()[0];
    let d = points.shape()[1];
    let m = grid.shape()[0];
    let p = weights.shape()[1];

    assert_eq!(grid.shape()[1], d, "points and grid must have the same shape at dimension 1: {} != {}", grid.shape()[1], d);
    assert!((precision.shape()[0] == d) && (precision.shape()[1] == d), "precision matrix must match point dimensions");

    let whitening = precision.cholesky(Lower)?;
    let white_points = points.dot(&whitening);
    let white_grid = grid.dot(&whitening);

    let mut norm = (2. * PI).powf((d as f32).mul(-1.).div(2.));
    for i in 0..d {
        norm *= whitening[[i, i]];
    }

    let mut estimate = Array2::zeros(Dim([m, p]));
    for i in 0..n {
        for j in 0..m {
            let mut arg = 0.;
            for k in 0..d {
                let residual = white_points[[i, k]] - white_grid[[j, k]];
                arg += residual * residual;
            }
            arg = ((-arg).div(2.0).exp()) * norm;
            for k in 0..p {
                estimate[[j, k]] += weights[[i, k]] * arg;
            }
        }
    }

    Ok(estimate)
}


#[cfg(test)]
mod tests {
    use ndarray::{arr2};
    use ndarray_linalg::assert_close_l1;
    use crate::training::node_estimation::multi_kde::gaussian_kernel_estimate::gaussian_kernel_estimate;

    #[test]
    fn test_1d() {
        let points = arr2(&[[1., 2., 3.]]);
        let weights = arr2(&[[1., 2., 3.]]);
        let grid = arr2(&[[1., 2., 3.]]);
        let precision = arr2(&[[0.5]]);
        let expected = arr2(&[[1.0328167 ], [1.44297216], [1.38945254]]);

        let result = gaussian_kernel_estimate(points.t(), weights.t(), grid.t().to_owned(), precision.view()).unwrap();
        assert_close_l1!(&result, &expected, 0.0000001)
    }
}
