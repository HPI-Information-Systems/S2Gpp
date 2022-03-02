use std::f32::consts::PI;
use ndarray::{Array1, Array2, ArrayView2, Dim, s};
use ndarray_linalg::{Cholesky, Inverse};
use ndarray_linalg::UPLO::Lower;
use ndarray_stats::CorrelationExt;
use anyhow::Result;

use crate::training::node_estimation::multi_kde::gaussian_kernel_estimate::gaussian_kernel_estimate;
use crate::utils::FloatFunctions;


/// Inspired by scipy's [gaussian_kde](https://github.com/scipy/scipy/blob/b5d8bab88af61d61de09641243848df63380a67f/scipy/stats/_kde.py#L495)
#[derive(Debug, Clone)]
pub(in crate::training::node_estimation::multi_kde) struct GaussianKDEBase<'a> {
    data: ArrayView2<'a, f32>,
    covariance: Option<Array2<f32>>,
    inv_covariance: Option<Array2<f32>>,
    weights: Option<Array2<f32>>,
    log_det: Option<f32>
}

impl<'a> GaussianKDEBase<'a> {
    pub fn new(data: ArrayView2<'a, f32>) -> Self {
        let mut kde = Self {
            data,
            covariance: None,
            inv_covariance: None,
            weights: None,
            log_det: None
        };
        kde.compute_covariance();
        kde
    }

    pub fn evaluate(&self, points: Array2<f32>) -> Result<Array1<f32>> {
        let result = gaussian_kernel_estimate(
            self.data.view(),
            self.weights.as_ref().expect("Before evaluating, the weights must be calculated!").view(),
            points,
            self.inv_covariance.as_ref().expect("Before evaluating, the inverse of the covariance must be calculated!").view()
        )?;

        Ok(result.slice(s![.., 0]).into_owned())
    }

    fn compute_covariance(&mut self) {
        self.calculate_weights();
        let factor = self.scotts_factor();

        let covariance = self.data.t().cov(1.).unwrap();
        let covariance_inv = covariance.inv().unwrap(); // todo: catch exception
        self.covariance = Some(covariance * factor.powi(2));
        self.inv_covariance = Some(covariance_inv / factor.powi(2));

        let l: Array2<f32> = (self.covariance.as_ref().unwrap() * 2.0 * PI).cholesky(Lower).unwrap();
        self.log_det = Some(2.0 * l.diag().into_owned().ln().sum());
    }

    fn scotts_factor(&self) -> f32 {
        let d = self.data.shape()[1];
        let exponent = -1.0 / ((d+4) as f32);
        self.neff().powf(exponent)
    }

    fn neff(&self) -> f32 {
        let weights = self.weights.as_ref().unwrap();
        1.0 / weights.clone().powi(2).sum()
    }

    fn calculate_weights(&mut self) {
        let n = self.data.shape()[0];
        self.weights = Some(Array2::ones(Dim([n, 1])) / (n as f32));
    }
}


mod tests {
    use ndarray::{arr1, arr2, Array1, Array2};
    use ndarray_linalg::assert_close_l1;
    use crate::training::node_estimation::multi_kde::gaussian_kde::GaussianKDEBase;

    fn setup() -> (Array2<f32>, Array2<f32>, Array1<f32>) {
        let data = arr2(&[[2., 2.1, 2.2, 8., 8.1, 8.2]]).t().into_owned();
        let grid = arr2(&[[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]]).t().into_owned();
        let expected: Array1<f32> = arr1(
            &[0.0573441 , 0.07811948, 0.08925389, 0.08777485, 0.07935172,
                0.07411137, 0.07774745, 0.0863332 , 0.0899083 , 0.08132743]
        );
        (data, grid, expected)
    }

    #[test]
    fn correct_fit() {
        let (data, _, _) = setup();
        let covariance = arr2(&[[5.27818777]]);
        let inv_covariance = arr2(&[[0.18945897]]);
        let log_det = 3.5014598793760214;

        let gkde = GaussianKDEBase::new(data.view());

        assert_eq!(gkde.covariance.unwrap(), covariance);
        assert_eq!(gkde.inv_covariance.unwrap(), inv_covariance);
        assert_eq!(gkde.log_det.unwrap(), log_det);
    }

    #[test]
    fn correct_evaluate() {
        let (data, grid, expected) = setup();
        let gkde = GaussianKDEBase::new(data.view());
        let result = gkde.evaluate(grid).unwrap();
        assert_close_l1!(&result, &expected, 0.0000001);
    }
}
