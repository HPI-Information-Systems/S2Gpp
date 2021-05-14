use actix::prelude::*;
use ndarray::{ArcArray2, Array2, Axis, ArrayBase, Array, Array3};
use crate::parameters::Parameters;
use num_traits::real::Real;

pub struct ReferenceDatasetBuilder {
    data: ArcArray2<f32>,
    parameters: Parameters
}

impl ReferenceDatasetBuilder {
    pub fn new(data: ArcArray2<f32>, parameters: Parameters) -> Self {
        Self {
            data,
            parameters
        }
    }

    pub fn build(&self) -> Array3<f32> {
        let min_cols = self.data.map_axis(Axis(0), |x| x.min());
        let max_cols = self.data.map_axis(Axis(0), |x| x.max());

        let length = 100;
        let width = self.parameters.pattern_length - self.parameters.latent;
        let dim = self.data.shape()[1];

        let mut data_ref = ArrayBase::zeros((length, width, dim));
        let mut tmp = ArrayBase::zeros((width, dim));
        let mut T = ArrayBase::zeros((self.parameters.pattern_length, dim));
        for (i, v) in Array::linspace(min_cols, max_cols, length).enumerate() {
            tmp.fill(0.0);
            T.fill(0.0);
            T = T + v;
            for j in 0..width {
                tmp[j] = T.slice(s![j..j+self.parameters.latent, ..]).sum_axis(Axis(0));
            }
            data_ref[i] = tmp.clone();
        }

        data_ref
    }
}