#[cfg(test)]
mod tests;

use crate::parameters::Parameters;
use ndarray::{s, Array2, Array3, ArrayBase, Axis};

use crate::data_manager::stats_collector::DatasetStats;
use crate::utils::linspace;

pub struct ReferenceDatasetBuilder {
    data_stats: DatasetStats,
    parameters: Parameters,
}

impl ReferenceDatasetBuilder {
    pub fn new(data_stats: DatasetStats, parameters: Parameters) -> Self {
        Self {
            data_stats,
            parameters,
        }
    }

    pub fn build(&self) -> Array3<f32> {
        let min_cols = self.data_stats.min_col.as_ref().unwrap();
        let max_cols = self.data_stats.max_col.as_ref().unwrap();

        let length = 100;
        let width = self.parameters.pattern_length - self.parameters.latent;
        let dim = min_cols.len();

        let mut data_ref = ArrayBase::zeros((length, width, dim));
        let mut tmp: Array2<f32> = ArrayBase::zeros((width, dim));
        let mut t: Array2<f32> = ArrayBase::zeros((self.parameters.pattern_length, dim));

        for (i, v) in linspace(min_cols.clone(), max_cols.clone(), length)
            .axis_iter(Axis(1))
            .enumerate()
        {
            tmp.fill(0.0);
            t.fill(0.0);
            t = t + v;

            for j in 0..width {
                tmp.slice_mut(s![j, ..]).assign(
                    &t.slice(s![j..j + self.parameters.latent, ..])
                        .sum_axis(Axis(0)),
                );
            }

            data_ref.index_axis_mut(Axis(0), i).assign(&tmp);
        }

        data_ref
    }
}
