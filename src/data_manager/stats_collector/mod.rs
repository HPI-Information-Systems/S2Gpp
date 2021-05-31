mod std;
mod messages;
mod minmax;
#[cfg(test)]
mod tests;

use ndarray::{Array1, ArcArray2, Array2, s};
use crate::utils::ClusterNodes;
use actix::prelude::*;
pub use crate::data_manager::stats_collector::std::{StdCalculator, StdCalculation};
pub use crate::data_manager::stats_collector::messages::*;
pub use crate::data_manager::stats_collector::minmax::{MinMaxCalculator, MinMaxCalculation};
use crate::parameters::Parameters;

#[derive(Default, Clone)]
pub struct DatasetStats {
    pub min_col: Option<Array1<f32>>,
    pub max_col: Option<Array1<f32>>,
    pub std_col: Option<Array1<f32>>
}

impl DatasetStats {
    pub fn new(std_col: Array1<f32>, min_col: Array1<f32>, max_col: Array1<f32>) -> Self {
        Self {
            min_col: Some(min_col),
            max_col: Some(max_col),
            std_col: Some(std_col)
        }
    }

    pub fn is_done(&self) -> bool {
        match (&self.std_col, &self.min_col, &self.max_col) {
            (Some(_), Some(_), Some(_)) => true,
            _ => false
        }
    }
}
