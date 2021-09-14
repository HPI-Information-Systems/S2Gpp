#[cfg(test)]
mod tests;

use ndarray::{ArcArray2, Array3, ArrayBase, Axis, s, Array2};
use crate::parameters::Parameters;

pub struct PhaseSpacer {
    data: ArcArray2<f32>,
    parameters: Parameters
}

impl PhaseSpacer {
    pub fn new(data: ArcArray2<f32>, parameters: Parameters) -> Self {
        Self {
            data,
            parameters
        }
    }

    pub fn build(&self) -> Array3<f32> {
        let mut phase_space = ArrayBase::zeros(
            (
                self.data.shape()[0] - (self.parameters.pattern_length - 1),
                self.parameters.pattern_length - self.parameters.latent,
                self.data.shape()[1]
            )
        );

        let shape = (self.parameters.pattern_length - self.parameters.latent, self.data.shape()[1]);
        let mut current_seq: Array2<f32> = ArrayBase::zeros(shape);
        let mut tmp = ArrayBase::zeros(shape);
        let mut first = true;

        for i in 0..phase_space.shape()[0] {
            tmp.fill(0.0);
            if first {
                first = false;
                for j in 0..(self.parameters.pattern_length - self.parameters.latent) {
                    tmp.index_axis_mut(Axis(0), j).assign(&self.data.slice(s![(i + j)..(i + j + self.parameters.latent), ..]).sum_axis(Axis(0)));
                }
                phase_space.index_axis_mut(Axis(0), i).assign(&tmp);
                current_seq = tmp.clone();
            } else {
                tmp.slice_mut(s![..-1, ..]).assign(&current_seq.slice(s![1.., ..]));
                tmp.slice_mut(s![-1, ..]).assign(&self.data.slice(s![(i + self.parameters.pattern_length - self.parameters.latent - 1)..(i + self.parameters.pattern_length - 1), ..]).sum_axis(Axis(0)));
                phase_space.index_axis_mut(Axis(0), i).assign(&tmp);
                current_seq = tmp.clone();
            }
        }

        phase_space
    }
}
