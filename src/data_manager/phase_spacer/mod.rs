use ndarray::{ArcArray2, Array3, ArrayBase, Axis};
use crate::parameters::Parameters;
use core::slice::SlicePattern;

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
                self.data.shape()[0] - (self.parameters.pattern_length - self.parameters.latent),
                self.parameters.pattern_length - self.parameters.latent,
                self.data.shape()[1]
            )
        );
        let shape = (self.parameters.pattern_length - self.parameters.latent, self.data.shape()[1]);
        let mut current_seq = ArrayBase::zeros(shape);
        let mut tmp = ArrayBase::zeros(shape);
        let mut first = true;

        for i in 0..phase_space.shape()[0] {
            tmp.fill(0.0);
            if first {
                first = false;
                for j in 0..(self.parameters.pattern_length - self.parameters.latent) {
                    tmp[j] = self.data.slice(s![(i + j)..(i + j + self.parameters.latent)]).sum_axis(Axis(0));
                }
                phase_space[i] = tmp.clone();
                current_seq = tmp.clone();
            } else {
                tmp.slice(s![..-1, ..]) = current_seq.slice(s![1.., ..]);
                tmp[-1] = self.data.slice(s![(i + self.parameters.pattern_length - self.parameters.latent)..(i + self.parameters.pattern_length)]).sum_axis(Axis(0));
                phase_space[i] = tmp.clone();
                current_seq = tmp.clone();
            }
        }

        phase_space
    }
}