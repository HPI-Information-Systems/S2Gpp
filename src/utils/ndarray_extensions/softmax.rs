use anyhow::{Error, Result};
use ndarray::{Array, ArrayBase, Axis, Data, Dim};
use num_traits::Float;
use std::fmt::Debug;
use std::ops::Div;

pub(crate) trait Softmax<A, D>
where
    A: Copy + Float + Debug,
{
    fn softmax(&self) -> Result<Array<A, D>>;
    fn softmax_axis(&self, axis: Axis) -> Result<Array<A, D>>;
}

impl<A, S> Softmax<A, Dim<[usize; 1]>> for ArrayBase<S, Dim<[usize; 1]>>
where
    A: Copy + Float + Debug,
    S: Data<Elem = A>,
{
    fn softmax(&self) -> Result<Array<A, Dim<[usize; 1]>>> {
        let max = self.sum();
        Ok(self.mapv(|x| x.div(max)))
    }

    fn softmax_axis(&self, axis: Axis) -> Result<Array<A, Dim<[usize; 1]>>> {
        let max = self.sum_axis(axis).insert_axis(axis);
        let shape = self.shape();
        Ok(self.div(
            &max.broadcast([shape[0]])
                .ok_or_else(|| Error::msg("Could not broadcast max"))?,
        ))
    }
}

impl<A, S> Softmax<A, Dim<[usize; 2]>> for ArrayBase<S, Dim<[usize; 2]>>
where
    A: Copy + Float + Debug,
    S: Data<Elem = A>,
{
    fn softmax(&self) -> Result<Array<A, Dim<[usize; 2]>>> {
        let max = self.sum();
        Ok(self.mapv(|x| x.div(max)))
    }

    fn softmax_axis(&self, axis: Axis) -> Result<Array<A, Dim<[usize; 2]>>> {
        let max = self.sum_axis(axis).insert_axis(axis);
        let shape = self.shape();
        Ok(self.div(
            &max.broadcast([shape[0], shape[1]])
                .ok_or_else(|| Error::msg("Could not broadcast max"))?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::ndarray_extensions::softmax::Softmax;
    use ndarray::{arr1, arr2, Axis};

    #[test]
    fn correct_result_1d() {
        let a = arr1(&[1., 2., 1.]);
        let expected = arr1(&[0.25, 0.5, 0.25]);
        assert_eq!(a.softmax().unwrap(), expected)
    }

    #[test]
    fn correct_result_2d() {
        let a = arr2(&[[1., 2., 1.]]);
        let expected = arr2(&[[0.25, 0.5, 0.25]]);
        assert_eq!(a.softmax_axis(Axis(1)).unwrap(), expected)
    }
}
