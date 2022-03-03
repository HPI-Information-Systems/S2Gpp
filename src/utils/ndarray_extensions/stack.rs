use ndarray::{Array, ArrayView, Axis, Dimension, RemoveAxis, stack};
use anyhow::Result;

pub(crate) trait Stack<A, O> {
    fn stack(self, axis: Axis) -> Result<Array<A, O>>;
}

impl<A, D, O> Stack<A, O> for Vec<Array<A, D>>
    where
        A: Copy,
        D: Dimension<Larger = O>,
        O: RemoveAxis
{
    fn stack(self, axis: Axis) -> Result<Array<A, O>> {
        let view_vec: Vec<ArrayView<A, D>> = self.iter().map(Array::view).collect();
        Ok(stack(axis, view_vec.as_slice())?)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2, Array1, Array2, Axis};
    use crate::utils::ndarray_extensions::stack::Stack;

    #[test]
    fn combine_dimensions() {
        let arrays: Vec<Array1<f32>> = vec![
            arr1(&[1., 2., 3.]),
            arr1(&[1., 2., 3.])
        ];
        let expected: Array2<f32> = arr2(&[[1., 1.], [2., 2.], [3., 3.]]);
        let combined = arrays.stack(Axis(1)).unwrap();
        assert_eq!(combined, expected)
    }
}
