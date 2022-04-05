use crate::utils::ndarray_extensions::index_arr::IndexArr;
use ndarray::{concatenate, Array, ArrayBase, Axis, Data, Dim};
use std::cmp::Ordering;
use std::iter::FromIterator;

pub(crate) trait Shift<A, D> {
    fn shift(&self, by: isize, axis: Axis) -> Array<A, D>;
}

impl<A, S> Shift<A, Dim<[usize; 1]>> for ArrayBase<S, Dim<[usize; 1]>>
where
    A: Copy,
    S: Data<Elem = A>,
{
    fn shift(&self, by: isize, axis: Axis) -> Array<A, Dim<[usize; 1]>> {
        let shape = self.shape();
        let axis_int = axis.0;
        let loc = match by.cmp(&0) {
            Ordering::Greater => {
                let first = Array::zeros(Dim([by as usize]));
                let second = Array::from_iter(0..(shape[axis_int] - by as usize));
                concatenate![Axis(0), first, second]
            }
            Ordering::Less => {
                let by = (-by) as usize;
                let first = Array::from_iter(by..shape[axis_int]);
                let second = Array::zeros(Dim([by])) + (shape[axis_int] - 1);
                concatenate![Axis(0), first, second]
            }
            Ordering::Equal => Array::from_iter(0..shape[axis_int]),
        };

        self.get_multiple(loc, axis).unwrap()
    }
}

impl<A, S> Shift<A, Dim<[usize; 2]>> for ArrayBase<S, Dim<[usize; 2]>>
where
    A: Copy,
    S: Data<Elem = A>,
{
    fn shift(&self, by: isize, axis: Axis) -> Array<A, Dim<[usize; 2]>> {
        let shape = self.shape();
        let axis_int = axis.0;
        let loc = match by.cmp(&0) {
            Ordering::Greater => {
                let first = Array::zeros(Dim([by as usize]));
                let second = Array::from_iter(0..(shape[axis_int] - by as usize));
                concatenate![Axis(0), first, second]
            }
            Ordering::Less => {
                let by = (-by) as usize;
                let first = Array::from_iter(by..shape[axis_int]);
                let second = Array::zeros(Dim([by])) + shape[axis_int];
                concatenate![Axis(0), first, second]
            }
            Ordering::Equal => Array::from_iter(0..shape[axis_int]),
        };

        self.get_multiple(loc, axis).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::ndarray_extensions::shift::Shift;
    use ndarray::{arr1, arr2, Axis};

    #[test]
    fn shift_forward_array1() {
        let arr = arr1(&[0., 1., 2., 3.]);
        let expected = arr1(&[0., 0., 1., 2.]);
        assert_eq!(arr.shift(1, Axis(0)), expected)
    }

    #[test]
    fn shift_forward_array2_axis0() {
        let arr = arr2(&[[0.], [1.], [2.], [3.]]);
        let expected = arr2(&[[0.], [0.], [1.], [2.]]);
        assert_eq!(arr.shift(1, Axis(0)), expected)
    }

    #[test]
    fn shift_forward_array2_axis1() {
        let arr = arr2(&[[0., 1., 2., 3.], [0., 1., 2., 3.]]);
        let expected = arr2(&[[0., 0., 1., 2.], [0., 0., 1., 2.]]);
        assert_eq!(arr.shift(1, Axis(1)), expected)
    }

    #[test]
    fn shift_backward_array1() {
        let arr = arr1(&[0., 1., 2., 3.]);
        let expected = arr1(&[1., 2., 3., 3.]);
        assert_eq!(arr.shift(-1, Axis(0)), expected)
    }
}
