pub(crate) mod boolean;
mod index_arr;
mod shift;
pub(crate) mod stack;

use ndarray::*;
use ndarray_linalg::Norm;
use num_traits::Float;
use std::fmt::Debug;
use std::iter::FromIterator;

pub fn norm(a: ArrayView2<f32>, axis: Axis) -> Array1<f32> {
    a.axis_iter(Axis(1 - axis.0)).map(|x| x.norm()).collect()
}

pub fn cross2d<A: Float>(
    a: ArrayView2<A>,
    b: ArrayView2<A>,
    axisa: Axis,
    axisb: Axis,
) -> Array2<A> {
    let axisa_other = Axis(1 - axisa.0);
    let axisb_other = Axis(1 - axisb.0);

    assert_eq!(a.shape()[axisa.0], 3);
    assert_eq!(b.shape()[axisb_other.0], 1);

    let crosses: Vec<Array1<A>> = a
        .axis_iter(axisa_other)
        .map(|a_| cross1d(a_, b.index_axis(axisb_other, 0)))
        .collect();
    stack_new_axis(
        axisa,
        crosses
            .iter()
            .map(|x| x.view())
            .collect::<Vec<ArrayView1<A>>>()
            .as_slice(),
    )
    .unwrap()
}

fn cross1d<A: Float>(a: ArrayView1<A>, b: ArrayView1<A>) -> Array1<A> {
    arr1(&[
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
}

pub fn repeat<A: Copy + Debug>(a: ArrayView2<A>, n: usize) -> Array1<A> {
    let flat = a.into_shape(a.len()).unwrap();
    let c = concatenate(Axis(0), vec![flat.view(); n].as_slice())
        .unwrap()
        .into_shape((n, a.len()))
        .unwrap();
    Array::from_iter(c.t().iter().cloned())
}

pub fn linspace<A>(start: Array1<A>, end: Array1<A>, n: usize) -> Array2<A>
where
    A: Float,
{
    assert!(start.len() == end.len());

    let linspaces: Vec<Array1<A>> = start
        .iter()
        .zip(end.iter())
        .map(|(s, e)| Array::linspace(*s, *e, n))
        .collect();

    stack(
        Axis(0),
        linspaces
            .iter()
            .map(|x| x.view())
            .collect::<Vec<ArrayView1<A>>>()
            .as_slice(),
    )
    .unwrap()
}

pub trait Stats<A> {
    fn min_axis(&self, axis: Axis) -> Array1<A>;
    fn max_axis(&self, axis: Axis) -> Array1<A>;
}

impl<A> Stats<A> for ArcArray2<A>
where
    A: Float,
{
    fn min_axis(&self, axis: Axis) -> Array1<A> {
        let _n = self.len_of(axis);
        let mut res = Array::zeros(self.raw_dim().remove_axis(axis));
        let stride = self.strides()[1 - axis.index()];
        if self.ndim() == 2 && stride == 1 {
            let ax = axis.index();
            for (i, elt) in res.iter_mut().enumerate() {
                let mut smallest = A::max_value();
                for v in self.index_axis(Axis(1 - ax), i).iter() {
                    if smallest.gt(v) {
                        smallest = *v
                    }
                }
                *elt = smallest;
            }
        } else {
            panic!("Not yet implemented!")
        }
        res
    }

    fn max_axis(&self, axis: Axis) -> Array1<A> {
        let mut res = Array::zeros(self.raw_dim().remove_axis(axis));
        let stride = self.strides()[1 - axis.index()];
        if self.ndim() == 2 && stride == 1 {
            let ax = axis.index();
            for (i, elt) in res.iter_mut().enumerate() {
                let mut largest = A::min_value();
                for v in self.index_axis(Axis(1 - ax), i).iter() {
                    if largest.lt(v) {
                        largest = *v
                    }
                }
                *elt = largest;
            }
        } else {
            panic!("Not yet implemented!")
        }
        res
    }
}

pub trait PolarCoords<A>
where
    A: Float,
{
    /// Takes the first two values as x and y respectively
    fn to_polar(&self) -> Array1<A>;
    fn to_cartesian(&self) -> Array1<A>;
}

impl<S, A> PolarCoords<A> for ArrayBase<S, Dim<[usize; 1]>>
where
    S: Data<Elem = A>,
    A: Float + std::ops::Mul<Output = A>,
{
    fn to_polar(&self) -> Array1<A> {
        let x = &self[0];
        let y = &self[1];
        let radius = (x.powi(2) + y.powi(2)).sqrt();
        let theta = y.atan2(*x);
        arr1(&[radius, theta])
    }

    fn to_cartesian(&self) -> Array1<A> {
        let radius = &self[0];
        let theta = &self[1];
        let x = radius.mul(theta.cos());
        let y = radius.mul(theta.sin());
        arr1(&[x, y])
    }
}

pub trait FloatFunctions<A, D>
where
    A: Float,
    D: Dimension,
{
    fn ln(self) -> Self;
    fn powi(self, exponent: i32) -> Self;
}

impl<S, A, D> FloatFunctions<A, D> for ArrayBase<S, D>
where
    S: DataMut<Elem = A>,
    A: Float + std::ops::Mul<Output = A>,
    D: Dimension,
{
    fn ln(mut self) -> Self {
        for v in self.iter_mut() {
            *v = v.ln();
        }
        self
    }

    fn powi(mut self, exponent: i32) -> Self {
        for v in self.iter_mut() {
            *v = v.powi(exponent);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::ndarray_extensions::cross2d;
    use ndarray::{arr2, Axis};
    use ndarray_linalg::close_l1;

    #[test]
    fn test_cross_product() {
        let a = arr2(&[
            [-9.99998576e-01, -9.99959697e-01, -9.99994247e-01],
            [9.45265121e-04, 8.29841247e-03, -2.44777798e-03],
            [1.39796979e-03, 3.42653727e-03, 2.34816447e-03],
        ]);
        let b = arr2(&[[0., 0., 1.]]);
        let expected = arr2(&[
            [9.45265121e-04, 9.99998576e-01, -0.00000000e+00],
            [8.29841247e-03, 9.99959697e-01, -0.00000000e+00],
            [-2.44777798e-03, 9.99994247e-01, 0.00000000e+00],
        ]);

        let actual = cross2d(a.view(), b.t(), Axis(0), Axis(0));
        close_l1(&actual, &expected, 0.005);
    }
}
