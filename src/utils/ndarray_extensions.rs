use ndarray::*;
use ndarray_linalg::Norm;
use num_traits::Float;
use std::fmt::Debug;
use std::iter::FromIterator;

pub fn norm(a: ArrayView2<f32>, axis: Axis) -> Array1<f32> {
    a.axis_iter(Axis(1 - axis.0)).map(|x| x.norm()).collect()
}

pub fn cross2d<A: Float>(a: ArrayView2<A>, b: ArrayView2<A>, axisa: Axis, axisb: Axis) -> Array2<A> {
    let axisa_other = Axis(1 - axisa.0);
    let axisb_other = Axis(1 - axisb.0);

    assert_eq!(a.shape()[axisa.0], 3);
    assert_eq!(b.shape()[axisb_other.0], 1);

    let crosses: Vec<Array1<A>> = a.axis_iter(axisa_other).map(|a_|
        cross1d(a_, b.index_axis(axisb_other, 0))
    ).collect();
    stack_new_axis(axisa, crosses.iter().map(|x| x.view()).collect::<Vec<ArrayView1<A>>>().as_slice()).unwrap()
}

fn cross1d<A: Float>(a: ArrayView1<A>, b: ArrayView1<A>) -> Array1<A> {
    arr1(&[
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])
}

pub fn repeat<A: Copy + Debug>(a: ArrayView2<A>, n: usize) -> Array1<A> {
    let flat = a.into_shape(a.len()).unwrap();
    let c = concatenate(Axis(0), vec![flat.view(); n].as_slice()).unwrap()
        .into_shape((n, a.len())).unwrap();
    Array::from_iter(c.t().iter().cloned())
}

pub fn linspace<A>(start: Array1<A>, end: Array1<A>, n: usize) -> Array2<A>
where
    A: Float
{
    assert!(start.len() == end.len());

    let linspaces: Vec<Array1<A>> = start.iter().zip(end.iter()).map(|(s,e)| {
        Array::linspace(s.clone(), e.clone(), n)
    }).collect();

    stack(Axis(0), linspaces.iter().map(|x| x.view()).collect::<Vec<ArrayView1<A>>>().as_slice()).unwrap()
}


pub trait Stats<A>
{
    fn min_axis(&self, axis: Axis) -> Array1<A>;
    fn max_axis(&self, axis: Axis) -> Array1<A>;
}

impl<A> Stats<A> for ArcArray2<A>
where
    A: Float
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


#[cfg(test)]
mod tests {
    use ndarray::{arr2, Axis};
    use crate::utils::ndarray_extensions::cross2d;
    use ndarray_linalg::close_l1;

    #[test]
    fn test_cross_product() {
        let a = arr2(
            &[[-9.99998576e-01, -9.99959697e-01, -9.99994247e-01],
                  [9.45265121e-04,  8.29841247e-03, -2.44777798e-03],
                  [1.39796979e-03,  3.42653727e-03,  2.34816447e-03]]
        );
        let b = arr2(&[[0., 0., 1.]]);
        let expected = arr2(
            &[[9.45265121e-04,  9.99998576e-01, -0.00000000e+00],
                [8.29841247e-03,  9.99959697e-01, -0.00000000e+00],
                [-2.44777798e-03,  9.99994247e-01,  0.00000000e+00]]
        );

        let actual = cross2d(a.view(), b.t(), Axis(0), Axis(0));
        close_l1(&actual, &expected, 0.005);
    }
}
