use std::hash::{Hash, Hasher};
use std::ops::Mul;
use ndarray::ArrayView1;
use num_traits::{Float};

#[repr(transparent)]
#[derive(Debug, Clone)]
pub(crate) struct FloatApprox<A>(A);

impl<A> FloatApprox<A>
    where
        A: Mul + Float
{
    pub fn approximate(&self, tolerance: usize) -> isize {
        self.0.mul(A::from(10 * tolerance).unwrap()).to_isize().unwrap()
    }

    pub fn from_array_view_clone(array: ArrayView1<A>) -> Vec<Self> {
        array.into_iter().map(|x| FloatApprox((*x).clone())).collect()
    }

    #[allow(dead_code)]
    pub fn from_array_view(array: ArrayView1<A>) -> Vec<FloatApprox<&A>> {
        array.into_iter().map(|x| FloatApprox(x)).collect()
    }
}

impl<A> Mul for FloatApprox<A>
    where
        A: Mul + Float
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.mul(rhs.0))
    }
}

impl<A: PartialEq<A>> PartialEq<Self> for FloatApprox<A> {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl Eq for FloatApprox<f32> {}
impl Eq for FloatApprox<f64> {}

impl<A> Hash for FloatApprox<A>
    where
        A: Mul + Float
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.approximate(8).hash(state);
    }
}


#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use ndarray::aview1;
    use crate::utils::float_approx::{FloatApprox};

    #[test]
    fn approx_float() {
        let a = FloatApprox(0.2);
        let b = FloatApprox(2.0);
        assert_eq!(a * b, FloatApprox(0.4))
    }

    #[test]
    fn approx_float_hash_key() {
        let a = FloatApprox(0.1);
        let mut set = HashSet::new();
        set.insert(a);
        assert_eq!(set.into_iter().last().unwrap(), FloatApprox(0.1))
    }

    #[test]
    fn from_array_view() {
        let arr = aview1(&[1., 2., 3.]);
        let expect = vec![FloatApprox(&1.), FloatApprox(&2.), FloatApprox(&3.)];
        assert_eq!(FloatApprox::from_array_view(arr), expect)
    }

    #[test]
    fn from_array_view_clone() {
        let arr = aview1(&[1., 2., 3.]);
        let expect = vec![FloatApprox(1.), FloatApprox(2.), FloatApprox(3.)];
        assert_eq!(FloatApprox::from_array_view_clone(arr), expect)
    }
}
