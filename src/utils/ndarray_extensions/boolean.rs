use ndarray::{ArrayBase, Data, Dimension};
use std::ops::{BitAnd, BitOr};

pub(crate) trait BooleanCollectives {
    fn all(&self) -> bool;
    fn any(&self) -> bool;
}

impl<S, D> BooleanCollectives for ArrayBase<S, D>
where
    S: Data<Elem = bool>,
    D: Dimension,
{
    fn all(&self) -> bool {
        self.fold(true, |accum, item| accum.bitand(item))
    }

    fn any(&self) -> bool {
        self.fold(false, |accum, item| accum.bitor(item))
    }
}
