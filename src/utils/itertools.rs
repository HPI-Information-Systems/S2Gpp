use log::*;
use ndarray::{Array1, Dimension};
use std::iter::{Enumerate, Skip};
use std::slice::Iter;

pub(crate) struct FromTo<I> {
    iter: Enumerate<Skip<I>>,
    from: usize,
    to: usize,
}

impl<I> FromTo<I> {
    pub fn new(iter: Enumerate<Skip<I>>, from: usize, to: usize) -> Self {
        Self { iter, from, to }
    }
}

impl<I: Iterator> Iterator for FromTo<I> {
    type Item = <I as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some((index, item)) => {
                if (index + self.from) < self.to {
                    Some(item)
                } else {
                    None
                }
            }

            None => None,
        }
    }
}

pub(crate) trait FromToAble
where
    Self: Iterator,
{
    fn fromto(self, from: usize, to: usize) -> FromTo<Self>
    where
        Self: Sized,
    {
        if from >= to {
            warn!(
                "FromTo Iterator will be empty, because from({}) >= to({})",
                from, to
            )
        }
        FromTo::new(self.skip(from).enumerate(), from, to)
    }
}

impl<T> FromToAble for Iter<'_, T> {}
impl<T, D: Dimension> FromToAble for ndarray::iter::Iter<'_, T, D> {}

pub(crate) trait LengthAble {
    fn get_length(&self) -> usize;
}

impl LengthAble for Vec<f32> {
    fn get_length(&self) -> usize {
        self.len()
    }
}

impl LengthAble for Array1<f32> {
    fn get_length(&self) -> usize {
        self.len()
    }
}
