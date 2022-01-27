use std::iter::{Enumerate, Skip};
use std::slice::Iter;
use log::*;

pub(crate) struct FromTo<I> {
    iter: Enumerate<Skip<I>>,
    from: usize,
    to: usize
}

impl<I> FromTo<I> {
    pub fn new(iter: Enumerate<Skip<I>>, from: usize, to: usize) -> Self {
        Self {
            iter,
            from,
            to
        }
    }
}

impl<I: Iterator> Iterator for FromTo<I> {
    type Item = <I as Iterator>::Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some((index, item)) => if (index + self.from) < self.to {
                Some(item)
            } else {
                None
            },

            None => None
        }

    }
}


pub(crate) trait FromToAble where Self: Iterator
{
    fn fromto(self, from: usize, to: usize) -> FromTo<Self> where Self: Sized {
        if from >= to {
            warn!("FromTo Iterator will be empty, because from({}) >= to({})", from, to)
        }
        FromTo::new(self.skip(from).enumerate(), from, to)
    }
}


impl<T> FromToAble for Iter<'_, T> {}
