use ndarray::{Array1, Array2};
use std::collections::HashMap;

#[derive(Clone, Hash, Eq, PartialEq, Debug)]
pub struct Transition(pub usize, pub usize);


pub type IntersectionsByTransition = HashMap<usize, Array1<f32>>;
