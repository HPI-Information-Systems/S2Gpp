use std::fmt::{Display, Formatter};
use std::sync::Arc;
use ndarray::{Array1, ArrayView1};
use crate::utils::PolarCoords;
use serde::{Serialize, Deserialize};
use crate::data_store::utils::get_segment_id;


#[derive(Clone, Serialize, Deserialize, Debug)]
pub(crate) struct Point {
    id: usize,
    coordinates: Array1<f32>,
    segment: usize
}

impl Point {
    pub fn new(id: usize, coordinates: Array1<f32>, segment: usize) -> Self {
        Self {
            id,
            coordinates,
            segment
        }
    }

    pub fn new_calculate_segment(id: usize, coordinates: Array1<f32>, n_segments: usize) -> Self {
        let segment = get_segment_id(coordinates.to_polar()[1], n_segments);
        Self::new(id, coordinates, segment)
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    pub fn get_coordinates(&self) -> ArrayView1<f32> {
        self.coordinates.view()
    }

    pub fn get_segment(&self) -> usize {
        self.segment
    }
}

impl Display for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Point({}-({}) => {})", self.id, self.segment, self.coordinates)
    }
}


pub(crate) type PointRef = Arc<Point>;
