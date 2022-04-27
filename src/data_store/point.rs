use crate::data_store::utils::get_segment_id;
use crate::utils::PolarCoords;
use ndarray::Array1;
#[cfg(test)]
use ndarray::ArrayView1;
use ndarray_stats::QuantileExt;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter};
use std::sync::{Arc, Mutex};

#[derive(Clone, Serialize, Deserialize, Debug)]
pub(crate) struct Point {
    id: usize,
    coordinates: Array1<f32>,
    segment: usize,
}

impl Point {
    pub fn new(id: usize, coordinates: Array1<f32>, segment: usize) -> Self {
        Self {
            id,
            coordinates,
            segment,
        }
    }

    pub fn new_calculate_segment(id: usize, coordinates: Array1<f32>, n_segments: usize) -> Self {
        let segment = get_segment_id(coordinates.to_polar()[1], n_segments);
        Self::new(id, coordinates, segment)
    }

    pub fn calculate_segment_id(&self, n_segments: usize) -> usize {
        get_segment_id(self.coordinates.to_polar()[1], n_segments)
    }

    pub fn get_id(&self) -> usize {
        self.id
    }

    #[cfg(test)]
    pub fn get_coordinates_view(&self) -> ArrayView1<f32> {
        self.coordinates.view()
    }

    pub fn get_segment(&self) -> usize {
        self.segment
    }

    pub fn into_ref(self) -> PointRef {
        PointRef::new(self)
    }

    pub fn mirror(&mut self, n_segments: usize) {
        if let Some(x_coord) = self.coordinates.get_mut(0) {
            *x_coord *= -1.0;
        }
        self.segment = self.calculate_segment_id(n_segments)
    }
}

impl Display for Point {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Point({}-({}) => {})",
            self.id, self.segment, self.coordinates
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PointRef(Arc<Mutex<Point>>);

impl PointRef {
    pub fn new(point: Point) -> Self {
        Self(Arc::new(Mutex::new(point)))
    }

    pub fn get_id(&self) -> usize {
        self.0.lock().unwrap().get_id()
    }

    pub fn get_max_coordinate(&self) -> f32 {
        let point = self.0.lock().unwrap();
        *point.coordinates.max().unwrap()
    }

    pub fn get_min_coordinate(&self) -> f32 {
        let point = self.0.lock().unwrap();
        *point.coordinates.min().unwrap()
    }

    pub fn clone_coordinates(&self) -> Array1<f32> {
        self.0.lock().unwrap().coordinates.clone()
    }

    pub fn get_segment(&self) -> usize {
        self.0.lock().unwrap().get_segment()
    }

    pub fn mirror(&mut self, n_segments: usize) {
        self.0.lock().unwrap().mirror(n_segments)
    }

    pub fn deref_clone(&self) -> Point {
        self.0.lock().unwrap().clone()
    }

    pub fn get_dims(&self) -> usize {
        self.0.lock().unwrap().coordinates.len()
    }
}
