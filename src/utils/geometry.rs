use ndarray::{Array2, s, Dim, Array1, concatenate, Axis, ShapeError};
use ndarray_linalg::solve::Inverse;
use ndarray_linalg::error::LinalgError;

#[derive(Debug)]
pub struct IntersectionError;

impl From<ShapeError> for IntersectionError {
    fn from(_: ShapeError) -> Self {
        Self
    }
}

impl From<LinalgError> for IntersectionError {
    fn from(_: LinalgError) -> Self {
        Self
    }
}

pub fn line_plane_intersection(line_points: Array2<f32>, plane_points: Array2<f32>) -> Result<Array1<f32>, IntersectionError> {
    let line_vector: Array1<f32> = line_points.slice(s![1, ..]).to_owned() - line_points.slice(s![0, ..]).to_owned();
    let new_dim = Dim((plane_points.shape()[0] - 1, plane_points.shape()[1]));
    let plane_vector: Array2<f32> = plane_points.slice(s![1.., ..]).to_owned() - plane_points.slice(s![0, ..]).broadcast(Dim(new_dim)).ok_or_else(|| IntersectionError)?;
    let new_dim = Dim((1, line_vector.shape()[0]));
    let vectors = concatenate(Axis(0), &[(line_vector.clone() * -1.).broadcast(new_dim).unwrap(), plane_vector.view()])?;
    let inv = vectors.inv()?;
    let vec_start = line_points.slice(s![0, ..]).to_owned() - plane_points.slice(s![0, ..]).to_owned();
    let vec_to_intersection: Array1<f32> = inv.t().dot(&vec_start);
    let intersection = line_points.slice(s![0, ..]).to_owned() + line_vector * vec_to_intersection.slice(s![0]);
    Ok(intersection)
}


#[cfg(test)]
mod tests {
    use ndarray::{arr2, arr1};
    use crate::utils::geometry::line_plane_intersection;
    use ndarray_linalg::close_l1;

    #[test]
    fn calculate_intersection() {
        let line_points = arr2(&[
            [-1., -1., -1.],
            [1., 1., 1.]
        ]);

        let plane_points = arr2(&[
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 0., 1.]
        ]);

        let intersections = line_plane_intersection(line_points, plane_points).unwrap();
        let expected = arr1(&[0., 0., 0.]);

        close_l1(&intersections, &expected, 0.00001)
    }
}
