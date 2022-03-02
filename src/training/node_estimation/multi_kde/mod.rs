use std::collections::{HashMap};
use std::ops::{BitAnd, BitAndAssign, Div, Mul, Sub};
use ndarray::{arr1, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_stats::{QuantileExt};
use anyhow::Result;
use num_traits::ToPrimitive;

use crate::training::node_estimation::multi_kde::gaussian_kde::GaussianKDEBase;
use crate::utils::boolean::BooleanCollectives;
use crate::utils::float_approx::FloatApprox;
use crate::utils::itertools::{FromToAble};
use crate::utils::stack::Stack;

mod gaussian_kde;
mod gaussian_kernel_estimate;
pub(crate) mod actors;


#[derive(Debug, Clone)]
pub(crate) struct MultiKDEBase {
    resolution: usize,
    peak_order: usize,
}


impl MultiKDEBase {
    pub fn new(resolution: usize, peak_order: usize) -> Self {
        Self {
            resolution,
            peak_order,
        }
    }

    pub fn cluster(&self, data: ArrayView2<f32>) -> Result<Array1<usize>> {
        let n_dims = data.shape()[1];
        let mut cluster_centers = Vec::with_capacity(n_dims);
        for points in data.axis_iter(Axis(1)) {
            let points = points.insert_axis(Axis(1));
            let gkde = GaussianKDEBase::new(points.clone());
            let grid_min = points.min()?.clone();
            let grid_max = points.max()?.clone();
            let grid = Array::linspace(grid_min, grid_max, self.resolution).insert_axis(Axis(1));
            let kernel_estimate = gkde.evaluate(grid)?;
            let peaks = self.find_peak_values(kernel_estimate.view(), grid_min, grid_max);
            let assigned_peak_values = if peaks.len() > 0 {
                self.assign_closest_peak_values(points, peaks)
            } else {
                Array1::from(vec![0.0; points.len()])
            };
            cluster_centers.push(assigned_peak_values);
        }
        let cluster_centers = cluster_centers.stack(Axis(1))?;
        let labels = self.extract_labels_from_centers(cluster_centers);
        Ok(Array::from(labels))
    }

    fn find_peak_values(&self, kernel_estimate: ArrayView1<f32>, grid_min: f32, grid_max: f32) -> Vec<f32> {
        let step_size = ((grid_max + 1.) - grid_min).div(self.resolution as f32);
        let result = self.find_peak_index(kernel_estimate)
            .iter()
            .map(|i| grid_min + step_size.mul(i.to_f32().unwrap()))
            .collect();
        result
    }

    fn find_peak_index(&self, kernel_estimate: ArrayView1<f32>) -> Vec<usize> {
        let mut results: Array1<bool> = arr1(vec![true; kernel_estimate.len()].as_slice());
        let datalen = results.len();
        for shift in 1..self.peak_order+1 {
            let last_element = vec![&kernel_estimate[datalen - 1]; shift];
            let first_element = vec![kernel_estimate[0]; shift];
            let plus_iter = kernel_estimate.iter().fromto(shift, datalen).chain(last_element);
            let minus_iter = first_element.iter().chain(kernel_estimate.iter().fromto(0, datalen - shift));

            let current_results = kernel_estimate.iter().zip(plus_iter).zip(minus_iter).map(|((main, plus), minus)|
                main.gt(plus).bitand(main.gt(minus))
            ).collect::<Array1<bool>>();

            results.bitand_assign(&current_results);

            if !results.any() {
                break
            }
        }

        results.indexed_iter()
            .filter_map(|(index, bit)| if *bit { Some(index) } else { None })
            .collect()
    }

    fn assign_closest_peak_values(&self, points: ArrayView2<f32>, peaks: Vec<f32>) -> Array1<f32> {
        let n_points = points.len();
        let n_peaks = peaks.len();
        let broadcast_shape = [n_points, n_peaks];
        let mut peaks_arr = Array1::from(peaks.clone()).broadcast(broadcast_shape).unwrap().to_owned();
        let broadcast_points = points.broadcast(broadcast_shape).unwrap();
        peaks_arr = peaks_arr.sub(broadcast_points);
        let p = peaks_arr.mapv(f32::abs).map_axis(Axis(1), |d| peaks[d.argmin().unwrap()]);
        p
    }

    fn extract_labels_from_centers(&self, cluster_centers: Array2<f32>) -> Vec<usize> {
        let mut key = 0;
        let mut unique_cluster_centers: HashMap<Vec<FloatApprox<f32>>, usize> = HashMap::new();
        let mut labels = vec![];
        for center in cluster_centers.axis_iter(Axis(0)) {
            let approx_center = FloatApprox::from_array_view_clone(center);
            match &unique_cluster_centers.get(&approx_center) {
                Some(k) => labels.push((*k).clone()),
                None => {
                    unique_cluster_centers.insert(approx_center, key);
                    labels.push(key);
                    key += 1;
                }
            }
        }
        labels
    }
}


impl Default for MultiKDEBase {
    fn default() -> Self {
        Self {
            resolution: 250,
            peak_order: 1,
        }
    }
}


mod tests {
    use ndarray::{arr1, arr2, Array1, Axis};
    use crate::data_manager::data_reader::read_data_;
    use crate::training::node_estimation::multi_kde::MultiKDEBase;

    #[test]
    fn find_peak() {
        let a = arr1(&[1., 2., 1.]);
        let expected = vec![1];
        let mkde = MultiKDEBase::default();
        let indices = mkde.find_peak_index(a.view());
        assert_eq!(indices, expected)
    }

    #[test]
    fn multiple_peaks() {
        let a = arr1(&[1., 2., 1., 2., 1.]);
        let expected = vec![1, 3];
        let mkde = MultiKDEBase::default();
        let indices = mkde.find_peak_index(a.view());
        assert_eq!(indices, expected)
    }

    #[test]
    fn no_peaks() {
        let a = arr1(&[1., 1., 1.]);
        let expected = vec![];
        let mkde = MultiKDEBase::default();
        let indices = mkde.find_peak_index(a.view());
        assert_eq!(indices, expected)
    }

    #[test]
    fn find_correct_peak_values() {
        let a = arr1(&[1., 2., 1., 2., 1.]);
        let expected = vec![1., 3.];
        let mkde = MultiKDEBase::new(5, 1);
        let peaks = mkde.find_peak_values(a.view(), 0., 4.);
        assert_eq!(peaks, expected)
    }

    #[test]
    fn assign_correct_peaks() {
        let points = arr2(&[[1., 1.], [2., 1.], [2.5, 1.], [4., 1.], [5., 1.], [5.5, 1.]]);
        let peaks = vec![2., 5.];
        let expected = arr1(&[2., 2., 2., 5., 5., 5.]);
        let mkde = MultiKDEBase::default();
        for col in points.axis_iter(Axis(1)) {
            let assignments = mkde.assign_closest_peak_values(col.insert_axis(Axis(1)), peaks.clone());
            assert_eq!(assignments, expected);
            break
        }
    }

    #[test]
    fn returns_labels_from_cluster_centers() {
        let arr = arr2(&[
            [1., 2.],
            [2., 1.],
            [1., 2.]
        ]);
        let expected = vec![0, 1, 0];
        let mkde = MultiKDEBase::default();
        let labels = mkde.extract_labels_from_centers(arr);
        assert_eq!(labels, expected)
    }

    #[test]
    fn multidim_clustering() {
        let points = arr2(&[
            [1.1, 0.9],
            [0.9, 1.1],
            [1.1, 1.1],
            [0.9, 0.9],
            [-1.1, -0.9],
            [-0.9, -1.1],
            [-1.1, -1.1],
            [-0.9, -0.9]
        ]);
        let expected = arr1(&[0, 0, 0, 0, 1, 1, 1, 1]);
        let mkde = MultiKDEBase::default();
        let assignments = mkde.cluster(points.view()).unwrap();
        assert_eq!(assignments, expected)
    }

    #[test]
    fn multidim_same_result_as_python() {
        let points = read_data_("data/cluster-test-data.csv");
        let mut expected = vec![0; 500];
        expected.extend(vec![1; 500]);
        let expected = Array1::from(expected);
        let mkde = MultiKDEBase::default();
        let assignments = mkde.cluster(points.view()).unwrap();
        assert_eq!(assignments, expected);
    }

    #[test]
    fn kde_timing() {
        let mut points = vec![1.0; 5];
        points.extend(vec![2.0; 5]);
        let points = Array1::from(points).insert_axis(Axis(1));
        let mkde = MultiKDEBase::default();
        let assignments = mkde.cluster(points.view()).unwrap().to_vec();

        let mut expected = vec![0; 5];
        expected.extend(vec![1; 5]);

        assert_eq!(assignments, expected);
    }
}
