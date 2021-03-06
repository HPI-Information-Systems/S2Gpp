use std::ops::{Mul, Sub};

use actix::{Addr, AsyncContext, Handler};

use ndarray::{
    arr1, concatenate, s, stack, ArcArray, Array1, Array2, Array3, ArrayBase, ArrayView2, Axis,
    Dim, Ix3,
};

use crate::parameters::Role;
pub use crate::training::rotation::messages::{RotationDoneMessage, RotationMatrixMessage};
pub use crate::training::rotation::pca::*;
use crate::training::rotation::pca::{PCAnalyzer, PCA};
use crate::training::Training;
use crate::utils::{cross2d, norm, repeat};

mod messages;
mod pca;

#[derive(Default, Clone)]
pub struct Rotation {
    phase_space: Option<ArcArray<f32, Ix3>>,
    data_ref: Option<ArcArray<f32, Ix3>>,
    reduced: Option<Array3<f32>>,
    reduced_ref: Option<Array3<f32>>,
    n_reduced: usize,
    broadcasted: bool,
    rotation_matrix_buffer: Option<RotationMatrixMessage>,
    pub pca: PCA,
}

pub trait Rotator {
    fn rotate(&mut self, phase_space: ArcArray<f32, Ix3>, data_ref: ArcArray<f32, Ix3>);
    fn run_pca(&mut self);
    fn reduce(&mut self);
    fn get_rotation_matrix(&mut self) -> Array3<f32>;
    fn broadcast_rotation_matrix(&mut self, addr: Addr<Training>);
    fn apply_rotation_matrix(&mut self, rotation_matrix: Array3<f32>);
}

impl Rotator for Training {
    fn rotate(&mut self, phase_space: ArcArray<f32, Ix3>, data_ref: ArcArray<f32, Ix3>) {
        let reduced = ArrayBase::zeros(Dim([phase_space.shape()[0], 3, phase_space.shape()[2]]));
        let reduced_ref = ArrayBase::zeros(Dim([data_ref.shape()[0], 3, data_ref.shape()[2]]));

        self.rotation.phase_space = Some(phase_space);
        self.rotation.data_ref = Some(data_ref);
        self.rotation.reduced = Some(reduced);
        self.rotation.reduced_ref = Some(reduced_ref);

        self.rotation.pca = PCA::new(self.cluster_nodes.get_own_idx(), 3);
        self.run_pca();
    }

    fn run_pca(&mut self) {
        self.rotation.pca.clear();
        let data = self
            .rotation
            .phase_space
            .as_ref()
            .unwrap()
            .slice(s![.., .., self.rotation.n_reduced])
            .to_shared();
        self.pca(data);
        self.rotation.n_reduced += 1
    }

    fn reduce(&mut self) {
        let components = self
            .rotation
            .pca
            .components
            .as_ref()
            .unwrap()
            .clone()
            .reversed_axes();
        let i = self.rotation.n_reduced - 1;
        self.rotation
            .reduced
            .as_mut()
            .unwrap()
            .index_axis_mut(Axis(2), i)
            .assign({
                let x = &self
                    .rotation
                    .phase_space
                    .as_ref()
                    .unwrap()
                    .slice(s![.., .., i]);
                let shape = x.shape();
                let x = x.sub(
                    &self
                        .rotation
                        .pca
                        .global_means
                        .as_ref()
                        .unwrap()
                        .broadcast([shape[0], shape[1]])
                        .unwrap(),
                );
                &x.dot(&components)
            });

        self.rotation
            .reduced_ref
            .as_mut()
            .unwrap()
            .index_axis_mut(Axis(2), i)
            .assign({
                let x = &self
                    .rotation
                    .data_ref
                    .as_ref()
                    .unwrap()
                    .slice(s![.., .., i]);
                let shape = x.shape();
                let x = x.sub(
                    &self
                        .rotation
                        .pca
                        .global_means
                        .as_ref()
                        .unwrap()
                        .broadcast([shape[0], shape[1]])
                        .unwrap(),
                );
                &x.dot(&components)
            });
    }

    fn get_rotation_matrix(&mut self) -> Array3<f32> {
        let curve_vec1 = self
            .rotation
            .reduced_ref
            .as_ref()
            .unwrap()
            .slice(s![0, .., ..])
            .to_owned();
        let curve_vec2 = arr1(&[0., 0., 1.]);

        let a = curve_vec1.clone()
            / norm(curve_vec1.view(), Axis(0))
                .into_shape((1, curve_vec1.shape()[1]))
                .unwrap();
        let b = curve_vec2.into_shape((3, 1)).unwrap();

        let v = cross2d(a.view(), b.view(), Axis(0), Axis(0));
        let c = b.t().dot(&a);
        let s_ = norm(v.t(), Axis(0)).into_shape((1, v.shape()[0])).unwrap();

        let identity: Array2<f32> = ArrayBase::eye(3);
        let i = repeat(identity.view(), v.shape()[0])
            .into_shape((3, 3, v.shape()[0]))
            .unwrap();

        let zeros: Array1<f32> = ArrayBase::zeros(v.shape()[0]);
        let v_ = v.t();
        let v_n = v_.mul(-1.0);

        let k = concatenate(
            Axis(0),
            &[
                zeros.view(),
                v_n.row(2),
                v_.row(1),
                v_.row(2),
                zeros.view(),
                v_n.row(0),
                v_n.row(1),
                v_.row(0),
                zeros.view(),
            ],
        )
        .unwrap()
        .into_shape((3, 3, v.shape()[0]))
        .unwrap();

        let k_: Vec<Array2<f32>> = k.axis_iter(Axis(2)).map(|x| x.dot(&x)).collect();
        i + k
            + stack(
                Axis(2),
                k_.iter()
                    .map(|x| x.view())
                    .collect::<Vec<ArrayView2<f32>>>()
                    .as_slice(),
            )
            .unwrap()
                * ((1.0 - c) / s_.clone().mul(s_))
    }

    fn broadcast_rotation_matrix(&mut self, addr: Addr<Self>) {
        if let Role::Main { .. } = &self.parameters.role {
            let rotation_matrix = self.get_rotation_matrix();
            for nodes in self.cluster_nodes.to_any_as(addr.clone(), "Training") {
                nodes.do_send(RotationMatrixMessage {
                    rotation_matrix: rotation_matrix.clone(),
                })
            }
        }
        if let Some(rotation_matrix_msg) = self.rotation.rotation_matrix_buffer.take() {
            addr.do_send(rotation_matrix_msg);
        }
    }

    fn apply_rotation_matrix(&mut self, rotation_matrix: Array3<f32>) {
        let rotations: Vec<Array2<f32>> = rotation_matrix
            .axis_iter(Axis(2))
            .zip(self.rotation.reduced.as_ref().unwrap().axis_iter(Axis(2)))
            .map(|(a, b)| b.dot(&a.t()))
            .collect();

        let rotated_3 = stack(
            Axis(2),
            rotations
                .iter()
                .map(|x| x.view())
                .collect::<Vec<ArrayView2<f32>>>()
                .as_slice(),
        )
        .unwrap();
        let rotated = rotated_3.slice(s![.., 0..2, ..]).to_owned();
        let shape = Dim([rotated.shape()[0], rotated.shape()[2] * 2]);

        let points: Vec<Array1<f32>> = rotated
            .into_shape(shape)
            .unwrap()
            .axis_iter(Axis(0))
            .map(|point| point.to_owned())
            .collect();

        self.num_rotated = Some(points.len());
        let points_per_node = self
            .dataset_stats
            .as_ref()
            .expect("DatasetStats should've been set by now!")
            .n
            .expect("DatasetStats.n should've been set by now!")
            / self.cluster_nodes.len_incl_own();
        let own_id = self.cluster_nodes.get_own_idx();
        self.data_store.add_points_with_offset(
            points,
            points_per_node * own_id,
            self.parameters.rate,
        );
    }
}

impl Handler<PCADoneMessage> for Training {
    type Result = ();

    fn handle(&mut self, _msg: PCADoneMessage, ctx: &mut Self::Context) -> Self::Result {
        if self.rotation.n_reduced < self.rotation.phase_space.as_ref().unwrap().shape()[2] {
            self.reduce();
            self.run_pca();
        } else {
            self.reduce();
            self.broadcast_rotation_matrix(ctx.address());
            self.rotation.broadcasted = true;
        }
    }
}

impl Handler<RotationMatrixMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: RotationMatrixMessage, ctx: &mut Self::Context) -> Self::Result {
        if !self.rotation.broadcasted {
            self.rotation.rotation_matrix_buffer = Some(msg);
            return;
        }

        self.apply_rotation_matrix(msg.rotation_matrix);
        ctx.address().do_send(RotationDoneMessage);
    }
}

#[cfg(test)]
mod tests;
