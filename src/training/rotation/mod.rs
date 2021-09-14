use std::ops::{Mul, Add, Sub};

use actix::{Actor, ActorContext, Addr, AsyncContext, Context, Handler, Recipient};
use actix_telepathy::prelude::*;
use ndarray::{ArcArray, arr1, Array1, Array2, Array3, ArrayBase, ArrayView2, Axis, concatenate, Dim, Ix3, s, stack};

use crate::parameters::{Parameters, Role};
use crate::training::Training;
use crate::utils::{ClusterNodes, cross2d, norm, repeat};
use crate::training::rotation::pca::{PCA, PCAnalyzer};
pub use crate::training::rotation::messages::{RotationMatrixMessage, RotationDoneMessage};
pub use crate::training::rotation::pca::*;
use num_traits::real::Real;

mod messages;
mod pca;

#[derive(Default)]
pub struct Rotation {
    phase_space: Option<ArcArray<f32, Ix3>>,
    data_ref: Option<ArcArray<f32, Ix3>>,
    reduced: Option<Array3<f32>>,
    reduced_ref: Option<Array3<f32>>,
    n_reduced: usize,
    pub pca: PCA,
    pub rotated: Option<Array2<f32>>
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
        let data = self.rotation.phase_space.as_ref().unwrap().slice(s![.., .., self.rotation.n_reduced]).to_shared();
        self.pca(data);
        self.rotation.n_reduced += 1
    }

    fn reduce(&mut self) {
        let components = self.rotation.pca.components.as_ref().unwrap().clone().reversed_axes();
        let i = self.rotation.n_reduced - 1;
        self.rotation.reduced.as_mut().unwrap().index_axis_mut(Axis(2), i).assign({
            let x = &self.rotation.phase_space.as_ref().unwrap().slice(s![.., .., i]);
            let x = x.sub(&self.rotation.pca.global_means.as_ref().unwrap().broadcast(x.shape()).unwrap());
            &x.dot(&components)
        });

        self.rotation.reduced_ref.as_mut().unwrap().index_axis_mut(Axis(2), i).assign({
            let x = &self.rotation.data_ref.as_ref().unwrap().slice(s![.., .., i]);
            let x = x.sub(&self.rotation.pca.global_means.as_ref().unwrap().broadcast(x.shape()).unwrap());
            &x.dot(&components)
        });
    }

    fn get_rotation_matrix(&mut self) -> Array3<f32> {
        let curve_vec1 = self.rotation.reduced_ref.as_ref().unwrap().slice(s![0, .., ..]).to_owned();
        let curve_vec2 = arr1(&[0., 0., 1.]);

        let a = curve_vec1.clone() / norm(curve_vec1.view(), Axis(0)).into_shape((1, curve_vec1.shape()[1])).unwrap();
        let b = curve_vec2.into_shape((3, 1)).unwrap();

        let v = cross2d(a.view(), b.view(), Axis(0), Axis(0));
        let c = b.t().dot(&a);
        let s_ = norm(v.t(), Axis(0)).into_shape((1, v.shape()[0])).unwrap();

        let identity: Array2<f32> = ArrayBase::eye(3);
        let i = repeat(identity.view(), v.shape()[0]).into_shape((3, 3, v.shape()[0])).unwrap();

        let zeros: Array1<f32> = ArrayBase::zeros(v.shape()[0]);
        let v_ = v.t();
        let v_n = v_.mul(-1.0);

        let k = concatenate(Axis(0), &[
            zeros.view(), v_n.row(2), v_.row(1),
            v_.row(2), zeros.view(), v_n.row(0),
            v_n.row(1), v_.row(0), zeros.view()]
        ).unwrap().into_shape((3, 3, v.shape()[0])).unwrap();

        let k_: Vec<Array2<f32>> = k.axis_iter(Axis(2)).map(|x| x.dot(&x)).collect();
        i + k + stack(Axis(2), k_.iter().map(|x| x.view())
            .collect::<Vec<ArrayView2<f32>>>().as_slice()
        ).unwrap() * ((1.0 - c) / s_.clone().mul(s_.clone()))
    }

    fn broadcast_rotation_matrix(&mut self, addr: Addr<Self>) {
        match &self.parameters.role {
            Role::Main { .. } => {
                let rotation_matrix = self.get_rotation_matrix();
                for nodes in self.cluster_nodes.to_any(addr) {
                    nodes.do_send(RotationMatrixMessage { rotation_matrix: rotation_matrix.clone() })
                }
            },
            _ => ()
        }
    }

    fn apply_rotation_matrix(&mut self, rotation_matrix: Array3<f32>) {
        let rotations: Vec<Array2<f32>> = rotation_matrix.axis_iter(Axis(2))
            .zip(self.rotation.reduced.as_ref().unwrap().axis_iter(Axis(2)))
            .map(|(a, b)| {
                a.dot(&b.t())
            }).collect();

        let rotated_3 = stack(Axis(2), rotations.iter().map(|x|
            x.view()).collect::<Vec<ArrayView2<f32>>>().as_slice()
        ).unwrap();

        let rotated = rotated_3.slice(s![0..2, .., ..])
            .into_shape(Dim([rotated_3.shape()[1], rotated_3.shape()[2] * 2])).unwrap().to_owned();

        self.rotation.rotated = Some(rotated);
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
        }
    }
}

impl Handler<RotationMatrixMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: RotationMatrixMessage, ctx: &mut Self::Context) -> Self::Result {
        self.apply_rotation_matrix(msg.rotation_matrix);
        ctx.address().do_send(RotationDoneMessage);
    }
}


#[cfg(test)]
mod tests;
