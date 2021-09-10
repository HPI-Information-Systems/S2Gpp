use std::ops::Mul;

use actix::{Actor, ActorContext, Addr, AsyncContext, Context, Handler, Recipient};
use actix_telepathy::prelude::*;
use ndarray::{ArcArray, arr1, Array1, Array2, Array3, ArrayBase, ArrayView2, Axis, concatenate, Dim, Ix3, s, stack};

use crate::parameters::{Parameters, Role};
use crate::training::Training;
use crate::utils::{ClusterNodes, cross2d, norm, repeat};
use crate::training::rotation::pca::{PCA, PCAnalyzer};
pub use crate::training::rotation::messages::{RotationMatrixMessage, RotationDoneMessage};
pub use crate::training::rotation::pca::*;

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
        let components = self.rotation.pca.components.as_ref().unwrap().clone();
        let i = self.rotation.n_reduced - 1;
        self.rotation.reduced.as_mut().unwrap().index_axis_mut(Axis(2), i).assign(
            &self.rotation.phase_space.as_ref().unwrap().slice(s![.., .., i]).dot(&components)
        );

        self.rotation.reduced_ref.as_mut().unwrap().index_axis_mut(Axis(2), i).assign(
            &self.rotation.data_ref.as_ref().unwrap().slice(s![.., .., i]).dot(&components)
        );
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
        let I = repeat(identity.view(), v.shape()[0]).into_shape((3, 3, v.shape()[0])).unwrap();

        let zeros: Array1<f32> = ArrayBase::zeros(v.shape()[0]);
        let v_ = v.t();
        let v_n = v_ .mul(-1.0);

        let k = concatenate(Axis(0), &[
            zeros.view(), v_n.row(2), v_.row(1),
            v_.row(2), zeros.view(), v_n.row(0),
            v_n.row(1), v_.row(0), zeros.view()]
        ).unwrap().into_shape((3, 3, v.shape()[0])).unwrap();

        let k_: Vec<Array2<f32>> = k.axis_iter(Axis(2)).map(|x| x.dot(&x)).collect();

        I + k + stack(Axis(2), k_.iter().map(|x|
            x.view()).collect::<Vec<ArrayView2<f32>>>().as_slice()
        ).unwrap() * ((1.0 - c) / &s_ * s_)
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

        let rotated = rotated_3.slice(s![.., 0..2, ..])
            .into_shape(Dim([rotated_3.shape()[0], rotated_3.shape()[2] * 2])).unwrap().to_owned();
    }
}

impl Handler<PCADoneMessage> for Training {
    type Result = ();

    fn handle(&mut self, _msg: PCADoneMessage, ctx: &mut Self::Context) -> Self::Result {
        if self.rotation.n_reduced < self.rotation.phase_space.as_ref().unwrap().shape()[2] {
            self.run_pca();
            self.reduce();
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
mod tests {
    use std::sync::{Arc, Mutex};

    use actix::{Actor, System};
    use ndarray::{arr3, Array3};
    use ndarray_linalg::close_l1;

    use crate::parameters::Parameters;
    use crate::training::Training;
    use crate::utils::ClusterNodes;
    use crate::training::rotation::Rotator;

    #[test]
    fn test_rotation_matrix() {
        let rotation_matrix: Arc<Mutex<Option<Array3<f32>>>> = Arc::new(Mutex::new(None));
        let rotation_matrix_clone = rotation_matrix.clone();

        let expects = arr3(&[
             [[ 1.39886206e-03,  3.49516576e-03],
              [ 9.43944169e-04,  8.26974144e-03],
              [ 9.99998576e-01,  9.99959697e-01]],
             [[ 9.43944169e-04,  8.26974144e-03],
              [ 9.99999108e-01,  9.99931372e-01],
              [-9.45265121e-04, -8.29841247e-03]],
             [[-9.99998576e-01, -9.99959697e-01],
              [ 9.45265121e-04,  8.29841247e-03],
              [ 1.39796979e-03,  3.42653727e-03]]]);

        let _system = System::run(move || {
            let mut training = Training::new(Parameters::default());
            let dummy_data = arr3(&[[[0.]]]);

            training.rotation.phase_space = Some(dummy_data.to_shared());
            training.rotation.data_ref = Some(dummy_data.to_shared());

            training.rotation.reduced_ref = Some(arr3(&[
                [[-2.32510113e+01, -1.84500066e+01],
                 [ 2.19784013e-02,  1.53111935e-01],
                 [ 3.25042576e-02,  6.32221831e-02]]
            ]));

            *(rotation_matrix_clone.lock().unwrap()) = Some(training.get_rotation_matrix());
            System::current().stop();
        });
        let truth = rotation_matrix.lock().unwrap();

        close_l1(truth.as_ref().unwrap(), &expects, 0.0005)
    }

    #[test]
    fn test_distributed_rotation() {
        // todo: test distributed rotation
    }
}
