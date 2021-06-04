use ndarray::{ArrayBase, ArcArray, Array3, Ix3, Array1, Array2, Axis, arr1, s, ArrayView2, concatenate, stack, Dim};
use actix::{Actor, Recipient, ActorContext, Context, Handler, AsyncContext, Addr};
use actix_telepathy::prelude::*;
use crate::pca::messages::{RotatedMessage, RotationMatrixMessage, StartRotation};
use crate::pca::{PCAResponse, PCA, PCAMessage};

use crate::utils::{ClusterNodes, norm, cross2d, repeat};
use std::ops::Mul;
use crate::parameters::{Parameters, Role};


#[derive(RemoteActor)]
#[remote_messages()]
pub struct Rotator {
    parameters: Parameters,
    cluster_nodes: ClusterNodes,
    source: Recipient<RotatedMessage>,
    phase_space: Option<ArcArray<f32, Ix3>>,
    data_ref: Option<ArcArray<f32, Ix3>>,
    reduced: Option<Array3<f32>>,
    pub(in crate::pca::rotator) reduced_ref: Option<Array3<f32>>,
    n_reduced: usize,
    pca: Option<Addr<PCA>>
}

impl Rotator {
    pub fn new(cluster_nodes: ClusterNodes, parameters: Parameters, source: Recipient<RotatedMessage>) -> Self {
        Self {
            parameters,
            cluster_nodes,
            source,
            phase_space: None,
            data_ref: None,
            reduced: None,
            reduced_ref: None,
            n_reduced: 0,
            pca: None
        }
    }

    fn run_pca(&mut self) {
        let data = self.phase_space.as_ref().unwrap().slice(s![.., .., self.n_reduced]).to_shared();
        self.pca.as_ref().unwrap().do_send(PCAMessage { data });
        self.n_reduced += 1
    }

    fn reduce(&mut self, components: Array2<f32>) {
        let i = self.n_reduced - 1;
        self.reduced.as_mut().unwrap().index_axis_mut(Axis(2), i).assign(
            &self.phase_space.as_ref().unwrap().slice(s![.., .., i]).dot(&components)
        );

        self.reduced_ref.as_mut().unwrap().index_axis_mut(Axis(2), i).assign(
            &self.data_ref.as_ref().unwrap().slice(s![.., .., i]).dot(&components)
        );
    }

    fn get_rotation_matrix(&mut self) -> Array3<f32> {
        let curve_vec1 = self.reduced_ref.as_ref().unwrap().slice(s![0, .., ..]).to_owned();
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

    fn rotate(&mut self, rotation_matrix: Array3<f32>) {
        let rotations: Vec<Array2<f32>> = rotation_matrix.axis_iter(Axis(2))
            .zip(self.reduced.as_ref().unwrap().axis_iter(Axis(2)))
            .map(|(a, b)| {
                a.dot(&b.t())
            }).collect();

        let rotated_3 = stack(Axis(2), rotations.iter().map(|x|
            x.view()).collect::<Vec<ArrayView2<f32>>>().as_slice()
        ).unwrap();

        let rotated = rotated_3.slice(s![.., 0..2, ..])
            .into_shape(Dim([rotated_3.shape()[0], rotated_3.shape()[2] * 2])).unwrap().to_owned();

        self.source.do_send(RotatedMessage { rotated }).unwrap();
    }
}

impl Actor for Rotator {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.pca = Some(PCA::new(
        self.cluster_nodes.clone(),
        Some(ctx.address().recipient()),
        self.cluster_nodes.get_own_idx(),
        3).start());
    }
}

impl Handler<StartRotation> for Rotator {
    type Result = ();

    fn handle(&mut self, msg: StartRotation, _ctx: &mut Self::Context) -> Self::Result {
        let reduced = ArrayBase::zeros(Dim([msg.phase_space.shape()[0], 3, msg.phase_space.shape()[2]]));
        let reduced_ref = ArrayBase::zeros(Dim([msg.data_ref.shape()[0], 3, msg.data_ref.shape()[2]]));

        self.phase_space = Some(msg.phase_space);
        self.data_ref = Some(msg.data_ref);
        self.reduced = Some(reduced);
        self.reduced_ref = Some(reduced_ref);

        self.run_pca();
    }
}

impl Handler<PCAResponse> for Rotator {
    type Result = ();

    fn handle(&mut self, msg: PCAResponse, ctx: &mut Self::Context) -> Self::Result {
        if self.n_reduced < self.phase_space.as_ref().unwrap().shape()[2] {
            self.run_pca();
            self.reduce(msg.components);
        } else {
            self.reduce(msg.components);
            self.broadcast_rotation_matrix(ctx.address());
        }
    }
}

impl Handler<RotationMatrixMessage> for Rotator {
    type Result = ();

    fn handle(&mut self, msg: RotationMatrixMessage, ctx: &mut Self::Context) -> Self::Result {
        self.rotate(msg.rotation_matrix);
        ctx.stop();
    }
}


#[cfg(test)]
mod tests {
    use crate::pca::{Rotator};
    use crate::utils::ClusterNodes;
    use actix::{Actor, System};
    use crate::training::Training;
    use crate::parameters::Parameters;
    use ndarray::{arr3, Array3};
    use std::sync::{Mutex, Arc};
    use ndarray_linalg::close_l1;

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
            let recipient = Training::new(Parameters::default()).start().recipient();
            let dummy_data = arr3(&[[[0.]]]);

            let mut rotator = Rotator::new(
                ClusterNodes::new(),
                Parameters::default(),
                recipient
            );

            rotator.phase_space = Some(dummy_data.to_shared());
            rotator.data_ref = Some(dummy_data.to_shared());

            rotator.reduced_ref = Some(arr3(&[
                [[-2.32510113e+01, -1.84500066e+01],
                 [ 2.19784013e-02,  1.53111935e-01],
                 [ 3.25042576e-02,  6.32221831e-02]]
            ]));

            *(rotation_matrix_clone.lock().unwrap()) = Some(rotator.get_rotation_matrix());
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
