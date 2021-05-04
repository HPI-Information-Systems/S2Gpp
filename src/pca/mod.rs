mod messages;

use log::*;
use ndarray::prelude::*;
use ndarray_linalg::qr::*;
use actix::prelude::*;
pub use crate::pca::messages::{PCAMessage, PCAMeansMessage, PCADecompositionMessage, PCAResponse};
use actix::dev::MessageResponse;
use ndarray::{ArcArray2, concatenate};
use std::ops::{Mul, Div};
use ndarray_linalg::SVD;


pub struct PCA {
    source: Option<Recipient<PCAResponse>>,
    id: usize,
    cluster_nodes: Vec<Addr<PCA>>,
    n_components: usize,
    components: Option<Array2<f32>>,
    data: Option<ArcArray2<f32>>,
    local_r: Option<Array2<f32>>,
    r_count: usize,
    column_means: Option<Array2<f32>>,
    n: Option<Array1<f32>>
}

impl PCA {
    pub fn new(source: Option<Recipient<PCAResponse>>, id: usize, n_components: usize) -> Self {
        Self {
            source,
            id,
            cluster_nodes: vec![],
            n_components,
            components: None,
            data: None,
            local_r: None,
            r_count: 0,
            column_means: None,
            n: None
        }
    }

    fn center_columns_decomposition(&mut self) {
        let data = self.data.as_ref().expect("PCA started before data is present!");
        self.column_means = Some(data.mean_axis(Axis(0)).unwrap().into_shape([1, data.shape()[1]]).unwrap());
        self.n = Some(arr1(&[data.shape()[0] as f32]));
        let col_centered = data - self.column_means.as_ref().unwrap();
        let (_q, r) = col_centered.qr().expect("Could not perform QR decomposition");
        self.local_r = Some(r);

        self.send_to_main();
        self.send_to_neighbor_or_finalize();
    }

    fn send_to_main(&mut self) {
        if self.id > 0 {
            self.cluster_nodes[0].do_send(PCAMeansMessage {
                columns_means: self.column_means.as_ref().unwrap().clone(),
                n: self.data.as_ref().unwrap().shape()[0]
            });
        }
    }

    fn send_to_neighbor_or_finalize(&mut self) {
        let threshold = self.cluster_nodes.len().div(2_usize.pow((self.r_count + 1) as u32));

        if self.id >= threshold && self.id > 0 {
            let neighbor_id = self.id - threshold;
            self.cluster_nodes[neighbor_id].do_send(PCADecompositionMessage {
                r: self.local_r.as_ref().unwrap().clone(), count: self.r_count + 1 });
        } else if self.id == 0 && self.cluster_nodes.len() == self.r_count + 1 {
            self.finalize()
        }
    }

    fn combine_remote_r(&mut self, remote_r: Array2<f32>) {
        match &self.local_r {
            Some(r) => {
                let (_q, r) = concatenate(Axis(0), &[r.view(), remote_r.view()]).unwrap().qr().unwrap();
                self.local_r = Some(r);
                self.send_to_neighbor_or_finalize();
            },
            None => panic!("Cannot combine sent and local R, because no local R exists.")
        }
    }

    fn finalize(&mut self) {
        let column_means = self.column_means.as_ref().unwrap().to_owned();
        let dim = column_means.shape()[1].clone();
        let n = self.n.as_ref().unwrap().view();
        let n_reshaped = n.broadcast((dim, n.len())).unwrap();
        let global_means = (n_reshaped.t().to_owned() * column_means.clone().to_owned()).sum_axis(Axis(0)) / n.sum();

        let squared_n = n_reshaped.t().mapv(f32::sqrt);
        let mean_diff = column_means.to_owned() - global_means.broadcast((n.len(), dim)).unwrap().to_owned();
        let squared_mul = squared_n * mean_diff;
        let (_q, r) = concatenate![Axis(0), squared_mul.view(), self.local_r.as_ref().unwrap().view()].qr().unwrap();

        let (_u, _s, v) = r.svd(false, true).unwrap();
        let v = v.expect("Could not calculate SVD.");
        let mut v_sliced = v.slice(s![0..self.n_components, ..]).to_owned();
        self.components = Some(self.normalize(&v_sliced));
        debug!("Principal components: {:?}", self.components.as_ref().unwrap());

        // todo: share principal components
    }

    fn normalize(&mut self, v: &Array2<f32>) -> Array2<f32> {
        let mut v = v.clone();

        for r in 0..v.shape()[0] {
            if v[[r, 0]] >= 0.0 {
                continue
            }

            for c in 0..v.shape()[1] {
                v[[r, c]] *= -1.0
            }
        }

        v
    }
}

impl Actor for PCA {
    type Context = Context<Self>;
}

impl Handler<PCAMessage> for PCA {
    type Result = ();

    fn handle(&mut self, msg: PCAMessage, ctx: &mut Self::Context) -> Self::Result {
        self.data = Some(msg.data);
        self.cluster_nodes = msg.cluster_nodes;
        self.center_columns_decomposition();
    }
}

impl Handler<PCAMeansMessage> for PCA {
    type Result = ();

    fn handle(&mut self, msg: PCAMeansMessage, ctx: &mut Self::Context) -> Self::Result {
        self.column_means = Some(concatenate![Axis(0),
            self.column_means.as_ref().unwrap().clone(),
            msg.columns_means.view().into_dimensionality().unwrap()
        ]);
        self.n = Some(concatenate![Axis(0),
            self.n.as_ref().unwrap().clone(),
            arr1(&[msg.n as f32])
        ]);
    }
}

impl Handler<PCADecompositionMessage> for PCA {
    type Result = ();

    fn handle(&mut self, msg: PCADecompositionMessage, ctx: &mut Self::Context) -> Self::Result {
        self.r_count += msg.count;
        self.combine_remote_r(msg.r);
    }
}
