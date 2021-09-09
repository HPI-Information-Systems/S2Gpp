mod messages;
#[cfg(test)]
mod tests;
mod rotator;

use log::*;
use ndarray::prelude::*;
use ndarray_linalg::qr::*;
use actix::prelude::*;
use actix_telepathy::prelude::*;
pub use crate::pca::messages::{PCAMessage, PCAMeansMessage, PCADecompositionMessage, PCAResponse, RotatedMessage, StartRotation};
pub use crate::pca::rotator::{Rotator};

use ndarray::{ArcArray2, concatenate};
use std::ops::{Div};
use ndarray_linalg::SVD;
use crate::pca::messages::PCAComponents;
use crate::utils::ClusterNodes;


#[derive(RemoteActor)]
#[remote_messages(PCAMeansMessage, PCADecompositionMessage, PCAComponents)]
pub struct PCA {
    source: Option<Recipient<PCAResponse>>,
    id: usize,
    cluster_nodes: ClusterNodes,
    n_components: usize,
    components: Option<Array2<f32>>,
    data: Option<ArcArray2<f32>>,
    local_r: Option<Array2<f32>>,
    r_count: usize,
    column_means: Option<Array2<f32>>,
    n: Option<Array1<f32>>,
    own_addr: Option<Addr<Self>>
}

impl PCA {
    pub fn new(cluster_nodes: ClusterNodes, source: Option<Recipient<PCAResponse>>, id: usize, n_components: usize) -> Self {
        Self {
            source,
            id,
            cluster_nodes,
            n_components,
            components: None,
            data: None,
            local_r: None,
            r_count: 0,
            column_means: None,
            n: None,
            own_addr: None
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
        match self.cluster_nodes.get(&0) {
            Some(addr) => {
                let mut addr = addr.clone();
                addr.change_id("PCA".to_string());
                addr.do_send(PCAMeansMessage {
                    columns_means: self.column_means.as_ref().unwrap().clone(),
                    n: self.data.as_ref().unwrap().shape()[0],
                })
            },
            None => ()
        }
    }

    fn next_2_power(&mut self) -> usize {
        let len = self.cluster_nodes.len() + 1;
        2_i32.pow((len as f32).log2().ceil() as u32) as usize
    }

    fn send_to_neighbor_or_finalize(&mut self) {
        let s = self.next_2_power();
        let threshold = s.div(2_usize.pow((self.r_count + 1) as u32));

        if self.id >= threshold && self.id > 0 {
            let neighbor_id = self.id - threshold;
            match self.cluster_nodes.get(&neighbor_id) {
                Some(node) => {
                    let mut addr = node.clone();
                    addr.change_id("PCA".to_string());
                    addr.do_send(PCADecompositionMessage {
                        r: self.local_r.as_ref().unwrap().clone(), count: self.r_count + 1 });
                },
                None => panic!("No cluster node with id {} exists!", &neighbor_id)
            }
        } else if self.r_count == 0 && (self.id + threshold) >= self.cluster_nodes.len_incl_own() {
            self.r_count += 1;
            self.send_to_neighbor_or_finalize()
        } else if self.id == 0 && s == self.r_count + 1 {
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
        let v_sliced = v.slice(s![0..self.n_components, ..]).to_owned();
        self.components = Some(self.normalize(&v_sliced));
        debug!("Principal components: {:?}", self.components.as_ref().unwrap());

        self.share_principal_components();
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

    fn share_principal_components(&mut self) {
        let msg = PCAComponents { components: self.components.as_ref().unwrap().clone() };

        for (_, node) in self.cluster_nodes.iter() {
            let mut addr = node.clone();
            addr.change_id("PCA".to_string());
            addr.do_send(msg.clone())
        }
        match &self.own_addr {
            Some(addr) => addr.do_send(msg.clone()),
            _ => ()
        }
    }
}

impl Actor for PCA {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.register(ctx.address().recipient(), "PCA".to_string());
        self.own_addr = Some(ctx.address());
    }
}

impl Handler<PCAMessage> for PCA {
    type Result = ();

    fn handle(&mut self, msg: PCAMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.data = Some(msg.data);
        self.center_columns_decomposition();
    }
}

impl Handler<PCAMeansMessage> for PCA {
    type Result = ();

    fn handle(&mut self, msg: PCAMeansMessage, _ctx: &mut Self::Context) -> Self::Result {
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

    fn handle(&mut self, msg: PCADecompositionMessage, _ctx: &mut Self::Context) -> Self::Result {
        self.r_count += msg.count;
        self.combine_remote_r(msg.r);
    }
}

impl Handler<PCAComponents> for PCA {
    type Result = ();

    fn handle(&mut self, msg: PCAComponents, ctx: &mut Self::Context) -> Self::Result {
        self.components = Some(msg.components);
        match &self.source {
            Some(source) => { source.do_send(PCAResponse { components: self.components.as_ref().unwrap().clone() }).unwrap(); },
            None => ()
        }
        ctx.stop();
    }
}
