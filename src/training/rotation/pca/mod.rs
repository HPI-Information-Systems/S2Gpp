mod helper;
mod messages;
#[cfg(test)]
mod tests;

use actix::prelude::*;
use ndarray::prelude::*;
use ndarray_linalg::qr::*;

use ndarray::{concatenate, ArcArray2};
use ndarray_linalg::SVD;
use std::ops::Div;

pub use crate::training::rotation::pca::messages::{
    PCAComponents, PCADecompositionMessage, PCADoneMessage, PCAMeansMessage, PCAMessage,
};
use crate::{messages::PoisonPill, training::Training};

use self::{helper::PCAHelper, messages::PCAHelperMessage};

#[derive(Default)]
#[allow(clippy::upper_case_acronyms)]
pub struct PCA {
    id: usize,
    n_components: usize,
    pub components: Option<Array2<f32>>,
    pub global_means: Option<Array1<f32>>,
    data: Option<ArcArray2<f32>>,
    local_r: Option<Array2<f32>>,
    r_count: usize,
    column_means: Option<Array2<f32>>,
    n: Option<Array1<f32>>,
    pub recipient: Option<Recipient<PCAComponents>>,
    helpers: Vec<Addr<PCAHelper>>,
    buffer: Vec<PCADecompositionMessage>,
    means_buffer: Vec<PCAMeansMessage>,
}

impl PCA {
    pub fn new(id: usize, n_components: usize) -> Self {
        PCA {
            id,
            n_components,
            ..Default::default()
        }
    }

    pub fn clear(&mut self) {
        *self = Self::new(self.id, self.n_components);
    }
}

pub trait PCAnalyzer {
    fn pca(&mut self, data: ArcArray2<f32>);
    fn resolve_buffer(&mut self);
    fn resolve_means_buffer(&mut self);
    fn center_columns_decomposition(&mut self);
    fn helper_center_columns_decomposition(&mut self);
    fn send_to_main(&mut self);
    fn next_2_power(&mut self) -> usize;
    fn send_to_neighbor_or_finalize(&mut self);
    fn combine_remote_r(&mut self, remote_r: Array2<f32>);
    fn finalize(&mut self);
    fn normalize(&mut self, v: &Array2<f32>) -> Array2<f32>;
    fn share_principal_components(&mut self, means: Array1<f32>);
}

impl PCAnalyzer for Training {
    fn pca(&mut self, data: ArcArray2<f32>) {
        self.rotation.pca.data = Some(data);
        if self.parameters.n_threads > 1 {
            self.helper_center_columns_decomposition();
        } else {
            self.center_columns_decomposition();
        }
    }

    fn resolve_buffer(&mut self) {
        let own_addr = self.own_addr.as_ref().unwrap();
        while let Some(msg) = self.rotation.pca.buffer.pop() {
            own_addr.do_send(msg);
        }
    }

    fn resolve_means_buffer(&mut self) {
        let own_addr = self.own_addr.as_ref().unwrap();
        while let Some(msg) = self.rotation.pca.means_buffer.pop() {
            own_addr.do_send(msg);
        }
    }

    fn center_columns_decomposition(&mut self) {
        let data = self
            .rotation
            .pca
            .data
            .as_ref()
            .expect("PCA started before data is present!");
        self.rotation.pca.column_means = Some(
            data.mean_axis(Axis(0))
                .unwrap()
                .into_shape([1, data.shape()[1]])
                .unwrap(),
        );
        self.rotation.pca.n = Some(arr1(&[data.shape()[0] as f32]));
        let col_centered = data - self.rotation.pca.column_means.as_ref().unwrap();
        let (_q, r) = col_centered
            .qr()
            .expect("Could not perform QR decomposition");
        self.rotation.pca.local_r = Some(r);

        self.send_to_main();
        self.send_to_neighbor_or_finalize();
    }

    fn helper_center_columns_decomposition(&mut self) {
        let own_addr = self.own_addr.as_ref().unwrap().clone().recipient();
        self.rotation.pca.helpers = (0..self.parameters.n_threads)
            .into_iter()
            .map(|i| PCAHelper::start_helper(i, own_addr.clone()))
            .collect();
        let helpers: Vec<Recipient<PCAHelperMessage>> = self
            .rotation
            .pca
            .helpers
            .iter()
            .map(|addr| addr.clone().recipient())
            .collect();
        let data = self
            .rotation
            .pca
            .data
            .as_ref()
            .expect("PCA started before data is present!");
        let chunk_size = num_integer::div_ceil(data.shape()[0], self.parameters.n_threads);
        for (chunk, helper) in data
            .axis_chunks_iter(Axis(0), chunk_size)
            .zip(self.rotation.pca.helpers.iter())
        {
            helper.do_send(PCAHelperMessage::Setup {
                neighbors: helpers.clone(),
                data: chunk.to_shared(),
            })
        }
    }

    fn send_to_main(&mut self) {
        if let Some(addr) = self.cluster_nodes.get(&0) {
            let mut addr = addr.clone();
            addr.change_id("Training".to_string());
            addr.do_send(PCAMeansMessage {
                columns_means: self.rotation.pca.column_means.as_ref().unwrap().clone(),
                n: self.rotation.pca.data.as_ref().unwrap().shape()[0],
            })
        }
    }

    fn next_2_power(&mut self) -> usize {
        let len = self.cluster_nodes.len_incl_own();
        2_i32.pow((len as f32).log2().ceil() as u32) as usize
    }

    fn send_to_neighbor_or_finalize(&mut self) {
        let s = self.next_2_power();
        let threshold = s.div(2_usize.pow((self.rotation.pca.r_count + 1) as u32));
        let id = self.rotation.pca.id;

        if id >= threshold && id > 0 {
            let neighbor_id = id - threshold;
            match self.cluster_nodes.get(&neighbor_id) {
                Some(node) => {
                    let mut addr = node.clone();
                    addr.change_id("Training".to_string());
                    addr.do_send(PCADecompositionMessage {
                        r: self.rotation.pca.local_r.as_ref().unwrap().clone(),
                        count: self.rotation.pca.r_count + 1,
                    });
                }
                None => panic!("No cluster node with id {} exists!", &neighbor_id),
            }
        } else if self.rotation.pca.r_count == 0
            && (id + threshold) >= self.cluster_nodes.len_incl_own()
        {
            self.rotation.pca.r_count += 1;
            self.send_to_neighbor_or_finalize()
        } else if id == 0 && threshold == 0 {
            self.finalize()
        }
    }

    fn combine_remote_r(&mut self, remote_r: Array2<f32>) {
        match &self.rotation.pca.local_r {
            Some(r) => {
                let (_q, r) = concatenate(Axis(0), &[r.view(), remote_r.view()])
                    .unwrap()
                    .qr()
                    .unwrap();
                self.rotation.pca.local_r = Some(r);
                self.send_to_neighbor_or_finalize();
            }
            None => panic!("Cannot combine sent and local R, because no local R exists."),
        }
    }

    fn finalize(&mut self) {
        let column_means = self.rotation.pca.column_means.as_ref().unwrap().to_owned();
        let dim = column_means.shape()[1];
        let n = self.rotation.pca.n.as_ref().unwrap().view();
        let n_reshaped = n.broadcast((dim, n.len())).unwrap();
        let global_means =
            (n_reshaped.t().to_owned() * column_means.to_owned()).sum_axis(Axis(0)) / n.sum();

        let squared_n = n_reshaped.t().mapv(f32::sqrt);
        let mean_diff =
            column_means.to_owned() - global_means.broadcast((n.len(), dim)).unwrap().to_owned();
        let squared_mul = squared_n * mean_diff;
        let (_q, r) = concatenate![
            Axis(0),
            squared_mul.view(),
            self.rotation.pca.local_r.as_ref().unwrap().view()
        ]
        .qr()
        .unwrap();

        let (_u, _s, v) = r.svd(false, true).unwrap();
        let v = v.expect("Could not calculate SVD.");
        let v_sliced = v
            .slice(s![0..self.rotation.pca.n_components, ..])
            .to_owned();
        self.rotation.pca.components = Some(self.normalize(&v_sliced));

        self.share_principal_components(global_means);
    }

    fn normalize(&mut self, v: &Array2<f32>) -> Array2<f32> {
        let mut v = v.clone();

        for r in 0..v.shape()[0] {
            if v[[r, 0]] >= 0.0 {
                continue;
            }

            for c in 0..v.shape()[1] {
                v[[r, c]] *= -1.0
            }
        }

        v
    }

    fn share_principal_components(&mut self, means: Array1<f32>) {
        let msg = PCAComponents {
            components: self.rotation.pca.components.as_ref().unwrap().clone(),
            means,
        };

        for (_, node) in self.cluster_nodes.iter() {
            let mut addr = node.clone();
            addr.change_id("Training".to_string());
            addr.do_send(msg.clone())
        }

        match &self.own_addr {
            Some(own_addr) => own_addr.do_send(msg),
            None => panic!("own_addr not yet set"),
        }
    }
}

impl Handler<PCAMeansMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: PCAMeansMessage, _ctx: &mut Self::Context) -> Self::Result {
        if self.rotation.pca.column_means.is_some() {
            self.rotation.pca.column_means = Some(concatenate![
                Axis(0),
                self.rotation.pca.column_means.as_ref().unwrap().clone(),
                msg.columns_means.view().into_dimensionality().unwrap()
            ]);
            self.rotation.pca.n = Some(concatenate![
                Axis(0),
                self.rotation.pca.n.as_ref().unwrap().clone(),
                arr1(&[msg.n as f32])
            ]);
        } else {
            self.rotation.pca.means_buffer.push(msg);
        }
    }
}

impl Handler<PCADecompositionMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: PCADecompositionMessage, _ctx: &mut Self::Context) -> Self::Result {
        if msg.count == self.rotation.pca.r_count + 1 && self.rotation.pca.local_r.is_some() {
            self.rotation.pca.r_count = msg.count;
            self.combine_remote_r(msg.r);
            self.resolve_buffer();
        } else {
            self.rotation.pca.buffer.push(msg);
        }
    }
}

impl Handler<PCAComponents> for Training {
    type Result = ();

    fn handle(&mut self, msg: PCAComponents, ctx: &mut Self::Context) -> Self::Result {
        self.rotation.pca.components = Some(msg.clone().components);
        self.rotation.pca.global_means = Some(msg.means.clone());
        match &self.rotation.pca.recipient {
            Some(rec) => {
                rec.do_send(msg).unwrap();
            }
            None => ctx.address().do_send(PCADoneMessage),
        }
    }
}

impl Handler<PCAHelperMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: PCAHelperMessage, _ctx: &mut Self::Context) -> Self::Result {
        if let PCAHelperMessage::Response { column_means, n, r } = msg {
            let n_columns = column_means.len();
            self.rotation.pca.column_means = Some(column_means.into_shape([1, n_columns]).unwrap());
            self.rotation.pca.n = Some(concatenate![Axis(0), arr1(&[n])]);
            self.rotation.pca.local_r = Some(r);

            let n_helpers = self.rotation.pca.helpers.len();
            let helpers = &mut self.rotation.pca.helpers;
            for _ in 0..n_helpers {
                let helper = helpers.pop().unwrap();
                helper.do_send(PoisonPill);
            }

            self.send_to_main();
            self.send_to_neighbor_or_finalize();
            self.resolve_buffer();
            self.resolve_means_buffer();
        }
    }
}
