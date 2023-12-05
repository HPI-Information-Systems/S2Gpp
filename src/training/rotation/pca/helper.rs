use actix::prelude::*;
use actix::SyncContext;
use ndarray::arr1;
use ndarray::Array1;
use ndarray::{concatenate, ArcArray2, Array2, Axis};
use ndarray_linalg::QR;
use std::ops::Div;

use crate::messages::PoisonPill;

use super::messages::PCAHelperMessage;

#[derive(Default)]
pub(crate) struct PCAHelper {
    id: usize,
    receiver: Option<Recipient<PCAHelperMessage>>,
    neighbors: Vec<Recipient<PCAHelperMessage>>,
    data: Option<ArcArray2<f32>>,
    column_means: Option<Array2<f32>>,
    n: Option<Array1<f32>>,
    local_r: Option<Array2<f32>>,
    r_count: usize,
    buffer: Vec<PCAHelperMessage>,
    means_buffer: Vec<PCAHelperMessage>,
}

impl PCAHelper {
    pub fn start_helper(id: usize, receiver: Recipient<PCAHelperMessage>) -> Addr<Self> {
        SyncArbiter::start(1, move || Self {
            id,
            receiver: Some(receiver.clone()),
            ..Default::default()
        })
    }

    fn center_columns_decomposition(&mut self) {
        let data = self.data.as_ref().unwrap();
        self.column_means = Some(
            data.mean_axis(Axis(0))
                .unwrap()
                .into_shape([1, data.shape()[1]])
                .unwrap(),
        );
        self.n = Some(arr1(&[data.shape()[0] as f32]));
        let col_centered = data - self.column_means.as_ref().unwrap();
        let (_q, r) = col_centered.qr().unwrap();
        self.local_r = Some(r);

        if self.id != 0 {
            self.send_to_main();
        }
        self.send_to_neighbor_or_finalize();
        self.resolve_buffer();
        self.resolve_means_buffer();
    }

    fn resolve_buffer(&mut self) {
        while let Some(msg) = self.buffer.pop() {
            self.neighbors.get(self.id).unwrap().do_send(msg);
        }
    }

    fn resolve_means_buffer(&mut self) {
        while let Some(msg) = self.means_buffer.pop() {
            self.neighbors.get(self.id).unwrap().do_send(msg);
        }
    }

    fn send_to_main(&mut self) {
        let main = self.neighbors.get(0).expect("Does not have neighbors yet");
        main.do_send(PCAHelperMessage::Means {
            columns_means: self.column_means.as_ref().unwrap().clone(),
            n: self.data.as_ref().unwrap().shape()[0],
        });
    }

    fn next_2_power(&mut self) -> usize {
        let len = self.neighbors.len();
        2_i32.pow((len as f32).log2().ceil() as u32) as usize
    }

    fn send_to_neighbor_or_finalize(&mut self) {
        let s = self.next_2_power();
        let threshold = s.div(2_usize.pow((self.r_count + 1) as u32));
        let id = self.id;

        if id >= threshold && id > 0 {
            let neighbor_id = id - threshold;
            match self.neighbors.get(neighbor_id) {
                Some(neighbor) => {
                    neighbor
                        .do_send(PCAHelperMessage::Decomposition {
                            r: self.local_r.as_ref().unwrap().clone(),
                            count: self.r_count + 1,
                        });
                }
                None => panic!("No neighbor with id {} exists!", &neighbor_id),
            }
        } else if self.r_count == 0 && (id + threshold) >= self.neighbors.len() {
            self.r_count += 1;
            self.send_to_neighbor_or_finalize();
        } else if id == 0 && threshold == 0 {
            self.finalize();
        }
    }

    fn combine_sent_r(&mut self, remote_r: Array2<f32>) {
        match &self.local_r {
            Some(local_r) => {
                let (_q, combined_r) = concatenate(Axis(0), &[local_r.view(), remote_r.view()])
                    .unwrap()
                    .qr()
                    .unwrap();
                self.local_r = Some(combined_r);
                self.send_to_neighbor_or_finalize();
            }
            None => panic!("Cannot combine sent and local R, because no local R exists"),
        }
    }

    fn finalize(&mut self) {
        let column_means = self.column_means.as_ref().unwrap().to_owned();
        let dim = column_means.shape()[1];
        let n = self.n.as_ref().unwrap().view();
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
            self.local_r.as_ref().unwrap().view()
        ]
        .qr()
        .unwrap();

        self.receiver
            .as_ref()
            .unwrap()
            .do_send(PCAHelperMessage::Response {
                column_means: global_means,
                n: n.sum(),
                r,
            });
    }
}

impl Actor for PCAHelper {
    type Context = SyncContext<Self>;
}

impl Handler<PCAHelperMessage> for PCAHelper {
    type Result = ();

    fn handle(&mut self, msg: PCAHelperMessage, _ctx: &mut Self::Context) -> Self::Result {
        if let PCAHelperMessage::Setup { neighbors, data } = msg {
            self.neighbors.extend(neighbors);
            self.data = Some(data);
            self.center_columns_decomposition();
            return;
        }

        if self.local_r.is_none() {
            self.buffer.push(msg);
        } else {
            match msg {
                PCAHelperMessage::Setup {
                    neighbors: _,
                    data: _,
                } => (),
                PCAHelperMessage::Decomposition { r, count } => {
                    if count == self.r_count + 1 {
                        self.r_count = count;
                        self.combine_sent_r(r);
                        self.resolve_buffer();
                    } else {
                        self.buffer
                            .push(PCAHelperMessage::Decomposition { r, count });
                    }
                }
                PCAHelperMessage::Means { columns_means, n } => {
                    if self.column_means.is_some() {
                        self.column_means = Some(concatenate![
                            Axis(0),
                            self.column_means.as_ref().unwrap().clone(),
                            columns_means.view().into_dimensionality().unwrap()
                        ]);
                        self.n = Some(concatenate![
                            Axis(0),
                            self.n.as_ref().unwrap().clone(),
                            arr1(&[n as f32])
                        ]);
                    } else {
                        self.means_buffer
                            .push(PCAHelperMessage::Means { columns_means, n });
                    }
                }
                PCAHelperMessage::Components {
                    components: _,
                    means: _,
                } => println!("Components received"),
                PCAHelperMessage::Response {
                    column_means: _,
                    n: _,
                    r: _,
                } => println!("Response received"),
            }
        }
    }
}

impl Handler<PoisonPill> for PCAHelper {
    type Result = ();

    fn handle(&mut self, _msg: PoisonPill, ctx: &mut Self::Context) -> Self::Result {
        ctx.stop();
    }
}
