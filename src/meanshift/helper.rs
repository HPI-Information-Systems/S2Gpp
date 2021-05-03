use actix::{Actor, ActorContext, SyncContext, Handler};
use crate::meanshift::messages::{MeanShiftHelperWorkMessage};
use actix::dev::MessageResponse;
use ndarray::prelude::*;
use kdtree::{KdTree, ErrorKind};
use std::collections::HashMap;
use ndarray::{ArcArray2, ArcArray1};
use num_traits::Float;
use crate::meanshift::{RefArray, MeanShiftHelperResponse, euclidean};
use std::sync::Arc;
use std::time::SystemTime;


pub struct MeanShiftHelper {
    data: ArcArray2<f32>,
    tree: Arc<KdTree<f32, usize, RefArray>>,
    bandwidth: f32
}

impl MeanShiftHelper {
    pub fn new(data: ArcArray2<f32>, tree: Arc<KdTree<f32, usize, RefArray>>, bandwidth: f32) -> Self {
        Self {
            data,
            tree,
            bandwidth
        }
    }

    fn mean_shift_single(&mut self, seed: usize, bandwidth: f32) -> (Array1<f32>, usize, usize) {
        //let start = SystemTime::now();
        let stop_threshold = 1e-3 * bandwidth;
        let max_iter = 300;

        let mut my_mean = self.data.select(Axis(0), &[seed]).mean_axis(Axis(0)).unwrap();
        let mut my_old_mean = my_mean.clone();
        let mut iterations: usize = 0;
        let mut points_within_len: usize = 0;

        loop {
            let within_result = self.tree.within(my_mean.as_slice().unwrap(), bandwidth, &euclidean);
            let neighbor_ids: Vec<usize> = match within_result {
                Ok(neighbors) => neighbors.into_iter().map(|(_, x)| x.clone()).collect(),
                Err(_) => break
            };
            let points_within = self.data.select(Axis(0), neighbor_ids.as_slice());
            points_within_len = points_within.shape()[0];
            my_old_mean = my_mean;
            my_mean = points_within.mean_axis(Axis(0)).unwrap();

            if euclidean(my_mean.as_slice().unwrap(), my_old_mean.as_slice().unwrap()) < stop_threshold || iterations >= max_iter {
                break
            }

            iterations += 1;
        }

        //println!("took {} microseconds", SystemTime::now().duration_since(start).unwrap().as_micros());

        (my_mean, points_within_len, iterations)
    }
}

impl Actor for MeanShiftHelper {
    type Context = SyncContext<Self>;
}

impl Handler<MeanShiftHelperWorkMessage> for MeanShiftHelper {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftHelperWorkMessage, ctx: &mut Self::Context) -> Self::Result {
        let (mean, points_within_len, iterations) = self.mean_shift_single(msg.start_center, self.bandwidth);
        msg.source.do_send(MeanShiftHelperResponse { source: ctx.address().recipient(), mean, points_within_len, iterations });
    }
}
