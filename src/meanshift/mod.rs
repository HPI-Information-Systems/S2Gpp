mod helper;
mod messages;
#[cfg(test)]
mod tests;

use actix::{Actor, ActorContext, Context, Addr, SyncArbiter, Handler, Recipient, AsyncContext};
use ndarray::prelude::*;
use ndarray::{Array2, Axis, ArcArray1, Array1, ArrayView2, concatenate};
use crate::meanshift::helper::MeanShiftHelper;
pub use crate::meanshift::messages::{MeanShiftMessage, MeanShiftResponse, MeanShiftHelperResponse, MeanShiftHelperWorkMessage};

use kdtree::KdTree;
use kdtree::distance::squared_euclidean;
use std::cmp::Ordering;
use num_traits::float::Float;
use std::sync::Arc;
use std::collections::HashMap;
use std::ops::{Sub};
use sortedvec::sortedvec;
use std::iter::FromIterator;
use std::time::{SystemTime};
use log::*;

sortedvec! {
    struct SortedVec {
        fn derive_key(x: &(Array1<f32>, usize, usize, usize)) -> usize {
            (*x).1
        }
    }
}

pub enum DistanceMeasure {
    SquaredEuclidean,
    Manhattan
}

impl DistanceMeasure {
    pub fn call(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::SquaredEuclidean => squared_euclidean(a, b),
            Self::Manhattan => {
                a.iter().zip(b.iter()).map(|(a_, b_)| {
                    a_.sub(b_).abs()
                }).sum()
            }
        }
    }
}

pub struct MeanShift {
    helpers: Option<Addr<MeanShiftHelper>>,
    n_threads: usize,
    dataset: Option<Array2<f32>>,
    tree: Option<Arc<KdTree<f32, usize, RefArray>>>,
    center_tree: Option<KdTree<f32, usize, RefArray>>,
    receiver: Option<Recipient<MeanShiftResponse>>,
    means: Vec<(Array1<f32>, usize, usize, usize)>,
    centers_sent: usize,
    bandwidth: f32,
    start_time: Option<SystemTime>
}

impl Actor for MeanShift {
    type Context = Context<Self>;
}

impl MeanShift {
    pub fn new(n_threads: usize) -> Self {
        Self {
            helpers: None,
            n_threads,
            dataset: None,
            tree: None,
            center_tree: None,
            receiver: None,
            means: vec![],
            centers_sent: 0,
            bandwidth: 0.0,
            start_time: None
        }
    }

    fn create_helpers(&mut self) {
        let data = self.dataset.as_ref().unwrap().clone().into_shared();
        let tree = self.tree.as_ref().unwrap().clone();
        let bandwidth = self.bandwidth;
        self.helpers = Some(SyncArbiter::start(self.n_threads, move || MeanShiftHelper::new(
            data.clone(), tree.clone(), bandwidth
        )))
    }

    fn estimate_bandwidth(&mut self) -> f32 {
        let quantile = 0.3_f32;

        match &self.dataset {
            Some(data) => {

                let n_neighbors = (data.shape()[0] as f32 * quantile).max(1.0) as usize;

                let mut tree = KdTree::new(data.shape()[1]);
                for (i, point) in data.axis_iter(Axis(0)).enumerate() {
                    tree.add(RefArray(point.to_shared()), i);
                }

                let bandwidth: f32 = data.axis_iter(Axis(0)).map(|x| {
                    let nearest = tree.nearest(x.to_slice().unwrap(), n_neighbors, &squared_euclidean).unwrap();
                    let sum = nearest.into_iter().map(|(dist, _)| dist).fold(f32::min_value(), f32::max);
                    sum.clone()
                }).sum();

                self.tree = Some(Arc::new(tree));
                bandwidth / data.shape()[0] as f32
            },
            _ => panic!("Data not yet set!")
        }
    }

    fn distribute_data(&mut self, rec: Recipient<MeanShiftHelperResponse>) {
        self.bandwidth = self.estimate_bandwidth();
        debug!("bandwidth {}", self.bandwidth);
        self.build_center_tree();
        self.create_helpers();
        self.start_time = Some(SystemTime::now());
        self.centers_sent = self.n_threads;
        for t in 0..self.n_threads {
            self.helpers.as_ref().unwrap().do_send(MeanShiftHelperWorkMessage {
                source: rec.clone(),
                start_center: t
            });
        }
    }

    fn build_center_tree(&mut self) {
        self.center_tree = Some(KdTree::new(self.dataset.as_ref().unwrap().shape()[1]));
    }

    fn add_means(&mut self, mean: Array1<f32>, points_within_len: usize, iterations: usize) {
        if points_within_len > 0 {
            let identifier = self.means.len();
            self.center_tree.as_mut().unwrap().add(RefArray(mean.to_shared()), identifier);
            self.means.push((mean, points_within_len, iterations, identifier));
        }
    }

    fn collect_means(&mut self) -> Array2<f32> {
        self.means.sort_by(|(a, a_intensity, _, _), (b, b_intensity, _, _)|
            {
                let intensity_cmp = a_intensity.cmp(b_intensity);
                match &intensity_cmp {
                    Ordering::Equal => {
                        a.slice_cmp(b).reverse()
                    },
                    _ => intensity_cmp.reverse()
                }
            }
        );
        debug!("duration {}", SystemTime::now().duration_since(self.start_time.unwrap()).unwrap().as_millis());
        self.means.dedup_by_key(|(x, _, _, _)| x.clone());

        let mut unique: HashMap<usize, bool> = HashMap::from_iter(self.means.iter().map(|(_, _, _, i)| (*i, true)));

        for (mean, _, _, i) in self.means.iter(){
            if unique[i] {
                let neighbor_idxs = self.center_tree.as_ref().unwrap().within(
                    mean.as_slice().unwrap(),
                    self.bandwidth,
                    &squared_euclidean).unwrap();
                for (_, neighbor) in neighbor_idxs {
                    match unique.get_mut(neighbor) {
                        None => {}
                        Some(val) => { *val = false}
                    }
                }
                *unique.get_mut(i).unwrap() = true;
            }
        }

        let dim = self.means[0].0.len();

        let cluster_centers: Vec<ArrayView2<f32>> =  self.means.iter().filter_map(|(mean, _, _, identifier)| {
            if unique[identifier] {
                Some(mean.view().into_shape((1, dim)).unwrap())
            } else {
                None
            }
        }).collect();

        concatenate(Axis(0), cluster_centers.as_slice()).unwrap()
    }
}

impl Handler<MeanShiftMessage> for MeanShift {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftMessage, ctx: &mut Self::Context) -> Self::Result {
        match &self.dataset {
            None => {
                self.dataset = Some(msg.data);
                self.receiver = msg.source;
                self.distribute_data(ctx.address().recipient())
            },
            _ => ()
        }
    }
}

impl Handler<MeanShiftHelperResponse> for MeanShift {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftHelperResponse, ctx: &mut Self::Context) -> Self::Result {
        // todo: do not send one by one but chunks
        self.add_means(msg.mean, msg.points_within_len, msg.iterations);

        if self.means.len() == self.dataset.as_ref().unwrap().shape()[0] {
            debug!("all means received");

            let cluster_centers = self.collect_means();
            debug!("cluster_centers {:?}", cluster_centers);
            match &self.receiver {
                Some(recipient) => { recipient.do_send(MeanShiftResponse { cluster_centers } ); },
                None => ()
            }
            ctx.stop();
        } else if self.centers_sent < self.dataset.as_ref().unwrap().shape()[0] {
            let start_center = self.centers_sent;
            self.centers_sent += 1;
            msg.source.do_send(MeanShiftHelperWorkMessage { source: ctx.address().recipient(), start_center });
        }
    }
}

pub struct RefArray(pub ArcArray1<f32>);

impl AsRef<[f32]> for RefArray {
    fn as_ref(&self) -> &[f32] {
        let arc_array = &self.0;
        arc_array.as_slice().unwrap()
    }
}

pub trait SliceComp {
    fn slice_cmp(&self, b: &Self) -> Ordering;
}

impl SliceComp for Array1<f32> {
    fn slice_cmp(&self, other: &Self) -> Ordering {
        debug_assert!(self.len() == other.len());
        let a = self.as_slice().unwrap();
        let b = other.as_slice().unwrap();
        for i in 0..b.len() {
            let cmp = a[i].partial_cmp(&b[i]).unwrap();
            if cmp.ne(&Ordering::Equal) {
                return cmp
            }
        }
        Ordering::Equal
    }
}
