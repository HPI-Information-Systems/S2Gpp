use actix::prelude::*;
use crate::training::Training;
use crate::parameters::Parameters;
use crate::training::segmentation::{Segmentation, SegmentedMessage};
use std::f32::consts::PI;
use ndarray::arr1;
use actix_telepathy::Cluster;
use port_scanner::request_open_port;
use tokio::time::{Duration, sleep};
use crate::training::intersection_calculation::IntersectionCalculationDone;
use std::sync::{Arc, Mutex};
use crate::data_store::DataStore;
use crate::data_store::point::Point;
use crate::data_store::transition::{Transition};


#[derive(Default)]
struct Checker {
    pub success: Arc<Mutex<bool>>
}

impl Actor for Checker {
    type Context = Context<Self>;
}

impl Handler<CheckingMessage> for Checker {
    type Result = ();

    fn handle(&mut self, _msg: CheckingMessage, _ctx: &mut Self::Context) -> Self::Result {
        *(self.success.lock().unwrap()) = true;
    }
}

impl Handler<IntersectionCalculationDone> for Checker {
    type Result = ();

    fn handle(&mut self, _msg: IntersectionCalculationDone, _ctx: &mut Self::Context) -> Self::Result {

    }
}


#[derive(Message)]
#[rtype(Result = "()")]
struct CheckingMessage {
    pub rec: Option<Recipient<CheckingMessage>>
}


impl Handler<CheckingMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: CheckingMessage, _ctx: &mut Self::Context) -> Self::Result {
        for intersection in self.data_store.get_intersections_from_segment(0).unwrap() {
            if intersection.get_from_id().eq(&49) {
                assert_eq!(51., intersection.get_coordinates()[0]);
            } else if intersection.get_from_id().eq(&100) {
                assert_eq!(102., intersection.get_coordinates()[0]);
            } else if intersection.get_from_id().eq(&151) {
                assert_eq!(153., intersection.get_coordinates()[0]);
            }
        }

        msg.rec.unwrap().do_send(CheckingMessage { rec: None }).unwrap();
    }
}


#[actix_rt::test]
async fn get_intersections() {
    let _cluster = Cluster::new(format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap(), vec![]);
    let parameters = Parameters::default();
    let mut training = Training::new(parameters);

    let success = Arc::new(Mutex::new(false));
    let checker = Checker { success: success.clone() }.start();

    training.segmentation = Segmentation::default();
    generate_segmented_transitions(&mut training.data_store);
    training.intersection_calculation.recipient = Some(checker.clone().recipient());
    let training_addr = training.start();
    training_addr.do_send(SegmentedMessage);
    sleep(Duration::from_millis(3000)).await;
    training_addr.do_send(CheckingMessage{ rec: Some(checker.recipient()) });
    sleep(Duration::from_millis(200)).await;
    assert!(*success.lock().unwrap())
}


fn generate_segmented_transitions(data_store: &mut DataStore) {
    let segments = 100;
    let segment_size = (2.0 * PI) / segments as f32;
    let spin_size = 51;
    for x in (1..1001).into_iter() {
        let theta = (2.0 * PI) * ((x % spin_size) as f32 / spin_size as f32);
        let segment_id = (theta / segment_size) as usize % segments;
        let radius = x as f32;
        let coords = arr1(&[radius * theta.cos(), radius * theta.sin()]);
        let point = Point::new(x-1, coords, segment_id);
        data_store.add_point(point);
    };

    let mut transitions = vec![];
    let mut last_point = None;

    for point in data_store.get_points() {
        match last_point {
            Some(last_point) => {
                let transition = Transition::new(last_point, point.clone());
                transitions.push(transition);
            },
            None => ()
        }
        last_point = Some(point);
    }

    data_store.add_transitions(transitions);
}
