use actix::prelude::*;
use crate::training::Training;
use crate::parameters::Parameters;
use crate::training::segmentation::{Segmentation, SegmentedPointWithId, PointWithId, SegmentedTransition, SegmentedMessage};
use std::f32::consts::PI;
use ndarray::arr1;
use actix_telepathy::Cluster;
use port_scanner::request_open_port;
use tokio::time::{Duration, sleep};
use crate::training::intersection_calculation::IntersectionCalculationDone;
use std::sync::{Arc, Mutex};


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
        assert_eq!([2, 3], self.intersection_calculation.intersections.get(&0).unwrap().as_slice());
        assert_eq!([4, 5], self.intersection_calculation.intersections.get(&1).unwrap().as_slice());
        assert_eq!([6, 7], self.intersection_calculation.intersections.get(&2).unwrap().as_slice());

        assert_eq!(51., self.intersection_calculation.intersection_coords_by_segment.get(&0).unwrap().get(&49).unwrap()[0]);
        assert_eq!(102., self.intersection_calculation.intersection_coords_by_segment.get(&0).unwrap().get(&100).unwrap()[0]);
        assert_eq!(153., self.intersection_calculation.intersection_coords_by_segment.get(&0).unwrap().get(&151).unwrap()[0]);

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
    training.segmentation.segments = generate_segmented_transitions();
    training.intersection_calculation.recipient = Some(checker.clone().recipient());
    let training_addr = training.start();
    training_addr.do_send(SegmentedMessage);
    sleep(Duration::from_millis(3000)).await;
    training_addr.do_send(CheckingMessage{ rec: Some(checker.recipient()) });
    sleep(Duration::from_millis(200)).await;
    assert!(*success.lock().unwrap())
}


fn generate_segmented_transitions() -> Vec<SegmentedTransition> {
    let segments = 100;
    let segment_size = (2.0 * PI) / segments as f32;
    let spin_size = 51;
    let points: Vec<SegmentedPointWithId> = (1..1001).into_iter().map(|x| {
        let theta = (2.0 * PI) * ((x % spin_size) as f32 / spin_size as f32);
        let segment_id = (theta / segment_size) as usize % segments;
        let radius = x as f32;
        let coords = arr1(&[radius * theta.cos(), radius * theta.sin()]);
        SegmentedPointWithId {
            segment_id,
            point_with_id: PointWithId { id: x-1, coords }
        }
    }).collect();

    let mut transitions = vec![];
    let mut last_point = None;

    for point in points {
        match last_point {
            Some(last_point) => {
                let transition = SegmentedTransition::new(last_point, point.clone());
                transitions.push(transition);
            },
            None => ()
        }
        last_point = Some(point);
    }

    transitions
}
