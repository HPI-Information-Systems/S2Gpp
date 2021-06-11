use actix::prelude::*;
use crate::training::Training;
use crate::parameters::Parameters;
use crate::training::messages::SegmentedMessage;
use crate::training::segmenter::{Segmentation, SegmentedPointWithId, PointWithId};
use std::f32::consts::PI;
use ndarray::{Array1, arr1};
use actix::dev::MessageResponse;
use std::thread::sleep;
use std::time::Duration;
use actix_telepathy::Cluster;
use port_scanner::request_open_port;
use actix_rt::time::delay_for;
use crate::training::intersection_calculation::Transition;


#[derive(Message)]
#[rtype(Result = "()")]
struct CheckingMessage {

}


impl Handler<CheckingMessage> for Training {
    type Result = ();

    fn handle(&mut self, msg: CheckingMessage, ctx: &mut Self::Context) -> Self::Result {
        println!("{:?}", self.intersection_calculation.intersections.get(&Transition(1,2)));
        //println!("{:?}", self.intersection_calculation.intersection_coords);
        ctx.stop();
        System::current().stop();
    }
}


#[actix_rt::test]
async fn get_intersections() {
    let cluster = Cluster::new(format!("127.0.0.1:{}", request_open_port().unwrap_or(8000)).parse().unwrap(), vec![]);
    let parameters = Parameters::default();
    let mut training = Training::new(parameters);

    training.segmentation = Segmentation {
        segments: vec![],
        own_segment: generate_segmented_points(),
        n_received: 0
    };
    let training_addr = training.start();
    training_addr.do_send(SegmentedMessage);
    delay_for(Duration::from_millis(3000)).await;
    training_addr.do_send(CheckingMessage{});
    delay_for(Duration::from_millis(200)).await;
}


fn generate_segmented_points() -> Vec<SegmentedPointWithId> {
    let segments = 100;
    let segment_size = (2.0 * PI) / segments as f32;
    let spin_size = 50;
    let points: Vec<SegmentedPointWithId> = (1..1001).into_iter().map(|x| {
        let theta = (2.0 * PI) * ((x % spin_size) as f32 / spin_size as f32);
        let segment_id = (theta / segment_size) as usize % segments;
        let radius = x as f32;
        let coords = arr1(&[radius * theta.cos(), radius * theta.sin()]);
        SegmentedPointWithId {
            segment_id,
            point_with_id: PointWithId { id: x, coords }
        }
    }).collect();
    points
}
