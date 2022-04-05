use crate::training::node_estimation::multi_kde::actors::messages::MultiKDEMessage;
use crate::training::node_estimation::multi_kde::actors::MultiKDEActor;
use actix::{Actor, Context, Handler};
use meanshift_rs::ClusteringResponse;
use ndarray::{Array1, Array2, Axis};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::time::sleep;

struct TestReceiver {
    labels: Arc<Mutex<Vec<usize>>>,
}

impl Actor for TestReceiver {
    type Context = Context<Self>;
}

impl Handler<ClusteringResponse<f32>> for TestReceiver {
    type Result = ();

    fn handle(&mut self, msg: ClusteringResponse<f32>, _ctx: &mut Self::Context) -> Self::Result {
        *(self.labels.lock().unwrap().deref_mut()) = msg.labels;
    }
}

fn setup_data(size: usize) -> (Array2<f32>, Vec<usize>) {
    let mut points = vec![1.0; size];
    points.extend(vec![2.0; size]);
    let data = Array1::from(points).insert_axis(Axis(1));
    let mut expected = vec![0; size];
    expected.extend(vec![1; size]);
    (data, expected)
}

#[actix_rt::test]
async fn correct_result() {
    let (data, expected) = setup_data(50);

    let labels = Arc::new(Mutex::new(vec![]));
    let receiver = (TestReceiver {
        labels: labels.clone(),
    })
    .start();
    let mkde = MultiKDEActor::new(receiver.recipient(), 1).start();
    mkde.do_send(MultiKDEMessage { data });
    sleep(Duration::from_millis(2000)).await;
    assert_eq!(labels.lock().unwrap().deref().clone(), expected);
}

#[actix_rt::test]
async fn correct_result_multithreading() {
    let (data, expected) = setup_data(50);

    let labels = Arc::new(Mutex::new(vec![]));
    let receiver = (TestReceiver {
        labels: labels.clone(),
    })
    .start();
    let mkde = MultiKDEActor::new(receiver.recipient(), 4).start();
    mkde.do_send(MultiKDEMessage { data });
    sleep(Duration::from_millis(1000)).await;
    assert_eq!(labels.lock().unwrap().deref().clone(), expected);
}
