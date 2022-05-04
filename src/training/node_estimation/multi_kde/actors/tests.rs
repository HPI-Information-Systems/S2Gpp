use crate::training::node_estimation::multi_kde::actors::messages::MultiKDEMessage;
use crate::training::node_estimation::multi_kde::actors::MultiKDEActor;
use actix::{Actor, Context, Handler};
use meanshift_rs::ClusteringResponse;
use ndarray::{arr2, Array1, Array2, Axis};
use ndarray_stats::CorrelationExt;
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

#[ignore]
#[actix_rt::test]
async fn test_cov() {
    let x = arr2(&[
        [-1.1044426],
        [3.3079557],
        [-2.6531453],
        [-0.7709985],
        [-1.1930218],
        [11.356522],
        [-1.1818404],
        [-2.8623846],
        [-1.6397605],
        [-1.9247131],
        [-0.33105826],
        [-0.57120085],
        [6.5464177],
        [0.91874397],
        [-3.3256016],
        [1.6511793],
        [-4.301383],
        [0.3333578],
        [-14.434255],
        [-1.6871758],
        [2.098528],
        [3.397635],
        [-1.4510679],
        [0.039141178],
        [-0.9916377],
        [1.8591633],
        [3.3673396],
        [-1.5271537],
        [-2.0864549],
        [0.05202365],
        [-6.286045],
        [6.923059],
        [-5.992464],
        [2.4069607],
        [-0.14090195],
        [-5.526738],
        [1.3183851],
        [-0.04570675],
        [-1.3094206],
        [1.5622854],
        [2.3346844],
        [-0.519022],
        [-0.01082468],
        [2.4892955],
        [1.676794],
        [-3.2251143],
        [-4.219629],
        [-1.8621855],
        [-2.6605353],
    ]);
    println!("{:?}", x.t().cov(1.).unwrap());
    assert!(false)
}
