use ndarray::prelude::*;
use actix::prelude::*;
use crate::pca::*;
use crate::data_manager::data_reader::*;
use std::sync::{Arc, Mutex};

struct PCAReceiver {
    result: Arc<Mutex<Option<Array2<f32>>>>
}

impl Actor for PCAReceiver {
    type Context = Context<Self>;

    fn stopped(&mut self, _ctx: &mut Context<Self>) {
        System::current().stop();
    }
}

impl Handler<PCAResponse> for PCAReceiver {
    type Result = ();

    fn handle(&mut self, msg: PCAResponse, ctx: &mut Self::Context) -> Self::Result {
        *(self.result.lock().unwrap()) = Some(msg.components);
        ctx.stop();
    }
}

/*#[test]
fn test_runs_pca_3_nodes() {
    // todo: distributed test
    let result = Arc::new(Mutex::new(None));
    let cloned = Arc::clone(&result);

    let system = System::run(move || {
        let dataset = read_data_("data/test.csv");
        let receiver = PCAReceiver {result: cloned}.start();
        let pca1 = PCA::new(Some(receiver.recipient()), 0, 2).start();
        let pca2 = PCA::new(None, 1, 2).start();
        let pca3 = PCA::new(None, 2, 2).start();
        pca1.do_send(PCAMessage { data: dataset.slice(s![..50, ..]).to_shared(), cluster_nodes: vec![pca1.clone(), pca2.clone(), pca3.clone()] });
        pca2.do_send(PCAMessage { data: dataset.slice(s![50..75, ..]).to_shared(), cluster_nodes: vec![pca1.clone(), pca2.clone(), pca3.clone()] });
        pca3.do_send(PCAMessage { data: dataset.slice(s![75.., ..]).to_shared(), cluster_nodes: vec![pca1.clone(), pca2.clone(), pca3.clone()] });
    });

    let expects: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543]
    ]);

    let received = (*result.lock().unwrap()).as_ref().unwrap().clone();

    assert!(expects[[0,0]] == received[[0,0]]);
    assert!(expects[[0,1]] == received[[0,1]]);
    assert!(expects[[0,2]] == received[[0,2]]);
    assert!(expects[[1,0]] == received[[1,0]]);
    assert!(expects[[1,1]] == received[[1,1]]);
    assert!(expects[[1,2]] == received[[1,2]]);
}

#[test]
fn test_runs_pca_2_nodes() {
    let result = Arc::new(Mutex::new(None));
    let cloned = Arc::clone(&result);

    let system = System::run(move || {
        let dataset = read_data_("data/test.csv");
        let receiver = PCAReceiver {result: cloned}.start();
        let pca1 = PCA::new(Some(receiver.recipient()), 0, 2).start();
        let pca2 = PCA::new(None, 1, 2).start();
        pca1.do_send(PCAMessage { data: dataset.slice(s![..50, ..]).to_shared(), cluster_nodes: vec![pca1.clone(), pca2.clone()] });
        pca2.do_send(PCAMessage { data: dataset.slice(s![50.., ..]).to_shared(), cluster_nodes: vec![pca1.clone(), pca2.clone()] });
    });

    let expects: Array2<f32> = arr2(&[
        [0.7265024, -0.39373094, 0.5631784],
        [0.57647973, -0.09682596, -0.8113543]
    ]);

    let received = (*result.lock().unwrap()).as_ref().unwrap().clone();

    println!("received {:?}", received);

    assert!((expects[[0,0]] - received[[0,0]]).abs() < 0.00001);
    assert!((expects[[0,1]] - received[[0,1]]).abs() < 0.00001);
    assert!((expects[[0,2]] - received[[0,2]]).abs() < 0.00001);
    assert!((expects[[1,0]] - received[[1,0]]).abs() < 0.00001);
    assert!((expects[[1,1]] - received[[1,1]]).abs() < 0.00001);
    assert!((expects[[1,2]] - received[[1,2]]).abs() < 0.00001);
}*/
