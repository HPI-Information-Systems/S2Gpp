use ndarray::prelude::*;
use actix::prelude::*;
use crate::meanshift::*;
use crate::data_reader::*;
use std::sync::{Arc, Mutex};

struct MeanShiftReceiver {
    result: Arc<Mutex<Option<Array2<f32>>>>
}

impl Actor for MeanShiftReceiver {
    type Context = Context<Self>;

    fn stopped(&mut self, _ctx: &mut Context<Self>) {
        System::current().stop();
    }
}

impl Handler<MeanShiftResponse> for MeanShiftReceiver {
    type Result = ();

    fn handle(&mut self, msg: MeanShiftResponse, ctx: &mut Self::Context) -> Self::Result {
        *(self.result.lock().unwrap()) = Some(msg.cluster_centers);
        ctx.stop();
    }
}

#[test]
fn test_runs_meanshift() {
    let result = Arc::new(Mutex::new(None));
    let cloned = Arc::clone(&result);

    let system = System::run(move || {
        let dataset = read_data_("data/test.csv");
        let receiver = MeanShiftReceiver {result: cloned}.start();
        let meanshift = MeanShift::new(20).start();
        meanshift.do_send(MeanShiftMessage { source: Some(receiver.recipient()), data: dataset });
    });

    let expects: Array2<f32> = arr2(&[
        [0.5185592, 0.43546146, 0.5697923]
    ]);

    let received = (*result.lock().unwrap()).as_ref().unwrap().clone();

    assert!(expects[[0,0]] == received[[0,0]]);
    assert!(expects[[0,1]] == received[[0,1]]);
    assert!(expects[[0,2]] == received[[0,2]]);
}
