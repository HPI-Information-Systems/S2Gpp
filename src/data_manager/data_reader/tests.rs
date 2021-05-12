use ndarray::prelude::*;
use actix::prelude::*;
use crate::data_manager::data_reader::*;
use std::sync::{Arc, Mutex};
use crate::data_manager::data_reader::messages::DataReceivedMessage;
use ndarray_linalg::assert::close_l1;

struct DataTestReceiver {
    result: Arc<Mutex<Option<Array2<f32>>>>
}

impl Actor for DataTestReceiver {
    type Context = Context<Self>;

    fn stopped(&mut self, _ctx: &mut Context<Self>) {
        System::current().stop();
    }
}

impl Handler<DataReceivedMessage> for DataTestReceiver {
    type Result = ();

    fn handle(&mut self, msg: DataReceivedMessage, ctx: &mut Self::Context) -> Self::Result {
        *(self.result.lock().unwrap()) = Some(msg.data);
        ctx.stop();
    }
}

#[test]
fn test_data_distribution() {
    let result: Arc<Mutex<Option<Array2<f32>>>> = Arc::new(Mutex::new(None));
    let cloned = Arc::clone(&result);

    let data = read_data_("data/test.csv");

    let system = System::run(move || {
        let test_receiver = DataTestReceiver { result: cloned }.start();
        let data_receiver = DataReceiver::new(Some(test_receiver.recipient())).start();
        let data_receiver2 = DataReceiver::new(None).start();
        let data_reader = DataReader::new("data/test.csv",
                                          vec![data_receiver.recipient(), data_receiver2.recipient()],
                                          5).start();
    });

    let received = (*result.lock().unwrap()).as_ref().unwrap().clone();

    close_l1(&received, &data.slice(s![..55, ..]), 0.5);
}
