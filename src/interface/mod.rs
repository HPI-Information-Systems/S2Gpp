mod sink;
#[cfg(test)]
mod tests;

use crate::interface::sink::{MySink, SinkActor};
use crate::training::DetectionResponse;
use crate::{Parameters, StartTrainingMessage, Training};
use actix::io::SinkWrite;
use actix::{Actor, Handler};
use anyhow::{Error, Result};
use ndarray::{Array1, Array2};
use tokio::sync::mpsc;

pub trait SyncInterface<A> {
    fn init(parameters: Parameters) -> Self;
    fn fit(&mut self, data: Array2<A>) -> Result<SyncResult>;
}

pub type SyncResult = Array1<f32>;

impl Handler<DetectionResponse> for SinkActor<SyncResult> {
    type Result = ();

    fn handle(&mut self, msg: DetectionResponse, _ctx: &mut Self::Context) -> Self::Result {
        let _ = self.sink.write(msg.anomaly_score);
        self.sink.close()
    }
}

pub async fn actor_fit(actor: Training, data: Array2<f32>) -> Result<SyncResult> {
    let (sender, mut receiver) = mpsc::unbounded_channel();

    let sink_actor = SinkActor::create(move |ctx| {
        let sink = MySink::new(sender);
        SinkActor::new(SinkWrite::new(sink, ctx))
    });

    let addr = actor.start();
    addr.do_send(StartTrainingMessage {
        nodes: Default::default(),
        source: Some(sink_actor.recipient()),
        data: Some(data),
    });

    if let Some(r) = receiver.recv().await {
        Ok(r)
    } else {
        Err(Error::msg("Await resulted in None value!"))
    }
}
