use actix::io::SinkWrite;
use actix::{Actor, System};
use futures_sink::Sink;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

pub struct MySink<T>
where
    T: Unpin,
{
    sender: mpsc::UnboundedSender<T>,
    content: Option<T>,
}

impl<T> MySink<T>
where
    T: Unpin,
{
    pub fn new(sender: mpsc::UnboundedSender<T>) -> Self {
        MySink {
            sender,
            content: None,
        }
    }
}

impl<T> Sink<T> for MySink<T>
where
    T: Unpin + Clone,
{
    type Error = ();

    fn poll_ready(self: Pin<&mut Self>, _ctx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        let this = self.get_mut();
        match &this.content {
            Some(content) => {
                let _r = this.sender.send(content.clone());
                Poll::Ready(Ok(()))
            }
            None => Poll::Ready(Ok(())),
        }
    }

    fn start_send(self: Pin<&mut Self>, item: T) -> Result<(), Self::Error> {
        self.get_mut().content = Some(item);
        Ok(())
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.poll_ready(cx)
    }

    fn poll_close(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Result<(), Self::Error>> {
        self.poll_ready(cx)
    }
}

pub struct SinkActor<T>
where
    T: Unpin + Clone,
{
    pub sink: SinkWrite<T, MySink<T>>,
    pub result: Option<T>,
}

impl<T> SinkActor<T>
where
    T: Unpin + Clone,
{
    pub fn new(sink: SinkWrite<T, MySink<T>>) -> Self {
        Self { sink, result: None }
    }
}

impl<T> Actor for SinkActor<T>
where
    T: Unpin + 'static + Clone,
{
    type Context = actix::Context<SinkActor<T>>;
}

impl<T> actix::io::WriteHandler<()> for SinkActor<T>
where
    T: Unpin + 'static + Clone,
{
    fn finished(&mut self, _ctxt: &mut Self::Context) {
        System::current().stop();
    }
}
