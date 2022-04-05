use actix::{Message, Recipient};
use actix_telepathy::RemoteMessage;

#[derive(Default)]
pub struct RotationProtocol<T: Message + RemoteMessage + Clone> {
    n_total: usize,
    pub n_received: usize,
    n_sent: usize,
    buffer: Vec<T>,
}

impl<T: Message + RemoteMessage + Clone> RotationProtocol<T> {
    pub fn start(&mut self, n_total: usize) {
        self.n_total = n_total;
    }

    pub fn received(&mut self, msg: &T) -> bool {
        if self.is_running() {
            self.n_received += 1;
            true
        } else {
            self.buffer.push(msg.clone());
            false
        }
    }

    pub fn sent(&mut self) {
        self.n_sent += 1;
    }

    pub fn is_running(&self) -> bool {
        (self.n_received < self.n_total) || (self.n_sent < self.n_total)
    }

    pub fn resolve_buffer(&mut self, rec: Recipient<T>)
    where
        T::Result: Send,
    {
        for _ in 0..self.buffer.len() {
            rec.do_send(self.buffer.remove(0)).unwrap();
        }
    }
}
