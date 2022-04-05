#[derive(Default)]
pub struct HelperProtocol {
    pub n_total: usize,
    pub n_sent: usize,
    pub n_received: usize,
}

impl HelperProtocol {
    pub fn sent(&mut self) {
        self.n_sent += 1;
    }

    pub fn received(&mut self) {
        self.n_sent -= 1;
        self.n_received += 1;
    }

    pub fn is_running(&self) -> bool {
        self.n_received < self.n_total
    }
}
