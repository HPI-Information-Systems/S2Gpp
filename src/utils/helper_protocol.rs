
#[derive(Default)]
pub struct HelperProtocol {
    pub n_total: usize,
    pub n_sent: usize,
    pub n_received: usize
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

    pub fn are_tasks_left(&self) -> bool {
        self.tasks_left() > 0
    }

    pub fn tasks_left(&self) -> usize {
        self.n_total - (self.n_received + self.n_sent)
    }
}
