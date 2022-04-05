use console::style;
use log::*;

pub struct ConsoleLogger {
    step: usize,
    from: usize,
    title: String,
}

impl ConsoleLogger {
    pub fn new(step: usize, from: usize, title: String) -> Self {
        Self { step, from, title }
    }

    fn format_step(&self) -> String {
        format!("[{}/{}]", self.step, self.from)
    }

    pub fn print(&self) {
        info!(
            "{} {}...",
            style(self.format_step()).bold().dim(),
            self.title
        );
    }
}
