use console::{style, Emoji};


pub struct ConsoleLogger {
    step: usize,
    from: usize,
    title: String
}

impl ConsoleLogger {
    pub fn new(step: usize, from: usize, title: String) -> Self {
        Self {
            step,
            from,
            title
        }
    }

    fn format_step(&self) -> &str {
        format!("[{}/{}]", self.step, self.from).as_str()
    }

    pub fn print(&self) {
        println!(
            "{} {}...",
            style(self.format_step()).bold().dim(),
            self.title
        );
    }
}
