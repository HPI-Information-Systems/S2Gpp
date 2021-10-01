use indicatif::ProgressBar;


#[derive(Default)]
pub struct S2GppProgressBar {
    progress_bar: Option<ProgressBar>,
    correct_env: bool
}


impl S2GppProgressBar {
    pub fn new(env: &str) -> Self {
        let correct_env = std::env::var("RUST_LOG").unwrap_or(String::from("info")).eq(env);
        Self {
            progress_bar: None,
            correct_env
        }
    }

    pub fn new_from_len(env: &str, len: usize) -> Self {
        let mut this = Self::new(env);
        this.create_pb(len);
        this
    }

    pub fn create_pb(&mut self, len: usize) {
        if self.correct_env {
            self.progress_bar = Some(ProgressBar::new(len as u64));
        }
    }

    pub fn inc_or_set(&mut self, len: usize) {
        if self.correct_env {
            match &self.progress_bar {
                None => self.create_pb(len),
                Some(_) => self.inc()
            }
        }
    }

    pub fn inc(&self) {
        match &self.progress_bar {
            Some(pb) => pb.inc(1),
            None => ()
        }
    }

    pub fn finish_and_clear(&self) {
        match &self.progress_bar {
            Some(pb) => pb.finish_and_clear(),
            None => ()
        }
    }
}
