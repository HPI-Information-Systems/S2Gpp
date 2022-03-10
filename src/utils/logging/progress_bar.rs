use indicatif::ProgressBar;


#[derive(Default)]
pub struct S2GppProgressBar {
    progress_bar: Option<ProgressBar>,
    correct_env: bool
}


impl S2GppProgressBar {
    pub fn new(env: &str) -> Self {
        let correct_env = Self::check_correct_env(env);
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

    pub fn check_correct_env(env: &str) -> bool {
        std::env::var("RUST_LOG").unwrap_or_else(|_| String::from("info")).eq(env)
    }

    pub fn create_pb(&mut self, len: usize) {
        if self.correct_env {
            self.progress_bar = Some(ProgressBar::new(len as u64));
        }
    }

    pub fn inc_or_set(&mut self, env: &str, len: usize) {
        self.correct_env = Self::check_correct_env(env);
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

    pub fn inc_by(&self, delta: u64) {
        match &self.progress_bar {
            Some(pb) => pb.inc(delta),
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
