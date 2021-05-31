mod messages;
mod helper;

use actix::{Handler, Addr, SyncArbiter, AsyncContext};
use ndarray::{ArcArray2};
pub use crate::data_manager::preprocessor::messages::{ProcessedColumnMessage, PreprocessColumnMessage, PreprocessingDoneMessage};

use crate::data_manager::preprocessor::helper::{PreprocessorHelper};




use crate::messages::PoisonPill;

use crate::data_manager::DataManager;


pub struct Preprocessing {
    helpers: Addr<PreprocessorHelper>,
    n_cols_total: usize,
    n_cols_processed: usize,
    n_cols_distributed: usize
}


impl Preprocessing {
    pub fn new(data: ArcArray2<f32>, n_threads: usize, window_size: usize) -> Self {
        let n_cols_total = data.ncols();
        let helpers = SyncArbiter::start(n_threads, move || {
            PreprocessorHelper::new(data.clone(), window_size)
        });

        Self {
            helpers,
            n_cols_total,
            n_cols_processed: 0,
            n_cols_distributed: 0
        }
    }
}


pub trait Preprocessor {
    fn distribute_work(&mut self, source: Addr<Self>) where Self: actix::Actor;
}


impl Preprocessor for DataManager {
    fn distribute_work(&mut self, source: Addr<Self>) {
        let preprocessing = self.preprocessing.as_mut().unwrap();
        let max_distribution = self.parameters.n_threads - (preprocessing.n_cols_distributed - preprocessing.n_cols_processed);

        for (i, column) in (preprocessing.n_cols_distributed..preprocessing.n_cols_total).enumerate() {
            if i < max_distribution {
                preprocessing.helpers.do_send(PreprocessColumnMessage { column, source: source.clone(), std: 0.0 });
                preprocessing.n_cols_distributed += 1;
            } else {
                break;
            }
        }
    }
}


impl Handler<ProcessedColumnMessage> for DataManager {
    type Result = ();

    fn handle(&mut self, _msg: ProcessedColumnMessage, ctx: &mut Self::Context) -> Self::Result {
        let preprocessing = self.preprocessing.as_mut().unwrap();
        preprocessing.n_cols_processed += 1;
        if preprocessing.n_cols_processed == preprocessing.n_cols_total {
            ctx.address().do_send(PreprocessingDoneMessage);
            preprocessing.helpers.do_send(PoisonPill);
        } else {
            self.distribute_work(ctx.address());
        }
    }
}
