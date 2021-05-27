mod messages;
mod helper;
#[cfg(test)]
mod tests;

use actix::{Actor, ActorContext, SyncContext, Context, Handler, Addr, SyncArbiter, Recipient, AsyncContext};
use ndarray::{ArcArray2, Axis, Array1, Array2};
pub use crate::data_manager::preprocessor::messages::{ProcessedColumnMessage, PreprocessColumnMessage, PreprocessingDoneMessage};
use actix::dev::MessageResponse;
use crate::data_manager::preprocessor::helper::{PreprocessorHelper};
use crate::parameters::Parameters;
use actix_telepathy::prelude::*;
use crate::main;
use std::net::SocketAddr;
use crate::messages::PoisonPill;
use crate::data_manager::stats_collector::DatasetStats;


pub struct Preprocessor {
    data: ArcArray2<f32>,
    parameters: Parameters,
    source: Recipient<PreprocessingDoneMessage>,
    helpers: Addr<PreprocessorHelper>,
    n_cols_total: usize,
    n_cols_processed: usize,
    n_cols_distributed: usize,
    dataset_stats: DatasetStats
}

impl Preprocessor {
    pub fn new(data: ArcArray2<f32>,
               parameters: Parameters,
               source: Recipient<PreprocessingDoneMessage>,
               dataset_stats: DatasetStats
    ) -> Self {
        let data_copy = data.clone();

        let cloned_parameters = parameters.clone();
        let helpers = SyncArbiter::start(cloned_parameters.n_threads, move || {
            PreprocessorHelper::new(data_copy.clone(), cloned_parameters.pattern_length)
        });

        let n_cols_total = data.ncols();

        Self {
            data,
            parameters,
            source,
            helpers,
            n_cols_total,
            n_cols_processed: 0,
            n_cols_distributed: 0,
            dataset_stats
        }
    }

    fn distribute_work(&mut self, source: Recipient<ProcessedColumnMessage>) {
        let max_distribution = self.parameters.n_threads - (self.n_cols_distributed - self.n_cols_processed);

        for (i, column) in (self.n_cols_distributed..self.n_cols_total).enumerate() {
            if i < max_distribution {
                self.helpers.do_send(PreprocessColumnMessage { column, source: source.clone(), std: 0.0 });
                self.n_cols_distributed += 1;
            } else {
                break;
            }
        }
    }

    fn set_processed_column(&mut self, column_id: usize, data: Array1<f32>) {
        self.n_cols_processed += 1;

    }
}

impl Actor for Preprocessor {
    type Context = Context<Self>;

    fn started(&mut self, ctx: &mut Self::Context) {
        self.distribute_work(ctx.address().recipient());
    }
}

impl Handler<ProcessedColumnMessage> for Preprocessor {
    type Result = ();

    fn handle(&mut self, msg: ProcessedColumnMessage, ctx: &mut Self::Context) -> Self::Result {
        self.set_processed_column(msg.column, msg.processed_column);
        if self.n_cols_processed == self.n_cols_total {
            self.source.do_send(PreprocessingDoneMessage );
            self.helpers.do_send(PoisonPill);
            ctx.stop();
        } else {
            self.distribute_work(ctx.address().recipient());
        }
    }
}
