mod messages;
mod helper;

use actix::{Actor, ActorContext, SyncContext, Context, Handler, Addr, SyncArbiter, Recipient, AsyncContext};
use ndarray::{ArcArray2, Axis, Array1, Array2};
pub use crate::data_manager::preprocessor::messages::{ProcessedColumnMessage, PreprocessColumnMessage, PreprocessingDoneMessage, StdNodeMessage, StdDoneMessage};
use actix::dev::MessageResponse;
use crate::data_manager::preprocessor::helper::{PreprocessorHelper};
use crate::parameters::Parameters;
use actix_telepathy::prelude::*;
use crate::main;
use std::net::SocketAddr;


#[derive(Default)]
struct StdCalculation {
    main_node: Option<AnyAddr<Preprocessor>>,
    std_nodes: Vec<RemoteAddr>,
    n: Option<usize>,
    mean: Option<Array1<f32>>,
    m2: Option<Array1<f32>>,
    std: Option<Array1<f32>>
}

impl StdCalculation {
    pub fn set_intermediate(&mut self,
                n: usize,
                mean: Array1<f32>,
                m2: Array1<f32>) {
        self.n = Some(n);
        self.mean = Some(mean);
        self.m2 = Some(m2);
    }
}


#[derive(RemoteActor)]
#[remote_messages(StdNodeMessage, StdDoneMessage)]
pub struct Preprocessor {
    data: ArcArray2<f32>,
    parameters: Parameters,
    source: Recipient<PreprocessingDoneMessage>,
    helpers: Addr<PreprocessorHelper>,
    n_cols_total: usize,
    n_cols_processed: usize,
    n_cols_distributed: usize,
    std_calculation: StdCalculation
}

impl Preprocessor {
    pub fn new(data: ArcArray2<f32>,
               parameters: Parameters,
               main_node: Option<AnyAddr<Self>>,
               source: Recipient<PreprocessingDoneMessage>
    ) -> Self {
        let data_copy = data.clone();

        let cloned_parameters = parameters.clone();
        let helpers = SyncArbiter::start(cloned_parameters.n_threads, move || {
            PreprocessorHelper::new(data_copy.clone(), cloned_parameters.pattern_length)
        });

        let n_cols_total = data.ncols();

        let mut std_calculation = StdCalculation::default();
        std_calculation.main_node = main_node;

        Self {
            data,
            parameters,
            source,
            helpers,
            n_cols_total,
            n_cols_processed: 0,
            n_cols_distributed: 0,
            std_calculation
        }
    }

    fn calculate_var(&mut self, addr: Addr<Self>) {
        let data = self.data.view();
        let n = data.nrows();
        let mean = data.mean_axis(Axis(0)).unwrap();
        let delta = data.to_owned() - mean.broadcast((self.data.nrows(), mean.len())).unwrap().to_owned();
        let delta_n = delta.clone() / (n as f32);
        let m2 = (delta * delta_n * (n as f32)).sum_axis(Axis(0));

        let main = match self.std_calculation.main_node.as_ref() {
            None => AnyAddr::Local(addr.clone()),
            Some(any_addr) => any_addr.clone()
        };
        main.do_send(StdNodeMessage { n, mean, m2, source: RemoteAddr::new_from_id(self.parameters.local_host, "Preprocessor") });
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
        self.calculate_var(ctx.address());
    }
}

impl Handler<StdNodeMessage> for Preprocessor {
    type Result = ();

    fn handle(&mut self, msg: StdNodeMessage, ctx: &mut Self::Context) -> Self::Result {
        self.std_calculation.std_nodes.push(msg.source);

        if self.std_calculation.std_nodes.len() < self.parameters.n_cluster_nodes {
            match (&self.std_calculation.n, &self.std_calculation.mean, &self.std_calculation.m2) {
                (Some(n), Some(mean), Some(m2)) => {
                    let global_n = n + msg.n;
                    let delta: Array1<f32> = msg.mean.clone() - mean;
                    self.std_calculation.m2 = Some(msg.m2 + m2 + delta.clone() * delta * ((n + msg.n) as f32 / global_n as f32));
                    self.std_calculation.mean = Some((mean * n.clone() as f32 + msg.mean * msg.n as f32) / global_n as f32);
                    self.std_calculation.n = Some(global_n);
                },
                _ => {
                    self.std_calculation.set_intermediate(msg.n, msg.mean, msg.m2);
                }
            }
        } else {
            match (&self.std_calculation.n, &self.std_calculation.mean, &self.std_calculation.m2) {
                (Some(n), Some(mean), Some(m2)) => {
                    let global_n = n + msg.n;
                    let delta: Array1<f32> = msg.mean.clone() - mean;
                    let m2 = msg.m2 + m2 + delta.clone() * delta * ((n + msg.n) as f32 / global_n as f32);
                    self.std_calculation.std = Some((m2 / (global_n - 1) as f32).iter().map(|x| x.sqrt()).collect());
                },
                _ => panic!("Some value for variance should have been set by now!")
            }

            for node in &self.std_calculation.std_nodes {

                node.do_send(StdDoneMessage { std: self.std_calculation.std.as_ref().unwrap().clone() });
            }
        }
    }
}

impl Handler<StdDoneMessage> for Preprocessor {
    type Result = ();

    fn handle(&mut self, msg: StdDoneMessage, ctx: &mut Self::Context) -> Self::Result {
        self.std_calculation.std = Some(msg.std);
        self.distribute_work(ctx.address().recipient());
    }
}

impl Handler<ProcessedColumnMessage> for Preprocessor {
    type Result = ();

    fn handle(&mut self, msg: ProcessedColumnMessage, ctx: &mut Self::Context) -> Self::Result {
        self.set_processed_column(msg.column, msg.processed_column);
        if self.n_cols_processed == self.n_cols_total {
            self.source.do_send(PreprocessingDoneMessage );
            // todo stop helpers
            ctx.stop();
        } else {
            self.distribute_work(ctx.address().recipient());
        }
    }
}
