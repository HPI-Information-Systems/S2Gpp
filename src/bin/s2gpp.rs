use log::*;
use structopt::StructOpt;

use env_logger::Env;
use s2gpp::{s2gpp, Parameters};
use std::io::Write;

fn main() {
    env_logger::Builder::from_env(Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{} [S2G++]: {}", record.level(), record.args()))
        .init();

    let params: Parameters = Parameters::from_args();
    if params.explainability && params.n_cluster_nodes > 1 {
        panic!("The explainability feature is only available in a non-distributed setting.")
    }
    debug!("Parameters: {:?}", params);

    s2gpp(params, None).expect("Series2Graph++ did not terminate correctly!");
}
