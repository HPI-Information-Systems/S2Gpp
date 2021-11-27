use structopt::StructOpt;
use std::net::SocketAddr;
use num_integer::Integer;

#[derive(Debug, StructOpt, Clone)]
#[structopt(name = "Role")]
pub enum Role {
    #[structopt(name = "main")]
    Main {
        #[structopt(short = "d", long = "data-path")]
        data_path: String
    },

    #[structopt(name = "sub")]
    Sub {
        #[structopt(short = "h", long = "mainhost")]
        mainhost: SocketAddr
    }
}

impl Default for Role {
    fn default() -> Self {
        Role::Main {data_path: "".to_string()}
    }
}


#[derive(StructOpt, Debug, Clone)]
pub struct Parameters {
    #[structopt(subcommand)]
    pub role: Role,

    #[structopt(short = "l", long = "local-host", default_value = "127.0.0.1:8000")]
    pub local_host: SocketAddr,

    #[structopt(short = "p", long = "pattern-length", default_value = "50")]
    pub pattern_length: usize,

    #[structopt(long = "latent", default_value = "16")]
    pub latent: usize,

    #[structopt(long = "rate", default_value = "100")]
    pub rate: usize,

    #[structopt(short = "t", long = "threads", default_value = "8")]
    pub n_threads: usize,

    #[structopt(short = "n", long = "cluster-nodes", default_value = "1")]
    pub n_cluster_nodes: usize,

    #[structopt(short = "q", long = "query-length", default_value = "75")]
    pub query_length: usize,

    #[structopt(long = "graph-output-path")]
    pub graph_output_path: Option<String>,

    #[structopt(long = "score-output-path")]
    pub score_output_path: Option<String>,
}

impl Parameters {
    pub fn is_main_addr(&self, addr: SocketAddr) -> bool {
        match &self.role {
            Role::Sub { mainhost} => addr.eq(mainhost),
            Role::Main { .. } => addr.eq(&self.local_host)
        }
    }

    pub fn segments_per_node(&self) -> usize {
        self.rate.div_floor(&self.n_cluster_nodes)
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            role: Role::default(),
            local_host: "127.0.0.1:8000".parse().unwrap(),
            pattern_length: 50,
            latent: 16,
            rate: 100,
            n_threads: 1,
            n_cluster_nodes: 1,
            query_length: 75,
            graph_output_path: None,
            score_output_path: None
        }
    }
}
