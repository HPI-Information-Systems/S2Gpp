use crate::training::Clustering;
use std::net::SocketAddr;
use structopt::StructOpt;

#[derive(Debug, StructOpt, Clone)]
#[structopt(name = "Role")]
pub enum Role {
    #[structopt(name = "main")]
    Main {
        #[structopt(short = "d", long = "data-path")]
        data_path: String,
    },

    #[structopt(name = "sub")]
    Sub {
        #[structopt(short = "h", long = "mainhost")]
        mainhost: SocketAddr,
    },
}

impl Default for Role {
    fn default() -> Self {
        Role::Main {
            data_path: "".to_string(),
        }
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

    #[structopt(long = "score-output-path")]
    pub score_output_path: Option<String>,

    #[structopt(long = "column-start-idx", default_value = "0")]
    pub column_start: usize,

    #[structopt(long = "column-end-idx", default_value = "0")]
    pub column_end: isize,

    #[structopt(long = "clustering", default_value = "meanshift")]
    pub clustering: Clustering,
}

impl Parameters {
    pub fn is_main_addr(&self, addr: SocketAddr) -> bool {
        match &self.role {
            Role::Sub { mainhost } => addr.eq(mainhost),
            Role::Main { .. } => addr.eq(&self.local_host),
        }
    }

    pub fn segments_per_node(&self) -> usize {
        num_integer::Integer::div_floor(&self.rate, &self.n_cluster_nodes)
    }

    pub fn segment_id_to_assignment(&self, segment_id: usize) -> usize {
        let segments_per_node = self.segments_per_node();
        let cluster_node_id = segment_id / segments_per_node;
        if cluster_node_id >= self.n_cluster_nodes {
            self.n_cluster_nodes - 1
        } else {
            cluster_node_id
        }
    }

    pub fn first_segment_of_i_next_cluster_node(&self, segment_id: usize, i: usize) -> usize {
        let i_next_cluster_node_id =
            (self.segment_id_to_assignment(segment_id) + i) % self.n_cluster_nodes;
        i_next_cluster_node_id * self.segments_per_node()
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
            score_output_path: None,
            column_start: 0,
            column_end: 0,
            clustering: Clustering::MeanShift,
        }
    }
}
