use structopt::StructOpt;
use std::net::SocketAddr;

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

    #[structopt(short = "l", long = "local-host")]
    pub local_host: SocketAddr,

    #[structopt(short = "p", long = "pattern-length")]
    pub pattern_length: usize,

    #[structopt(long = "latent")]
    pub latent: usize,

    #[structopt(long = "rate")]
    pub rate: usize,

    #[structopt(short = "t", long = "threads")]
    pub n_threads: usize,

    #[structopt(short = "n", long = "cluster-nodes")]
    pub n_cluster_nodes: usize
}

impl Parameters {
    pub fn get_main_addr(&self) -> Option<SocketAddr> {
        match self.role {
            Role::Sub { mainhost} => Some(mainhost),
            Role::Main { .. } => None
        }
    }

    pub fn is_main_addr(&self, addr: SocketAddr) -> bool {
        match &self.role {
            Role::Sub { mainhost} => addr.eq(mainhost),
            Role::Main { .. } => addr.eq(&self.local_host)
        }
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
            n_cluster_nodes: 1
        }
    }
}
