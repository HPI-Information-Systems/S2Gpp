use actix::prelude::*;
use anyhow::Result;

pub use crate::parameters::{Parameters, Role};

use crate::cluster_listener::ClusterMemberListener;
use crate::interface::SyncInterface;
use crate::training::{StartTrainingMessage, Training};
use crate::utils::ClusterNodes;
use actix_telepathy::Cluster;
use ndarray::{Array1, Array2};

mod cluster_listener;
mod data_manager;
mod data_store;
mod messages;
mod parameters;
#[cfg(test)]
mod tests;
mod training;
mod utils;

mod interface;
#[cfg(feature = "python")]
mod python_binding;

pub fn s2gpp(params: Parameters, data: Option<Array2<f32>>) -> Result<Option<Array1<f32>>> {
    if let Some(data) = data {
        let mut training = Training::init(params);
        let anomaly_score = training.fit(data)?;

        Ok(Some(anomaly_score))
    } else {
        let system = System::new();
        system
            .block_on(s2gpp_async(params))
            .expect("Series2Graph++ did not terminate correctly!");
        system.run().unwrap();

        Ok(None)
    }
}

pub async fn s2gpp_async(params: Parameters) -> Result<()> {
    let host = params.local_host;
    let seed_nodes = match &params.role {
        Role::Sub { mainhost } => vec![*mainhost],
        _ => vec![],
    };

    let training = Training::init(params.clone()).start();
    if params.n_cluster_nodes > 1 {
        let _cluster = Cluster::new(host, seed_nodes);
        let _cluster_listener = ClusterMemberListener::new(params, training).start();
    } else {
        let nodes = ClusterNodes::new();
        training.do_send(StartTrainingMessage {
            nodes,
            source: None,
            data: None,
        });
    }

    Ok(())
}
