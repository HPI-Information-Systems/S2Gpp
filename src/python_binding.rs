use crate::s2gpp as orig_s2gpp;
use crate::training::Clustering;
use crate::{Parameters, Role};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2};
use pyo3::exceptions;
use pyo3::prelude::*;
use std::panic;
use std::str::FromStr;

#[pyfunction]
fn s2gpp_local_array<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f32>,
    pattern_length: usize,
    latent: usize,
    query_length: usize,
    rate: usize,
    n_threads: usize,
    clustering: String,
    self_correction: bool,
) -> PyResult<&'py PyArray1<f32>> {
    let mut params = Parameters::default();
    params.pattern_length = pattern_length;
    params.latent = latent;
    params.query_length = query_length;
    params.rate = rate;
    params.n_threads = n_threads;
    params.clustering = Clustering::from_str(&clustering).unwrap();
    params.self_correction = self_correction;

    let data = data.as_array().to_owned();
    let anomaly_scores =
        orig_s2gpp(params, Some(data)).expect("Series2Graph++ did not terminate correctly!");

    match anomaly_scores {
        Some(res) => Ok(res.into_pyarray(py)),
        None => Err(exceptions::PyTypeError::new_err("Error message")),
    }
}

#[pyfunction]
fn s2gpp_local_file<'py>(
    _py: Python<'py>,
    data_path: String,
    pattern_length: usize,
    latent: usize,
    query_length: usize,
    rate: usize,
    n_threads: usize,
    score_output_path: Option<String>,
    column_start: usize,
    column_end: isize,
    clustering: String,
    self_correction: bool,
    local_host: String,
) -> PyResult<()> {
    let result = panic::catch_unwind(|| {
        let mut params = Parameters::default();

        params.role = Role::Main {
            data_path: Some(data_path),
        };
        params.pattern_length = pattern_length;
        params.latent = latent;
        params.query_length = query_length;
        params.rate = rate;
        params.n_threads = n_threads;
        params.score_output_path = score_output_path;
        params.column_start = column_start;
        params.column_end = column_end;
        params.clustering = Clustering::from_str(&clustering).unwrap();
        params.self_correction = self_correction;
        params.local_host = local_host.parse().unwrap();

        orig_s2gpp(params, None).expect("Series2Graph++ did not terminate correctly!");
    });
    match result {
        Ok(_) => Ok(()),
        Err(_) => Err(exceptions::PyTypeError::new_err("Error message")),
    }
}

#[pyfunction]
fn s2gpp_distributed_main<'py>(
    _py: Python<'py>,
    data_path: String,
    pattern_length: usize,
    latent: usize,
    query_length: usize,
    rate: usize,
    n_threads: usize,
    score_output_path: Option<String>,
    column_start: usize,
    column_end: isize,
    clustering: String,
    self_correction: bool,
    local_host: String,
    n_cluster_nodes: usize,
) -> PyResult<()> {
    let mut params = Parameters::default();

    params.role = Role::Main {
        data_path: Some(data_path),
    };
    params.pattern_length = pattern_length;
    params.latent = latent;
    params.query_length = query_length;
    params.rate = rate;
    params.n_threads = n_threads;
    params.score_output_path = score_output_path;
    params.column_start = column_start;
    params.column_end = column_end;
    params.clustering = Clustering::from_str(&clustering).unwrap();
    params.self_correction = self_correction;
    params.local_host = local_host.parse()?;
    params.n_cluster_nodes = n_cluster_nodes;

    orig_s2gpp(params, None).expect("Series2Graph++ did not terminate correctly!");
    Ok(())
}

#[pyfunction]
fn s2gpp_distributed_sub<'py>(
    _py: Python<'py>,
    pattern_length: usize,
    latent: usize,
    query_length: usize,
    rate: usize,
    n_threads: usize,
    score_output_path: Option<String>,
    column_start: usize,
    column_end: isize,
    clustering: String,
    self_correction: bool,
    local_host: String,
    n_cluster_nodes: usize,
    mainhost: String,
) -> PyResult<()> {
    let mut params = Parameters::default();

    params.role = Role::Sub {
        mainhost: mainhost.parse()?,
    };
    params.pattern_length = pattern_length;
    params.latent = latent;
    params.query_length = query_length;
    params.rate = rate;
    params.n_threads = n_threads;
    params.score_output_path = score_output_path;
    params.column_start = column_start;
    params.column_end = column_end;
    params.clustering = Clustering::from_str(&clustering).unwrap();
    params.self_correction = self_correction;
    params.local_host = local_host.parse()?;
    params.n_cluster_nodes = n_cluster_nodes;

    orig_s2gpp(params, None).expect("Series2Graph++ did not terminate correctly!");
    Ok(())
}

#[pymodule]
fn s2gpp(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(s2gpp_local_file, m)?)?;
    m.add_function(wrap_pyfunction!(s2gpp_local_array, m)?)?;
    m.add_function(wrap_pyfunction!(s2gpp_distributed_main, m)?)?;
    m.add_function(wrap_pyfunction!(s2gpp_distributed_sub, m)?)?;

    Ok(())
}
