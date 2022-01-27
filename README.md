# Series2Graph++

[![pipeline status](https://gitlab.hpi.de/akita/s2gpp/badges/main/pipeline.svg)](https://gitlab.hpi.de/akita/s2gpp/-/commits/main)
[![release info](https://img.shields.io/badge/Release-0.3.1-blue)](https://gitlab.hpi.de/phillip.wenig/s2gpp/-/releases/0.3.1)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

_Series2Graph++_ (S2G++) is a time series anomaly detection algorithm based on the [Series2Graph](https://helios2.mi.parisdescartes.fr/~themisp/series2graph/) (S2G) and the [DADS](https://hpi.de/naumann/s/dads) algorithms. 
S2G++ can handle multivariate time series whereas S2G and DADS can cope with only univariate time series. 
Moreover, S2G++ takes ideas from DADS to run distributedly in a computer cluster.
S2G++ is written in _Rust_ and leverages the [actix](https://github.com/actix/actix) and [actix-telepathy](https://github.com/wenig/actix-telepathy) libraries.

## Quick Start

### Requirements

- Rust 1.58
- openblas
- (Docker)

To have `openblas` available to the Rust build process, do the following on Debian (Linux):

```shell
sudo apt install build-essential gfortran libopenblas-base libopenblas-dev gcc
```

### Installation

#### From source

```shell
git pull https://gitlab.hpi.de/akita/s2gpp
cd s2gpp
cargo build
```

#### Docker

The base image `akita/rust-base` must be available to your machine.

```shell
docker build s2gpp .
```

### Usage

#### Parameters

Pattern:
```shell
s2gpp --local-host <IP:Port> --pattern-length <Int> --latent <Int> --query-length <Int> --rate <Int> --threads <Int> --cluster-nodes <Int> --score-output-path <Path> [main --data-path <Path> | sub --mainhost <IP:Port>]
```

S2G++ expects one of two sub-commands with its specific parameters:

- `main` (The head computer in a cluster)
  - `data-path` (The path to the input time series)
- `sub` (The other computers in a cluster; only necessary in a distributed setting)
  - `mainhost` (The ip-address to the main computer in a cluster)

Before these sub-commands are used, general parameters must be defined:

- `local-host` (The ip-address with port to bind the listener on.)
- `pattern-length` (Size of the sliding window, independent of anomaly length, but should in the best case be larger.)
- `latent` (Size of latent embedding space. This space is the input for the PCA calculation afterwards.)
- `query-length` (Size of the sliding windows used to find anomalies (query subsequences). query-length must be >= pattern-length!)
- `rate` (Number of angles used to extract pattern nodes. A higher value will lead to high precision, but at the cost of increased computation time.)
- `threads` (Number of helper threads started besides the main thread. (min=1))
- `cluster-nodes` (Size of the computer cluster.)
- `score-output-path` (Path the score are written to.)
- `column-start-idx` (How many columns to skip)
- `column-end-idx` (Until which column to use (exclusive). Can also take negative numbers to count from the end.)


#### Input Format

The input format of the time series is expected to be a CSV with header. Each column represents a channel of the timeseries.
Sometimes, time series files include also the labels and an index. You can skip columns with the `column-start-idx` / `column-end-idx` range pattern. It behave like Python ranges.

## Cite

Please cite this work, when using it!

## References

[1] P. Boniol and T. Palpanas, Series2Graph: Graph-based Subsequence Anomaly Detection in Time Series, PVLDB (2020) [link](https://helios2.mi.parisdescartes.fr/~themisp/series2graph/data/Series2Graph.pdf)

[2] Schneider, J., Wenig, P. & Papenbrock, T. Distributed detection of sequential anomalies in univariate time series. The VLDB Journal 30, 579–602 (2021). [link](https://doi.org/10.1007/s00778-021-00657-6)

## TODO

- [ ] Reduce dimensions (channels) in graph
- [ ] Python Binding
- [ ] Dimension responsibility (check how far off a node is from the mean of all other nodes, per dimension, use this as indicator)
- [ ] add own citation
