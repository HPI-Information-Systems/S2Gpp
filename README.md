# S2G++

[![pipeline status](https://gitlab.hpi.de/akita/s2gpp/badges/main/pipeline.svg)](https://gitlab.hpi.de/akita/s2gpp/-/commits/main)
[![release info](https://img.shields.io/badge/Release-0.3.0-blue)](https://gitlab.hpi.de/phillip.wenig/s2gpp/-/releases/0.3.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Checklist

- [x] File reader
  - [x] read
  - [x] distribute
- [x] PhaseSpacer
- [x] Distributed PCA
  - [x] enumerate cluster nodes
- [x] Rotation
- [x] Intersection calculator
  - [x] Intersection sharing (in ring?)
- [x] Node estimator
    - [x] MeanShift
- [x] Edge estimator
- [x] Graph creation
  - [x] Graph merging (edges are reduced to main node and graph is created there)
  - [x] Graph output
- [x] Scorer
  - [x] distributed Scorer
  - [x] parallel scoring
- [x] Result Writing
- [ ] Docu in README.md

## Nice to Have
- [x] Node storage
- [x] Edge storage
- [x] No need to build Graph
- [ ] Python Binding
- [ ] Dimension responsibility (check how far off a node is from the mean of all other nodes, per dimension, use this as indicator)
