# S2G++

[![pipeline status](https://gitlab.hpi.de/akita/s2gpp/badges/main/pipeline.svg)](https://gitlab.hpi.de/akita/s2gpp/-/commits/main)
[![release info](https://img.shields.io/badge/Release-0.1.0-blue)](https://gitlab.hpi.de/phillip.wenig/s2gpp/-/releases/v0.1.0)
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
  - [ ] distributed Scorer
- [x] Result Writing

## Nice to Have
- [ ] Node storage
- [ ] Edge storage
- [ ] No need to build Graph
- [ ] Python Binding
- [ ] Dimension responsibility (check how far off a node is from the mean of all other nodes, per dimension, use this as indicator)
