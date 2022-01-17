#!/usr/bin/env bash

cargo run --release --package s2gpp --bin s2gpp -- -l $1 -n $2 $3
