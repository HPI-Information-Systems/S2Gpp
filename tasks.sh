#!/usr/bin/env bash

function test-build {
  maturin build --cargo-extra-args="--features python" -o wheels -i $(which python)
}

function test-install {
  test-build
  cd wheels && pip install --force-reinstall -U s2gpp-*.whl && cd ..
}

function release-build {
  maturin build --release --cargo-extra-args="--features python" -o wheels -i $(which python)
}

function release-install {
  release-build
  cd wheels && pip install --force-reinstall -U s2gpp-*.whl && cd ..
}

function build-run-tests {
  test-install
  pytest tests
}

function test {
  maturin develop
  pytest tests
}

"$@"
