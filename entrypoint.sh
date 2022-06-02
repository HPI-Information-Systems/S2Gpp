#!/usr/bin/env bash

if [[ "$1" = "build" ]]; then
    cp -r /app/wheels /results
elif [[ "$1" = "deploy" ]]; then
    TWINE_PASSWORD=${CI_JOB_TOKEN} TWINE_USERNAME=gitlab-ci-token maturin upload --repository-url ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/pypi /app/wheels/s2gpp-*.whl
else
    "$@"
fi
