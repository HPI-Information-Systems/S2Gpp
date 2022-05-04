#/bin/bash

VERSION=$1
DATA_FOLDER=$2
FILE=$3
PARAMS=$3

docker run --rm -v $DATA_FOLDER:/data:ro -v $(pwd):/results:rw -e LOCAL_GID=1000 -e LOCAL_UID=1000 -e RUST_LOG=debug sopedu:5000/akita/s2gpp:$VERSION execute-algorithm '{"dataInput": "/data/$FILE", "dataOutput": "/results/anomaly_scores.ts", "executionType": "execute", "customParameters": $PARAMS}'
