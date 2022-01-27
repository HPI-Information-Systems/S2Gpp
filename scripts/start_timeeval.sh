#!/usr/bin/env bash

EXECUTIONTYPE=$(jq -r .executionType <<< $1)

if [ $EXECUTIONTYPE == "execute" ]; then
  CUSTOMPARAMS=$(jq '.customParameters | to_entries[] | "--" + .key + " " + (.value|tostring)' <<< $1 | xargs)
  MAINPARAMS=$(jq -r '. | "--score-output-path " + .dataOutput + " main -d " + .dataInput' <<< $1)
  PARAMS="--column-start-idx 1 --column-end-idx=-1 $CUSTOMPARAMS $MAINPARAMS"

  /app/s2gpp $PARAMS
else
  echo "Series2Graph++ is an unsupervised algorithm and therefore it has only the execution-type 'execute'!"
fi
