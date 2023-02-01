#!/usr/bin/bash

base_port=1993
cluster_nodes=12
mainhost=172.20.11.103:1992
# params: number of processes_per_node
processes_per_node=$1
# start processes 
for ((i=0; i<$processes_per_node; i++)); do
    echo "Starting process $i"
    
    # start process and add i to port
    port=$(($base_port + $i))
    time_results_path="time_results_${processes_per_node}_${i}.txt"
    
    /usr/bin/time -v target/release/s2gpp \
                    --local-host $(hostname -i):$port \
                    --pattern-length 100 \
                    --latent 25 \
                    --query-length 150 \
                    --rate 100 \
                    --threads 20 \
                    --cluster-nodes $cluster_nodes \
                    --score-output-path results.txt \
                    --column-start-idx 1 \
                    --column-end-idx=-1 \
                    --clustering kde \
                    sub --mainhost $mainhost &
done