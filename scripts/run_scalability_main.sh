#!/usr/bin/bash

base_port=1993
cluster_nodes=12
mainhost=172.20.11.103:1992
# params: number of processes_per_node
processes_per_node=$1
# start processes 
for ((i=0; i<$processes_per_node; i++)); do
    echo "Starting process $i"
    
    # if it's the first process, start the main node
    if [ $i -eq 0 ]; then
        # write time results to file with filename including number of processes
        time_results_path="time_results_${processes_per_node}_main.txt"
        # Starting main node
        /usr/bin/time -v target/release/s2gpp \
                        --local-host $(hostname -i):1992 \
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
                        main --data-path=/home/phillip.wenig/datasets/timeseries/scalability/ecg-5120000-1/test.csv > $time_results_path &
    else
    
        # start process and add i to port
        port=$(($base_port + $i))
        # write time results to file with filename including number of processes
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
                        sub --mainhost $mainhost > $time_results_path &
    fi
