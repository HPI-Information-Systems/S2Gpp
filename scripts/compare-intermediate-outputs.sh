#!/usr/bin/env bash

n_nodes=${2:-2}
directory=$(echo $1 | sed 's:/*$::')

compare_local_dist() {
  sort -o $directory/$1.local.0 $directory/$1.local.0
  cat $(awk -v n_nodes=$n_nodes -v directory=$directory -v name=$1 'BEGIN { for(i=0; i<n_nodes; i++) print directory"/"name".dist."i }' | xargs) > $directory/$1.dist
  sort -o $directory/$1.dist $directory/$1.dist
  diff -q $directory/$1.local.0 $directory/$1.dist
}

# transitions
compare_local_dist transitions

# intersections
compare_local_dist intersections

# nodes
compare_local_dist nodes

# edges
compare_local_dist edges

# edges transposed
compare_local_dist edges-transposed
