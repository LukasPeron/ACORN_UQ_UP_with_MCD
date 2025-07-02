#!/bin/bash

for drop in 0.9 0.95;
do
    echo "launching job for $drop"
    sbatch /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/eval_drop.sh $drop
done
