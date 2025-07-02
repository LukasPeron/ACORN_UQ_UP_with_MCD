#!/bin/bash

for n_train in 100 200 400 800 1400;
do
    echo "launching job for $n_train"
    # sbatch /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/all_pipeline.sh $n_train
    sbatch /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/submit_UQ_propagation.sh $n_train
done
