#!/bin/bash



for drop in 0.05 0.1 0.15 0.2 0.25 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95
do
#     cp /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/1400/gnn_eval.yaml /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/1400/multi_dropout/gnn_eval_$drop.yaml
#     sed -i "s/input_dropout: 0.1/input_dropout: $drop/g" /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/1400/multi_dropout/gnn_eval_$drop.yaml
#     sed -i "s/hidden_dropout: 0.1/hidden_dropout: $drop/g" /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/1400/multi_dropout/gnn_eval_$drop.yaml
#     sed -i "s/multi_dropout: false/multi_dropout: true/g" /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/1400/multi_dropout/gnn_eval_$drop.yaml
#     echo "Created gnn_eval_$drop.yaml with input and hidden dropout set to $drop"
    sed -i '/graph_roc_curve:/,/title: Interaction GNN Efficiency in RZ/d' /pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/1400/multi_dropout/gnn_eval_$drop.yaml
done