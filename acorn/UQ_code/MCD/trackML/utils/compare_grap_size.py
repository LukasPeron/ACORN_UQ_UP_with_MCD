import torch 
from torch_geometric.data import Data
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tqdm

for set in ["trainset", "valset", "testset"]:
    list_sizes_graph = [[], [], []]
    list_node_sizes = [[], [], []]
    # Loop over the event IDs
    for event_id in tqdm.tqdm(range(21000, 22755+1)):
        for i, step in enumerate(["metric_learning", "filter", "gnn"]):
            results_path = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/pt>1GeV/no_filter_step/"
            if list(Path(f"{results_path}/{step}/{set}/").glob(f"*0000{event_id}*"))!=[]:   
                post_step = list(Path(f"{results_path}/{step}/{set}/").glob(f"*0000{event_id}*"))[0]
                data_post_step = torch.load(post_step, map_location="cpu")
                # print(f"Data loaded from {post_step}")
                # graph_size = sum(data_post_step.scores>0.5) 
                graph_size = data_post_step.edge_index.shape[1]
                # print(data_post_step.edge_index)
                number_of_nodes = data_post_step.num_nodes
                # print(f"Graph size post {step} : {graph_size}")
                list_sizes_graph[i].append(graph_size)
                list_node_sizes[i].append(number_of_nodes)
                
    # Plot the graph sizes
    plt.figure(figsize=(10, 6))
    plt.plot(list_sizes_graph[0], label='Metric Learning', marker='o')
    plt.plot(list_sizes_graph[1], label='Filter', marker='o')
    plt.plot(list_sizes_graph[2], label='GNN', marker='o')
    plt.title(f'Number of Edges at Different Stages from {set} ({len(list_sizes_graph[0])} graphs)')
    plt.xlabel('Event ID (arbitrary)')
    plt.ylabel('Number of Edges')
    plt.legend()
    plt.savefig(f'graph_sizes_{set}.svg')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(list_node_sizes[0], label='Metric Learning', marker='o')
    plt.plot(list_node_sizes[1], label='Filter', marker='o')
    plt.plot(list_node_sizes[2], label='GNN', marker='o')
    plt.title(f'Number of Nodes at Different Stages from {set} ({len(list_node_sizes[0])} graphs)')
    plt.xlabel('Event ID (arbitrary)')
    plt.ylabel('Number of Nodes')
    plt.legend()
    plt.savefig(f'graph_nodes_{set}.svg')
    plt.close()