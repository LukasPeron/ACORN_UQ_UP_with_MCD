import torch
import numpy as np
import matplotlib.pyplot as plt
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
from tqdm import tqdm

event = torch.load("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/track_building/valset/event[['000021000']].pyg", weights_only=False)


# Function to plot edges, reconstructed tracks, and truth tracks in x-y and r-z planes
def plot_edges(event):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax1, ax2, ax3 = axes[0]  # x-y plane (top row)
    ax4, ax5, ax6 = axes[1]  # r-z plane (bottom row)
    
    # step 1 : plot all edges
    for n_edge in tqdm(range(len(event.edge_index[0]))):
        edge_in = event.edge_index[0][n_edge].item()
        edge_out = event.edge_index[1][n_edge].item()
        r1, phi1, z1 = event.r[edge_in], event.phi[edge_in], event.z[edge_in]
        r2, phi2, z2 = event.r[edge_out], event.phi[edge_out], event.z[edge_out]
        # convert to x, y coordinates for x-y plane
        x1, y1 = r1 * np.cos(phi1), r1 * np.sin(phi1)
        x2, y2 = r2 * np.cos(phi2), r2 * np.sin(phi2)
        # plot the edges in x-y plane
        ax1.plot([x1, x2], [y1, y2], color='black', alpha=0.25, linewidth=0.1)
        # plot the edges in r-z plane
        ax4.plot([z1, z2], [r1, r2], color='black', alpha=0.25, linewidth=0.1)
        
    #step 2 : plot all reconstructed tracks
    track_file_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/track_building/valset_tracks/event[['000021000']].txt"
    with open(track_file_path, 'r') as f:
        num_tracks = sum(1 for _ in f)
    
    for n in tqdm(range(num_tracks)):
        track = np.loadtxt(track_file_path, skiprows=n, max_rows=1).astype(int).tolist()
        if type(track) is int:
            track = [track]
        if len(track) < 2: # skip tracks with less than 2 hits
            continue
        track = np.array(track)
        node_id = np.searchsorted(event.hit_id, track)  
        full_track = {"n_hits": track, "r": event.r[node_id], "phi": event.phi[node_id], "z": event.z[node_id]}
        # sort the track by r
        sorted_track = sorted(zip(full_track["n_hits"], full_track["r"], full_track["phi"], full_track["z"]), key=lambda x: x[1])
        # plot the edges pairwise in increasing r
        for i in range(len(sorted_track) - 1):
            n_hit1, r1, phi1, z1 = sorted_track[i]
            n_hit2, r2, phi2, z2 = sorted_track[i + 1]
            # convert to x, y coordinates for x-y plane
            x1, y1 = r1 * np.cos(phi1), r1 * np.sin(phi1)
            x2, y2 = r2 * np.cos(phi2), r2 * np.sin(phi2)
            # plot in x-y plane
            ax2.plot([x1, x2], [y1, y2], color='blue', alpha=0.25, linewidth=0.1)
            # plot in r-z plane
            ax5.plot([z1, z2], [r1, r2], color='blue', alpha=0.25, linewidth=0.1)

    
    # step 3 : plot all Truth tracks $p_T > 1$GeV
    for n_edge in tqdm(range(len(event.track_edges[0]))):
        edge_in = event.track_edges[0][n_edge].item()
        edge_out = event.track_edges[1][n_edge].item()
        r1, phi1, z1 = event.r[edge_in], event.phi[edge_in], event.z[edge_in]
        r2, phi2, z2 = event.r[edge_out], event.phi[edge_out], event.z[edge_out]
        # convert to x, y coordinates for x-y plane
        x1, y1 = r1 * np.cos(phi1), r1 * np.sin(phi1)
        x2, y2 = r2 * np.cos(phi2), r2 * np.sin(phi2)
        # plot in x-y plane
        if event.pt[n_edge] < 1000:
            ax3.plot([x1, x2], [y1, y2], color='red', alpha=0.25, linewidth=0.1)
            # plot in r-z plane
            ax6.plot([z1, z2], [r1, r2], color='red', alpha=0.25, linewidth=0.1)
        else:
            ax3.plot([x1, x2], [y1, y2], color='green', alpha=0.5, linewidth=0.4)
            # plot in r-z plane
            ax6.plot([z1, z2], [r1, r2], color='green', alpha=0.5, linewidth=0.4)
    
    # set the limits for x-y plane (top row)
    ax1.set_xlim(-1100, 1100)
    ax1.set_ylim(-1100, 1100)
    ax2.set_xlim(-1100, 1100)
    ax2.set_ylim(-1100, 1100)
    ax3.set_xlim(-1100, 1100)
    ax3.set_ylim(-1100, 1100)
    
    # set the limits for r-z plane (bottom row)
    ax4.set_xlim(-3000, 3000)
    ax4.set_ylim(0, 1100)
    ax5.set_xlim(-3000, 3000)
    ax5.set_ylim(0, 1100)
    ax6.set_xlim(-3000, 3000)
    ax6.set_ylim(0, 1100)
    
    # set the aspect ratio to be equal for x-y plane
    ax1.set_aspect('equal', adjustable='box')
    ax2.set_aspect('equal', adjustable='box')
    ax3.set_aspect('equal', adjustable='box')
    
    # labels for x-y plane (top row)
    ax1.set_ylabel("y [mm]", fontsize=14, ha="right", y=0.95)
    ax2.set_ylabel("y [mm]", fontsize=14, ha="right", y=0.95)
    ax3.set_ylabel("y [mm]", fontsize=14, ha="right", y=0.95)
    ax1.set_xlabel("x [mm]", fontsize=14, ha="right", x=0.95)
    ax2.set_xlabel("x [mm]", fontsize=14, ha="right", x=0.95)
    ax3.set_xlabel("x [mm]", fontsize=14, ha="right", x=0.95)
    
    # labels for r-z plane (bottom row)
    ax4.set_xlabel("z [mm]", fontsize=14, ha="right", x=0.95)
    ax4.set_ylabel("r [mm]", fontsize=14, ha="right", y=0.95)
    ax5.set_xlabel("z [mm]", fontsize=14, ha="right", x=0.95)
    ax5.set_ylabel("r [mm]", fontsize=14, ha="right", y=0.95)
    ax6.set_xlabel("z [mm]", fontsize=14, ha="right", x=0.95)
    ax6.set_ylabel("r [mm]", fontsize=14, ha="right", y=0.95)
    
    # set the title for x-y plane
    atlasify("Post GNN edges", axes=ax1, outside=True)
    atlasify("Reconstructed tracks", axes=ax2, outside=True)
    atlasify("Truth tracks", axes=ax3, outside=True)
    
    # set the title for r-z plane
    atlasify("Post GNN edges", axes=ax4, outside=True)
    atlasify("Reconstructed tracks", axes=ax5, outside=True)
    atlasify("Truth tracks", axes=ax6, outside=True)
    
    fig.tight_layout()
    fig.savefig("/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/tracks_edges_visualisation.png", dpi=300)

plot_edges(event)

def plot_track_edges(event):
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    for n_edge in tqdm(range(len(event.track_edges[0]))):
        edge_in = event.track_edges[0][n_edge].item()
        edge_out = event.track_edges[1][n_edge].item()
        r1, phi1, z1 = event.r[edge_in], event.phi[edge_in], event.z[edge_in]
        r2, phi2, z2 = event.r[edge_out], event.phi[edge_out], event.z[edge_out]
        # convert to x, y coordinates for x-y plane
        x1, y1 = r1 * np.cos(phi1), r1 * np.sin(phi1)
        x2, y2 = r2 * np.cos(phi2), r2 * np.sin(phi2)
        # plot in x-y plane
    if event.pt[n_edge] < 1000:
        ax.plot([x1, x2], [y1, y2], color='red', alpha=0.25, linewidth=0.1)
    else:
        ax.plot([x1, x2], [y1, y2], color='green', alpha=0.5, linewidth=0.4)

    # set the limits for x-y plane
    ax.set_xlim(-1100, 1100)
    ax.set_ylim(-1100, 1100)
    
    # set the aspect ratio to be equal for x-y plane
    ax.set_aspect('equal', adjustable='box')
    
    # labels for x-y plane
    ax.set_ylabel("y [mm]", fontsize=14, ha="right", y=0.95)
    ax.set_xlabel("x [mm]", fontsize=14, ha="right", x=0.95)
    
    # set the title for x-y plane
    atlasify("Truth tracks", outside=True)
    
    fig.tight_layout()
    fig.savefig("/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/plots/true_track_edges_xy.png", dpi=300)

plot_track_edges(event)

# load csv raw data
import pandas as pd
raw_data = pd.read_csv("/pscratch/sd/l/lperon/ATLAS/data_dir/Example_3/trackml_1500_events/event000021000-hits.csv")

# Function to plot raw hits data in x-y and r-z planes
def plot_raw_data(raw_data):
    # Create subplot with 1 row, 2 columns
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot 1: hits in the x-y plane
    x, y = raw_data['x'], raw_data['y']
    ax.scatter(x, y, s=0.1, alpha=0.25, color="black")
    ax.set_xlabel("x [mm]", fontsize=14, ha="right", x=0.95)
    ax.set_ylabel("y [mm]", fontsize=14, ha="right", y=0.95)
    ax.set_xlim(-1100, 1100)
    ax.set_ylim(-1100, 1100)
    ax.set_aspect('equal', adjustable='box')
    atlasify("Detector hits in x-y plane", axes=ax, outside=True)
    
    fig.tight_layout()
    fig.savefig("/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/raw_hits_visualisation_xy_plane.png", dpi=300)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot 2: hits in the r-z plane
    z = raw_data['z']
    r = np.sqrt(x**2 + y**2)  # Calculate r from x and y coordinates
    ax.scatter(z, r, s=0.1, alpha=0.25, color="black")
    ax.set_xlabel("z [mm]", fontsize=14, ha="right", x=0.95)
    ax.set_ylabel("r [mm]", fontsize=14, ha="right", y=0.95)
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(0, 1100)
    atlasify("Detector hits in r-z plane", axes=ax, outside=True)
    
    fig.tight_layout()
    fig.savefig("/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/raw_hits_visualisation_rz_plane.png", dpi=300)

plot_raw_data(raw_data)

# Function to plot 3D visualization of edges, reconstructed and truth tracks
def plot_tracks_3d(event):
    """
    Plot 3D visualization of edges, reconstructed and truth tracks using x, y, z coordinates.
    """
    fig = plt.figure(figsize=(13, 16))
    
    # Create three 3D subplots in column layout
    ax1 = fig.add_subplot(311, projection='3d')  # Top: Full edge graph
    ax2 = fig.add_subplot(312, projection='3d')  # Middle: Reconstructed tracks
    ax3 = fig.add_subplot(313, projection='3d')  # Bottom: Truth tracks
    
    # Plot 1: Full edge graph in black
    for n_edge in tqdm(range(len(event.edge_index[0])), desc="Plotting all edges"):
        edge_in = event.edge_index[0][n_edge].item()
        edge_out = event.edge_index[1][n_edge].item()
        r1, phi1, z1 = event.r[edge_in], event.phi[edge_in], event.z[edge_in]
        r2, phi2, z2 = event.r[edge_out], event.phi[edge_out], event.z[edge_out]
        
        # Convert to cartesian coordinates
        x1, y1 = r1 * np.cos(phi1), r1 * np.sin(phi1)
        x2, y2 = r2 * np.cos(phi2), r2 * np.sin(phi2)
        
        # Plot the edges
        ax1.plot([x1, x2], [y1, y2], [z1, z2], color='black', alpha=0.25, linewidth=0.1)
    
    # Plot 2: Reconstructed tracks
    track_file_path = f"/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/track_building/valset_tracks/event[['000021000']].txt"
    with open(track_file_path, 'r') as f:
        num_tracks = sum(1 for _ in f)
    
    for n in tqdm(range(num_tracks), desc="Plotting reconstructed tracks"):
        track = np.loadtxt(track_file_path, skiprows=n, max_rows=1).astype(int).tolist()
        if type(track) is int:
            track = [track]
        if len(track) < 3:  # skip tracks with less than 3 hits for better 3D visualization
            continue
        
        track = np.array(track)
        node_id = np.searchsorted(event.hit_id, track)  
        full_track = {"n_hits": track, "r": event.r[node_id], "phi": event.phi[node_id], "z": event.z[node_id]}
        
        # Convert to cartesian coordinates
        x = full_track["r"] * np.cos(full_track["phi"])
        y = full_track["r"] * np.sin(full_track["phi"])
        z = full_track["z"]
        
        # Sort by r for consistent ordering
        sorted_indices = np.argsort(full_track["r"])
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        z_sorted = z[sorted_indices]
        
        # Plot track as connected line
        ax2.plot(x_sorted, y_sorted, z_sorted, color="blue", alpha=0.5, linewidth=0.2)
    
    # Plot 3: Truth tracks using particle_id
    # Group edges by particle_id
    particle_tracks = {}
    
    for i in tqdm(range(len(event.track_edges[0])), desc="Processing truth tracks"):
        edge_in = event.track_edges[0][i].item()
        edge_out = event.track_edges[1][i].item()
        particle_id = event.particle_id[i].item()
        pt_value = event.pt[i].item()
        
        if particle_id not in particle_tracks:
            particle_tracks[particle_id] = {
                'edges': [],
                'pt': pt_value
            }
        
        particle_tracks[particle_id]['edges'].append((edge_in, edge_out))
    
    # Build track sequences from edges for each particle
    truth_tracks = []
    
    for particle_id, track_data in tqdm(particle_tracks.items(), desc="Building truth tracks"):
        edges = track_data['edges']
        pt_value = track_data['pt']
        
        if len(edges) < 2:  # Skip tracks with less than 2 edges
            continue
            
        # Build hit sequence from edges
        # Create adjacency list
        adjacency = {}
        all_hits = set()
        
        for edge_in, edge_out in edges:
            all_hits.add(edge_in)
            all_hits.add(edge_out)
            if edge_in not in adjacency:
                adjacency[edge_in] = []
            if edge_out not in adjacency:
                adjacency[edge_out] = []
            adjacency[edge_in].append(edge_out)
            adjacency[edge_out].append(edge_in)
        
        # Find the start of the track (hit with only one neighbor or arbitrary start)
        start_hit = None
        for hit in all_hits:
            if len(adjacency[hit]) == 1:  # End point
                start_hit = hit
                break
        
        if start_hit is None and all_hits:  # If no clear end point, start with any hit
            start_hit = next(iter(all_hits))
        
        if start_hit is None:
            continue
            
        # Build the track sequence
        track_hits = [start_hit]
        visited = {start_hit}
        current_hit = start_hit
        
        while True:
            next_hits = [h for h in adjacency[current_hit] if h not in visited]
            if not next_hits:
                break
            
            # Choose the next hit (in case of multiple options, pick one)
            next_hit = next_hits[0]
            track_hits.append(next_hit)
            visited.add(next_hit)
            current_hit = next_hit
        
        if len(track_hits) >= 3:  # Only include tracks with at least 3 hits
            truth_tracks.append({'hits': track_hits, 'pt': pt_value, 'particle_id': particle_id})
    
    for n, track_info in enumerate(tqdm(truth_tracks, desc="Plotting truth tracks")):
        track_hits = track_info['hits']
        pt_value = track_info['pt']
        
        # Get coordinates
        r_vals = event.r[track_hits]
        phi_vals = event.phi[track_hits]
        z_vals = event.z[track_hits]
        
        # Convert to cartesian
        x = r_vals * np.cos(phi_vals)
        y = r_vals * np.sin(phi_vals)
        z = z_vals
        
        # Sort by r for proper track ordering
        sorted_indices = np.argsort(r_vals)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        z_sorted = z[sorted_indices]
        
        # Color based on pT
        color = 'red' if pt_value < 1000 else 'green'
        alpha = 0.1 if pt_value < 1000 else 0.5
        linewidth = 0.2 if pt_value < 1000 else 0.4
        
        # Plot track
        ax3.plot(x_sorted, y_sorted, z_sorted, color=color, alpha=alpha, linewidth=linewidth)
    
    # Configure all three plots
    titles = ['Post GNN Edges', 'Reconstructed Tracks', 'Truth Tracks']
    for ax, title in zip([ax1, ax2, ax3], titles):
        ax.set_xlabel('x [mm]', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('y [mm]', fontsize=14, ha="right", y=0.95)
        ax.set_zlabel('z [mm]', fontsize=14, ha="right", y=0.95)
        
        # Add subtitle for each plot
        ax.set_title(title, fontsize=14, pad=10)
        
        # Set limits
        ax.set_xlim(-1100, 1100)
        ax.set_ylim(-1100, 1100)
        ax.set_zlim(-3000, 3000)
        ax.set_box_aspect([1, 1, 3])  # Keep 1:1:3 ratio
        
        # Set viewing angle so z-axis (beamline) is horizontal
        ax.view_init(elev=20, azim=45, roll=90)

        # Make pane edges more subtle
        ax.xaxis.pane.set_edgecolor('gray')
        ax.yaxis.pane.set_edgecolor('gray')
        ax.zaxis.pane.set_edgecolor('gray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
    
    # Add suptitle
    fig.suptitle('TrackML Dataset', fontsize=16, y=0.98, fontweight='bold', style='italic')
    
    # Adjust layout for column layout
    plt.subplots_adjust(hspace=0)
    
    fig.tight_layout()
    fig.savefig("/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/all_pt/tracks_3d_visualization.png", dpi=300, bbox_inches='tight')

plot_tracks_3d(event)

def plot_pipeline():
    """
    Plot side-by-side (x-y) projections of:
      1. Metric learning graph edges
      2. GNN graph edges
      3. Reconstructed tracks
    Uses fixed file paths for event000021000.
    """
    # File paths
    ml_event_path = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/metric_learning/valset/event000021000.pyg"
    gnn_event_path = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/gnn/valset/event[['000021000']].pyg"
    track_file_path = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/track_building/valset_tracks/event[['000021000']].txt"
    
    # Load events
    ml_event = torch.load(ml_event_path, weights_only=False)
    gnn_event = torch.load(gnn_event_path, weights_only=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_ml, ax_gnn, ax_rec = axes
    
    # 1. Metric learning graph edges
    for idx in tqdm(range(len(ml_event.edge_index[0])), desc="Metric learning edges"):
        e_in = ml_event.edge_index[0][idx].item()
        e_out = ml_event.edge_index[1][idx].item()
        r1, phi1 = ml_event.r[e_in], ml_event.phi[e_in]
        r2, phi2 = ml_event.r[e_out], ml_event.phi[e_out]
        x1, y1 = r1 * np.cos(phi1), r1 * np.sin(phi1)
        x2, y2 = r2 * np.cos(phi2), r2 * np.sin(phi2)
        ax_ml.plot([x1, x2], [y1, y2], color='grey', alpha=0.35, linewidth=0.10)
    
    # 2. GNN graph edges
    for idx in tqdm(range(len(gnn_event.edge_index[0])), desc="GNN edges"):
        e_in = gnn_event.edge_index[0][idx].item()
        e_out = gnn_event.edge_index[1][idx].item()
        r1, phi1 = gnn_event.r[e_in], gnn_event.phi[e_in]
        r2, phi2 = gnn_event.r[e_out], gnn_event.phi[e_out]
        x1, y1 = r1 * np.cos(phi1), r1 * np.sin(phi1)
        x2, y2 = r2 * np.cos(phi2), r2 * np.sin(phi2)
        ax_gnn.plot([x1, x2], [y1, y2], color='black', alpha=0.35, linewidth=0.10)
    
    # 3. Reconstructed tracks (reuse logic from plot_edges)
    with open(track_file_path, 'r') as f:
        num_tracks = sum(1 for _ in f)
    # Need a reference event for hit geometry (use gnn_event which has r,phi,z,hit_id)
    ref_event = gnn_event
    for n in tqdm(range(num_tracks), desc="Reconstructed tracks"):
        track = np.loadtxt(track_file_path, skiprows=n, max_rows=1).astype(int).tolist()
        if isinstance(track, int):
            track = [track]
        if len(track) < 2:
            continue
        track = np.array(track)
        node_id = np.searchsorted(ref_event.hit_id, track)
        r_vals = ref_event.r[node_id]
        phi_vals = ref_event.phi[node_id]
        # Order by radius
        order = np.argsort(r_vals)
        r_sorted = r_vals[order]
        phi_sorted = phi_vals[order]
        x = r_sorted * np.cos(phi_sorted)
        y = r_sorted * np.sin(phi_sorted)
        ax_rec.plot(x, y, color='blue', alpha=0.35, linewidth=0.12)
    
    # Common styling
    for ax in axes:
        ax.set_xlim(-1100, 1100)
        ax.set_ylim(-1100, 1100)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [mm]", fontsize=12, ha="right", x=0.95)
        ax.set_ylabel("y [mm]", fontsize=12, ha="right", y=0.95)
    
    atlasify("Metric learning graph", axes=ax_ml, outside=True)
    atlasify("GNN graph", axes=ax_gnn, outside=True)
    atlasify("Reconstructed tracks", axes=ax_rec, outside=True)
    
    fig.tight_layout()
    out_path = "/pscratch/sd/l/lperon/ATLAS/acorn/UQ_code/MCD/trackML/plots/pipeline_visualisation_xy.png"
    fig.savefig(out_path, dpi=300)
    return fig

plot_pipeline()
