import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
import torch
from tqdm import tqdm
import yaml
from pytorch_lightning import Trainer
from scipy.stats import kurtosis, skew
from acorn.stages.edge_classifier.models.filter import Filter
from acorn.stages.edge_classifier.models.interaction_gnn import InteractionGNN

from acorn.stages.track_building.utils import rearrange_by_distance
from acorn.utils.plotting_utils import (
    get_ratio,
    plot_1d_histogram,
    plot_eff_pur_region,
    plot_efficiency_rz,
    plot_score_histogram,
    plot_efficiency_2D,
)
from acorn.utils.version_utils import get_pyg_data_keys
from acorn.stages.graph_construction.models.utils import graph_intersection


def dump_edges(event, config):
    src, dst = event.track_edges
    df = pd.DataFrame()
    df["track_to_edge_map_bool"] = event.track_to_edge_map[event.track_target_mask] >= 0
    df["track_particle_radius"] = event.track_particle_radius[event.track_target_mask]
    df["track_particle_pt"] = event.track_particle_pt[event.track_target_mask]
    df["track_particle_eta"] = event.track_particle_eta[event.track_target_mask]
    df["track_particle_pdgId"] = event.track_particle_pdgId[event.track_target_mask]
    df["track_particle_primary"] = event.track_particle_primary[event.track_target_mask]
    df["track_particle_id"] = event.track_particle_id[event.track_target_mask]
    df["hit_id_src"] = event.hit_id[src[event.track_target_mask]]
    df["hit_id_dst"] = event.hit_id[dst[event.track_target_mask]]

    delta_eta = (
        event.hit_eta[src[event.track_target_mask]]
        - event.hit_eta[dst[event.track_target_mask]]
    )
    df["deta"] = delta_eta

    delta_phi = (
        event.hit_phi[dst[event.track_target_mask]]
        - event.hit_phi[src[event.track_target_mask]]
    )
    # Reset angles
    delta_phi[delta_phi > np.pi] = delta_phi[delta_phi > np.pi] - 2 * np.pi
    delta_phi[delta_phi < -np.pi] = delta_phi[delta_phi < -np.pi] + 2 * np.pi
    df["dphi"] = delta_phi

    delta_z = (
        event.hit_z[dst[event.track_target_mask]]
        - event.hit_z[src[event.track_target_mask]]
    )
    delta_r = (
        event.hit_r[dst[event.track_target_mask]]
        - event.hit_r[src[event.track_target_mask]]
    )
    z0 = event.hit_z[src[event.track_target_mask]] - (
        event.hit_r[src[event.track_target_mask]] * delta_z / delta_r
    )
    z0[delta_r == 0] = 0
    df["z0"] = z0

    phi_slope = delta_phi / delta_r
    phi_slope[delta_r == 0] = 0
    df["phi_slope"] = phi_slope

    df.to_csv(
        os.path.join(config["stage_dir"], f"dump_track_edges_event{event.event_id}.csv")
    )


def graph_construction_efficiency(lightning_module, plot_config, config):
    """
    Plot the graph construction efficiency vs. pT of the edge.
    """
    all_y_truth, target_pt = [], []
    target_eta = []
    graph_size = []

    n_train = config["n_train"]
    dataset_name = config["dataset"]
    calibration = config.get("calibration", False)
    dataset = getattr(lightning_module, dataset_name)

    for event in tqdm(dataset):
        event = event.to(lightning_module.device)
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.track_target_mask = torch.ones(
                event.track_to_edge_map.shape[0], dtype=torch.bool
            ).to(lightning_module.device)

        all_y_truth.append(event.track_to_edge_map[event.track_target_mask] >= 0)
        target_pt.append(event.track_particle_pt[event.track_target_mask])

        target_eta.append(event.hit_eta[event.track_edges[:, event.track_target_mask][0]])
        graph_size.append(event.edge_index.size(1))

        if plot_config.get("dump_edges", False):
            dump_edges(event, config)

    #  TODO: Handle different pT units!
    target_pt = torch.cat(target_pt).cpu().numpy()
    target_eta = torch.cat(target_eta).cpu().numpy()
    all_y_truth = torch.cat(all_y_truth).cpu().numpy()

    # Get the edgewise efficiency
    # Build a histogram of true pTs, and a histogram of true-positive pTs
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    pt_max = 50
    if pt_min==0:
        pt_min+= 1e-1
    print(f"pt_min: {pt_min}")
    nb_pt_bins = 10 if pt_min != 1e-1 else 17
    if "pt_units" in plot_config and plot_config["pt_units"] == "MeV":
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), nb_pt_bins)
    if "eta_lim" in plot_config:
        eta_min, eta_max = plot_config["eta_lim"]
    else:
        eta_min, eta_max = [-4, 4]
    eta_bins = np.linspace(eta_min, eta_max)

    true_pt_hist, _ = np.histogram(target_pt, bins=pt_bins)
    true_pos_pt_hist, _ = np.histogram(target_pt[all_y_truth], bins=pt_bins)

    true_eta_hist, true_eta_bins = np.histogram(target_eta, bins=eta_bins)
    true_pos_eta_hist, _ = np.histogram(target_eta[all_y_truth], bins=eta_bins)

    pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]

    for true_pos_hist, true_hist, bins, xlabel, xlim, logx, filename, filename_svg in zip(
        [true_pos_pt_hist, true_pos_eta_hist],
        [true_pt_hist, true_eta_hist],
        [pt_bins, eta_bins],
        [f"$p_T [{pt_units}]$", r"$\eta$"],
        [[pt_min, pt_max], [eta_min, eta_max]],
        [True, False],
        [f"edgewise_efficiency_pt_{n_train}.png", f"edgewise_efficiency_eta_{n_train}.png"],
        [f"edgewise_efficiency_pt_{n_train}.svg", f"edgewise_efficiency_eta_{n_train}.svg"],
    ):
        hist, err = get_ratio(true_pos_hist, true_hist)
        if plot_config.get("filename_template") is not None:
            filename = config["filename_template"] + "_" + filename
            filename_svg = config["filename_template"] + "_" + filename_svg
        fig, ax = plot_1d_histogram(
            hist,
            bins,
            err,
            xlabel,
            plot_config["title"],
            plot_config.get("ylim", [0.8, 1.06]),
            xlim,
            "Efficiency",
            logx=logx,
        )
        # Save the plot
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$GeV, $|\eta| < 4$" + "\n"
            r"Mean graph size: "
            + f"{np.mean(graph_size):.2e}"
            + r"$\pm$"
            + f"{np.std(graph_size):.2e}"
            + "\n"
            + f"Global efficiency: {all_y_truth.sum() / target_pt.shape[0] :.4f}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}",
        )
        calib_folder = "calibrated" if calibration else "uncalibrated"
        if not config["input_cut"]:
            calib_folder += "/no_input_cut"
        else:
            calib_folder += "/with_input_cut"
        fig.savefig(os.path.join(config["stage_dir"]+f"", filename))
        fig.savefig(os.path.join(config["stage_dir"]+f"", filename_svg))

        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(config["stage_dir"]+f"", filename)}'
        )


def graph_construction_efficiency_2D(lightning_module, plot_config: dict, config: dict):
    """_summary_

    Args:
        plot_config (dict): any plotting config
        config (dict): config

    Plot graph construction efficiency efficiency against (r,z) or (x,y)
    """
    vars = plot_config.get("vars")

    print(f"Plotting graph construction efficiency as a function of {vars})")
    print(f"Using  events from {config['dataset']}")

    if "target_tracks" in config:
        print(f"Track selection criteria: \n{yaml.dump(config.get('target_tracks'))}")
    else:
        print("No track selection criteria found, accepting all tracks.")

    target = dict()
    for x, y in vars:
        target[x] = torch.empty(0).to(lightning_module.device)
        target[y] = torch.empty(0).to(lightning_module.device)
    all_target = target.copy()
    graph_size, n_graphs = (0, 0)

    dataset_name = config["dataset"]
    dataset = getattr(lightning_module, dataset_name)

    for event in tqdm(dataset):
        event = event.to(lightning_module.device)

        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.track_target_mask = torch.ones(
                event.track_to_edge_map.shape[0], dtype=torch.bool
            ).to(lightning_module.device)

        # mm -> m scaling
        for v in ["hit_x", "hit_z", "hit_r"]:
            if v in target:
                event[v] /= 1000

        if "hit_y" in target and "hit_y" not in event.keys():
            # DEPRECATED : linked to old naming scheme, y of SP is missing because it is overwritting by the true/fake boolean
            event.hit_y = event.hit_r * torch.sin(event.hit_phi)

        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)

        # indices of all target edges present in the input graph
        track_map = graph_intersection(
            event.edge_index,
            event.track_edges,
            return_y_pred=False,
            return_y_truth=False,
            return_truth_to_pred=True,
        )
        target_edges = event.track_edges[
            :, event.track_target_mask & (track_map > -1)
        ]

        # indices of all target edges (may or may not be present in the input graph)
        all_target_edges = event.track_edges[:, event.track_target_mask]

        # get target z r
        for key, item in target.items():
            target[key] = torch.cat([item, event[key][target_edges[0]]], dim=0)
        for key, item in all_target.items():
            all_target[key] = torch.cat([item, event[key][all_target_edges[0]]], dim=0)

        graph_size += event.edge_index.size(1)
        n_graphs += 1

    for x, y in vars:
        fig, ax = plot_efficiency_2D(
            all_target[x].cpu(),
            all_target[y].cpu(),
            target[x].cpu(),
            target[y].cpu(),
            plot_config,
            x,
            y,
        )
        n_train = config["n_train"]
        # Save the plot
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        atlasify(f"{n_train} train events",
            r" \bar{t}$ and soft interactions " + "\n"
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$ " + "\n"
            "Graph Construction Efficiency:"
            f" {(target[x].shape[0] / all_target[x].shape[0]):.4f}, "
            + f"Mean graph size: {graph_size / n_graphs :.2e} \n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}",
        )
        plt.tight_layout()
        save_dir = os.path.join(
            config["stage_dir"],
            f"{plot_config.get('filename', f'graph_construction_edgewise_efficiency_{x}{y}')}.png",
        )
        fig.savefig(save_dir)
        print(f"Finish plotting. Find the plot at {save_dir}")
        plt.close()


def graph_scoring_efficiency(lightning_module, plot_config, config):
    """
    Plot the graph construction efficiency vs. pT of the edge.
    """
    print("Plotting efficiency against pT and eta")
    true_positive, target_pt, target_eta = [], [], []
    pred = []
    track = []
    
    calibration = config.get("calibration", False)
    n_train = config["n_train"]
    score_cut = config["score_cut"]
    print(
        f"Using score cut: {score_cut}, events from {config['dataset']}"
    )
    if "target_tracks" in config:
        print(f"Track selection criteria: \n{yaml.dump(config.get('target_tracks'))}")
    else:
        print("No track selection criteria found, accepting all tracks.")

    dataset_name = config["dataset"]
    dataset = getattr(lightning_module, dataset_name)

    for event in tqdm(dataset):
        event = event.to(lightning_module.device)

        # Need to apply score cut and remap the track_to_edge_map

        lightning_module.apply_score_cut(event, score_cut)
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.track_target_mask = torch.ones(
                event.track_to_passing_edge_map.shape[0], dtype=torch.bool
            ).to(lightning_module.device)

        # get all target true positives
        true_positive.append(
            (event.track_to_passing_edge_map[event.track_target_mask] > -1).to(
                lightning_module.device
            )
        )
        print(event.track_target_mask.sum(), len(event.track_target_mask))

        # get all target pt. Length = number of target true. This includes ALL truth edges in the event,
        # even those not included in the input graph. We MUST filter them out to isolate the inefficiency from model.
        # Otherwise, we are plotting cumilative edge efficiency.
        target_pt.append(event.track_particle_pt[event.track_target_mask])

        # similarly for target eta
        target_eta.append(event.hit_eta[event.track_edges[0, event.track_target_mask]])

        # get all edges passing edge cut
        if "edge_scores" in get_pyg_data_keys(event):
            pred.append(event.edge_scores >= score_cut)
        else:
            pred.append(event.edge_y)
        # get a boolean array answer the question is this target edge in input graph
        track.append((event.track_to_edge_map[event.track_target_mask] > -1))

    # concat all target pt and eta
    target_pt = torch.cat(target_pt).cpu().numpy()
    target_eta = torch.cat(target_eta).cpu().numpy()

    # get all true positive
    true_positive = torch.cat(true_positive).cpu().numpy()

    # get all positive
    track = torch.cat(track).cpu().numpy()

    # count number of graphs to calculate mean efficiency
    n_graphs = len(pred)

    # get all predictions
    pred = torch.cat(pred).cpu().numpy()

    # get mean graph size
    mean_graph_size = pred.sum() / n_graphs

    # get mean target efficiency
    target_efficiency = true_positive.sum() / len(target_pt[track])
    target_purity = true_positive[track].sum() / pred.sum()
    cumulative_efficiency = true_positive.sum() / len(target_pt)

    # get graph construction efficiency
    graph_construction_efficiency = track.mean()

    # Get the edgewise efficiency
    # Build a histogram of true pTs, and a histogram of true-positive pTs
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    pt_max = 50
    if pt_min==0:
        pt_min+= 1e-1
    print(f"pt_min: {pt_min}")
    nb_pt_bins = 10 if pt_min != 1e-1 else 17
    if "pt_units" in plot_config and plot_config["pt_units"] == "MeV":
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), nb_pt_bins)

    eta_bins = np.linspace(-4, 4)

    true_pt_hist, true_pt_bins = np.histogram(target_pt[track], bins=pt_bins)
    true_pos_pt_hist, _ = np.histogram(target_pt[true_positive], bins=pt_bins)

    true_eta_hist, true_eta_bins = np.histogram(target_eta[track], bins=eta_bins)
    true_pos_eta_hist, _ = np.histogram(target_eta[true_positive], bins=eta_bins)

    pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]

    filename = plot_config.get("filename", "edgewise_efficiency")

    if calibration:
        calib_folder = "calibrated"
    else:
        calib_folder = "uncalibrated"

    if not config["input_cut"]:
        calib_folder += "/no_input_cut"
    else:
        calib_folder += "/with_input_cut"
    
    for true_pos_hist, true_hist, bins, xlabel, logx, filename, filename_svg in zip(
        [true_pos_pt_hist, true_pos_eta_hist],
        [true_pt_hist, true_eta_hist],
        [true_pt_bins, true_eta_bins],
        [f"$p_T [{pt_units}]$", r"$\eta$"],
        [True, False],
        [f"{filename}_pt.png", f"{filename}_eta.png"],
        [f"{filename}_pt.svg", f"{filename}_eta.svg"],
    ):
        # Divide the two histograms to get the edgewise efficiency
        hist, err = get_ratio(true_pos_hist, true_hist)
        nb_of_true_pos = true_pos_hist.sum()
        nb_of_true = true_hist.sum()
        ylim = plot_config.get("ylim", [np.min(hist) - 1.1*np.max(err), 1.05] if logx else [0.85, 1.05])
        fig, ax = plot_1d_histogram(
            hist,
            bins,
            err,
            xlabel,
            ylabel=plot_config["title"],
            xlim=[pt_min, pt_max] if logx else [-4, 4],
            ylim=ylim,
            label="Efficiency",
            logx=logx,
        )
        # Save the plot
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: " + str(score_cut) + "\n"
            f"Input graph size: {pred.shape[0]/n_graphs:.2e}, Graph Construction"
            f" Efficiency: {graph_construction_efficiency:.4f}" + "\n"
            f"Mean graph size: {mean_graph_size:.2e}, Signal Efficiency:"
            f" {target_efficiency:.4f}"
            + "\n"
            + f"Cumulative Signal Efficiency: {cumulative_efficiency:.4f}\n"
            + f"Number of true positives: {nb_of_true_pos}, Number of true: {nb_of_true}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}",
        )

        fig.savefig(os.path.join(config["stage_dir"]+f"/", filename))
        fig.savefig(os.path.join(config["stage_dir"]+f"/", filename_svg))
        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(config["stage_dir"], filename)}'
        )
        if logx==r"$\eta$":
            np.savetxt(os.path.join(config["stage_dir"]+f"/", f"edgewise_efficiency_eta_hist.txt"), hist)
            np.savetxt(os.path.join(config["stage_dir"]+f"/", f"edgewise_efficiency_eta_err.txt"), err)
            np.savetxt(os.path.join(config["stage_dir"]+f"/", f"edgewise_efficiency_eta_bins.txt"), bins)


def graph_scoring_efficiency_compar_calib(lightning_module, plot_config, config):
    filename = "edgewise_efficiency_eta"
    n_train = config["n_train"]
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    score_cut = config["score_cut"]
    dataset_name = config["dataset"]
    dataset = getattr(lightning_module, dataset_name)
    fig, ax = plt.subplots(figsize=(8, 6))
    for calib in ["calibrated", "uncalibrated"]:
        calib_folder = calib
        if not config["input_cut"]:
            calib_folder += "/no_input_cut"
        else:
            calib_folder += "/with_input_cut"
        hist = os.path.join(config["stage_dir"]+f"/", f"{filename}_efficiency_hist.txt")
        err = os.path.join(config["stage_dir"]+f"/", f"{filename}_efficiency_err.txt")
        bins = os.path.join(config["stage_dir"]+f"/", f"{filename}_efficiency_bins.txt")
        xvals = (bins[1:] + bins[:-1]) / 2
        xerrs = (bins[1:] - bins[:-1]) / 2
        ax.errorbar(xvals, hist, xerr=xerrs, yerr=err, fmt="o", color="black" if calib=="uncalibrated" else "red", label="calibrated" if calib == "calibrated" else "uncalibrated")
    ax.set_xlabel(r"$\eta$", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel(plot_config["title"], ha="right", y=0.95, fontsize=14)
    atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: " + str(score_cut) + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}",
        )
    fig.savefig(os.path.join(config["stage_dir"]+f"/plots/", f"{filename}_efficiency_compar.png"))
    fig.savefig(os.path.join(config["stage_dir"]+f"/plots/", f"{filename}_efficiency_compar.svg"))


def multi_edgecut_graph_scoring_efficiency(lightning_module, plot_config, config):
    """Plot graph scoring efficiency across multiple score cuts

    Args:
        lightning_module (_type_): lightning module from which to draw evaluation
        plot_config (_type_): Plot config, must contain
            'score_cuts: LIST OF CUTS
            'filename_template': A TEMPLATE FOR FILENAME
        config (_type_): Usual config from lightning module and evaluation config
    """

    filenames = [
        f"{plot_config['template_filename']}_{cut*100:.0f}"
        for cut in plot_config["score_cuts"]
    ]
    config_ = config.copy()
    for score_cut, filename in zip(plot_config["score_cuts"], filenames):
        config_["score_cut"] = score_cut
        plot_config["filename"] = filename
        graph_scoring_efficiency(lightning_module, plot_config, config_)


def graph_roc_curve(lightning_module, plot_config, config):
    """
    Plot the ROC curve for the graph construction efficiency.
    """
    print(
        "Plotting the ROC curve and score distribution, events from"
        f" {config['dataset']}"
    )
    all_y_truth, all_scores, masked_scores, masked_y_truth = [], [], [], []
    masks = []
    dataset_name = config["dataset"]
    print("dataset_name ROC Curve", dataset_name)
    calibration = config.get("calibration", False)
    n_train = config["n_train"]

    dataset = getattr(lightning_module, dataset_name)

    for event in tqdm(dataset):
        event = event.to(lightning_module.device)
        # Need to apply score cut and remap the track_to_edge_map
        if "edge_weights" in get_pyg_data_keys(event):
            target_y = event.edge_weights.bool() & event.edge_y.bool()
            mask = event.edge_weights > 0
        else:
            target_y = event.edge_y.bool()
            mask = torch.ones_like(target_y).bool().to(target_y.device)

        all_y_truth.append(target_y)
        all_scores.append(event.edge_scores)
        masked_scores.append(event.edge_scores[mask])
        masked_y_truth.append(target_y[mask])
        masks.append(mask)

    all_scores = torch.cat(all_scores).cpu().numpy()
    all_y_truth = torch.cat(all_y_truth).cpu().numpy()
    masked_scores = torch.cat(masked_scores).cpu().numpy()
    masked_y_truth = torch.cat(masked_y_truth).cpu().numpy()
    masks = torch.cat(masks).cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    # Get the ROC curve
    fpr, tpr, _ = roc_curve(all_y_truth, all_scores)
    full_auc_score = auc(fpr, tpr)

    # Plot the ROC curve
    ax.plot(fpr, tpr, color="black", label="ROC curve")

    # Get the ROC curve
    fpr, tpr, _ = roc_curve(masked_y_truth, masked_scores)
    masked_auc_score = auc(fpr, tpr)

    # Plot the ROC curve
    ax.plot(fpr, tpr, color="green", label="masked ROC curve")

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random classifier")
    ax.set_xlabel("False Positive Rate", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("True Positive Rate", ha="right", y=0.95, fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=14)
    ax.text(
        0.95,
        0.20,
        f"Full AUC: {full_auc_score:.3f}, Masked AUC: {masked_auc_score: .3f}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=14,
    )
    if calibration:
        calib_folder = "calibrated"
    else:
        calib_folder = "uncalibrated"
    
    if not config["input_cut"]:
        calib_folder += "/no_input_cut"
    else:
        calib_folder += "/with_input_cut"
    n_train = config["n_train"]
    # Save the plot
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    atlasify(f"{n_train} train events",
        f"{plot_config['title']} \n"
        r"Target: $p_T >" + f"{pt_min}"+"$GeV, $|\eta| < 4$"
        + "\n"
        + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}",
    )
    filename_template = plot_config.get("filename")
    filename = (
        f"{filename_template}_roc_curve.png"
        if filename_template is not None
        else f"roc_curve.png"
    )
    filename_svg = (
        f"{filename_template}_roc_curve.svg"
        if filename_template is not None
        else f"roc_curve.svg"
    )
    filename = os.path.join(config["stage_dir"]+f"/", filename)
    filename_svg = os.path.join(config["stage_dir"]+f"/", filename_svg)
    fig.savefig(filename)
    fig.savefig(filename_svg)
    print("Finish plotting. Find the ROC curve at" f" {filename}")
    plt.close()

    np.savetxt(os.path.join(config["stage_dir"]+f"/", f"roc_curve_fpr.txt"), fpr)
    np.savetxt(os.path.join(config["stage_dir"]+f"/", f"roc_curve_tpr.txt"), tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    all_y_truth = all_y_truth.astype(np.int16)
    all_y_truth[~masks] = 2
    labels = np.array(["Fake"] * len(all_y_truth))
    labels[all_y_truth == 1] = "Target True"
    labels[all_y_truth == 2] = "Non-target True"
    weight = 1 / dataset.len()
    ax = plot_score_histogram(all_scores, labels, ax=ax, inverse_dataset_length=weight)
    ax.set_xlabel("Edge score", ha="right", x=0.95, fontsize=14)
    ax.set_ylabel("Count/event", ha="right", y=0.95, fontsize=14)
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    # Compute the histograms integrals for fake, target true and non-target true
    fake_hist = np.histogram(all_scores[all_y_truth == 0], bins=50)[0]
    target_true_hist = np.histogram(all_scores[all_y_truth == 1], bins=50)[0]
    non_target_true_hist = np.histogram(all_scores[all_y_truth == 2], bins=50)[0]
    atlasify(f"{n_train} train events",
        "Score Distribution \n"
        r"Target: $p_T >" + f"{pt_min}"+"$GeV, $|\eta| < 4$"
        + "\n"
        + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}\n"
        + f"Number of fake : {fake_hist.sum()}, target true : {target_true_hist.sum()}, non-target true : {non_target_true_hist.sum()}",
    )
    filename = (
        f"{filename_template}_score_distribution.png"
        if filename_template is not None
        else f"score_distribution.png"
    )
    filename_svg = (
        f"{filename_template}_score_distribution.svg"
        if filename_template is not None
        else f"score_distribution.svg"
    )
    filename = os.path.join(config["stage_dir"]+f"/", filename)
    filename_svg = os.path.join(config["stage_dir"]+f"/", filename_svg)
    fig.savefig(filename)
    fig.savefig(filename_svg)
    print("Finish plotting. Find the score distribution at" f" {filename}")


def graph_region_efficiency_purity(lightning_module, plot_config, config):
    print(f"Plotting efficiency and purity by region , events from {config['dataset']}")
    edge_truth, edge_regions, edge_positive = [], [], []
    node_r, node_z, node_regions = [], [], []

    dataset_name = config["dataset"]
    dataset = getattr(lightning_module, dataset_name)
    for event in tqdm(dataset):
        with torch.no_grad():
            eval_dict = lightning_module.shared_evaluation(
                event.to(lightning_module.device), 0
            )
        event = eval_dict["batch"]
        event.edge_scores = torch.sigmoid(eval_dict["output"])
        edge_truth.append(event.edge_y)
        edge_regions.append(
            event.hit_region[event.edge_index[0]]
        )  # Assign region depending on first node in edge
        edge_positive.append(event.edge_scores > config["edge_cut"])

        node_r.append(event.hit_r)
        node_z.append(event.hit_z)
        node_regions.append(event.hit_region)

    edge_truth = torch.cat(edge_truth).cpu().numpy()
    edge_regions = torch.cat(edge_regions).cpu().numpy()
    edge_positive = torch.cat(edge_positive).cpu().numpy()

    node_r = torch.cat(node_r).cpu().numpy()
    node_z = torch.cat(node_z).cpu().numpy()
    node_regions = torch.cat(node_regions).cpu().numpy()

    fig, ax = plot_eff_pur_region(
        edge_truth,
        edge_positive,
        edge_regions,
        node_r,
        node_z,
        node_regions,
        plot_config,
    )
    fig.savefig(os.path.join(config["stage_dir"], "region_eff_pur.png"))
    fig.savefig(os.path.join(config["stage_dir"], "region_eff_pur.svg"))
    print(
        "Finish plotting. Find the plot at"
        f' {os.path.join(config["stage_dir"], "region_eff_pur.png")}'
    )


def gnn_efficiency_rz(lightning_module, plot_config: dict, config: dict):
    """_summary_

    Args:
        plot_config (dict): any plotting config
        config (dict): config

    Plot GNN edgewise efficiency against rz
    """

    print("Plotting edgewise efficiency as a function of rz")
    print(
        f"Using score cut: {config.get('score_cut')}, events from {config['dataset']}"
    )
    if "target_tracks" in config:
        print(f"Track selection criteria: \n{yaml.dump(config.get('target_tracks'))}")
    else:
        print("No track selection criteria found, accepting all tracks.")

    target = {
        "hit_z": torch.empty(0).to(lightning_module.device),
        "hit_r": torch.empty(0).to(lightning_module.device),
    }
    all_target = target.copy()
    true_positive = target.copy()
    input_graph_size, graph_size, n_graphs = (0, 0, 0)

    dataset_name = config["dataset"]
    dataset = getattr(lightning_module, dataset_name)

    for event in tqdm(dataset):
        event = event.to(lightning_module.device)

        # Need to apply score cut and remap the track_to_edge_map
        if "score_cut" in config:
            lightning_module.apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.track_target_mask = torch.ones(
                event.track_to_passing_edge_map.shape[0], dtype=torch.bool
            ).to(lightning_module.device)

        # scale r and z
        event.hit_r /= 1000
        event.hit_z /= 1000

        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)

        # indices of all target edges present in the input graph
        target_edges = event.track_edges[
            :, event.track_target_mask & (event.track_to_edge_map > -1)
        ]

        # indices of all target edges (may or may not be present in the input graph)
        all_target_edges = event.track_edges[:, event.track_target_mask]

        # get target z r
        for key, item in target.items():
            target[key] = torch.cat([item, event[key][target_edges[0]]], dim=0)
        for key, item in all_target.items():
            all_target[key] = torch.cat([item, event[key][all_target_edges[0]]], dim=0)

        # indices of all true positive target edges
        target_true_positive_edges = event.track_edges[
            :, event.track_target_mask & (event.track_to_passing_edge_map > -1)
        ]
        for key in ["hit_r", "hit_z"]:
            true_positive[key] = torch.cat(
                [true_positive[key], event[key][target_true_positive_edges[0]]], dim=0
            )

        input_graph_size += event.edge_index.size(1)
        graph_size += event.edge_pred.sum().cpu().numpy()
        n_graphs += 1

    fig, ax = plot_efficiency_rz(
        target["hit_z"].cpu(),
        target["hit_r"].cpu(),
        true_positive["hit_z"].cpu(),
        true_positive["hit_r"].cpu(),
        plot_config,
    )
    calibration = config.get("calibration", False)
    if calibration:
        calib_folder = "calibrated"
    else:
        calib_folder = "uncalibrated"
    
    if not config["input_cut"]:
        calib_folder += "/no_input_cut"
    else:
        calib_folder += "/with_input_cut"
    n_train = config["n_train"]
    # Save the plot
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$ " + "\n"
        "Graph Construction Efficiency:"
        f" {(target['hit_z'].shape[0] / all_target['hit_z'].shape[0]):.4f}, Input graph size:"
        f" {input_graph_size / n_graphs: .2e} \n"
        r"Edge score cut: "
        + str(config["score_cut"])
        + f", Mean graph size: {graph_size / n_graphs :.2e} \n"
        "Signal Efficiency:"
        f" {true_positive['hit_z'].shape[0] / target['hit_z'].shape[0] :.4f} \n"
        "Cumulative signal efficiency:"
        f" {true_positive['hit_z'].shape[0] / all_target['hit_z'].shape[0]: .4f}"
        + "\n"
        + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}",
    )
    plt.tight_layout()
    save_dir = os.path.join(
        config["stage_dir"]+f"/",
        f"{plot_config.get('filename', 'edgewise_efficiency_rz')}.png",
    )
    save_dir_svg = os.path.join(
        config["stage_dir"]+f"/",
        f"{plot_config.get('filename', 'edgewise_efficiency_rz')}.svg",
    )
    fig.savefig(save_dir)
    fig.savefig(save_dir_svg)
    print(f"Finish plotting. Find the plot at {save_dir}")
    plt.close()

    fig, ax = plot_efficiency_rz(
        all_target["hit_z"].cpu(),
        all_target["hit_r"].cpu(),
        true_positive["hit_z"].cpu(),
        true_positive["hit_r"].cpu(),
        plot_config,
    )
    n_train = config["n_train"]
    # Save the plot
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$ " + "\n"
        "Graph Construction Efficiency:"
        f" {(target['hit_z'].shape[0] / all_target['hit_z'].shape[0]):.4f}, Input graph size:"
        f" {input_graph_size / n_graphs: .2e} \n"
        r"Edge score cut: "
        + str(config["score_cut"])
        + f", Mean graph size: {graph_size / n_graphs :.2e} \n"
        "Signal Efficiency:"
        f" {true_positive['hit_z'].shape[0] / target['hit_z'].shape[0] :.4f} \n"
        "Cumulative signal efficiency:"
        f" {true_positive['hit_z'].shape[0] / all_target['hit_z'].shape[0]: .4f}"
        + "\n"
        + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}",
    )
    plt.tight_layout()
    save_dir = os.path.join(
        config["stage_dir"]+f"/",
        f"cumulative_{plot_config.get('filename', 'edgewise_efficiency_rz')}.png",
    )
    save_dir_svg = os.path.join(
        config["stage_dir"]+f"/",
        f"cumulative_{plot_config.get('filename', 'edgewise_efficiency_rz')}.svg",
    )
    fig.savefig(save_dir)
    fig.savefig(save_dir_svg)
    print(f"Finish plotting. Find the plot at {save_dir}")
    plt.close()


def gnn_purity_rz(lightning_module, plot_config: dict, config: dict):
    """_summary_

    Args:
        plot_config (dict): any plotting config
        config (dict): config

    Plot GNN edgewise efficiency against rz
    """

    print("Plotting edgewise purity as a function of rz")
    print(
        f"Using score cut: {config.get('score_cut')}, events from {config['dataset']}"
    )
    if "target_tracks" in config:
        print(f"Track selection criteria: \n{yaml.dump(config.get('target_tracks'))}")
    else:
        print("No track selection criteria found, accepting all tracks.")

    true_positive = {
        key: torch.empty(0).to(lightning_module.device) for key in ["hit_z", "hit_r"]
    }
    target_true_positive = true_positive.copy()

    pred = true_positive.copy()
    masked_pred = true_positive.copy()

    dataset_name = config["dataset"]
    n_train = config["n_train"]
    dataset = getattr(lightning_module, dataset_name)

    for event in tqdm(dataset):
        event = event.to(lightning_module.device)
        # Need to apply score cut and remap the track_to_edge_map
        if "score_cut" in config:
            lightning_module.apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.track_target_mask = torch.ones(
                event.track_to_edge_map.shape[0], dtype=torch.bool
            ).to(lightning_module.device)

        # scale r and z
        event.hit_r /= 1000
        event.hit_z /= 1000

        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)

        # target true positive edge indices, used as numerator of target purity and purity
        target_true_positive_edges = event.track_edges[
            :, event.track_target_mask & (event.track_to_passing_edge_map > -1)
        ]

        print(len(target_true_positive_edges))

        # true positive edge indices, used as numerator of total purity
        true_positive_edges = event.track_edges[
            :, (event.track_to_passing_edge_map > -1)
        ]

        # all positive edges, used as denominator of total and target purity
        positive_edges = event.edge_index[:, event.edge_pred]

        # masked positive edge indices, including true positive target edges and all false positive edges
        fake_positive_edges = event.edge_index[:, event.edge_pred & (event.edge_y == 0)]
        masked_positive_edges = torch.cat(
            [target_true_positive_edges, fake_positive_edges], dim=1
        )

        for key in ["hit_r", "hit_z"]:
            target_true_positive[key] = torch.cat(
                [
                    target_true_positive[key].float(),
                    event[key][target_true_positive_edges[0]].float(),
                ],
                dim=0,
            )
            true_positive[key] = torch.cat(
                [
                    true_positive[key].float(),
                    event[key][true_positive_edges[0]].float(),
                ],
                dim=0,
            )
            pred[key] = torch.cat(
                [pred[key].float(), event[key][positive_edges[0]].float()], dim=0
            )
            masked_pred[key] = torch.cat(
                [
                    masked_pred[key].float(),
                    event[key][masked_positive_edges[0]].float(),
                ],
                dim=0,
            )

    purity_definition_label = {
        "target_purity": "Target Purity",
        "masked_purity": "Masked Purity",
        "total_purity": "Total Purity",
    }
    for numerator, denominator, suffix in zip(
        [true_positive, target_true_positive, target_true_positive],
        [pred, pred, masked_pred],
        ["total_purity", "target_purity", "masked_purity"],
    ):
        fig, ax = plot_efficiency_rz(
            denominator["hit_z"].cpu(),
            denominator["hit_r"].cpu(),
            numerator["hit_z"].cpu(),
            numerator["hit_r"].cpu(),
            plot_config,
        )
        calibration = config.get("calibration", False)
        if calibration:
            calib_folder = "calibrated"
        else:
            calib_folder = "uncalibrated"
        
        if not config["input_cut"]:
            calib_folder += "/no_input_cut"
        else:
            calib_folder += "/with_input_cut"
        n_train = config["n_train"]
        # Save the plot
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        atlasify(f"{n_train} train events",

            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: "
            + str(config["score_cut"])
            + "\n"
            + purity_definition_label[suffix]
            + ": "
            + f"{numerator['hit_z'].size(0) / denominator['hit_z'].size(0) : .5f}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}",
        )
        plt.tight_layout()
        save_dir = os.path.join(
            config["stage_dir"]+f"/",
            f"{plot_config.get('filename', 'edgewise')}_{suffix}_rz.png",
        )
        save_dir_svg = os.path.join(
            config["stage_dir"]+f"/",
            f"{plot_config.get('filename', 'edgewise')}_{suffix}_rz.svg",
        )
        fig.savefig(save_dir)
        fig.savefig(save_dir_svg)
        print(f"Finish plotting. Find the plot at {save_dir}")
        plt.close()


def graph_scoring_efficiency_purity(lightning_module, plot_config, config):
    """
    Plot the graph construction efficiency vs. pT of the edge.
    """
    print("Plotting efficiency against pT and eta")
    print("Plotting efficiency and purity against r/z")
    true_positive, target_pt, target_eta, pred, track = [], [], [], [], []
    target_rz = {
        "z": torch.empty(0).to(lightning_module.device),
        "r": torch.empty(0).to(lightning_module.device),
    }
    all_target_rz = target_rz.copy()
    true_positive_rz = target_rz.copy()
    target_true_positive_rz = target_rz.copy()
    pred_rz = target_rz.copy()
    masked_pred_rz = target_rz.copy()
    input_graph_size, graph_size, n_graphs = (0, 0, 0)

    print(f"Using score cut: {config.get('score_cut')}")
    if "target_tracks" in config:
        print(f"Track selection criteria: \n{yaml.dump(config.get('target_tracks'))}")
    else:
        print("No track selection criteria found, accepting all tracks.")

    dataset_name = config["dataset"]
    dataset = getattr(lightning_module, dataset_name)

    for event in tqdm(dataset):
        event = event.to(lightning_module.device)
        # phi_region = float(event.hit_phi_region_id[0])
        # eta_region = float(event.hit_eta_region_id[0])

        # Need to apply score cut and remap the truth_map
        if "score_cut" in config:
            lightning_module.apply_score_cut(event, config["score_cut"])
        if "target_tracks" in config:
            lightning_module.apply_target_conditions(event, config["target_tracks"])
        else:
            event.track_target_mask = torch.ones(
                event.track_to_edge_map.shape[0], dtype=torch.bool
            ).to(lightning_module.device)

        event = event.to(lightning_module.device)

        # get all target true positives
        true_positive.append(event.track_to_edge_map[event.track_target_mask] > -1)

        # get all target pt. Length = number of target true. This includes ALL truth edges in the event,
        # even those not included in the input graph. We MUST filter them out to isolate the inefficiency from model.
        # Otherwise, we are plotting cumilative edge efficiency.
        target_pt.append(event.track_particle_pt[event.track_target_mask])
        target_eta.append(event.hit_eta[event.track_edges[0, event.track_target_mask]])

        # get all edges passing edge cut
        if "scores" in event.keys:
            pred.append(event.edge_scores >= config["score_cut"])
        else:
            pred.append(event.edge_y)
        # get a boolean array answer the question is this target edge in input graph
        track.append(event.track_map[event.track_target_mask] > -1)

        # scale r and z
        event.hit_r /= 1000
        event.hit_z /= 1000

        # flip edges that point inward if not undirected, since if undirected is True, lightning_module.apply_score_cut takes care of this
        event.edge_index = rearrange_by_distance(event, event.edge_index)
        event.track_edges = rearrange_by_distance(event, event.track_edges)

        # indices of all target edges present in the input graph
        target_edges = event.track_edges[
            :, event.track_target_mask & (event.track_map > -1)
        ]

        # indices of all target edges (may or may not be present in the input graph)
        all_target_edges = event.track_edges[:, event.track_target_mask]

        # get target z r
        for key, item in target_rz.items():
            target_rz[key] = torch.cat([item, event[key][target_edges[0]]], dim=0)
        for key, item in all_target_rz.items():
            all_target_rz[key] = torch.cat(
                [item, event[key][all_target_edges[0]]], dim=0
            )

        # indices of all true positive target edges
        target_true_positive_edges_rz = event.track_edges[
            :, event.track_target_mask & (event.track_to_edge_map > -1)
        ]

        # true positive edge indices, used as numerator of total purity
        true_positive_edges_rz = event.track_edges[:, event.track_to_edge_map > -1]

        # all positive edges, used as denominator of total and target purity
        positive_edges_rz = event.edge_index[:, event.pred]

        # masked positive edge indices, including true positive target edges and all false positive edges
        fake_positive_edges_rz = event.edge_index[:, event.pred & (event.edge_y == 0)]
        masked_positive_edges_rz = torch.cat(
            [target_true_positive_edges_rz, fake_positive_edges_rz], dim=1
        )

        for key in ["r", "z"]:
            true_positive_rz[key] = torch.cat(
                [true_positive_rz[key], event[key][target_true_positive_edges_rz[0]]],
                dim=0,
            )
            target_true_positive_rz[key] = torch.cat(
                [
                    target_true_positive_rz[key].float(),
                    event[key][target_true_positive_edges_rz[0]].float(),
                ],
                dim=0,
            )
            pred_rz[key] = torch.cat(
                [pred_rz[key].float(), event[key][positive_edges_rz[0]].float()], dim=0
            )
            masked_pred_rz[key] = torch.cat(
                [
                    masked_pred_rz[key].float(),
                    event[key][masked_positive_edges_rz[0]].float(),
                ],
                dim=0,
            )

    # concat all target pt and eta
    target_pt = torch.cat(target_pt).cpu().numpy()
    target_eta = torch.cat(target_eta).cpu().numpy()

    # get all true positive
    true_positive = torch.cat(true_positive).cpu().numpy()

    # get all positive
    track = torch.cat(track).cpu().numpy()

    # count number of graphs to calculate mean efficiency
    n_graphs = len(pred)

    # get all predictions
    pred = torch.cat(pred).cpu().numpy()

    # get mean graph size
    mean_graph_size = pred.sum() / n_graphs

    # get mean target efficiency
    target_efficiency = true_positive.sum() / len(target_pt[track])
    target_purity = true_positive[track].sum() / pred.sum()
    cumulative_efficiency = true_positive.sum() / len(target_pt)
    # get graph construction efficiency
    graph_construction_efficiency = track.mean()

    # Get the edgewise efficiency
    # Build a histogram of true pTs, and a histogram of true-positive pTs
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    pt_max = 50
    if pt_min==0:
        pt_min+= 1e-1
    print(f"pt_min: {pt_min}")
    nb_pt_bins = 10 if pt_min != 1e-1 else 17
    if "pt_units" in plot_config and plot_config["pt_units"] == "MeV":
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), nb_pt_bins)
    if "eta_lim" in plot_config:
        eta_min, eta_max = plot_config["eta_lim"]
    else:
        eta_min, eta_max = [-4, 4]
    eta_bins = np.linspace(eta_min, eta_max)

    true_pt_hist, true_pt_bins = np.histogram(target_pt[track], bins=pt_bins)
    true_pos_pt_hist, _ = np.histogram(target_pt[true_positive], bins=pt_bins)

    true_eta_hist, true_eta_bins = np.histogram(target_eta[track], bins=eta_bins)
    true_pos_eta_hist, _ = np.histogram(target_eta[true_positive], bins=eta_bins)

    pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]

    filename = plot_config.get("filename", "gnn_edgewise_efficiency")

    for true_pos_hist, true_hist, bins, xlabel, xlim, logx, filename in zip(
        [true_pos_pt_hist, true_pos_eta_hist],
        [true_pt_hist, true_eta_hist],
        [true_pt_bins, true_eta_bins],
        [f"$p_T [{pt_units}]$", r"$\eta$"],
        [[pt_min, pt_max], [eta_min, eta_max]],
        [True, False],
        [f"{filename}_pt", f"{filename}_eta"],
    ):
        # Divide the two histograms to get the edgewise efficiency
        hist, err = get_ratio(true_pos_hist, true_hist)

        fig, ax = plot_1d_histogram(
            hist,
            bins,
            err,
            xlabel,
            plot_config["title"],
            plot_config.get("ylim", [0.8, 1.06]),
            xlim,
            "Efficiency",
            logx=logx,
        )
        n_train = config["n_train"]
        # Save the plot
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: " + str(config["score_cut"]) + "\n"
            f"Input graph size: {pred.shape[0]/n_graphs:.2e}, Graph Construction"
            f" Efficiency: {graph_construction_efficiency:.4f}" + "\n"
            f"Mean graph size: {mean_graph_size:.2e}, Signal Efficiency:"
            f" {target_efficiency:.4f}"
            + "\n"
            + f"Cumulative Signal Efficiency: {cumulative_efficiency:.4f}",
        )

        fig.savefig(os.path.join(config["stage_dir"], filename + f"_{n_train}_evts.png"))
        fig.savefig(os.path.join(config["stage_dir"], filename + f"_{n_train}_evts.svg"))
        print(
            "Finish plotting. Find the plot at"
            f' {os.path.join(config["stage_dir"], filename)}'
        )

    fig, ax = plot_efficiency_rz(
        target_rz["z"],
        target_rz["r"],
        true_positive_rz["z"],
        true_positive_rz["r"],
        plot_config,
    )
    n_train = config["n_train"]
    # Save the plot
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$ "
        "\nGraph Construction Efficiency:"
        f" {graph_construction_efficiency:.4f}, Input graph size: {pred.shape[0]/n_graphs: .2e} \n"
        r"Edge score cut: "
        + str(config["score_cut"])
        + f", Mean graph size: {mean_graph_size:.2e} \n"
        "Signal Efficiency:"
        f" {true_positive_rz['z'].shape[0] / target_rz['z'].shape[0] :.4f} \n"
        f"Cumulative signal efficiency: {true_positive_rz['z'].shape[0] / all_target_rz['z'].shape[0]: .4f}",
    )
    plt.tight_layout()
    save_dir = os.path.join(
        config["stage_dir"],
        f"{plot_config.get('filename', 'gnn_edgewise_efficiency_rz')}_{n_train}_evts.png",
    )
    save_dir_svg = os.path.join(
        config["stage_dir"],
        f"{plot_config.get('filename', 'gnn_edgewise_efficiency_rz')}_{n_train}_evts.svg",
    )
    fig.savefig(save_dir)
    fig.savefig(save_dir_svg)
    print(f"Finish plotting. Find the plot at {save_dir}")
    plt.close()

    fig, ax = plot_efficiency_rz(
        all_target_rz["z"],
        all_target_rz["r"],
        true_positive_rz["z"],
        true_positive_rz["r"],
        plot_config,
    )
    
    # Save the plot
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$ "
        "\nGraph Construction Efficiency:"
        f" {graph_construction_efficiency:.4f}, Input graph size: {pred.shape[0]/n_graphs: .2e} \n"
        r"Edge score cut: "
        + str(config["score_cut"])
        + f", Mean graph size: {mean_graph_size:.2e} \n"
        "Signal Efficiency:"
        f" {true_positive_rz['z'].shape[0] / target_rz['z'].shape[0] :.4f} \n"
        f"Cumulative signal efficiency: {true_positive_rz['z'].shape[0] / all_target_rz['z'].shape[0]: .4f}",
    )
    plt.tight_layout()
    save_dir = os.path.join(
        config["stage_dir"],
        f"cumulative_{plot_config.get('filename', 'gnn_edgewise_efficiency_rz')}_{n_train}_evts.png",
    )
    save_dir_svg = os.path.join(
        config["stage_dir"],
        f"cumulative_{plot_config.get('filename', 'gnn_edgewise_efficiency_rz')}_{n_train}_evts.svg",
    )
    fig.savefig(save_dir)
    fig.savefig(save_dir_svg)
    print(f"Finish plotting. Find the plot at {save_dir}")
    plt.close()

    purity_definition_label = {
        "target_purity": "Target Purity",
        "masked_purity": "Masked Purity",
        "total_purity": "Total Purity",
    }
    for numerator, denominator, suffix in zip(
        [true_positive_rz, target_true_positive_rz, target_true_positive_rz],
        [pred_rz, pred_rz, masked_pred_rz],
        ["total_purity", "target_purity", "masked_purity"],
    ):
        fig, ax = plot_efficiency_rz(
            denominator["z"].cpu(),
            denominator["r"].cpu(),
            numerator["z"].cpu(),
            numerator["r"].cpu(),
            plot_config,
        )
        # Save the plot
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: "
            + str(config["score_cut"])
            + "\n"
            + purity_definition_label[suffix]
            + ": "
            + f"{numerator['z'].size(0) / denominator['z'].size(0) : .5f}",
        )
        plt.tight_layout()
        save_dir = os.path.join(
            config["stage_dir"],
            f"{plot_config.get('filename', 'gnn_edgewise')}_{suffix}_rz_{n_train}_evts.png",
        )
        save_dir_svg = os.path.join(
            config["stage_dir"],
            f"{plot_config.get('filename', 'gnn_edgewise')}_{suffix}_rz_{n_train}_evts.svg",
        )
        fig.savefig(save_dir)
        fig.savefig(save_dir_svg)
        print(f"Finish plotting. Find the plot at {save_dir}")
        plt.close()


def plot_uncertainty_distribution(all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False):
    """
    Plot histogram of uncertainties split by target, non-target, and false edges.
    Creates separate plots for each edge type.
    """
    
    # Create separate plots for target, non-target, and false edges
    for edge_type, truth_mask, suffix in [
        ("Target", all_flat_target_truth, "target"),
        ("Non-target", all_flat_non_target_truth, "non_target"),
        ("False", all_flat_false, "false")
    ]:
        
        # Filter data for this edge type
        edge_uncertainties = all_flat_uncertainties[truth_mask]
        
        if len(edge_uncertainties) == 0:  # Skip if no edges of this type
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create histogram for this edge type
        ax.hist(edge_uncertainties, bins=100, histtype="step", linewidth=2, density=True, 
                color='tab:blue' if edge_type == "Target" else 'tab:green' if edge_type == "Non-target" else 'tab:orange',
                label=f'{edge_type} Edges')
        
        # Switch to log-scale Y-axis for improved visibility
        ax.set_yscale('log')
        
        ax.set_xlabel('Uncertainty (Std Dev)', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Density', fontsize=14, ha="right", y=0.95)
        ax.legend(fontsize=14)
        
        n_train = config["n_train"]
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        if not UQ_propagation:
            atlasify(f"{n_train} train events",
                r"Target: $p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {dropout_value}"
                + "\n"
                + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                + "\n"
                + f"{edge_type} Edges Only",
            )
        else:
            atlasify(f"{n_train} train events",
                r"Target: $p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {dropout_value}"
                + "\n"
                + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                + "\n"
                + f"{edge_type} Edges Only",
            )

        fig.tight_layout()
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_distribution_{suffix}{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_distribution_{suffix}{dropout_str}.svg"
        )
        
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} edges uncertainty distribution plot saved to {save_path}")
    
    # Also create a combined plot showing all three edge types together for comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot all three edge types on the same plot
    for edge_type, truth_mask, color in [
        ("Target", all_flat_target_truth, 'tab:blue'),
        ("Non-target", all_flat_non_target_truth, 'tab:green'),
        ("False", all_flat_false, 'tab:orange')
    ]:
        edge_uncertainties = all_flat_uncertainties[truth_mask]
        if len(edge_uncertainties) > 0:
            ax.hist(edge_uncertainties, bins=100, histtype="step", linewidth=2, density=True, 
                    color=color, label=f'{edge_type} Edges')
    
    # Switch to log-scale Y-axis for improved visibility
    ax.set_yscale('log')
    
    ax.set_xlabel('Uncertainty (Std Dev)', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Density', fontsize=14, ha="right", y=0.95)
    ax.legend(fontsize=14)
    if not UQ_propagation:
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
            + "\n"
            + f"Dropout rate: {dropout_value}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}",
        )
    else:
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
            + "\n"
            + f"Dropout rate: {dropout_value}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}",
        )

    fig.tight_layout()
    save_path_combined = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_distribution_combined{dropout_str}.png"
    )
    save_path_combined_svg = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_distribution_combined{dropout_str}.svg"
    )
    fig.savefig(save_path_combined)
    fig.savefig(save_path_combined_svg)
    plt.close(fig)
    print(f"Combined uncertainty distribution plot saved to {save_path_combined}")


def plot_calibration_curve(all_scores, all_truth, dataset, config, plot_config, dropout_str, dropout_value, from_calibration_stage=False):
    """Plot calibration curve comparing uncertainties to prediction accuracy."""
    fig, ax = plt.subplots(figsize=(8, 6))

    score_bins = np.linspace(0, 1, 101)
    bin_indices = np.digitize(all_scores, score_bins) - 1

    accuracies = []
    bin_centers = []
    accuracy_errors = []
    for i in range(len(score_bins) - 1):
        mask = bin_indices == i
        if np.sum(mask) > 10:
            bin_truth = all_truth[mask]
            bin_scores = all_scores[mask]

            accuracy = np.mean(bin_truth == (bin_scores > 0.5))
            accuracy_error = np.sqrt(accuracy * (1 - accuracy) / np.sum(mask))
            
            bin_center = (score_bins[i] + score_bins[i + 1]) / 2
            
            accuracies.append(accuracy)
            accuracy_errors.append(accuracy_error)
            bin_centers.append(bin_center)

    ax.plot(bin_centers, np.abs(np.array(accuracies) - 0.5) * 2, '-', linewidth=2, label='Calibration Curve')
    ax.fill_between(
            bin_centers, 
            np.abs(np.array(accuracies) - 0.5) * 2 - np.array(accuracy_errors) * 2, 
            np.abs(np.array(accuracies) - 0.5) * 2 + np.array(accuracy_errors) * 2, 
            alpha=0.3,
            edgecolor=None
        )
    ax.plot([0, 0.5], [1, 0], 'k--', label='Perfect calibration')
    ax.plot([0.5, 1], [0, 1], 'k--')
    ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95,)
    ax.set_ylabel(r'|Accuracy - 0.5| $\times$ 2', fontsize=14, ha="right", y=0.95,)
    ax.legend(fontsize=14)
    
    # Adjust y-axis for better visibility
    ax.set_ylim(0, 1.1)
    n_train = config["n_train"]

    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    if not from_calibration_stage:
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
            + "\n"
            + f"Dropout rate: {dropout_value}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
            + "\n"
            +f"Edge score for minimal " +r"|Accuracy - 0.5| $\times$ 2 : " + f"{bin_centers[np.argmin(np.abs(np.array(accuracies) - 0.5) * 2)]:.2f}",
        )
    else:
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
            + "\n"
            +f"Edge score for minimal " +r"|Accuracy - 0.5| $\times$ 2 : " + f"{bin_centers[np.argmin(np.abs(np.array(accuracies) - 0.5) * 2)]:.2f}",
        )
    
    fig.tight_layout()

    if not from_calibration_stage:
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_calibration_{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_calibration_{dropout_str}.svg"
        )
    else:
        save_path = os.path.join(
            config["stage_dir"], 
            f"calibration/calibration_calibration_stage_{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"calibration/calibration_calibration_stage_{dropout_str}.svg"
        )
    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)


def plot_reliability_diagram(all_scores, all_truth, dataset, config, plot_config, dropout_str, dropout_value, from_calibration_stage=False):
    """Plot reliability diagram showing calibration of predicted scores."""
    fig, ax = plt.subplots(figsize=(8, 6))

    score_bins = np.linspace(0, 1, 101)
    bin_indices = np.digitize(all_scores, score_bins) - 1

    bin_centers = []
    reliability = []
    reliability_errors = []
    for i in range(len(score_bins) - 1):
        mask = bin_indices == i
        bin_size = np.sum(mask)
        
        if bin_size > 10:
            bin_truth = all_truth[mask]
            true_count = np.sum(bin_truth)
            bin_reliability = true_count / bin_size if bin_size > 0 else 0
            se = (
                np.sqrt(bin_reliability * (1 - bin_reliability) / bin_size)
                if bin_size > 0
                else 0
            )
            
            bin_center = (score_bins[i] + score_bins[i + 1]) / 2
            bin_centers.append(bin_center)
            reliability.append(bin_reliability)
            reliability_errors.append(se)

    if reliability:
        ax.plot(bin_centers, reliability, '-', linewidth=2, label='Reliability Diagram')
        ax.fill_between(
            bin_centers, 
            np.array(reliability) - np.array(reliability_errors), 
            np.array(reliability) + np.array(reliability_errors), 
            alpha=0.3,
            edgecolor=None
        )
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95,)
        ax.set_ylabel('Fraction of True Edges', fontsize=14, ha="right", y=0.95,)
        ax.legend(fontsize=14)
        n_train = config["n_train"]
        
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        if not from_calibration_stage:
            atlasify(f"{n_train} train events",
                r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {dropout_value}"
                + "\n"
                + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}",
            )
        else:
            atlasify(f"{n_train} train events",
                r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}",
            )

        fig.tight_layout()

        if not from_calibration_stage:
            save_path = os.path.join(
                config["stage_dir"], 
                f"{plot_config.get('filename', 'mc_dropout')}_reliability{dropout_str}.png"
            )
            save_path_svg = os.path.join(
                config["stage_dir"], 
                f"{plot_config.get('filename', 'mc_dropout')}_reliability{dropout_str}.svg"
            )
        else:
            save_path = os.path.join(
                config["stage_dir"], 
                f"calibration/reliability_calibration_stage{dropout_str}.png"
            )
            save_path_svg = os.path.join(
                config["stage_dir"], 
                f"calibration/reliability_calibration_stage{dropout_str}.svg"
            )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)


def plot_uncertainty_vs_score(all_flat_scores, all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False):
    """
    Plot the distribution of uncertainties vs. edge scores with error bands.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Separate data for target truth, non-target truth, and false edges
    target_truth_mask = all_flat_target_truth
    non_target_truth_mask = all_flat_non_target_truth
    false_mask = all_flat_false
    
    # Create score bins and calculate statistics in each bin
    score_bins = np.linspace(0, 1, 101)
    bin_indices = np.digitize(all_flat_scores, score_bins) - 1
    
    target_truth_means = []
    non_target_truth_means = []
    false_means = []
    target_truth_errors = []
    non_target_truth_errors = []
    false_errors = []
    bin_centers = []
    
    for i in range(len(score_bins) - 1):
        bin_mask = bin_indices == i
        
        target_truth_bin_mask = bin_mask & target_truth_mask
        non_target_truth_bin_mask = bin_mask & non_target_truth_mask
        false_bin_mask = bin_mask & false_mask
        
        if np.sum(target_truth_bin_mask) > 10:
            target_truth_means.append(np.mean(all_flat_uncertainties[target_truth_bin_mask]))
            target_truth_errors.append(np.std(all_flat_uncertainties[target_truth_bin_mask]) / np.sqrt(np.sum(target_truth_bin_mask)))
        else:
            target_truth_means.append(np.nan)
            target_truth_errors.append(np.nan)
            
        if np.sum(non_target_truth_bin_mask) > 10:
            non_target_truth_means.append(np.mean(all_flat_uncertainties[non_target_truth_bin_mask]))
            non_target_truth_errors.append(np.std(all_flat_uncertainties[non_target_truth_bin_mask]) / np.sqrt(np.sum(non_target_truth_bin_mask)))
        else:
            non_target_truth_means.append(np.nan)
            non_target_truth_errors.append(np.nan)
            
        if np.sum(false_bin_mask) > 10:
            false_means.append(np.mean(all_flat_uncertainties[false_bin_mask]))
            false_errors.append(np.std(all_flat_uncertainties[false_bin_mask]) / np.sqrt(np.sum(false_bin_mask))) 
        else:
            false_means.append(np.nan)
            false_errors.append(np.nan)
            
        bin_centers.append((score_bins[i] + score_bins[i+1]) / 2)
    
    # Convert to numpy arrays for easier manipulation
    target_truth_means = np.array(target_truth_means)
    non_target_truth_means = np.array(non_target_truth_means)
    false_means = np.array(false_means)
    target_truth_errors = np.array(target_truth_errors)
    non_target_truth_errors = np.array(non_target_truth_errors)
    false_errors = np.array(false_errors)
    bin_centers = np.array(bin_centers)
    
    # Plot mean uncertainty lines with error bands
    ax.plot(bin_centers, target_truth_means, '-', linewidth=2, color='tab:blue', label='Target True Edges')
    ax.fill_between(
        bin_centers, 
        target_truth_means - target_truth_errors, 
        target_truth_means + target_truth_errors, 
        alpha=0.3,
    )
    
    ax.plot(bin_centers, false_means, '-', linewidth=2, color='tab:orange', label='False Edges')
    ax.fill_between(
        bin_centers, 
        false_means - false_errors, 
        false_means + false_errors, 
        alpha=0.3,
    )
    
    ax.plot(bin_centers, non_target_truth_means, '-', linewidth=2, color='tab:green', label='Non-target True Edges')
    ax.fill_between(
        bin_centers, 
        non_target_truth_means - non_target_truth_errors, 
        non_target_truth_means + non_target_truth_errors, 
        alpha=0.3,
    )
    
    # Set labels
    ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95,)
    ax.set_ylabel('Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95,)
    
    # Set axis limits
    ax.set_xlim(0, 1)
    y_data = np.concatenate([target_truth_means + target_truth_errors, non_target_truth_means + non_target_truth_errors, false_means + false_errors])
    y_data = y_data[~np.isnan(y_data)]
    if len(y_data) > 0:
        # ylim_max = min(np.percentile(y_data, 99), 0.5)  # Cap at 0.5 or 99th percentile
        ax.set_ylim(0, 0.7)
    
    # Add vertical line at score cut
    score_cut = config["score_cut"]
    ax.axvline(x=score_cut, color='black', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
    n_train = config["n_train"]
    # Apply ATLAS styling
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    if not UQ_propagation:
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
            + "\n"
            + f"Dropout rate: {dropout_value}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}",
        )
    else:
        atlasify(f"{n_train} train events",
            r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
            + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
            + "\n"
            + f"Dropout rate: {dropout_value}"
            + "\n"
            + f"UQ propagation evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}",
        )
    fig.tight_layout()
    
    # Save the figure
    save_path = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_score.png"
    )
    save_path_svg = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_score.svg"
    )

    fig.savefig(save_path)
    fig.savefig(save_path_svg)
    plt.close(fig)
    print(f"Uncertainty vs. score plot saved to {save_path}")


def plot_uncertainty_vs_pt(all_flat_pt, all_flat_scores_track, all_flat_uncertainties_track, all_flat_target_truth_track, all_flat_non_target_truth_track, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False):
    """
    Plot uncertainties vs. pT for track edges.
    """
    
    # Define PT bins based on units
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    pt_max = 50
    if pt_min==0:
        pt_min += 1e-1
    if ("pt_units" in plot_config and plot_config["pt_units"] == "MeV") or UQ_propagation:
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
        
    pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]
    
    # Create bins for PT (log scale) and scores
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 100 if pt_min == 1000 else 167)
    score_bins = np.linspace(0, 1, 101)
    
    # Create separate plots for target and non-target edges
    for edge_type, truth_mask, suffix in [
        ("Target", all_flat_target_truth_track, "target"),
        ("Non-target", all_flat_non_target_truth_track, "non_target")
    ]:
        
        # Filter data for this edge type
        edge_pt = all_flat_pt[truth_mask]
        edge_scores = all_flat_scores_track[truth_mask]
        edge_uncertainties = all_flat_uncertainties_track[truth_mask]
        
        if len(edge_pt) == 0:  # Skip if no edges of this type
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        H, xedges, yedges = np.histogram2d(
            edge_pt,
            edge_scores,
            bins=[pt_bins, score_bins]
        )
        
        # Compute sum of uncertainties in each bin
        uncertainty_sum, _, _ = np.histogram2d(
            edge_pt,
            edge_scores,
            bins=[pt_bins, score_bins],
            weights=edge_uncertainties
        )
        
        # Calculate mean uncertainty per bin (avoid divide by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_uncertainty = np.where(H > 0, uncertainty_sum / H, np.nan)
        
        # Create heatmap
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        im = ax.pcolormesh(X, Y, mean_uncertainty, cmap='viridis', 
                            vmin=0, vmax=min(0.5, np.nanpercentile(mean_uncertainty, 99)))
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Mean Uncertainty (Std Dev)', fontsize=12)
        
        # Configure axis
        ax.set_xscale('log')
        ax.set_xlabel(f'$p_T$ [{pt_units}]', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Edge Score', fontsize=14, ha="right", y=0.95)
        
        # Add horizontal line at score cut
        score_cut = config["score_cut"]
        ax.axhline(y=score_cut, color='red', linestyle='--', alpha=0.7)
        n_train = config["n_train"]
        # Apply ATLAS styling
        pt_min_legend = config["target_tracks"]['track_particle_pt'][0]/1e3
        if not UQ_propagation:
            atlasify(f"{n_train} train events",
                r"$p_T >" + f"{pt_min_legend if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes, dropout rate: {dropout_value}"
                + "\n"
                + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}, {edge_type} edges only", outside=True
            )
        else:
            atlasify(f"{n_train} train events",
                r"$p_T >" + f"{pt_min_legend if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes, dropout rate: {dropout_value}"
                + "\n"
                + f"UQ propagation evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}, {edge_type} edges only", outside=True
            )

        fig.tight_layout()

        

        # Save the figure
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_pt_heatmap_{suffix}{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_pt_heatmap_{suffix}{dropout_str}.svg"
        )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} edges uncertainty vs. PT heatmap saved to {save_path}")
        
        # Create a second plot showing 10 lines for different score bins
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create fewer bins for clearer visualization
        pt_bins_1d = np.logspace(np.log10(pt_min), np.log10(pt_max), 20 if pt_min==1000 else 34)
        print(pt_min)
        print(pt_max)
        # Create 10 score bins
        score_bins_1d = np.linspace(0, 1, 11)
        
        # Create a colormap for the lines
        cmap = plt.cm.viridis
        
        # Plot one line for each score bin
        for i in range(10):
            score_min, score_max = score_bins_1d[i], score_bins_1d[i+1]
            score_mask = (edge_scores >= score_min) & (edge_scores < score_max)
            
            if np.sum(score_mask) < 50:  # Skip if not enough data
                continue
                
            # Filter data for this score bin
            bin_pt = edge_pt[score_mask]
            bin_uncertainties = edge_uncertainties[score_mask]
            
            # Compute statistics in each pT bin
            pt_indices = np.digitize(bin_pt, pt_bins_1d) - 1
            
            mean_uncertainties = []
            bin_centers = []
            uncertainty_errors = []
            for j in range(len(pt_bins_1d) - 1):
                bin_mask = (pt_indices == j)
                
                if np.sum(bin_mask) > 10:  # Only include bins with sufficient data
                    uncertainties_in_bin = bin_uncertainties[bin_mask]
                    mean_uncertainty = np.mean(uncertainties_in_bin)
                    error = np.std(uncertainties_in_bin) / np.sqrt(np.sum(bin_mask))
                    
                    bin_center = np.sqrt(pt_bins_1d[j] * pt_bins_1d[j+1])  # Geometric mean for log scale
                    
                    mean_uncertainties.append(mean_uncertainty)
                    uncertainty_errors.append(error)
                    bin_centers.append(bin_center)
            mean_uncertainties = np.array(mean_uncertainties)
            bin_centers = np.array(bin_centers)
            uncertainty_errors = np.array(uncertainty_errors)
            
            if len(mean_uncertainties) > 0:
                # Plot line for this score bin with color from colormap
                color = cmap(i / 10)
                ax.plot(bin_centers, mean_uncertainties, '-o', linewidth=2, 
                        color=color, label=f'Score [{score_min:.1f}-{score_max:.1f})')
                ax.fill_between(
                        bin_centers, 
                        mean_uncertainties - uncertainty_errors, 
                        mean_uncertainties + uncertainty_errors, 
                        alpha=0.3,
                        color=color,
                        edgecolor=None
                        )
        
        # Configure axis
        ax.set_xscale('log')
        ax.set_xlabel(f'$p_T$ [{pt_units}]', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
        
        # Set y-axis limits
        ax.set_ylim(0, 0.5)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=14)
        
        # Apply ATLAS styling
        if not UQ_propagation:
            atlasify(f"{n_train} train events",
                r"$p_T >" + f"{pt_min_legend if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {dropout_value}"
                + "\n"
                + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                + "\n"
                + f"{edge_type} Edges Only",
            )
        else:
            atlasify(f"{n_train} train events",
                r"$p_T >" + f"{pt_min_legend if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {dropout_value}"
                + "\n"
                + f"UQ propagation evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                + "\n"
                + f"{edge_type} Edges Only",
            )
        
        fig.tight_layout()
        
        # Save the figure
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_pt_{suffix}{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_pt_{suffix}{dropout_str}.svg"
        )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} edges uncertainty vs. PT plot saved to {save_path}")


def plot_uncertainty_vs_eta(all_flat_eta, all_flat_scores, all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False):
    """
    Plot heatmaps of uncertainties vs. eta and score for target, non-target, and false edges.
    """
    
    # Define eta bins
    if "eta_lim" in plot_config:
        eta_min, eta_max = plot_config["eta_lim"]
    else:
        eta_min, eta_max = [-4, 4]
    
    # Create bins for eta and scores
    eta_bins = np.linspace(eta_min, eta_max, 100)
    score_bins = np.linspace(0, 1, 101)
    
    # Create separate plots for target, non-target, and false edges
    for edge_type, truth_mask, suffix in [
        ("Target", all_flat_target_truth, "target"),
        ("Non-target", all_flat_non_target_truth, "non_target"),
        ("False", all_flat_false, "false")
    ]:
        
        # Filter data for this edge type
        edge_eta = all_flat_eta[truth_mask]
        edge_scores = all_flat_scores[truth_mask]
        edge_uncertainties = all_flat_uncertainties[truth_mask]
        
        if len(edge_eta) == 0:  # Skip if no edges of this type
            continue
            
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Compute 2D histogram for counts
        H, xedges, yedges = np.histogram2d(
            edge_eta,
            edge_scores,
            bins=[eta_bins, score_bins]
        )
        
        # Compute sum of uncertainties in each bin
        uncertainty_sum, _, _ = np.histogram2d(
            edge_eta,
            edge_scores,
            bins=[eta_bins, score_bins],
            weights=edge_uncertainties
        )
        
        # Calculate mean uncertainty per bin (avoid divide by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            mean_uncertainty = np.where(H > 0, uncertainty_sum / H, np.nan)
        
        # Create heatmap
        X, Y = np.meshgrid(xedges[:-1], yedges[:-1], indexing='ij')
        im = ax.pcolormesh(X, Y, mean_uncertainty, cmap='viridis', 
                          vmin=0, vmax=min(0.5, np.nanpercentile(mean_uncertainty, 99)))
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Mean Uncertainty (Std Dev)', fontsize=12)
        
        # Configure axis
        ax.set_xlabel(r'$\eta$', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Edge Score', fontsize=14, ha="right", y=0.95)
        
        # Add horizontal line at score cut
        score_cut = config["score_cut"]
        ax.axhline(y=score_cut, color='red', linestyle='--', alpha=0.7)
        n_train = config["n_train"]
        # Apply ATLAS styling
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        if not UQ_propagation:
            atlasify(f"{n_train} train events",
                r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes, dropout rate: {dropout_value}"
                + "\n"
                + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}, {edge_type} edges only", outside=True
            )
        else:
            atlasify(f"{n_train} train events",
                r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes, dropout rate: {dropout_value}"
                + "\n"
                + f"UQ propagation evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}, {edge_type} edges only", outside=True
            )
        fig.tight_layout()

        
        
        # Save the figure
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_eta_heatmap_{suffix}{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_eta_heatmap_{suffix}{dropout_str}.svg"
        )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} edges uncertainty vs. eta heatmap saved to {save_path}")

        # Create 1D plots with 10 lines for different score bins
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create fewer bins for clearer visualization
        eta_bins_1d = np.linspace(eta_min, eta_max, 20)
        
        # Create 10 score bins
        score_bins_1d = np.linspace(0, 1, 11)
        
        # Create a colormap for the lines
        cmap = plt.cm.viridis
        
        # Plot one line for each score bin
        for i in range(10):
            score_min, score_max = score_bins_1d[i], score_bins_1d[i+1]
            score_mask = (edge_scores >= score_min) & (edge_scores < score_max)
            
            if np.sum(score_mask) < 50:  # Skip if not enough data
                continue
                
            # Filter data for this score bin
            bin_eta = edge_eta[score_mask]
            bin_uncertainties = edge_uncertainties[score_mask]
            
            # Compute statistics in each eta bin
            eta_indices = np.digitize(bin_eta, eta_bins_1d) - 1
            
            mean_uncertainties = []
            bin_centers = []
            uncertainty_errors = []
            
            for j in range(len(eta_bins_1d) - 1):
                bin_mask = (eta_indices == j)
                
                if np.sum(bin_mask) > 10:  # Only include bins with sufficient data
                    uncertainties_in_bin = bin_uncertainties[bin_mask]
                    mean_uncertainty = np.mean(uncertainties_in_bin)
                    error = np.std(uncertainties_in_bin) / np.sqrt(np.sum(bin_mask))
                    bin_center = (eta_bins_1d[j] + eta_bins_1d[j+1]) / 2  # Arithmetic mean for linear scale
                    
                    mean_uncertainties.append(mean_uncertainty)
                    uncertainty_errors.append(error)
                    bin_centers.append(bin_center)
            mean_uncertainties = np.array(mean_uncertainties)
            bin_centers = np.array(bin_centers)
            uncertainty_errors = np.array(uncertainty_errors)
            
            if len(mean_uncertainties) > 0:
                # Plot line for this score bin with color from colormap
                color = cmap(i / 10)
                ax.plot(bin_centers, mean_uncertainties, '-o', linewidth=2, 
                        color=color, label=f'Score [{score_min:.1f}-{score_max:.1f})')
                ax.fill_between(
                        bin_centers, 
                        mean_uncertainties - uncertainty_errors, 
                        mean_uncertainties + uncertainty_errors, 
                        alpha=0.3,
                        color=color,
                        edgecolor=None
                        )
        
        # Configure axis
        ax.set_xlabel(r'$\eta$', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Uncertainty (Std Dev)', fontsize=14, ha="right", y=0.95)
        
        # Set y-axis limits
        ax.set_ylim(0, 0.5)
        
        # Set x-axis limits
        ax.set_xlim(eta_min, eta_max)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=14)
        
        # Apply ATLAS styling
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        if not UQ_propagation:
            atlasify(f"{n_train} train events",
                r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {dropout_value}"
                + "\n"
                + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                + "\n"
                + f"{edge_type} Edges Only",
            )
        else:
            atlasify(f"{n_train} train events",
                r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                + "\n"
                + f"Dropout rate: {dropout_value}"
                + "\n"
                + f"UQ propagation evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                + "\n"
                + f"{edge_type} Edges Only",
            )
        fig.tight_layout()
        
        # Save the figure
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_eta_{suffix}{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_uncertainty_vs_eta_{suffix}{dropout_str}.svg"
        )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} edges uncertainty vs. eta plot saved to {save_path}")


def plot_edge_scores_distribution(all_flat_scores, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_value, UQ_propagation=False):
    """
    Plot histogram of edge scores split by target, non-target, and false edges.
    Creates separate plots for each edge type and a combined plot.
    """
    dataset_name = config["dataset"]
    n_mcd_passes = config.get("nb_MCD_passes", 100)
    n_target_edges = np.sum(all_flat_target_truth)
    n_non_target_edges = np.sum(all_flat_non_target_truth)
    n_false_edges = np.sum(all_flat_false)
    # Create separate plots for target, non-target, and false edges
    for edge_type, truth_mask, suffix, color, hist_integral in [
        ("Target", all_flat_target_truth, "target", 'tab:blue', n_target_edges),
        ("Non-target", all_flat_non_target_truth, "non_target", 'tab:green', n_non_target_edges),
        ("False", all_flat_false, "false", 'tab:orange', n_false_edges)
    ]:
        # Filter data for this edge type
        edge_scores = all_flat_scores[truth_mask]
        if len(edge_scores) == 0:  # Skip if no edges of this type
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create histogram for this edge type
        ax.hist(edge_scores, bins=100, histtype='step', label=f'{edge_type} Edges - {hist_integral} edges', 
                linewidth=2, color=color)
        
        ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Count', fontsize=14, ha="right", y=0.95)
        ax.set_yscale('log')
        ax.legend(fontsize=14)
        ax.set_xlim([0, 1])
        
        score_cut = config["score_cut"]
        ax.axvline(x=score_cut, color='black', linestyle='--', alpha=0.7,
                   label=f'Score Cut ({score_cut})')
        
        n_train = config["n_train"]
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        
        atlasify(f"{n_train} train events",
            r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
            r"Edge score cut: " + str(score_cut) + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}" + "\n"
            + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
            + f"Dropout rate: {dropout_value}" + "\n"
            + f"{edge_type} Edges Only",
        )
        
        fig.tight_layout()

        # Save the plot
        if not UQ_propagation:
            save_path = os.path.join(
                config["stage_dir"], 
                f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_distribution_{suffix}.png"
            )
            save_path_svg = os.path.join(
                config["stage_dir"], 
                f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_distribution_{suffix}.svg"
            )
        else:
            save_path = os.path.join(
                config["stage_dir"], 
                f"{plot_config.get('filename', 'mc_dropout')}_UQ_propagation_edge_scores_distribution_{suffix}.png"
            )
            save_path_svg = os.path.join(
                config["stage_dir"], 
                f"{plot_config.get('filename', 'mc_dropout')}_UQ_propagation_edge_scores_distribution_{suffix}.svg"
            )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} edges score distribution plot saved to {save_path}")
    
    # Also create a combined plot showing all three edge types together for comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot all three edge types on the same plot
    for edge_type, truth_mask, color, hist_integral in [
        ("Target", all_flat_target_truth, 'tab:blue', n_target_edges),
        ("Non-target", all_flat_non_target_truth, 'tab:green', n_non_target_edges),
        ("False", all_flat_false, 'tab:orange', n_false_edges)
    ]:
        edge_scores = all_flat_scores[truth_mask]
        if len(edge_scores) > 0:
            ax.hist(edge_scores, bins=100, histtype='step', label=f'{edge_type} Edges - {hist_integral} edges', 
                    linewidth=2, color=color)
    
    ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Count', fontsize=14, ha="right", y=0.95)
    ax.set_yscale('log')
    ax.legend(fontsize=14)
    ax.set_xlim([0, 1])
    
    score_cut = config["score_cut"]
    ax.axvline(x=score_cut, color='black', linestyle='--', alpha=0.7,
               label=f'Score Cut ({score_cut})')
    
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
        r"Edge score cut: " + str(score_cut) + "\n"
        + f"Evaluated on {config.get('dataset_size', 50)} events in {dataset_name}" + "\n"
        + f"MC Dropout with {n_mcd_passes} forward passes" + "\n"
        + f"Dropout rate: {dropout_value}",
    )
    
    fig.tight_layout()

    # Save the combined plot
    save_path_combined = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_distribution_combined.png"
    )
    save_path_combined_svg = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_distribution_combined.svg"
    )
    fig.savefig(save_path_combined)
    fig.savefig(save_path_combined_svg)
    plt.close(fig)
    print(f"Combined edge scores distribution plot saved to {save_path_combined}")


def find_edge_indices(graph_edges, edges_prop):
    """
    Returns a 1D array containing the position indices of edges_prop in graph_edges.
    
    Parameters:
    - graph_edges: numpy array of shape (2, n_edges) representing all edges
    - edges_prop: numpy array of shape (2, m) representing a subset of edges
    
    Returns:
    - indices: numpy array of shape (m,) containing position indices
    """
    # Create a dictionary mapping edge pairs to their indices
    # Use frozenset to handle undirected edges (same hash regardless of order)
    edge_to_idx = {frozenset([a, b]): i for i, (a, b) in enumerate(zip(graph_edges[0], graph_edges[1]))}
    
    # Find indices using dictionary lookup in a vectorized manner
    indices = np.array([edge_to_idx.get(frozenset([a, b]), 0) for a, b in zip(edges_prop[0], edges_prop[1])])
    
    return indices


def plot_number_edges_vs_eta(all_flat_eta, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_value, UQ_propagation=False):
    """
    Plot number of edges vs. eta for target, non-target, and false edges.
    Creates separate plots for each edge type and a combined plot.
    """
    
    # Create separate plots for target, non-target, and false edges
    for edge_type, truth_mask, suffix, color in [
        ("Target", all_flat_target_truth, "target", 'tab:blue'),
        ("Non-target", all_flat_non_target_truth, "non_target", 'tab:green'),
        ("False", all_flat_false, "false", 'tab:orange')
    ]:
        
        # Filter data for this edge type
        edge_eta = all_flat_eta[truth_mask]
        
        if len(edge_eta) == 0:  # Skip if no edges of this type
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create eta bins
        eta_bins = np.linspace(-4, 4, 50)

        # Create histogram for this edge type
        ax.hist(edge_eta, bins=eta_bins, label=f'{edge_type} Edges', 
                linewidth=2, histtype="step", color=color)
        
        ax.set_xlim(-4, 4)
        ax.set_xlabel(r'$\eta$', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Number of Edges', fontsize=14, ha="right", y=0.95)
        ax.legend(fontsize=14)
        
        n_train = config["n_train"]
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        
        atlasify(f"{n_train} train events",
            r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
            + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes, dropout rate: {dropout_value}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
            + "\n"
            + f"{edge_type} Edges Only",
        )

        fig.tight_layout()
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_number_edges_vs_eta_{suffix}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_number_edges_vs_eta_{suffix}.svg"
        )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} edges number vs. eta plot saved to {save_path}")
    
    # Also create a combined plot showing all three edge types together for comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create eta bins
    eta_bins = np.linspace(-4, 4, 50)
    
    # Plot all three edge types on the same plot
    for edge_type, truth_mask, color in [
        ("Target", all_flat_target_truth, 'tab:blue'),
        ("Non-target", all_flat_non_target_truth, 'tab:green'),
        ("False", all_flat_false, 'tab:orange')
    ]:
        edge_eta = all_flat_eta[truth_mask]
        if len(edge_eta) > 0:
            ax.hist(edge_eta, bins=eta_bins, label=f'{edge_type} Edges', 
                    linewidth=2, histtype="step", color=color)
    
    ax.set_xlim(-4, 4)
    ax.set_xlabel(r'$\eta$', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Number of Edges', fontsize=14, ha="right", y=0.95)
    ax.legend(fontsize=14)
    
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes, dropout rate: {dropout_value}"
        + "\n"
        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}",
    )

    fig.tight_layout()

    # Save the combined plot
    save_path_combined = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_number_edges_vs_eta_combined.png"
    )
    save_path_combined_svg = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_number_edges_vs_eta_combined.svg"
    )
    fig.savefig(save_path_combined)
    fig.savefig(save_path_combined_svg)
    plt.close(fig)
    print(f"Combined number of edges vs. eta plot saved to {save_path_combined}")


def plot_edges_score_vs_eta(all_flat_eta, all_flat_scores, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False):
    """
    Plot mean edge scores vs. eta with error bands for target, non-target, and false edges.
    Creates separate plots for each edge type and a combined plot.
    """
    
    # Define eta bins
    if "eta_lim" in plot_config:
        eta_min, eta_max = plot_config["eta_lim"]
    else:
        eta_min, eta_max = [-4, 4]
    
    eta_bins = np.linspace(eta_min, eta_max, 20)
    
    # Create separate plots for target, non-target, and false edges
    for edge_type, truth_mask, suffix, color in [
        ("Target", all_flat_target_truth, "target", 'tab:blue'),
        ("Non-target", all_flat_non_target_truth, "non_target", 'tab:green'),
        ("False", all_flat_false, "false", 'tab:orange')
    ]:
        
        # Filter data for this edge type
        edge_eta = all_flat_eta[truth_mask]
        edge_scores = all_flat_scores[truth_mask]
        
        if len(edge_eta) == 0:  # Skip if no edges of this type
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Compute statistics in each eta bin
        eta_indices = np.digitize(edge_eta, eta_bins) - 1
        
        mean_scores = []
        bin_centers = []
        score_errors = []
        
        for i in range(len(eta_bins) - 1):
            bin_mask = (eta_indices == i)
            
            if np.sum(bin_mask) > 10:  # Only include bins with sufficient data
                scores_in_bin = edge_scores[bin_mask]
                mean_score = np.mean(scores_in_bin)
                error = np.std(scores_in_bin) / np.sqrt(np.sum(bin_mask))
                
                bin_center = (eta_bins[i] + eta_bins[i+1]) / 2
                
                mean_scores.append(mean_score)
                score_errors.append(error)
                bin_centers.append(bin_center)
        
        # Plot line for this edge type
        if mean_scores:
            ax.plot(bin_centers, mean_scores, '-o', linewidth=2, color=color, label=f'{edge_type} Edges')
            ax.fill_between(
                bin_centers, 
                np.array(mean_scores) - np.array(score_errors), 
                np.array(mean_scores) + np.array(score_errors), 
                alpha=0.3,
                color=color,
                edgecolor=None
            )
        
        # Configure axis
        ax.set_xlabel(r'$\eta$', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Edge Score', fontsize=14, ha="right", y=0.95)
        
        # Set y-axis limits to the range of scores
        ax.set_ylim(0, 1)
        
        # Set x-axis limits
        ax.set_xlim(eta_min, eta_max)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=14)
        
        # Add score cut line if available
        score_cut = config["score_cut"]
        ax.axhline(y=score_cut, color='black', linestyle='--', alpha=0.7, 
                   label=f'Score Cut ({score_cut})')
        
        n_train = config["n_train"]
        pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
        
        # Apply ATLAS styling
        atlasify(f"{n_train} train events",
            r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
            + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
            + "\n"
            + f"Dropout rate: {dropout_value}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
            + "\n"
            + f"{edge_type} Edges Only",
        )
        
        fig.tight_layout()

        # Save the figure
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_vs_eta_{suffix}{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_vs_eta_{suffix}{dropout_str}.svg"
        )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} edges score vs. eta plot saved to {save_path}")
    
    # Also create a combined plot showing all three edge types together for comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot all three edge types on the same plot
    for edge_type, truth_mask, color in [
        ("Target", all_flat_target_truth, 'tab:blue'),
        ("Non-target", all_flat_non_target_truth, 'tab:green'),
        ("False", all_flat_false, 'tab:orange')
    ]:
        edge_eta = all_flat_eta[truth_mask]
        edge_scores = all_flat_scores[truth_mask]
        
        if len(edge_eta) > 0:
            # Compute statistics in each eta bin
            eta_indices = np.digitize(edge_eta, eta_bins) - 1
            
            mean_scores = []
            bin_centers = []
            score_errors = []
            
            for i in range(len(eta_bins) - 1):
                bin_mask = (eta_indices == i)
                
                if np.sum(bin_mask) > 10:  # Only include bins with sufficient data
                    scores_in_bin = edge_scores[bin_mask]
                    mean_score = np.mean(scores_in_bin)
                    error = np.std(scores_in_bin) / np.sqrt(np.sum(bin_mask))
                    
                    bin_center = (eta_bins[i] + eta_bins[i+1]) / 2
                    
                    mean_scores.append(mean_score)
                    score_errors.append(error)
                    bin_centers.append(bin_center)
            
            # Plot line for this edge type
            if mean_scores:
                ax.plot(bin_centers, mean_scores, '-o', linewidth=2, color=color, label=f'{edge_type} Edges')
                ax.fill_between(
                    bin_centers, 
                    np.array(mean_scores) - np.array(score_errors), 
                    np.array(mean_scores) + np.array(score_errors), 
                    alpha=0.3,
                    color=color,
                    edgecolor=None
                )
    
    # Configure axis
    ax.set_xlabel(r'$\eta$', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Edge Score', fontsize=14, ha="right", y=0.95)
    
    # Set y-axis limits to the range of scores
    ax.set_ylim(0, 1)
    
    # Set x-axis limits
    ax.set_xlim(eta_min, eta_max)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
    
    # Add score cut line if available
    score_cut = config["score_cut"]
    ax.axhline(y=score_cut, color='black', linestyle='--', alpha=0.7, 
               label=f'Score Cut ({score_cut})')
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
        + "\n"
        + f"Dropout rate: {dropout_value}"
        + "\n"
        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}",
    )
    
    fig.tight_layout()

    # Save the combined plot
    save_path_combined = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_vs_eta_combined{dropout_str}.png"
    )
    save_path_combined_svg = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_vs_eta_combined{dropout_str}.svg"
    )
    fig.savefig(save_path_combined)
    fig.savefig(save_path_combined_svg)
    plt.close(fig)
    print(f"Combined edge scores vs. eta plot saved to {save_path_combined}")


def plot_edges_score_vs_pt(all_flat_pt, all_flat_scores_track, all_flat_target_truth_track, all_flat_non_target_truth_track, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False):
    """
    Plot mean edge scores vs. pT with error bands for target and non-target track edges.
    Creates separate plots for each edge type and a combined plot.
    """
    
    # Define PT bins based on units
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    pt_max = 50
    if pt_min==0:
        pt_min += 1e-1
    if ("pt_units" in plot_config and plot_config["pt_units"] == "MeV") or UQ_propagation:
        pt_min, pt_max = pt_min * 1000, pt_max * 1000
        
    pt_units = "GeV" if "pt_units" not in plot_config else plot_config["pt_units"]
    
    # Create bins for PT (log scale)
    pt_bins = np.logspace(np.log10(pt_min), np.log10(pt_max), 20)
    
    # Create separate plots for target and non-target track edges
    for edge_type, truth_mask, suffix, color in [
        ("Target", all_flat_target_truth_track, "target", 'tab:blue'),
        ("Non-target", all_flat_non_target_truth_track, "non_target", 'tab:green')
    ]:
        
        # Filter data for this edge type
        edge_pt = all_flat_pt[truth_mask]
        edge_scores = all_flat_scores_track[truth_mask]
        
        if len(edge_pt) == 0:  # Skip if no edges of this type
            continue
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        pt_indices = np.digitize(edge_pt, pt_bins) - 1
        
        mean_scores = []
        bin_centers = []
        score_errors = []
        
        for i in range(len(pt_bins) - 1):
            bin_mask = (pt_indices == i)
            
            if np.sum(bin_mask) > 10:  # Only include bins with sufficient data
                scores_in_bin = edge_scores[bin_mask]
                mean_score = np.mean(scores_in_bin)
                error = np.std(scores_in_bin) / np.sqrt(np.sum(bin_mask))
                
                # Geometric mean for log scale
                bin_center = np.sqrt(pt_bins[i] * pt_bins[i+1])
                
                mean_scores.append(mean_score)
                score_errors.append(error)
                bin_centers.append(bin_center)
        
        # Plot line for this edge type
        if mean_scores:
            ax.plot(bin_centers, mean_scores, '-o', linewidth=2, color=color, label=f'{edge_type} Track Edges')
            ax.fill_between(
                bin_centers, 
                np.array(mean_scores) - np.array(score_errors), 
                np.array(mean_scores) + np.array(score_errors), 
                alpha=0.3,
                color=color,
                edgecolor=None
            )
        
        # Configure axis
        ax.set_xscale('log')
        ax.set_xlabel(f'$p_T$ [{pt_units}]', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel('Edge Score', fontsize=14, ha="right", y=0.95)
        
        # Set y-axis limits to the range of scores
        ax.set_ylim(0, 1)
        
        # Set x-axis limits
        ax.set_xlim(pt_min, pt_max)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=14)
        
        # Add score cut line if available
        score_cut = config["score_cut"]
        ax.axhline(y=score_cut, color='black', linestyle='--', alpha=0.7, 
                   label=f'Score Cut ({score_cut})')
        
        n_train = config["n_train"]
        pt_min_legend = config["target_tracks"]['track_particle_pt'][0]/1e3
        
        # Apply ATLAS styling
        atlasify(f"{n_train} train events",
            r"$p_T >" + f"{pt_min_legend if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
            + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
            + "\n"
            + f"Dropout rate: {dropout_value}"
            + "\n"
            + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
            + "\n"
            + f"{edge_type} Track Edges Only",
        )
        
        fig.tight_layout()
        # Save the figure
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_vs_pt_{suffix}{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_vs_pt_{suffix}{dropout_str}.svg"
        )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} track edges score vs. PT plot saved to {save_path}")
    
    # Also create a combined plot showing both edge types together for comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot both edge types on the same plot
    for edge_type, truth_mask, color in [
        ("Target", all_flat_target_truth_track, 'tab:blue'),
        ("Non-target", all_flat_non_target_truth_track, 'tab:green')
    ]:
        edge_pt = all_flat_pt[truth_mask]
        edge_scores = all_flat_scores_track[truth_mask]
        
        if len(edge_pt) > 0:
            pt_indices = np.digitize(edge_pt, pt_bins) - 1
            
            mean_scores = []
            bin_centers = []
            score_errors = []
            
            for i in range(len(pt_bins) - 1):
                bin_mask = (pt_indices == i)
                
                if np.sum(bin_mask) > 10:  # Only include bins with sufficient data
                    scores_in_bin = edge_scores[bin_mask]
                    mean_score = np.mean(scores_in_bin)
                    error = np.std(scores_in_bin) / np.sqrt(np.sum(bin_mask))
                    
                    # Geometric mean for log scale
                    bin_center = np.sqrt(pt_bins[i] * pt_bins[i+1])
                    
                    mean_scores.append(mean_score)
                    score_errors.append(error)
                    bin_centers.append(bin_center)
            
            # Plot line for this edge type
            if mean_scores:
                ax.plot(bin_centers, mean_scores, '-o', linewidth=2, color=color, label=f'{edge_type} Track Edges')
                ax.fill_between(
                    bin_centers, 
                    np.array(mean_scores) - np.array(score_errors), 
                    np.array(mean_scores) + np.array(score_errors), 
                    alpha=0.3,
                    color=color,
                    edgecolor=None
                )
    
    # Configure axis
    ax.set_xscale('log')
    ax.set_xlabel(f'$p_T$ [{pt_units}]', fontsize=14, ha="right", x=0.95)
    ax.set_ylabel('Edge Score', fontsize=14, ha="right", y=0.95)
    
    # Set y-axis limits to the range of scores
    ax.set_ylim(0, 1)
    
    # Set x-axis limits
    ax.set_xlim(pt_min, pt_max)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=14)
    
    # Add score cut line if available
    score_cut = config["score_cut"]
    ax.axhline(y=score_cut, color='black', linestyle='--', alpha=0.7, 
               label=f'Score Cut ({score_cut})')
    
    # Apply ATLAS styling
    atlasify(f"{n_train} train events",
        r"Target: $p_T >" + f"{pt_min_legend}"+"$ GeV, $ | \eta | < 4$" + "\n"
        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
        + "\n"
        + f"Dropout rate: {dropout_value}"
        + "\n"
        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}",
    )
    
    fig.tight_layout()
    # Save the combined plot
    save_path_combined = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_vs_pt_combined{dropout_str}.png"
    )
    save_path_combined_svg = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_scores_vs_pt_combined{dropout_str}.svg"
    )
    fig.savefig(save_path_combined)
    fig.savefig(save_path_combined_svg)
    plt.close(fig)
    print(f"Combined track edges score vs. PT plot saved to {save_path_combined}")


def plot_edge_skewness_kurtosis(all_flat_scores, all_flat_skewness, all_flat_kurtosis, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False):
    """
    Plot skewness and kurtosis vs. edge scores for target, non-target, and false edges as separate plots.
    Creates separate plots for each edge type and combined plots.
    Uses precomputed skewness and kurtosis values.
    """
    
    # Create score bins
    score_bins = np.linspace(0, 1, 101)  # 100 bins from 0 to 1
    
    n_train = config["n_train"]
    
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    
    # Create separate plots for target, non-target, and false edges
    for edge_type, truth_mask, suffix, color in [
        ("Target", all_flat_target_truth, "target", 'tab:blue'),
        ("Non-target", all_flat_non_target_truth, "non_target", 'tab:green'),
        ("False", all_flat_false, "false", 'tab:orange')
    ]:
        
        # Filter data for this edge type
        edge_scores = all_flat_scores[truth_mask]
        edge_skewness = all_flat_skewness[truth_mask]
        edge_kurtosis = all_flat_kurtosis[truth_mask]
        
        if len(edge_scores) == 0:  # Skip if no edges of this type
            continue
        
        # Process bins for this edge type
        mean_skewness = []
        mean_kurtosis = []
        skewness_errors = []
        kurtosis_errors = []
        bin_centers = []
        
        for i in range(len(score_bins) - 1):
            bin_min, bin_max = score_bins[i], score_bins[i+1]
            bin_center = (bin_min + bin_max) / 2
            
            # Get indices of edges with scores in this bin
            score_mask = (edge_scores >= bin_min) & (edge_scores < bin_max)
            
            if np.sum(score_mask) > 10:  # Only include bins with sufficient data
                bin_skewness = edge_skewness[score_mask]
                bin_kurtosis = edge_kurtosis[score_mask]
                
                mean_skewness.append(np.mean(bin_skewness))
                skewness_errors.append(np.std(bin_skewness) / np.sqrt(np.sum(score_mask)))
                
                mean_kurtosis.append(np.mean(bin_kurtosis))
                kurtosis_errors.append(np.std(bin_kurtosis) / np.sqrt(np.sum(score_mask)))
                
                bin_centers.append(bin_center)
        
        # SKEWNESS PLOT for this edge type
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        
        if mean_skewness:
            ax1.plot(bin_centers, mean_skewness, '-', linewidth=2, color=color, label=f'{edge_type} Edges')
            ax1.fill_between(
                bin_centers, 
                np.array(mean_skewness) - np.array(skewness_errors), 
                np.array(mean_skewness) + np.array(skewness_errors), 
                alpha=0.3,
                color=color,
                edgecolor=None
            )
            ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax1.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
        ax1.set_ylabel('Skewness', fontsize=14, ha="right", y=0.95)
        ax1.legend(fontsize=14)
        ax1.set_xlim(0, 1)
        
        # Apply ATLAS styling
        if not UQ_propagation:
            atlasify(f"{n_train} train events", 
                    r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        + "\n" + f"{edge_type} Edges Only")
        else:
            atlasify(f"{n_train} train events", 
                    r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        + "\n" + f"{edge_type} Edges Only (UQ Propagation)")
            
        fig1.tight_layout()
        
        # Save the skewness figure
        save_path_skewness = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_edge_skewness_{suffix}{dropout_str}.png"
        )
        save_path_skewness_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_edge_skewness_{suffix}{dropout_str}.svg"
        )
        fig1.savefig(save_path_skewness)
        fig1.savefig(save_path_skewness_svg)
        plt.close(fig1)
        print(f"{edge_type} edges skewness vs. score plot saved to {save_path_skewness}")
        
        # KURTOSIS PLOT for this edge type
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        
        if mean_kurtosis:
            ax2.plot(bin_centers, mean_kurtosis, '-', linewidth=2, color=color, label=f'{edge_type} Edges')
            ax2.fill_between(
                bin_centers, 
                np.array(mean_kurtosis) - np.array(kurtosis_errors), 
                np.array(mean_kurtosis) + np.array(kurtosis_errors), 
                alpha=0.3,
                color=color,
                edgecolor=None
            )
            ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        ax2.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
        ax2.set_ylabel('Kurtosis', fontsize=14, ha="right", y=0.95)
        ax2.legend(fontsize=14)
        ax2.set_xlim(0, 1)
        
        # Apply ATLAS styling
        if not UQ_propagation:
            atlasify(f"{n_train} train events", 
                    r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        + "\n" + f"{edge_type} Edges Only")
        else:
            atlasify(f"{n_train} train events", 
                    r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        + "\n" + f"{edge_type} Edges Only (UQ Propagation)")
        
        fig2.tight_layout()
        
        # Save the kurtosis figure
        save_path_kurtosis = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_edge_kurtosis_{suffix}{dropout_str}.png"
        )
        save_path_kurtosis_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_edge_kurtosis_{suffix}{dropout_str}.svg"
        )
        fig2.savefig(save_path_kurtosis)
        fig2.savefig(save_path_kurtosis_svg)
        plt.close(fig2)
        print(f"{edge_type} edges kurtosis vs. score plot saved to {save_path_kurtosis}")
    
    # Create combined plots showing all three edge types together
    
    # Process bins for all edge types for combined plots
    combined_data = {}
    for edge_type, truth_mask, color in [
        ("Target", all_flat_target_truth, 'tab:blue'),
        ("Non-target", all_flat_non_target_truth, 'tab:green'),
        ("False", all_flat_false, 'tab:orange')
    ]:
        edge_scores = all_flat_scores[truth_mask]
        edge_skewness = all_flat_skewness[truth_mask]
        edge_kurtosis = all_flat_kurtosis[truth_mask]
        
        if len(edge_scores) == 0:
            continue
            
        mean_skewness = []
        mean_kurtosis = []
        skewness_errors = []
        kurtosis_errors = []
        bin_centers = []
        
        for i in range(len(score_bins) - 1):
            bin_min, bin_max = score_bins[i], score_bins[i+1]
            bin_center = (bin_min + bin_max) / 2
            
            score_mask = (edge_scores >= bin_min) & (edge_scores < bin_max)
            
            if np.sum(score_mask) > 10:
                bin_skewness = edge_skewness[score_mask]
                bin_kurtosis = edge_kurtosis[score_mask]
                
                mean_skewness.append(np.mean(bin_skewness))
                skewness_errors.append(np.std(bin_skewness) / np.sqrt(np.sum(score_mask)))
                
                mean_kurtosis.append(np.mean(bin_kurtosis))
                kurtosis_errors.append(np.std(bin_kurtosis) / np.sqrt(np.sum(score_mask)))
                
                bin_centers.append(bin_center)
        
        combined_data[edge_type] = {
            'color': color,
            'skewness': mean_skewness,
            'skewness_errors': skewness_errors,
            'kurtosis': mean_kurtosis,
            'kurtosis_errors': kurtosis_errors,
            'bin_centers': bin_centers
        }
    
    # COMBINED SKEWNESS PLOT
    fig_combined_skew, ax_combined_skew = plt.subplots(figsize=(8, 6))
    
    for edge_type, data in combined_data.items():
        if data['skewness']:
            ax_combined_skew.plot(data['bin_centers'], data['skewness'], '-', 
                                  linewidth=2, color=data['color'], label=f'{edge_type} Edges')
            ax_combined_skew.fill_between(
                data['bin_centers'], 
                np.array(data['skewness']) - np.array(data['skewness_errors']), 
                np.array(data['skewness']) + np.array(data['skewness_errors']), 
                alpha=0.3,
                color=data['color'],
                edgecolor=None
            )
    
    ax_combined_skew.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_combined_skew.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
    ax_combined_skew.set_ylabel('Skewness', fontsize=14, ha="right", y=0.95)
    ax_combined_skew.legend(fontsize=14)
    ax_combined_skew.set_xlim(0, 1)
    if not UQ_propagation:
        atlasify(f"{n_train} train events", 
                r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        )
    else:
        atlasify(f"{n_train} train events", 
                r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        + "\n"
                        + "UQ Propagation")
        
    fig_combined_skew.tight_layout()
    
    # Save the combined skewness figure
    save_path_combined_skewness = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_skewness_combined{dropout_str}.png"
    )
    save_path_combined_skewness_svg = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_skewness_combined{dropout_str}.svg"
    )
    fig_combined_skew.savefig(save_path_combined_skewness)
    fig_combined_skew.savefig(save_path_combined_skewness_svg)
    plt.close(fig_combined_skew)
    print(f"Combined edge skewness vs. score plot saved to {save_path_combined_skewness}")
    
    # COMBINED KURTOSIS PLOT
    fig_combined_kurt, ax_combined_kurt = plt.subplots(figsize=(8, 6))
    
    for edge_type, data in combined_data.items():
        if data['kurtosis']:
            ax_combined_kurt.plot(data['bin_centers'], data['kurtosis'], '-', 
                                  linewidth=2, color=data['color'], label=f'{edge_type} Edges')
            ax_combined_kurt.fill_between(
                data['bin_centers'], 
                np.array(data['kurtosis']) - np.array(data['kurtosis_errors']), 
                np.array(data['kurtosis']) + np.array(data['kurtosis_errors']), 
                alpha=0.3,
                color=data['color'],
                edgecolor=None
            )
    
    ax_combined_kurt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax_combined_kurt.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
    ax_combined_kurt.set_ylabel('Kurtosis', fontsize=14, ha="right", y=0.95)
    ax_combined_kurt.legend(fontsize=14)
    ax_combined_kurt.set_xlim(0, 1)
    if not UQ_propagation:
        atlasify(f"{n_train} train events", 
                r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        )
    else:
        atlasify(f"{n_train} train events", 
                r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        + "\n"
                        + "UQ Propagation")
    fig_combined_kurt.tight_layout()
    
    # Save the combined kurtosis figure
    save_path_combined_kurtosis = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_kurtosis_combined{dropout_str}.png"
    )
    save_path_combined_kurtosis_svg = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_edge_kurtosis_combined{dropout_str}.svg"
    )
    fig_combined_kurt.savefig(save_path_combined_kurtosis)
    fig_combined_kurt.savefig(save_path_combined_kurtosis_svg)
    plt.close(fig_combined_kurt)
    print(f"Combined edge kurtosis vs. score plot saved to {save_path_combined_kurtosis}")


def generate_matching_gaussians(A):
    """
    Generate random Gaussian samples with the same mean and std as A along the first axis.
    Works with variable-length arrays where different events have different numbers of edges.
    
    Parameters:
        A: List where A[t][n] is an array/list of scores for event n in pass t
        
    Returns:
        B: List with the same structure as A, containing Gaussian samples
    """
    
    T = len(A)  # Number of MC dropout passes
    
    # Initialize output with same structure as A
    B = [[] for _ in range(T)]
    
    # Process each event
    for n in range(len(A[0])):
        # Create arrays to store means and stds for each edge position
        edge_counts = [len(A[t][n]) if n < len(A[t]) else 0 for t in range(T)]
        max_edges = max(edge_counts)
            
        # Pre-allocate arrays for edge statistics
        edge_values = [[] for _ in range(max_edges)]
        
        # Collect values for each edge position across all passes
        for t in range(T):
            for e, val in enumerate(A[t][n]):
                edge_values[e].append(val)
        
        # Calculate mean and std for each edge position
        means = np.zeros(max_edges)
        stds = np.zeros(max_edges)
        
        for e in range(max_edges):
            means[e] = np.mean(edge_values[e])
            stds[e] = np.std(edge_values[e]) if len(edge_values[e]) > 1 else 0
        
        # Generate Gaussian samples for each pass
        for t in range(T):
            ne = len(A[t][n])  # Number of edges for this pass and event
            # Vectorized generation of samples
            random_samples = np.random.normal(0, 1, size=ne)
            gaussian_scores = means[:ne] + stds[:ne] * random_samples
            gaussian_scores = np.clip(gaussian_scores, 0, 1)
            B[t].append(gaussian_scores)
    
    return B
  

def compare_entropy(A, B):
    """
    Compare the entropy of two data structures A and B along the first axis.
    Works with variable-length arrays where different events have different numbers of edges.
    
    Parameters:
        A, B: Lists where A[t][n]/B[t][n] is an array/list of scores for event n in pass t
        
    Returns:
        diff_list: List of arrays, one per event, containing entropy differences for each edge
    """
    
    T = len(A)  # Number of MC dropout passes
    
    N = min(len(A[0]), len(B[0]))  # Number of events
    diff_list = []
    total_entropy_A = []
    total_entropy_B = []
    
    for n in range(N):
        # Find the minimum number of edges across all passes for both A and B
        A_edge_counts = [len(A[t][n]) if n < len(A[t]) else 0 for t in range(T)]
        B_edge_counts = [len(B[t][n]) if n < len(B[t]) else 0 for t in range(T)]
            
        min_edge_count = min(max(A_edge_counts), max(B_edge_counts))
    
        # Pre-allocate arrays for entropy calculation
        entropy_A = np.zeros(min_edge_count)
        entropy_B = np.zeros(min_edge_count)
        
        # Calculate entropy for each edge position
        for e in range(min_edge_count):
            # Collect values for this edge across all passes
            for t in range(T):
                val = A[t][n][e]
                entropy_A[e] -= val * np.log(val + 1e-10)
                
                val = B[t][n][e]
                entropy_B[e] -= val * np.log(val + 1e-10)
        
        # Store the difference
        diff_list.append(entropy_A - entropy_B)
        total_entropy_A.append(entropy_A)
        total_entropy_B.append(entropy_B)
    
    return diff_list, total_entropy_A, total_entropy_B


def plot_entropy_difference(all_flat_scores, all_flat_entropy_diff, all_flat_entropy_scores, all_flat_entropy_gaussians, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False):
    """
    Plot the entropy of original scores and Gaussian samples, and their difference.
    Creates separate plots for each edge type and combined plots.
    """
    import matplotlib.gridspec as gridspec
    
    # Create score bins
    score_bins = np.linspace(0, 1, 101)  # 100 bins from 0 to 1
    
    n_train = config["n_train"]
    
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    
    # Create separate plots for target, non-target, and false edges
    for edge_type, truth_mask, suffix, color in [
        ("Target", all_flat_target_truth, "target", 'tab:blue'),
        ("Non-target", all_flat_non_target_truth, "non_target", 'tab:green'),
        ("False", all_flat_false, "false", 'tab:orange')
    ]:
        
        # Filter data for this edge type
        edge_scores = all_flat_scores[truth_mask]
        edge_entropy_diff = all_flat_entropy_diff[truth_mask]
        edge_entropy_scores = all_flat_entropy_scores[truth_mask]
        edge_entropy_gaussians = all_flat_entropy_gaussians[truth_mask]
        
        if len(edge_scores) == 0:  # Skip if no edges of this type
            continue
        
        # Process bins for this edge type
        mcd_entropy = []
        gaussian_entropy = []
        entropy_diff = []
        mcd_errors = []
        gaussian_errors = []
        diff_errors = []
        bin_centers = []
        
        for i in range(len(score_bins) - 1):
            bin_min, bin_max = score_bins[i], score_bins[i+1]
            bin_center = (bin_min + bin_max) / 2
            
            # Get indices of edges with scores in this bin
            score_mask = (edge_scores >= bin_min) & (edge_scores < bin_max)
            
            if np.sum(score_mask) > 10:  # Only include bins with sufficient data
                # MCD entropy
                mcd_entropy_in_bin = edge_entropy_scores[score_mask]
                mcd_entropy.append(np.mean(mcd_entropy_in_bin))
                mcd_errors.append(np.std(mcd_entropy_in_bin) / np.sqrt(np.sum(score_mask)))
                
                # Gaussian entropy
                gaussian_entropy_in_bin = edge_entropy_gaussians[score_mask]
                gaussian_entropy.append(np.mean(gaussian_entropy_in_bin))
                gaussian_errors.append(np.std(gaussian_entropy_in_bin) / np.sqrt(np.sum(score_mask)))
                
                # Entropy difference
                entropy_diff_in_bin = edge_entropy_diff[score_mask]
                entropy_diff.append(np.mean(entropy_diff_in_bin))
                diff_errors.append(np.std(entropy_diff_in_bin) / np.sqrt(np.sum(score_mask)))
                
                bin_centers.append(bin_center)
        
        # Create a figure with two subplots stacked vertically
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3:1 ratio between plots
        
        # Top plot for entropy values
        ax1 = plt.subplot(gs[0])
        
        # Plot MCD entropy
        if mcd_entropy:
            ax1.plot(bin_centers, mcd_entropy, '-', linewidth=2, color=color, label=f'MCD {edge_type} Edges')
            ax1.fill_between(
                bin_centers, 
                np.array(mcd_entropy) - np.array(mcd_errors), 
                np.array(mcd_entropy) + np.array(mcd_errors), 
                alpha=0.3, color=color, edgecolor=None
            )
        
        # Plot Gaussian entropy with dashed line and lighter color
        gaussian_color = color.replace('tab:', 'tab:') if 'tab:' in color else color
        if edge_type == "Target":
            gaussian_color = 'tab:cyan'
        elif edge_type == "Non-target":
            gaussian_color = 'lightgreen'
        else:  # False
            gaussian_color = 'moccasin'
            
        if gaussian_entropy:
            ax1.plot(bin_centers, gaussian_entropy, '--', linewidth=2, color=gaussian_color, label=f'Gaussian {edge_type} Edges')
            ax1.fill_between(
                bin_centers, 
                np.array(gaussian_entropy) - np.array(gaussian_errors), 
                np.array(gaussian_entropy) + np.array(gaussian_errors), 
                alpha=0.3, color=gaussian_color, edgecolor=None
            )
        
        ax1.set_ylabel('Entropy', fontsize=14, ha="right", y=0.95)
        ax1.legend(fontsize=14, loc='upper right')
        ax1.set_xlim(0, 1)
        
        # Bottom plot for entropy difference
        ax2 = plt.subplot(gs[1], sharex=ax1)
        
        # Plot entropy difference
        if entropy_diff:
            ax2.plot(bin_centers, entropy_diff, '-', linewidth=2, color=color, label=f'{edge_type} Edges')
            ax2.fill_between(
                bin_centers, 
                np.array(entropy_diff) - np.array(diff_errors), 
                np.array(entropy_diff) + np.array(diff_errors), 
                alpha=0.3, color=color, edgecolor=None
            )
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
        ax2.set_ylabel('Entropy Difference\n(MCD - Gaussian)', fontsize=14, ha="right", y=0.95)
        
        for ax in [ax1, ax2]:
            ax.label_outer()
        
        # Apply ATLAS styling to the whole figure
        if not UQ_propagation:
            atlasify(f"{n_train} train events", 
                    r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}" + "\n" + f"{edge_type} Edges Only", axes=ax1)
        else:
            atlasify(f"{n_train} train events", 
                    r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}" + "\n" + f"{edge_type} Edges Only (UQ Propagation)", axes=ax1)
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)  # Reduce space between subplots

        # Save the entropy comparison figure
        save_path = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_entropy_comparison_{suffix}{dropout_str}.png"
        )
        save_path_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_entropy_comparison_{suffix}{dropout_str}.svg"
        )
        fig.savefig(save_path)
        fig.savefig(save_path_svg)
        plt.close(fig)
        print(f"{edge_type} edges entropy comparison plot saved to {save_path}")
    
    # Create combined plots showing all three edge types together
    combined_data = {}
    for edge_type, truth_mask, color in [
        ("Target", all_flat_target_truth, 'tab:blue'),
        ("Non-target", all_flat_non_target_truth, 'tab:green'),
        ("False", all_flat_false, 'tab:orange')
    ]:
        edge_scores = all_flat_scores[truth_mask]
        edge_entropy_diff = all_flat_entropy_diff[truth_mask]
        edge_entropy_scores = all_flat_entropy_scores[truth_mask]
        edge_entropy_gaussians = all_flat_entropy_gaussians[truth_mask]
        
        if len(edge_scores) == 0:
            continue
            
        mcd_entropy = []
        gaussian_entropy = []
        entropy_diff = []
        mcd_errors = []
        gaussian_errors = []
        diff_errors = []
        bin_centers = []
        
        for i in range(len(score_bins) - 1):
            bin_min, bin_max = score_bins[i], score_bins[i+1]
            bin_center = (bin_min + bin_max) / 2
            
            score_mask = (edge_scores >= bin_min) & (edge_scores < bin_max)
            
            if np.sum(score_mask) > 10:
                # MCD entropy
                mcd_entropy_in_bin = edge_entropy_scores[score_mask]
                mcd_entropy.append(np.mean(mcd_entropy_in_bin))
                mcd_errors.append(np.std(mcd_entropy_in_bin) / np.sqrt(np.sum(score_mask)))
                
                # Gaussian entropy
                gaussian_entropy_in_bin = edge_entropy_gaussians[score_mask]
                gaussian_entropy.append(np.mean(gaussian_entropy_in_bin))
                gaussian_errors.append(np.std(gaussian_entropy_in_bin) / np.sqrt(np.sum(score_mask)))
                
                # Entropy difference
                entropy_diff_in_bin = edge_entropy_diff[score_mask]
                entropy_diff.append(np.mean(entropy_diff_in_bin))
                diff_errors.append(np.std(entropy_diff_in_bin) / np.sqrt(np.sum(score_mask)))
                
                bin_centers.append(bin_center)
        
        combined_data[edge_type] = {
            'color': color,
            'mcd_entropy': mcd_entropy,
            'gaussian_entropy': gaussian_entropy,
            'entropy_diff': entropy_diff,
            'mcd_errors': mcd_errors,
            'gaussian_errors': gaussian_errors,
            'diff_errors': diff_errors,
            'bin_centers': bin_centers
        }
    
    # Create combined entropy comparison plot
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # 3:1 ratio between plots
    
    # Top plot for entropy values (combined)
    ax1 = plt.subplot(gs[0])
    
    for edge_type, data in combined_data.items():
        color = data['color']
        
        # Plot MCD entropy
        if data['mcd_entropy']:
            ax1.plot(data['bin_centers'], data['mcd_entropy'], '-', linewidth=2, color=color, 
                    label=f'MCD {edge_type} Edges')
            ax1.fill_between(
                data['bin_centers'], 
                np.array(data['mcd_entropy']) - np.array(data['mcd_errors']), 
                np.array(data['mcd_entropy']) + np.array(data['mcd_errors']), 
                alpha=0.3, color=color, edgecolor=None
            )
        
        # Plot Gaussian entropy with dashed line and lighter color
        gaussian_color = color
        if edge_type == "Target":
            gaussian_color = 'tab:cyan'
        elif edge_type == "Non-target":
            gaussian_color = 'lightgreen'
        else:  # False
            gaussian_color = 'moccasin'
            
        if data['gaussian_entropy']:
            ax1.plot(data['bin_centers'], data['gaussian_entropy'], '--', linewidth=2, color=gaussian_color, 
                    label=f'Gaussian {edge_type} Edges')
            ax1.fill_between(
                data['bin_centers'], 
                np.array(data['gaussian_entropy']) - np.array(data['gaussian_errors']), 
                np.array(data['gaussian_entropy']) + np.array(data['gaussian_errors']), 
                alpha=0.3, color=gaussian_color, edgecolor=None
            )
    
    ax1.set_ylabel('Entropy', fontsize=14, ha="right", y=0.95)
    ax1.legend(fontsize=14, loc='upper right')
    ax1.set_xlim(0, 1)
    
    # Bottom plot for entropy difference (combined)
    ax2 = plt.subplot(gs[1], sharex=ax1)
    
    for edge_type, data in combined_data.items():
        if data['entropy_diff']:
            ax2.plot(data['bin_centers'], data['entropy_diff'], '-', linewidth=2, color=data['color'], 
                    label=f'{edge_type} Edges')
            ax2.fill_between(
                data['bin_centers'], 
                np.array(data['entropy_diff']) - np.array(data['diff_errors']), 
                np.array(data['entropy_diff']) + np.array(data['diff_errors']), 
                alpha=0.3, color=data['color'], edgecolor=None
            )
    
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
    ax2.set_ylabel('Entropy Difference\n(MCD - Gaussian)', fontsize=14, ha="right", y=0.95)
    
    for ax in [ax1, ax2]:
        ax.label_outer()
    
    # Apply ATLAS styling to the whole figure
    if not UQ_propagation:
        atlasify(f"{n_train} train events", 
                r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        , axes=ax1)
    else:
        atlasify(f"{n_train} train events", 
                r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        + "\n"
                        + "UQ Propagation", axes=ax1)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)  # Reduce space between subplots

    # Save the combined entropy comparison figure
    save_path_combined = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_entropy_comparison_combined{dropout_str}.png"
    )
    save_path_combined_svg = os.path.join(
        config["stage_dir"], 
        f"{plot_config.get('filename', 'mc_dropout')}_entropy_comparison_combined{dropout_str}.svg"
    )
    fig.savefig(save_path_combined)
    fig.savefig(save_path_combined_svg)
    plt.close(fig)
    print(f"Combined entropy comparison plot saved to {save_path_combined}")


def plot_aleatoric_epistemic_uncertainty(all_flat_scores, all_flat_epistemic_uncertainty, all_flat_score_entropy, all_flat_total_entropy, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, UQ_propagation=False):
    """
    Plot total entropy, aleatoric entropy and epistemic uncertainty as a function of edge scores.
    Creates separate plots for each edge type and combined plots for each uncertainty type.
    
    Parameters:
    -----------
    all_flat_scores: numpy.ndarray
        Edge scores for all edges
    all_flat_epistemic_uncertainty: numpy.ndarray
        Epistemic uncertainty values for all edges
    all_flat_score_entropy: numpy.ndarray
        Aleatoric entropy values for all edges
    all_flat_total_entropy: numpy.ndarray
        Total entropy values for all edges
    all_flat_target_truth: numpy.ndarray
        Boolean array indicating target edges
    all_flat_non_target_truth: numpy.ndarray
        Boolean array indicating non-target true edges
    all_flat_false: numpy.ndarray
        Boolean array indicating false edges
    dataset: Dataset
        The dataset used for evaluation
    config: dict
        Configuration dictionary
    plot_config: dict
        Plot configuration dictionary
    dropout_str: str
        String indicating the dropout rate for file naming
    calibration: bool
        Whether the model is calibrated or not
    """
    
    # Create score bins
    score_bins = np.linspace(0, 1, 101)  # 100 bins from 0 to 1
    
    # Define the data to be plotted with corresponding titles and filenames
    entropy_types = [
        (all_flat_total_entropy, "Total Entropy", "total_entropy"),
        (all_flat_score_entropy, "Aleatoric Entropy", "score_entropy"),
        (all_flat_epistemic_uncertainty, "Epistemic Uncertainty", "epistemic_uncertainty")
    ]
    
    n_train = config["n_train"]
    
    pt_min = config["target_tracks"]['track_particle_pt'][0]/1e3
    
    # Create separate plots for each entropy type and each edge type
    for entropy_data, plot_title, filename_suffix in entropy_types:
        
        # Create individual plots for each edge type
        for edge_type, truth_mask, edge_suffix, color in [
            ("Target", all_flat_target_truth, "target", 'tab:blue'),
            ("Non-target", all_flat_non_target_truth, "non_target", 'tab:green'),
            ("False", all_flat_false, "false", 'tab:orange')
        ]:
            
            # Filter data for this edge type
            edge_scores = all_flat_scores[truth_mask]
            edge_entropy = entropy_data[truth_mask]
            
            if len(edge_scores) == 0:  # Skip if no edges of this type
                continue
            
            # Process bins for this edge type
            means = []
            errors = []
            bin_centers = []
            
            for i in range(len(score_bins) - 1):
                bin_min, bin_max = score_bins[i], score_bins[i+1]
                bin_center = (bin_min + bin_max) / 2
                
                # Get indices of edges with scores in this bin
                score_mask = (edge_scores >= bin_min) & (edge_scores < bin_max)
                
                if np.sum(score_mask) > 10:  # Only include bins with sufficient data
                    bin_values = edge_entropy[score_mask]
                    means.append(np.mean(bin_values))
                    errors.append(np.std(bin_values) / np.sqrt(np.sum(score_mask)))
                    bin_centers.append(bin_center)
            
            # Create the figure for this edge type
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Plot this edge type
            if means:
                ax.plot(bin_centers, means, '-', linewidth=2, color=color, label=f'{edge_type} Edges')
                ax.fill_between(
                    bin_centers, 
                    np.array(means) - np.array(errors), 
                    np.array(means) + np.array(errors), 
                    alpha=0.3,
                    color=color,
                    edgecolor=None
                )
            
            # Configure axis
            ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
            ax.set_ylabel(plot_title, fontsize=14, ha="right", y=0.95)
            ax.legend(fontsize=14)
            ax.set_xlim(0, 1)
            
            # Set y-axis limits based on data
            if means:
                ymax = max(np.array(means) + np.array(errors))
                # ymax = min(ymax * 1.1, 2.0)  # Cap at 2.0 for entropy plots
                ax.set_ylim(0, 0.8)
            
            # Apply ATLAS styling
            if not UQ_propagation:
                atlasify(f"{n_train} train events", 
                        r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}" + "\n" + f"{edge_type} Edges Only")
            else:
                atlasify(f"{n_train} train events", 
                        r"$p_T >" + f"{pt_min if edge_type=='Target' else 0}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        + "\n"
                        + "UQ Propagation")
                
            fig.tight_layout()
            
            # Save the figure
            save_path = os.path.join(
                config["stage_dir"], 
                f"{plot_config.get('filename', 'mc_dropout')}_{filename_suffix}_{edge_suffix}{dropout_str}.png"
            )
            save_path_svg = os.path.join(
                config["stage_dir"], 
                f"{plot_config.get('filename', 'mc_dropout')}_{filename_suffix}_{edge_suffix}{dropout_str}.svg"
            )
            fig.savefig(save_path)
            fig.savefig(save_path_svg)
            plt.close(fig)
            print(f"{edge_type} edges {plot_title} plot saved to {save_path}")
        
        # Create combined plots showing all three edge types together
        combined_data = {}
        for edge_type, truth_mask, color in [
            ("Target", all_flat_target_truth, 'tab:blue'),
            ("Non-target", all_flat_non_target_truth, 'tab:green'),
            ("False", all_flat_false, 'tab:orange')
        ]:
            edge_scores = all_flat_scores[truth_mask]
            edge_entropy = entropy_data[truth_mask]
            
            if len(edge_scores) == 0:
                continue
                
            means = []
            errors = []
            bin_centers = []
            
            for i in range(len(score_bins) - 1):
                bin_min, bin_max = score_bins[i], score_bins[i+1]
                bin_center = (bin_min + bin_max) / 2
                
                score_mask = (edge_scores >= bin_min) & (edge_scores < bin_max)
                
                if np.sum(score_mask) > 10:
                    bin_values = edge_entropy[score_mask]
                    means.append(np.mean(bin_values))
                    errors.append(np.std(bin_values) / np.sqrt(np.sum(score_mask)))
                    bin_centers.append(bin_center)
            
            combined_data[edge_type] = {
                'color': color,
                'means': means,
                'errors': errors,
                'bin_centers': bin_centers
            }
        
        # Create combined plot for this entropy type
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for edge_type, data in combined_data.items():
            if data['means']:
                ax.plot(data['bin_centers'], data['means'], '-', 
                        linewidth=2, color=data['color'], label=f'{edge_type} Edges')
                ax.fill_between(
                    data['bin_centers'], 
                    np.array(data['means']) - np.array(data['errors']), 
                    np.array(data['means']) + np.array(data['errors']), 
                    alpha=0.3,
                    color=data['color'],
                    edgecolor=None
                )
        
        # Configure axis
        ax.set_xlabel('Edge Score', fontsize=14, ha="right", x=0.95)
        ax.set_ylabel(plot_title, fontsize=14, ha="right", y=0.95)
        ax.legend(fontsize=14)
        ax.set_xlim(0, 1)
        
        # Set y-axis limits based on data
        all_means = []
        all_errors = []
        for data in combined_data.values():
            if data['means']:
                all_means.extend(data['means'])
                all_errors.extend(data['errors'])
        
        if all_means:
            ymax = max(np.array(all_means) + np.array(all_errors))
            # ymax = min(ymax * 1.1, 2.0)  # Cap at 2.0 for entropy plots
            ax.set_ylim(0, 0.8)
        
        # Apply ATLAS styling
        if not UQ_propagation:
            atlasify(f"{n_train} train events", 
                    r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}")
        else:
            atlasify(f"{n_train} train events", 
                    r"Target: $p_T >" + f"{pt_min}"+"$ GeV, $ | \eta | < 4$" + "\n"
                        + f"MC Dropout with {config.get('nb_MCD_passes', 100)} forward passes"
                        + "\n"
                        + f"Dropout rate: {dropout_value}"
                        + "\n"
                        + f"Evaluated on {config.get('dataset_size', 50)} events in {config['dataset']}"
                        + "\n"
                        + "UQ Propagation")
            
        fig.tight_layout()
        
        # Save the combined figure
        save_path_combined = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_{filename_suffix}_combined{dropout_str}.png"
        )
        save_path_combined_svg = os.path.join(
            config["stage_dir"], 
            f"{plot_config.get('filename', 'mc_dropout')}_{filename_suffix}_combined{dropout_str}.svg"
        )
        fig.savefig(save_path_combined)
        fig.savefig(save_path_combined_svg)
        plt.close(fig)
        print(f"Combined {plot_title} plot saved to {save_path_combined}")


def graph_mcdropout_uncertainty(lightning_module, plot_config, config):
    """
    Evaluate the uncertainty of edge scores using MC Dropout and plot the results.
    """
    dataset_name = config["dataset"]
    dataset = getattr(lightning_module, dataset_name)
    n_train = config["n_train"]
    dropout_value = config.get('hidden_dropout', 0.0)
    dropout_str = f"_dropout{dropout_value:.1f}".replace('.', 'p')
    config["dataset_size"] = len(dataset)
    if isinstance(lightning_module, Filter):
        true_target_weight = 2
        true_non_target_weight = 0
        false_weight = 1
    if isinstance(lightning_module, InteractionGNN):
        true_target_weight = 1
        true_non_target_weight = 0
        false_weight = 0.1

    n_mcd_passes = config.get("nb_MCD_passes", 100)
    n_event = len(dataset)
    lightning_module.calibrated = config.get("calibration", False)
    if lightning_module.calibrated:
        # print("Using calibrated model")
        # rescaling_parameters = np.loadtxt(os.path.join(config["stage_dir"], "rescaling.txt"), skiprows=1, usecols=(0, 1))
        # lightning_module.rescaling_parameters = rescaling_parameters
        print("Using calibrated model")
        # Load spline object instead of text file
        import pickle
        spline_path = os.path.join(config["stage_dir"], "calibration", "rescaling_spline.pkl")
        with open(spline_path, 'rb') as f:
            rescaling_parameters = pickle.load(f)
        lightning_module.rescaling_parameters = rescaling_parameters
        print(f"Loaded spline rescaling parameters from {spline_path}")
        calib_folder = "calibrated"     
   
    else:
        print("Using uncalibrated model")
        calib_folder = "uncalibrated"
    
    if config.get("multi_dropout", False):
        calib_folder += f"/multi_drop/{dropout_value}"
    else:
        if not config["input_cut"]:
            calib_folder += "/no_input_cut"
        else:
            calib_folder += "/with_input_cut"
    
    config["stage_dir"] = os.path.join(config["stage_dir"], "plots", calib_folder)

    all_target_truth = [None for _ in range(len(dataset))]     # Store target truth once per event
    all_non_target_truth = [None for _ in range(len(dataset))] # Store non-target truth once per event
    all_false = [None for _ in range(len(dataset))]            # Store false edges once per event
    target_pt = [None for _ in range(len(dataset))]     # Store truth once per event
    target_eta = [None for _ in range(len(dataset))]    # Store truth once per event
    all_scores = [[] for _ in range(n_mcd_passes)]
    all_score_entropy = [[] for _ in range(n_mcd_passes)]
    all_track_scores = [[] for _ in range(n_mcd_passes)]
    
    # Add new storage for track edge truth values
    all_target_truth_track = [None for _ in range(len(dataset))]
    all_non_target_truth_track = [None for _ in range(len(dataset))]

    for t in tqdm(range(n_mcd_passes)):
        for num_event, filter_event in enumerate(dataset):
            if not config["input_cut"]: # if input_cut is 0.0 we apply the mask by hand
                input_cut_mask = filter_event.edge_scores < 0.05 # default gnn input cut
            with torch.inference_mode():
                lightning_module.train()
                eval_dict = lightning_module.shared_evaluation(filter_event.to(lightning_module.device), 0)
            gnn_event = eval_dict["batch"]
            edge_scores = gnn_event.edge_scores.cpu().numpy()
            # put all scores that were below the input cut to 0
            if not config["input_cut"]: # if input_cut is 0.0 we apply the mask by hand
                edge_scores[input_cut_mask.cpu().numpy()] = 0.0
            all_scores[t].append(edge_scores)
            
            # Calculate the entropy of the edge scores, useful for epistemic/aleatoric uncertainty
            score_entropy = -edge_scores*np.log(edge_scores + 1e-10) - (1-edge_scores)*np.log(1-edge_scores + 1e-10)
            all_score_entropy[t].append(score_entropy)

            # Handling of true edges
            if all_target_truth[num_event] is None:
                all_target_truth[num_event] = gnn_event.edge_y.cpu().numpy() & (gnn_event.edge_weights==true_target_weight).cpu().numpy() 
                #* select only true target edges (ie. with pT>1GeV and nhits>3)
                all_non_target_truth[num_event] = gnn_event.edge_y.cpu().numpy() & (gnn_event.edge_weights==true_non_target_weight).cpu().numpy() 
                #* select only true non-target edges
                all_false[num_event] = (~gnn_event.edge_y.cpu().numpy()) & (gnn_event.edge_weights==false_weight).cpu().numpy() 
                #* select only false edges
                
            edge_index = gnn_event.edge_index.cpu().numpy()
            track_edges = gnn_event.track_edges.cpu().numpy()
            track_mask = find_edge_indices(edge_index, track_edges)
            all_track_scores[t].append(edge_scores[track_mask])
            
            # Store track edge truth values
            if all_target_truth_track[num_event] is None:
                track_edge_y = gnn_event.edge_y.cpu().numpy()[track_mask]
                track_edge_weights = gnn_event.edge_weights.cpu().numpy()[track_mask]
                all_target_truth_track[num_event] = track_edge_y & (track_edge_weights==1)
                all_non_target_truth_track[num_event] = track_edge_y & (track_edge_weights==0)
                
            if target_pt[num_event] is None:
                target_pt[num_event] = gnn_event.track_particle_pt.cpu().numpy()
                target_eta[num_event] = gnn_event.hit_eta[edge_index[0]].cpu().numpy()
    
    # Prepare flatten arrays for analysis
    all_mean_scores = [[] for _ in range(len(dataset))]
    all_mean_uncertainties = [[] for _ in range(len(dataset))]
    all_mean_score_entropy = [[] for _ in range(len(dataset))]
    all_mean_kurtosis = [[] for _ in range(len(dataset))]
    all_mean_skewness = [[] for _ in range(len(dataset))]
    all_mean_track_scores = [[] for _ in range(len(dataset))]
    all_mean_uncertainties_track = [[] for _ in range(len(dataset))]

    for n_event in range(len(dataset)):
        temp_all_scores = []
        temp_all_scores_track = []
        temp_all_score_entropy = []
        temp_all_kurtosis = []
        temp_all_skewness = []
        for t in range(n_mcd_passes):
            temp_all_scores.append(all_scores[t][n_event])
            temp_all_scores_track.append(all_track_scores[t][n_event])
            temp_all_score_entropy.append(all_score_entropy[t][n_event])
            temp_all_kurtosis.append(kurtosis(all_scores[t][n_event]))
            temp_all_skewness.append(skew(all_scores[t][n_event]))
        all_mean_scores[n_event] = np.mean(temp_all_scores, axis=0)
        all_mean_uncertainties[n_event] = np.std(temp_all_scores, axis=0)
        all_mean_score_entropy[n_event] = np.mean(temp_all_score_entropy, axis=0)
        all_mean_track_scores[n_event] = np.mean(temp_all_scores_track, axis=0)
        all_mean_uncertainties_track[n_event] = np.std(temp_all_scores_track, axis=0)
        all_mean_kurtosis[n_event] = kurtosis(temp_all_scores, axis=0)
        all_mean_skewness[n_event] = skew(temp_all_scores, axis=0)

    if n_train>=800:
        all_gaussian_scores = generate_matching_gaussians(all_scores)
        all_entropy_diff, entropy_scores, entropy_gaussians = compare_entropy(all_scores, all_gaussian_scores) #entropy scores - gaussian scores

    # # Flatten arrays for analysis
    all_flat_scores = np.concatenate(all_mean_scores)
    all_flat_uncertainties = np.concatenate(all_mean_uncertainties)
    all_flat_BCE_score_entropy = np.concatenate(all_mean_score_entropy)
    all_flat_target_truth = np.concatenate(all_target_truth)
    all_flat_non_target_truth = np.concatenate(all_non_target_truth)
    all_flat_false = np.concatenate(all_false)
    all_flat_pt = np.concatenate(target_pt)
    all_flat_eta = np.concatenate(target_eta)
    all_flat_scores_track = np.concatenate(all_mean_track_scores)
    all_flat_uncertainties_track = np.concatenate(all_mean_uncertainties_track)
    all_flat_kurtosis = np.concatenate(all_mean_kurtosis)
    all_flat_skewness = np.concatenate(all_mean_skewness)
    all_flat_truth = all_flat_target_truth | all_flat_non_target_truth
    if n_train>=800:
        all_flat_entropy_diff = np.concatenate(all_entropy_diff)
        all_flat_entropy_scores = np.concatenate(entropy_scores)
        all_flat_entropy_gaussians = np.concatenate(entropy_gaussians)
    
    # # Flatten track edge truth arrays
    all_flat_target_truth_track = np.concatenate(all_target_truth_track)
    all_flat_non_target_truth_track = np.concatenate(all_non_target_truth_track)

    all_flat_total_uncertainty = -all_flat_scores*np.log(all_flat_scores + 1e-10) - (1-all_flat_scores)*np.log(1-all_flat_scores + 1e-10)
    all_flat_epistemic_uncertainty = all_flat_total_uncertainty - all_flat_BCE_score_entropy # mutual information

    # Save the results to text files
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_scores_{n_train}.txt"), all_flat_scores)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_uncertainties_{n_train}.txt"), all_flat_uncertainties)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_target_truth_{n_train}.txt"), all_flat_target_truth)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_non_target_truth_{n_train}.txt"), all_flat_non_target_truth)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_false_{n_train}.txt"), all_flat_false)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_pt_{n_train}.txt"), all_flat_pt)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_eta_{n_train}.txt"), all_flat_eta)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_scores_track_{n_train}.txt"), all_flat_scores_track)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_uncertainties_track_{n_train}.txt"), all_flat_uncertainties_track)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_BCE_score_entropy_{n_train}.txt"), all_flat_BCE_score_entropy)
    np.savetxt(os.path.join(config["stage_dir"], f"Total_uncertainty_{n_train}.txt"), all_flat_total_uncertainty)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_epistemic_uncertainty_{n_train}.txt"), all_flat_epistemic_uncertainty)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_target_truth_track_{n_train}.txt"), all_flat_target_truth_track)
    np.savetxt(os.path.join(config["stage_dir"], f"all_flat_non_target_truth_track_{n_train}.txt"), all_flat_non_target_truth_track)

    # Plot the results
    plot_uncertainty_vs_score(
        all_flat_scores, all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, lightning_module.calibrated
    )

    plot_uncertainty_vs_pt(
        all_flat_pt, all_flat_scores_track, all_flat_uncertainties_track, all_flat_target_truth_track, all_flat_non_target_truth_track, dataset, config, plot_config, dropout_str, dropout_value, lightning_module.calibrated
    )

    plot_uncertainty_vs_eta(
        all_flat_eta, all_flat_scores, all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, lightning_module.calibrated
    )
    
    plot_uncertainty_distribution(
        all_flat_uncertainties, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, lightning_module.calibrated
    )

    plot_calibration_curve(
        all_flat_scores, all_flat_truth,
        dataset, config, plot_config, dropout_str, dropout_value, from_calibration_stage=False
    )
    
    plot_reliability_diagram(
        all_flat_scores, all_flat_truth, dataset, config, plot_config, dropout_str, dropout_value, from_calibration_stage=False
    )

    plot_edge_scores_distribution(
        all_flat_scores, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_value, lightning_module.calibrated
    )
    
    plot_number_edges_vs_eta(
        all_flat_eta, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_value, lightning_module.calibrated
    )

    plot_edges_score_vs_eta(
        all_flat_eta, all_flat_scores, all_flat_target_truth, all_flat_non_target_truth, all_flat_false, dataset, config, plot_config, dropout_str, dropout_value, lightning_module.calibrated
    )

    plot_edges_score_vs_pt( 
        all_flat_pt, all_flat_scores_track, all_flat_target_truth_track, all_flat_non_target_truth_track, dataset, config, plot_config, dropout_str, dropout_value, lightning_module.calibrated
    )

    plot_edge_skewness_kurtosis(
        all_flat_scores, all_flat_skewness, all_flat_kurtosis, all_flat_target_truth, all_flat_non_target_truth, all_flat_false,
        dataset, config, plot_config, dropout_str, dropout_value, lightning_module.calibrated
    )

    plot_aleatoric_epistemic_uncertainty(
        all_flat_scores, all_flat_epistemic_uncertainty, all_flat_BCE_score_entropy, 
        all_flat_total_uncertainty, all_flat_target_truth, all_flat_non_target_truth, all_flat_false,
        dataset, config, plot_config, dropout_str, dropout_value, lightning_module.calibrated
    )

    if n_train >= 800:
        plot_entropy_difference(
            all_flat_scores, all_flat_entropy_diff, all_flat_entropy_scores, all_flat_entropy_gaussians, all_flat_target_truth, all_flat_non_target_truth, all_flat_false,
            dataset, config, plot_config, dropout_str, dropout_value, lightning_module.calibrated
        )