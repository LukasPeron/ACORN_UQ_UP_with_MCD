import numpy as np
import matplotlib.pyplot as plt
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify
 
fig, ax = plt.subplots(figsize=(8, 6))
for calib in ["Uncalibrated", "Calibrated"]:
    if calib == "Uncalibrated":
        path = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/track_building/uncal_res/track_reconstruction_eff_vs_pt_ATLAS_Uncalibrated_"
        color="black"
    else:
        path = "/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/track_building/cal_res/track_reconstruction_eff_vs_pt_ATLAS_Calibrated_"
        color="red"
    # Load the data
    eff = np.loadtxt(path + "eff.txt")
    err = np.loadtxt(path + "err.txt")
    xerrs = np.loadtxt(path + "xerrs.txt")
    xvals = np.loadtxt(path + "xvals.txt")
    # Plot the data
    ax.errorbar(xvals, eff, yerr=err, xerr=xerrs, fmt='o', markersize=5, label=calib, color=color)
ax.set_xlabel(r'$p_T$ [GeV]', fontsize=14, ha="right", x=0.95)
ax.set_ylabel("Track Efficiency", fontsize=14, ha="right", y=0.95)
ax.set_ylim(ymin=0.75, ymax=1)
ax.legend(fontsize=14, loc='upper right')
atlasify("1400 events",
        "CCandWalk v0 algorithm\n"
        + r"$p_T > 1$GeV, $|\eta| < 4$",
    )
fig.tight_layout()
fig.savefig("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/track_building/"+"track_reconstruction_eff_vs_pt.png", dpi=300)
fig.savefig("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/track_building/"+"track_reconstruction_eff_vs_pt.svg", dpi=300)
