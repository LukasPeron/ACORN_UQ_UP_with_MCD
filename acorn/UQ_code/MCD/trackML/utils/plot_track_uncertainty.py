import matplotlib.pyplot as plt
import numpy as np
import atlasify as atl
atl.ATLAS = "TrackML dataset"
from atlasify import atlasify

track_eff = np.loadtxt("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/UQ_propagation/track_building/efficiency.txt")

n_bins = 7
# Plot histogram of track efficiency
plt.figure(figsize=(8, 6))
# Use fewer bins since we only have 5 data points
plt.hist(track_eff, bins=n_bins, label='Track Efficiency')
plt.xlabel('Track Efficiency', fontsize=14, ha="right", x=0.95)
plt.ylabel('Number of Entries', fontsize=14, ha="right", y=0.95)

# Add gaussian fit (scaled to match histogram normalization)
mean = np.mean(track_eff)
std = np.std(track_eff)

# track_eff_gauss = np.random.normal(loc=mean, scale=std, size=100)  # Simulated data for demonstration
# plt.hist(track_eff_gauss, bins=n_bins, alpha=0.7, label='Track Efficiency\n Gaussian samples')

x = np.linspace(mean - 3*std, mean + 3*std, 100)
# Scale the gaussian to match the probability histogram
bin_width = (track_eff.max() - track_eff.min()) / 7
y = len(track_eff) * bin_width * (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
plt.plot(x, y, linewidth=3, label=f'Gaussian Fit\n mean: {mean*100:.2f}%\n std: {std*100:.2f}%')
plt.legend(loc='upper right', fontsize=14)
atlasify(f"1400 train events",
        r"Target: $p_T >1$ GeV, $ | \eta | < 4$" + "\n"
        + "Track efficiency with CC&Walk algorithm" + "\n"
        + f"MC Dropout with {len(track_eff)} forward passes on Filter and GNN" + "\n"
        + f"Evaluated on 50 events in valset" + "\n"
        + f"Dropout rate: 0.1" + "\n"
    )
plt.tight_layout()
plt.savefig("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/UQ_propagation/track_building/track_efficiency_histogram.svg")
plt.savefig("/pscratch/sd/l/lperon/UQ_data/MCD/trackML/all_pt/1400/UQ_propagation/track_building/track_efficiency_histogram.pdf")
print(mean, std)

# Compute skewness and kurtosis
# from scipy.stats import skew, kurtosis
# skewness = skew(track_eff)
# kurt = kurtosis(track_eff)
# print(f"Skewness: {skewness:.4f}, Kurtosis: {kurt:.4f}")

# # Compute indicator of normality
# from scipy.stats import normaltest
# stat, p_value = normaltest(track_eff)
# print(f"Normality test statistic: {stat:.4f}, p-value: {p_value:.4f}")
# if p_value < 0.05:
#     print("The data is not normally distributed (reject H0)")
# else:
#     print("The data is normally distributed (fail to reject H0)")

# track_eff_entropy = -np.sum(track_eff * np.log(track_eff + 1e-10))  # Adding a small constant to avoid log(0)
# track_eff_gauss_entropy = -np.sum(track_eff_gauss * np.log(track_eff_gauss + 1e-10))  # Adding a small constant to avoid log(0)
# print(f"Entropy difference: {track_eff_entropy - track_eff_gauss_entropy:.4f}")