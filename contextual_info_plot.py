import pandas as pd
import matplotlib.pyplot as plt
import os

# Make fonts bigger and bold
plt.rcParams.update({
    'font.size': 14,            # base font size
    'axes.labelweight': 'bold', # bold axis labels
    'axes.titlesize': 16,       # title font size
    'axes.titleweight': 'bold', # bold title
    'legend.fontsize': 12,      # legend font size
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Paths to each CSV file
folders = {
    "U-Net": "plots_unet-/contextual_info_data.csv",
    "nnU-Net": "plots_nn-unet/contextual_info_data.csv",
    "Swin UNETR": "plots_swin_unetr/contextual_info_data.csv"
}

# -------------------
# Figure 1: average_score ± std_dev
# -------------------
plt.figure(figsize=(8, 6))

for label, path in folders.items():
    if not os.path.exists(path):
        print(f"Warning: file not found: {path}")
        continue

    df = pd.read_csv(path).sort_values(by="distance_mm")
    x = df["distance_mm"]
    y = df["average_score"]
    std = df["std_dev"]

    plt.plot(x, y, label=label, linewidth=2)
    plt.fill_between(x, y - std, y + std, alpha=0.2)

plt.xlabel("Distance from lesion (mm)", fontweight='bold')
plt.ylabel("Average Prediction Score", fontweight='bold')
plt.legend(frameon=True, loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('contextual_info_nets_plot1.png', dpi=300)


# -------------------
# Figure 2: n_predictions step plot
# -------------------
plt.figure(figsize=(8, 6))

for label, path in folders.items():
    if not os.path.exists(path):
        continue

    df = pd.read_csv(path).sort_values(by="distance_mm")
    x = df["distance_mm"]
    y2 = df["n_predictions"]

    plt.step(x, y2, label=label, where='mid', linewidth=2)

plt.xlabel("Distance from lesion (mm)", fontweight='bold')
plt.ylabel("Number of Predictions", fontweight='bold')
plt.legend(frameon=True, loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('contextual_info_nets_plot2.png', dpi=300)

