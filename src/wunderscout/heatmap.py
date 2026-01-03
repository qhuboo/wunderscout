import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
from scipy.stats import gaussian_kde

# === Load the raw tracking data CSV ===
# NOTE: "header=2" → skip the first two rows (team/labels) and use row 3 as header
df = pd.read_csv(
    "./data/Sample_Game_1_RawTrackingData_Away_Team.csv",
    header=2,
)

# === Clean column names so each player has _X and _Y ===
cleaned_colums = []
colnames = df.columns.tolist()
i = 0
while i < len(colnames):
    col = colnames[i]
    if col.startswith("Player") or col.startswith("Ball"):
        cleaned_colums.append(f"{col}_X")
        cleaned_colums.append(f"{col}_Y")
        i += 2
    else:
        cleaned_colums.append(col)
        i += 1
df.columns = cleaned_colums

print("Columns cleaned. First few rows:")
print(df.head())

# === Extract Player17 (drop NaN values where tracking failed) ===
player17 = df[["Player17_X", "Player17_Y"]].dropna()
x = player17["Player17_X"].to_numpy()
y = player17["Player17_Y"].to_numpy()

# === Detect scale (normalized [0,1] or real meters) ===
if x.max() <= 1.5 and y.max() <= 1.5:
    print("Scaling Player17 data from normalized [0,1] to meters...")
    x = x * 105  # pitch length in meters
    y = y * 68  # pitch width in meters
else:
    print("Data appears to already be in meters, leaving as is.")

print("First 10 points:", list(zip(x[:10], y[:10])))

# =============================================================================
# 1. Scatter Plot (sanity check, raw positions)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))
# Pitch outline
ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color="black")
ax.plot([52.5, 52.5], [0, 68], color="black")  # halfway line
# Player positions
ax.scatter(x, y, s=1, alpha=0.3, color="blue")
ax.set_xlim(0, 105)
ax.set_ylim(0, 68)
ax.set_title("Player17 Movement Scatter (raw positions)")
plt.savefig("./heatmap/player17_scatter.png", dpi=150, bbox_inches="tight")

# =============================================================================
# 2. Histogram Heatmap (occupancy grid)
# =============================================================================
heatmap, xedges, yedges = np.histogram2d(x, y, bins=(50, 34), range=[[0, 105], [0, 68]])

fig, ax = plt.subplots(figsize=(10, 7))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
im = ax.imshow(
    heatmap.T, origin="lower", extent=extent, cmap="Blues", alpha=0.7, aspect="auto"
)
ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color="black")
ax.plot([52.5, 52.5], [0, 68], color="black")
fig.colorbar(im, ax=ax, label="Frames")
ax.set_title("Player17 Heatmap (Histogram)")
plt.savefig("./heatmap/player17_histogram.png", dpi=150, bbox_inches="tight")

# === Export histogram data as JSON for three.js ===
heatmap_data = {
    "xedges": xedges.tolist(),
    "yedges": yedges.tolist(),
    "values": heatmap.T.tolist(),  # transpose so rows correspond to y-axis correctly
}
with open("./heatmap/player17_histogram.json", "w") as f:
    json.dump(heatmap_data, f)

# =============================================================================
# 3. KDE Heatmap (smoothed density field)
# =============================================================================
values = np.vstack([x, y])
kde = gaussian_kde(values)

# Define mesh grid
X, Y = np.meshgrid(np.linspace(0, 105, 100), np.linspace(0, 68, 68))
Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

fig, ax = plt.subplots(figsize=(10, 7))
sns.kdeplot(x=x, y=y, fill=True, cmap="Blues", alpha=0.7, thresh=0.05, levels=50, ax=ax)
ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color="black")
ax.plot([52.5, 52.5], [0, 68], color="black")
ax.set_xlim(0, 105)
ax.set_ylim(0, 68)
ax.set_title("Player17 Heatmap (KDE Smoothed)")
plt.savefig("./heatmap/player17_kde.png", dpi=150, bbox_inches="tight")

# === Export KDE density field for three.js ===
kde_data = {
    "x": X[0].tolist(),  # x grid coordinates
    "y": Y[:, 0].tolist(),  # y grid coordinates
    "values": Z.tolist(),  # density values
}
with open("./heatmap/player17_kde.json", "w") as f:
    json.dump(kde_data, f)

print("Outputs saved: scatter, histogram PNG+JSON, KDE PNG+JSON for Player17")
