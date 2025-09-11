"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Dynamic Penrose semantic anchor mapping (Red↔Time, Green↔Space, Blue↔Light)

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- LOAD DATA ---
CSV_PATH = '/data/nltk_color_similarity_summary.csv'
summary = pd.read_csv(CSV_PATH)

clusters_path = '/data/la_matrice_clusters.csv'
clusters = pd.read_csv(clusters_path)

# Make sure column names align
summary.columns = [c.strip() for c in summary.columns]
clusters.columns = [c.strip() for c in clusters.columns]

# Find the color name column
col_color = None
for cand in ["Color", "color", "Closest Color", "color", "closest_color"]:
    if cand in summary.columns:
        col_color = cand
        break
if col_color is None:
    raise ValueError("No color name column found in summary CSV!")

# Merge clusters onto summary by color name
if "color" in clusters.columns:
    clusters = clusters.rename(columns={"color": col_color})
elif "Closest Color" in clusters.columns:
    clusters = clusters.rename(columns={"Closest Color": col_color})

merged = summary.merge(clusters, on=col_color, how="left")


# Extract needed columns
R = merged["R"].to_numpy(dtype=float)
G = merged["G"].to_numpy(dtype=float)
B = merged["B"].to_numpy(dtype=float)

time_sem  = merged["time_sem"].to_numpy(dtype=float)
space_sem = merged["space_sem"].to_numpy(dtype=float)
light_sem = merged["light_sem"].to_numpy(dtype=float)


# === 1) Dynamic (change-based) correlation ===
sem_cols = ["time_sem", "space_sem", "light_sem"]
rgb_cols = ["R", "G", "B"]

# Compute diffs row-to-row
diff_df = merged[rgb_cols + sem_cols].diff().dropna()

def spearman_simple(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan, mask.sum()
    xr = x[mask].rank(method="average")
    yr = y[mask].rank(method="average")
    rho = np.corrcoef(xr, yr)[0, 1]
    return float(rho), mask.sum()

corr_rows = []
pairs = [("R", "time_sem"), ("G", "space_sem"), ("B", "light_sem")]

for rgb_col, sem_col in pairs:
    pear = np.corrcoef(diff_df[rgb_col], diff_df[sem_col])[0, 1]
    spear, n = spearman_simple(diff_df[rgb_col], diff_df[sem_col])
    corr_rows.append({
        "Δrgb_channel": rgb_col,
        "Δconcept": sem_col,
        "pearson": pear,
        "spearman": spear,
        "n_pairs": n
    })

delta_corr_df = pd.DataFrame(corr_rows)

# --- SAVE OUTPUTS ---
delta_corr_df.to_csv("dynamic_change_correlation.csv", index=False)

print("Dynamic (Δ) correlations:")
display(delta_corr_df)

print("\nSaved CSVs:")
print("dynamic_change_correlation.csv")


# === 2) Concept space visualization ===
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter points in concept space, colored by RGB
ax.scatter(merged["time_sem"], merged["space_sem"], merged["light_sem"],
           c=merged[rgb_cols].astype(float) / 255.0, s=50, alpha=0.8)

# Anchors for concepts (unit points)
anchors = {
    "time":  np.array([1.0, 0.0, 0.0]),
    "space": np.array([0.0, 1.0, 0.0]),
    "light": np.array([0.0, 0.0, 1.0]),
}

# Draw vectors from each anchor to all points (thin, semi-transparent)
for name, anchor in anchors.items():
    diffs = merged[sem_cols].values - anchor
    for point in merged[sem_cols].values:
        ax.plot([anchor[0], point[0]],
                [anchor[1], point[1]],
                [anchor[2], point[2]],
                color='gray', alpha=0.1, linewidth=0.5)

ax.set_xlabel("Time (SEM similarity)")
ax.set_ylabel("Space (SEM similarity)")
ax.set_zlabel("Light (SEM similarity)")
ax.set_title("Concept Space with RGB Coloring & Anchor Vectors")
plt.show()

# Anchor concept mapped to colors

concept_map = {0: "Time", 1: "Space", 2: "Light"}
merged["DominantConcept"] = merged[sem_cols].values.argmax(axis=1)
merged["DominantConcept"] = merged["DominantConcept"].map(concept_map)


# --- FINAL TABLE ---
concept_table = merged[[col_color, "cluster", "R", "G", "B",
                        "time_sem", "space_sem", "light_sem", "DominantConcept"]]

# Save for later use

print("\nColor → Concept Anchoring Table:")
display(concept_table.head(15))

# Save results
concept_table.to_csv("color_concept_anchoring.csv", index=False)
print("\nSaved: color_concept_anchoring.csv")
