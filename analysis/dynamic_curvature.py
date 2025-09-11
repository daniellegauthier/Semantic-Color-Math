"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Dynamic Penrose semantic curvature 

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --- LOAD DATA ---
CSV_PATH = '/data/nltk_color_similarity_summary.csv'
summary = pd.read_csv(CSV_PATH)

# Extract needed columns
R = summary["R"].to_numpy(dtype=float)
G = summary["G"].to_numpy(dtype=float)
B = summary["B"].to_numpy(dtype=float)

time_sem  = summary["time_sem"].to_numpy(dtype=float)
space_sem = summary["space_sem"].to_numpy(dtype=float)
light_sem = summary["light_sem"].to_numpy(dtype=float)

sem_cols = ["time_sem", "space_sem", "light_sem"]
rgb_cols = ["R", "G", "B"]

# 1) Compute step-to-step deltas
df = summary[sem_cols + rgb_cols].copy()
diff_df = df.diff().dropna()

# Helper: average semantic movement when a given channel moves up vs down
def avg_sem_vector_for_channel(delta_channel: str):
    up_mask = diff_df[delta_channel] > 0
    down_mask = diff_df[delta_channel] < 0
    v_up = diff_df.loc[up_mask, sem_cols].mean().to_numpy(dtype=float) if up_mask.any() else np.array([np.nan, np.nan, np.nan])
    v_dn = diff_df.loc[down_mask, sem_cols].mean().to_numpy(dtype=float) if down_mask.any() else np.array([np.nan, np.nan, np.nan])
    return v_up, v_dn, int(up_mask.sum()), int(down_mask.sum())

vR_up, vR_dn, nRu, nRd = avg_sem_vector_for_channel("R")
vG_up, vG_dn, nGu, nGd = avg_sem_vector_for_channel("G")
vB_up, vB_dn, nBu, nBd = avg_sem_vector_for_channel("B")

# 2) Base point for arrows: centroid of concept space
centroid = df[sem_cols].mean().to_numpy(dtype=float)

# 3) Scale arrows to fit nicely in [0,1]^3 box
vectors = np.vstack([vR_up, vR_dn, vG_up, vG_dn, vB_up, vB_dn])
finite_norms = np.linalg.norm(vectors[~np.isnan(vectors).any(axis=1)], axis=1)
scale = 0.5 / (finite_norms.max() + 1e-12) if finite_norms.size else 1.0

# 4) Plot: points + centroid + six arrows (default colors; no custom styles)
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

# scatter (default color)
ax.scatter(df["time_sem"], df["space_sem"], df["light_sem"], s=40, alpha=0.6)

# centroid
ax.scatter([centroid[0]], [centroid[1]], [centroid[2]], s=80, marker='o')
ax.text(centroid[0], centroid[1], centroid[2], " centroid", fontsize=9)

def draw_arrow(vec, label_suffix):
    if np.isnan(vec).any():
        return
    end = centroid + scale * vec
    ax.plot([centroid[0], end[0]],
            [centroid[1], end[1]],
            [centroid[2], end[2]],
            linewidth=2)
    ax.text(end[0], end[1], end[2], label_suffix, fontsize=9)

# Draw six average movement arrows
draw_arrow(vR_up,  f" +ΔR (n={nRu})")
draw_arrow(vR_dn,  f" −ΔR (n={nRd})")
draw_arrow(vG_up,  f" +ΔG (n={nGu})")
draw_arrow(vG_dn,  f" −ΔG (n={nGd})")
draw_arrow(vB_up,  f" +ΔB (n={nBu})")
draw_arrow(vB_dn,  f" −ΔB (n={nBd})")

ax.set_xlabel("Time (SEM similarity)")
ax.set_ylabel("Space (SEM similarity)")
ax.set_zlabel("Light (SEM similarity)")
ax.set_title("Average Semantic Movement Vectors for ±ΔR, ±ΔG, ±ΔB")
plt.show()

# 5) Also print the raw vectors & norms so you can read the directions numerically
def vec_info(name, v, n):
    if np.isnan(v).any():
        return {"vector": name, "n_steps": n, "vx": np.nan, "vy": np.nan, "vz": np.nan, "norm": np.nan}
    return {"vector": name, "n_steps": n, "vx": float(v[0]), "vy": float(v[1]), "vz": float(v[2]), "norm": float(np.linalg.norm(v))}

vector_table = pd.DataFrame([
    vec_info("+ΔR", vR_up, nRu),
    vec_info("−ΔR", vR_dn, nRd),
    vec_info("+ΔG", vG_up, nGu),
    vec_info("−ΔG", vG_dn, nGd),
    vec_info("+ΔB", vB_up, nBu),
    vec_info("−ΔB", vB_dn, nBd),
])
print("Average semantic movement vectors (components are Δtime_sem, Δspace_sem, Δlight_sem):")
display(vector_table)

# --- SAVE OUTPUTS ---
vector_table.to_csv("semantic_change_vectors.csv", index=False)

print("\nSaved CSVs:")
print("semantic_change_vectors.csv")

# Define ideal axes for Penrose mapping
ideal_axes = {
    "R": np.array([1.0, 0.0, 0.0]),  # R ↔ Time
    "G": np.array([0.0, 1.0, 0.0]),  # G ↔ Space
    "B": np.array([0.0, 0.0, 1.0])   # B ↔ Light
}

# Compute signed projection strength onto the ideal axis
projection_rows = []
for _, row in vector_table.iterrows():
    chan = row['vector'][2]  # Extract 'R', 'G', or 'B'
    v_obs = np.array([row['vx'], row['vy'], row['vz']])
    ideal = ideal_axes[chan]

    if np.isnan(v_obs).any() or np.linalg.norm(v_obs) == 0:
        proj_len = np.nan
        alignment_ratio = np.nan
    else:
        proj_len = np.dot(v_obs, ideal)                 # scalar projection
        alignment_ratio = proj_len / np.linalg.norm(v_obs)  # -1 to +1

    projection_rows.append({
        'vector': row['vector'],
        'n_steps': row['n_steps'],
        'norm': row['norm'],
        'proj_len': proj_len,
        'alignment_ratio': alignment_ratio
    })

projection_df = pd.DataFrame(projection_rows)
print("Curvature + projection alignment analysis:")
display(projection_df)

projection_df.to_csv("curvature_alignment.csv", index=False)
print("\nSaved CSVs:")
print("curvature_alignment.csv")
