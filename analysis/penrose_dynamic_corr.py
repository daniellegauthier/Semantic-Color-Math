"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Penrose dynamic mapping correlations (Red↔Time, Green↔Space, Blue↔Light)

'''

import numpy as np
import pandas as pd


def safe_corr_pearson(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])

def safe_corr_spearman(x, y):
    xs = pd.Series(x, dtype=float)
    ys = pd.Series(y, dtype=float)
    m = xs.notna() & ys.notna()
    if m.sum() < 2:
        return np.nan
    return float(xs[m].rank().corr(ys[m].rank()))

# --- LOAD DATA ---
CSV_PATH = '/data/nltk_color_similarity_summary.csv'
df = pd.read_csv(CSV_PATH)

# Extract needed columns
R = df["R"].to_numpy(dtype=float)
G = df["G"].to_numpy(dtype=float)
B = df["B"].to_numpy(dtype=float)

time_sem  = df["time_sem"].to_numpy(dtype=float)
space_sem = df["space_sem"].to_numpy(dtype=float)
light_sem = df["light_sem"].to_numpy(dtype=float)


# --- Compositional Channels ---
RG   = R + G
GB   = G + B
RB   = R + B
RGB  = R + G + B

# --- Compositional Concepts ---
time_space = time_sem + space_sem
space_light = space_sem + light_sem
time_light  = time_sem + light_sem
all_three   = time_sem + space_sem + light_sem

# --- Compare each composition ---
results = []
compositions = [
    (RG, time_space, "R+G", "Time+Space"),
    (GB, space_light, "G+B", "Space+Light"),
    (RB, time_light, "R+B", "Time+Light"),
    (RGB, all_three, "R+G+B", "All Concepts")
]

for chan_vec, concept_vec, chan_label, concept_label in compositions:
    results.append({
        "channels": chan_label,
        "concepts": concept_label,
        "pearson": safe_corr_pearson(chan_vec, concept_vec),
        "spearman": safe_corr_spearman(chan_vec, concept_vec)
    })

results_df = pd.DataFrame(results)

# --- SAVE OUTPUTS ---
results_df.to_csv("dynamic_correlation.csv", index=False)


print("\nSaved CSVs:")
print("dynamic_correlation.csv")

print("Dynamic / Compositional Mapping Correlations:")
display(results_df)
