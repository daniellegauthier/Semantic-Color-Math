"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Penrose semantic mapping correlations (Red↔Time, Green↔Space, Blue↔Light)

'''

import pandas as pd
import numpy as np

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

# Helper for Spearman without SciPy
def spearmanr_simple(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan, mask.sum()
    xr = x[mask].rank(method="average")
    yr = y[mask].rank(method="average")
    rho = np.corrcoef(xr, yr)[0, 1]
    return float(rho), mask.sum()

# Build correlation table
rows = []
pairs = [("R", "time_sem"), ("G", "space_sem"), ("B", "light_sem")]

for rgb_col, sem_col in pairs:
    mask = summary[rgb_col].notna() & summary[sem_col].notna()
    if mask.sum() > 1:
        pearson = np.corrcoef(summary.loc[mask, rgb_col], summary.loc[mask, sem_col])[0, 1]
        spearman, n = spearmanr_simple(summary[rgb_col], summary[sem_col])
    else:
        pearson, spearman, n = np.nan, np.nan, mask.sum()

    rows.append({
        "rgb_channel": rgb_col,
        "concept": sem_col,
        "pearson": pearson,
        "spearman": spearman,
        "n_pairs": n
    })

sem_corr_df = pd.DataFrame(rows)

# --- SAVE OUTPUTS ---
sem_corr_df.to_csv("semantic_penrose_correlation.csv", index=False)

print("RGB ↔ SEM concept correlations:")
display(sem_corr_df)

print("\nSaved CSVs:")
print("semantic_penrose_correlation.csv")