"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Color ↔ Word similarity via color vectors with cosine similarity to identify most meaningful color sentiments using Lab color space

'''

import re, ast, math, itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = '/data/la_matrice_color_word_similarity.csv'


# === HELPERS ===
def norm_cols(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}
    return df, lower

def find_col(lower_map, candidates):
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None

def parse_lab(val):
    """
    Robust parser for Lab strings:
    - '[L, a, b]' or '(L, a, b)'
    - 'np.float32([L, a, b])'
    - 'L a b' or 'L,a,b'
    Returns (L, a, b) or (np.nan, np.nan, np.nan)
    """
    if pd.isna(val): return (np.nan, np.nan, np.nan)
    s = str(val)
    # strip wrappers like 'np.float32(...)'
    s = re.sub(r'^\s*np\.\s*float32\s*\((.*)\)\s*$', r'\1', s)
    # try literal eval
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, (list, tuple)) and len(obj) == 3:
            return tuple(float(x) for x in obj)
    except Exception:
        pass
    # fallback: split by commas/spaces
    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if len(nums) >= 3:
        return (float(nums[0]), float(nums[1]), float(nums[2]))
    return (np.nan, np.nan, np.nan)

def parse_top_terms(val):
    """
    Parse color_word_top_terms into dict(term -> score).
    Accepts:
      - JSON-like list of [("term", score), ...] or [{"term": "...", "score": ...}, ...]
      - 'term1:score1, term2:score2, ...'
    """
    if pd.isna(val): return {}
    s = str(val).strip()
    # try literal eval / json-ish
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            # maybe {"term": score, ...}
            return {str(k): float(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            out = {}
            for item in obj:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    out[str(item[0])] = float(item[1])
                elif isinstance(item, dict):
                    # expect keys like 'term','score'
                    k = item.get('term') or item.get('word') or item.get('token')
                    v = item.get('score') or item.get('sim') or item.get('similarity')
                    if k is not None and v is not None:
                        out[str(k)] = float(v)
            if out:
                return out
    except Exception:
        pass
    # fallback: "term:score" comma-separated
    out = {}
    for part in s.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            k = k.strip()
            try:
                out[k] = float(v.strip())
            except:
                continue
    return out

def cosine(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))

def euclidean(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return float(np.linalg.norm(u - v))

def spearmanr_from_arrays(x, y):
    """
    Spearman via Pearson of ranks (no SciPy dependency).
    Returns (rho, n).
    """
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    mask = x.notna() & y.notna()
    x = x[mask]; y = y[mask]
    if len(x) < 2: return (np.nan, len(x))
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    r = np.corrcoef(xr, yr)[0,1]
    return (float(r), int(len(x)))

# === LOAD ===
df = pd.read_csv(CSV_PATH)
df, lower = norm_cols(df)

# Identify needed columns
lab_col = "Lab"
r_col   = find_col(lower, ["r"])
g_col   = find_col(lower, ["g"])
b_col   = find_col(lower, ["b"])
terms_col = find_col(lower, ["english-words_top_terms", "top_terms", "terms"])
self_sim_col = find_col(lower, ["english-words_similarity"])

# Parse Lab into numeric columns
L_vals, a_vals, b_vals = [], [], []
for v in df[lab_col]:
    L,a_,b_ = parse_lab(v)
    L_vals.append(L); a_vals.append(a_); b_vals.append(b_)
df["L*"] = L_vals; df["a*"] = a_vals; df["b*"] = b_vals

# Parse top-term dicts
term_dicts = [parse_top_terms(v) for v in df[terms_col]]

# === TEST 1: Word similarity summary (from top-term scores) ===
# Build vocabulary across all colors
vocab = sorted({term for d in term_dicts for term in d.keys()})
# Build word vectors (len = |vocab|)
word_mat = np.zeros((len(df), len(vocab)), dtype=float)
for i, d in enumerate(term_dicts):
    for j, term in enumerate(vocab):
        if term in d:
            word_mat[i, j] = d[term]

# Simple summaries per color
word_sum = word_mat.sum(axis=1)
word_mean = np.where((word_mat != 0).sum(axis=1) > 0,
                     word_mat.sum(axis=1) / np.maximum(1, (word_mat != 0).sum(axis=1)),
                     0.0)
word_max = np.where(word_mat.shape[1] > 0, word_mat.max(axis=1), 0.0)

word_summary = pd.DataFrame({
    "word_score_sum": word_sum,
    "word_score_mean_nonzero": word_mean,
    "word_score_max": word_max
}, index=df.index)

# Add color names alongside scores
color_name_col = find_col(lower, ["color", "closest color", "closest_color", "name", "hue"])
if color_name_col is None:
    color_name_col = "_label"
    df[color_name_col] = [f"color_{i}" for i in range(len(df))]

word_summary = pd.concat([df[[color_name_col]], word_summary], axis=1)

print("Word similarity summary (per color):")
print(word_summary.head(20).to_string(index=False))


# === TEST 2: Word–color vector alignment ===
# Pairwise comparisons across colors
pairs = []
for i, j in itertools.combinations(range(len(df)), 2):
    # word cosine
    wcos = cosine(word_mat[i], word_mat[j]) if word_mat.shape[1] > 0 else np.nan
    # Lab cosine + distance
    li = np.array([df.at[i,"L*"], df.at[i,"a*"], df.at[i,"b*"]], dtype=float)
    lj = np.array([df.at[j,"L*"], df.at[j,"a*"], df.at[j,"b*"]], dtype=float)
    lcos = cosine(li, lj)
    ldist = euclidean(li, lj)
    pairs.append((i, j, wcos, lcos, ldist))

pair_df = pd.DataFrame(pairs, columns=["i","j","word_cosine","lab_cosine","lab_distance"])

# Correlate word cosine vs Lab cosine
pear1 = np.corrcoef(pair_df["word_cosine"].dropna(), pair_df["lab_cosine"].dropna())[0,1] if pair_df[["word_cosine","lab_cosine"]].dropna().shape[0] > 1 else np.nan
spr1, n1 = spearmanr_from_arrays(pair_df["word_cosine"], pair_df["lab_cosine"])

# Correlate word cosine vs (-) Lab distance (expect positive if aligned)
pear2 = np.corrcoef(pair_df["word_cosine"].dropna(), (-pair_df["lab_distance"]).dropna())[0,1] if pair_df[["word_cosine","lab_distance"]].dropna().shape[0] > 1 else np.nan
spr2, n2 = spearmanr_from_arrays(pair_df["word_cosine"], -pair_df["lab_distance"])

corr_pairwise = pd.DataFrame([
    {"x":"word_cosine", "y":"lab_cosine", "pearson": pear1, "spearman": spr1, "n": n1},
    {"x":"word_cosine", "y":"-lab_distance", "pearson": pear2, "spearman": spr2, "n": n2},
])

print("\nPairwise alignment correlations:")
display(corr_pairwise)

color_name_col = find_col(
    lower,
    ["color", "closest color", "closest_color", "name", "hue", "english-words", "matrice"]
)
if color_name_col is None:
    # Fall back to a synthetic name if nothing reasonable exists
    color_name_col = "_label"
    df[color_name_col] = [f"color_{i}" for i in range(len(df))]

# 2) Build display colors from RGB if available; fallback to gray
def srgb_tuple(row):
    try:
        r = float(row[r_col]); g = float(row[g_col]); b = float(row[b_col])
        return (max(0,min(1,r/255.0)), max(0,min(1,g/255.0)), max(0,min(1,b/255.0)))
    except Exception:
        return (0.5, 0.5, 0.5)

disp_colors = [srgb_tuple(df.iloc[i]) for i in range(len(df))]

# Self similarity vs channels (per color)
self_series = df[self_sim_col] if self_sim_col else pd.Series([np.nan]*len(df))
corr_self_lab = pd.Series({
    "L*": np.corrcoef(self_series.dropna(), df.loc[self_series.notna(), "L*"])[0,1] if self_series.notna().sum()>1 else np.nan,
    "a*": np.corrcoef(self_series.dropna(), df.loc[self_series.notna(), "a*"])[0,1] if self_series.notna().sum()>1 else np.nan,
    "b*": np.corrcoef(self_series.dropna(), df.loc[self_series.notna(), "b*"])[0,1] if self_series.notna().sum()>1 else np.nan,
})
corr_self_rgb = pd.Series({
    "R":  np.corrcoef(self_series.dropna(), df.loc[self_series.notna(), find_col(lower,["r"])])[0,1] if (self_series.notna().sum()>1 and r_col) else np.nan,
    "G":  np.corrcoef(self_series.dropna(), df.loc[self_series.notna(), find_col(lower,["g"])])[0,1] if (self_series.notna().sum()>1 and g_col) else np.nan,
    "B":  np.corrcoef(self_series.dropna(), df.loc[self_series.notna(), find_col(lower,["b"])])[0,1] if (self_series.notna().sum()>1 and b_col) else np.nan,
})

print("\nSelf word-similarity vs LAB (Pearson):")
display(pd.DataFrame(corr_self_lab, columns=["pearson"]).T)
print("\nSelf word-similarity vs RGB (Pearson):")
display(pd.DataFrame(corr_self_rgb, columns=["pearson"]).T)

def annotate_extremes(ax, x, y, labels, n=8):
    """Annotate top/bottom n points by y magnitude."""
    s = pd.Series(y)
    if s.notna().sum() == 0:
        return
    # indices of largest |y| values (top n)
    idxs = s.abs().sort_values(ascending=False).index[:n]
    for i in idxs:
        if pd.isna(x[i]) or pd.isna(y[i]):
            continue
        ax.annotate(
            str(labels[i]),
            xy=(x[i], y[i]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=9
        )

def annotate_selected(ax, x, y, labels, where_mask, n=10):
    """Annotate up to n points where mask True (e.g., highest/lowest)."""
    idxs = np.where(where_mask)[0][:n]
    for i in idxs:
        if pd.isna(x[i]) or pd.isna(y[i]):
            continue
        ax.annotate(
            str(labels[i]),
            xy=(x[i], y[i]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=9
        )

# Scatter: Word cosine vs Lab cosine (pairwise)
plt.figure(figsize=(7,5))
plt.scatter(pair_df["word_cosine"], pair_df["lab_cosine"])
plt.xlabel("Word cosine")
plt.ylabel("Lab cosine")
plt.title("Pairwise: Word Cosine vs Lab Cosine")
plt.tight_layout()
plt.show()

# Word cosine vs Lab distance (pairwise)
plt.figure(figsize=(7,5))
plt.scatter(pair_df["word_cosine"], pair_df["lab_distance"])
plt.xlabel("Word cosine")
plt.ylabel("Lab distance")
plt.title("Pairwise: Word Cosine vs Lab Distance")
plt.tight_layout()
plt.show()

# Point-wise plots colored by actual RGB and labeled by color name
# Self similarity vs L*, a*, b* 
if self_sim_col:
    for ch in ["L*","a*","b*"]:
        xv = df[ch].to_numpy(dtype=float)
        yv = df[self_sim_col].to_numpy(dtype=float)
        labels = df[color_name_col].astype(str).to_numpy()

        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(xv, yv, c=disp_colors, s=40)
        ax.set_xlabel(ch)
        ax.set_ylabel(self_sim_col)
        ax.set_title(f"Self Word Similarity vs {ch} (colored by RGB; labels = hue)")
        # annotate a few most extreme by |y|
        annotate_extremes(ax, xv, yv, labels, n=10)
        plt.tight_layout()
        plt.show()

# Self similarity vs R,G,B
for ch_col in [r_col, g_col, b_col]:
    if ch_col:
        xv = df[ch_col].to_numpy(dtype=float)
        yv = df[self_sim_col].to_numpy(dtype=float) if self_sim_col else np.full(len(df), np.nan)
        labels = df[color_name_col].astype(str).to_numpy()

        fig, ax = plt.subplots(figsize=(7,5))
        ax.scatter(xv, yv, c=disp_colors, s=40)
        ax.set_xlabel(ch_col)
        ax.set_ylabel(self_sim_col if self_sim_col else "self_similarity (missing)")
        ax.set_title(f"Self Word Similarity vs {ch_col} (colored by RGB; labels = hue)")
        annotate_extremes(ax, xv, yv, labels, n=10)
        plt.tight_layout()
        plt.show()


# pair_df is pairwise across indices
pair_df_out = pair_df.copy()
pair_df_out["color_i"] = pair_df["i"].apply(lambda k: df.at[k, color_name_col])
pair_df_out["color_j"] = pair_df["j"].apply(lambda k: df.at[k, color_name_col])

# Save 
word_summary.to_csv("lab_vector_word_similarity_summary.csv", index=False)
pair_df_out.to_csv("pairwise_word_lab_metrics.csv", index=False)
corr_pairwise.to_csv("pairwise_alignment_correlations.csv", index=False)


print("\nSaved CSVs:")
print("lab_vector_word_similarity_summary.csv")
print("pairwise_word_lab_metrics.csv")
print("pairwise_alignment_correlations.csv")
