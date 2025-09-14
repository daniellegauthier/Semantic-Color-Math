"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
NLTK-POWERED Penrose ANCHOR TEST (Red↔Time, Green↔Space, Blue↔Light)

'''

# --- IMPORTS ---
import re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NLTK setup
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- CONFIG ---
EXCEL_PATH = '/data/semantic_rgb_mapping_with_similarity.xlsx'
RGB_COLS = ["r","g","b"]

TIME_SEEDS  = ["time","temporal","duration","interval","era","epoch","moment","history","chronology"]
SPACE_SEEDS = ["space","spatial","distance","dimension","geometry","location","area","room","cosmos","universe"]
LIGHT_SEEDS = ["light","luminance","brightness","illumination","glow","radiance","luminosity","beam","shine"]

WN_EXPAND_DEPTH = 1

# --- HELPERS ---
wnl = WordNetLemmatizer()
stopset = set(stopwords.words("english"))

def normalize_term(term):
    """Lowercase, lemmatize, remove stopwords/punctuation."""
    t = term.lower().strip()
    t = re.sub(r"[_\-]+", " ", t)
    tokens = [w for w in re.findall(r"[A-Za-z]+", t) if w not in stopset]
    if not tokens:
        return set()
    lemmas = set(tokens)
    for w in tokens:
        lemmas.add(wnl.lemmatize(w, pos="n"))
        lemmas.add(wnl.lemmatize(w, pos="v"))
        lemmas.add(wnl.lemmatize(w, pos="a"))
    return {x for x in lemmas if x}

def expand_with_wordnet(seed_words, depth=1):
    expanded = set()
    for w in seed_words:
        expanded |= normalize_term(w)
        for pos in (wn.NOUN, wn.VERB, wn.ADJ):
            for s in wn.synsets(w, pos=pos):
                if depth >= 1:
                    for lemma in s.lemmas():
                        expanded.add(lemma.name().replace("_", " ").lower())
                if depth >= 2:
                    for rel in (s.hypernyms() + s.hyponyms()):
                        for lemma in rel.lemmas():
                            expanded.add(lemma.name().replace("_", " ").lower())
    return expanded

def synsets_for_term(term):
    syns = []
    for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
        syns.extend(wn.synsets(term.replace(" ", "_"), pos=pos))
    return syns

def max_wup_similarity(term, anchor_syns):
    tsyns = synsets_for_term(term)
    if not tsyns or not anchor_syns:
        return np.nan
    best = -1.0
    for t in tsyns:
        for a in anchor_syns:
            sim = t.wup_similarity(a)
            if sim is not None and sim > best:
                best = sim
    return best if best >= 0 else np.nan

def lexical_score(words, anchor_lex):
    if not words: return np.nan
    norms = set()
    for w in words:
        norms |= normalize_term(w)
    return 1.0 if norms & anchor_lex else 0.0

def semantic_score(words, anchor_syns):
    if not words: return np.nan
    sims = []
    for w in words:
        sim = max_wup_similarity(w, anchor_syns)
        if sim is not None and not math.isnan(sim):
            sims.append(sim)
    return float(np.mean(sims)) if sims else np.nan

def safe_corr(x, y, method="pearson"):
    xs, ys = pd.Series(x), pd.Series(y)
    m = xs.notna() & ys.notna()
    if m.sum() < 2: return np.nan
    if method=="pearson":
        return float(xs[m].corr(ys[m], method="pearson"))
    else:
        return float(xs[m].corr(ys[m], method="spearman"))

# --- LOAD DATA ---
df = pd.read_excel(EXCEL_PATH)
lower_map = {c.lower(): c for c in df.columns}

terms_col     = lower_map.get("original words")
sent_col      = lower_map.get("sentiment score")
color_col     = lower_map.get("closest color") or lower_map.get("color") or lower_map.get("name")

R = df[lower_map["r"]].astype(float).to_numpy()
G = df[lower_map["g"]].astype(float).to_numpy()
B = df[lower_map["b"]].astype(float).to_numpy()
sentiment = df[sent_col].astype(float).to_numpy()

# Split words
word_lists = [str(v).split(",") if pd.notna(v) else [] for v in df[terms_col]]

# --- Anchors ---
time_lex  = expand_with_wordnet(TIME_SEEDS,  depth=WN_EXPAND_DEPTH)
space_lex = expand_with_wordnet(SPACE_SEEDS, depth=WN_EXPAND_DEPTH)
light_lex = expand_with_wordnet(LIGHT_SEEDS, depth=WN_EXPAND_DEPTH)

time_syns  = [s for w in TIME_SEEDS+list(time_lex)  for s in synsets_for_term(w)]
space_syns = [s for w in SPACE_SEEDS+list(space_lex) for s in synsets_for_term(w)]
light_syns = [s for w in LIGHT_SEEDS+list(light_lex) for s in synsets_for_term(w)]

# --- Scores ---
time_lex_scores  = np.array([lexical_score(ws, time_lex)  for ws in word_lists], dtype=float)
space_lex_scores = np.array([lexical_score(ws, space_lex) for ws in word_lists], dtype=float)
light_lex_scores = np.array([lexical_score(ws, light_lex) for ws in word_lists], dtype=float)

time_sem_scores  = np.array([semantic_score(ws, time_syns)  for ws in word_lists], dtype=float)
space_sem_scores = np.array([semantic_score(ws, space_syns) for ws in word_lists], dtype=float)
light_sem_scores = np.array([semantic_score(ws, light_syns) for ws in word_lists], dtype=float)

# --- Correlations: Sentiment vs Semantic Anchors ---

rgb_anchor_corrs = pd.DataFrame([
    {"pair": "R vs time_sem",  "pearson": safe_corr(R, time_sem_scores,"pearson"), 
                                "spearman": safe_corr(R, time_sem_scores,"spearman")},
    {"pair": "G vs space_sem", "pearson": safe_corr(G, space_sem_scores,"pearson"), 
                                "spearman": safe_corr(G, space_sem_scores,"spearman")},
    {"pair": "B vs light_sem", "pearson": safe_corr(B, light_sem_scores,"pearson"), 
                                "spearman": safe_corr(B, light_sem_scores,"spearman")},
])
print("\nRGB ↔ Anchor Correlations:")
print(rgb_anchor_corrs)
rgb_anchor_corrs.to_csv("rgb_anchor_correlations.csv", index=False)

# --- Summary ---
summary = pd.DataFrame({
    "Color": df[color_col],
    "R":R,"G":G,"B":B,
    "Sentiment": sentiment,
    "time_lex":time_lex_scores,"space_lex":space_lex_scores,"light_lex":light_lex_scores,
    "time_sem":time_sem_scores,"space_sem":space_sem_scores,"light_sem":light_sem_scores
})
summary.to_csv("nltk_color_sentiment_summary.csv", index=False)

# --- Scatter Plots ---
RGB = np.vstack([R,G,B]).T
labels = df[color_col].astype(str).to_numpy()

def scatter_with_colors(x,y,rgb,labels,xlab,ylab,title):
    plt.figure(figsize=(8,6))
    plt.scatter(x,y,c=rgb/255.0,s=60,edgecolors="k")
    for xi,yi,lab in zip(x,y,labels):
        if not (np.isnan(xi) or np.isnan(yi)):
            plt.text(xi,yi,lab,fontsize=7,ha="left",va="bottom")
    plt.xlabel(xlab); plt.ylabel(ylab); plt.title(title); plt.tight_layout(); plt.show()

# --- RGB ↔ Anchors (direct mapping) ---
pairs = [
    (R, time_sem_scores, "R", "time_sem", "R vs Time (WordNet sim)"),
    (G, space_sem_scores, "G", "space_sem", "G vs Space (WordNet sim)"),
    (B, light_sem_scores, "B", "light_sem", "B vs Light (WordNet sim)"),
]

for (x, y, xl, yl, title) in pairs:
    scatter_with_colors(x, y, RGB, labels, xl, yl, title)

# --- CLUSTER ANALYSIS ---

import seaborn as sns

CLUSTER_INPUT = '/data/la_matrice_clusters.csv'
CLUSTER_OUTPUT = "anchor_cluster_summary.csv"

# Load
clusters_df = pd.read_csv(CLUSTER_INPUT)

# Assume cluster label is in a column called "cluster"
# (adjust this to your file — maybe "ClusterID" or similar)
if "cluster" not in clusters_df.columns:
    raise ValueError("Expected a column named 'cluster' in the cluster CSV.")

# Merge with your semantic summary (from previous code)
# Detect possible color column in clusters_df
possible_color_cols = ["Color", "color", "Closest Color", "closest_color", "hue", "name"]
cluster_color_col = None
for cand in possible_color_cols:
    if cand in clusters_df.columns:
        cluster_color_col = cand
        break

if cluster_color_col is None:
    raise ValueError("Could not find a color identifier column in clusters_df")

# Also check summary
if "Color" not in summary.columns:
    raise ValueError("Expected 'Color' column in summary")

# Merge using aligned columns
merged = clusters_df.merge(summary, left_on=cluster_color_col, right_on="Color", how="left")


# Group by cluster
cluster_summary = merged.groupby("cluster").agg({
    "Sentiment":"mean",
    "R":"mean","G":"mean","B":"mean",
    "time_sem":"mean","space_sem":"mean","light_sem":"mean"
}).reset_index()

# Save
cluster_summary.to_csv(CLUSTER_OUTPUT, index=False)
print(f"Cluster summary saved to {CLUSTER_OUTPUT}")

# --- VISUALIZATIONS ---

# Heatmap of semantic anchors per cluster
plt.figure(figsize=(8,6))
sns.heatmap(cluster_summary[["time_sem","space_sem","light_sem"]], 
            annot=True, cmap="coolwarm", xticklabels=["Time","Space","Light"])
plt.title("Semantic Anchor Means per Cluster")
plt.ylabel("Cluster")
plt.show()

# Scatterplot sentiment vs cluster (jittered)
plt.figure(figsize=(8,6))
sns.stripplot(x="cluster", y="Sentiment", data=merged, jitter=True, palette="Set2")
plt.title("Sentiment Distribution by Cluster")
plt.show()

