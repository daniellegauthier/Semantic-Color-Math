"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
NLTK-POWERED Penrose ANCHOR TEST (Red↔Time, Green↔Space, Blue↔Light)

'''

# --- INSTALL / IMPORTS ---
import sys, re, ast, itertools, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NLTK setup
import nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- CONFIG ---
CSV_PATH = '/data/la_matrice_with_color_word_similarity.csv'
TOP_TERMS_COL_CANDIDATES = ["color_word_top_terms", "top_terms", "terms"]
RGB_COLS = ["r", "g", "b"]

# Base seed words for the anchors
TIME_SEEDS  = ["time", "temporal", "duration", "interval", "era", "epoch", "moment", "history", "chronology"]
SPACE_SEEDS = ["space", "spatial", "distance", "dimension", "geometry", "location", "area", "room", "cosmos", "universe"]
LIGHT_SEEDS = ["light", "luminance", "brightness", "illumination", "glow", "radiance", "luminosity", "beam", "shine"]

# Depth for WordNet expansion (0 = just seeds; 1 = include synsets' lemmas; 2 = also include hypernyms/hyponyms lemmas)
WN_EXPAND_DEPTH = 1

# --- HELPERS ---

def parse_top_terms(val):
    """
    Parse the color_word_top_terms column into dict(term -> score).
    Handles python-literal-ish lists/dicts and 'term:score' strings.
    """
    if pd.isna(val):
        return {}
    s = str(val).strip()
    # Try literal eval
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return {str(k).lower(): float(v) for k, v in obj.items() if _is_num(v)}
        if isinstance(obj, (list, tuple)):
            out = {}
            for item in obj:
                if isinstance(item, (list, tuple)) and len(item) >= 2 and _is_num(item[1]):
                    out[str(item[0]).lower()] = float(item[1])
                elif isinstance(item, dict):
                    k = item.get("term") or item.get("word") or item.get("token")
                    v = item.get("score") or item.get("sim") or item.get("similarity")
                    if k is not None and _is_num(v):
                        out[str(k).lower()] = float(v)
            return out
    except Exception:
        pass
    # Fallback "term:score, term:score"
    out = {}
    for part in s.split(","):
        if ":" in part:
            k, v = part.split(":", 1)
            try:
                out[k.strip().lower()] = float(v.strip())
            except:
                pass
    return out

def _is_num(x):
    try:
        _ = float(x); return True
    except:
        return False

def normalize_term(term, wnl=None, lang_stop=None):
    """
    Normalize a term: lowercase, lemmatize (noun/verb/adj), drop stopwords/punctuation.
    If the term has spaces, we keep both the whole phrase and its head lemma.
    """
    if wnl is None: wnl = WordNetLemmatizer()
    if lang_stop is None: lang_stop = set(stopwords.words("english"))
    t = term.lower().strip()
    t = re.sub(r"[_\-]+", " ", t)
    tokens = [w for w in re.findall(r"[A-Za-z]+", t) if w not in lang_stop]
    if not tokens:
        return set()
    lemmas = set()
    # keep original (joined) form too, if multiword
    if len(tokens) > 1:
        lemmas.add(" ".join(tokens))
    # lemmatize each token with multiple POS guesses
    for w in tokens:
        lemmas.add(wnl.lemmatize(w, pos="n"))
        lemmas.add(wnl.lemmatize(w, pos="v"))
        lemmas.add(wnl.lemmatize(w, pos="a"))
    # Also keep single-token forms
    lemmas.update(tokens)
    # Clean empties
    lemmas = {x for x in lemmas if x}
    return lemmas

def expand_with_wordnet(seed_words, depth=1):
    """
    Expand a set of seed words using WordNet:
    depth=0: just lemmas of the seeds
    depth=1: include lemmas from seed synsets
    depth=2: also include lemmas from hypernyms/hyponyms of seed synsets
    """
    wnl = WordNetLemmatizer()
    expanded = set()
    base = set()
    for w in seed_words:
        base |= normalize_term(w, wnl)
        for pos in (wn.NOUN, wn.VERB, wn.ADJ):
            for s in wn.synsets(w, pos=pos):
                if depth >= 1:
                    for lemma in s.lemmas():
                        expanded.add(lemma.name().replace("_", " ").lower())
                if depth >= 2:
                    for rel in (s.hypernyms() + s.hyponyms()):
                        for lemma in rel.lemmas():
                            expanded.add(lemma.name().replace("_", " ").lower())
    return base | expanded

def synsets_for_term(term):
    """
    Get synsets for a (possibly multiword) term across several POS.
    """
    term = term.replace(" ", "_")
    syns = []
    for pos in (wn.NOUN, wn.VERB, wn.ADJ, wn.ADV):
        syns.extend(wn.synsets(term, pos=pos))
    return syns

def max_wup_similarity(term, anchor_syns):
    """
    Max Wu-Palmer similarity between any synset of term and any anchor synset.
    Returns NaN if no synset available on either side.
    """
    tsyns = synsets_for_term(term)
    if not tsyns or not anchor_syns:
        return np.nan
    best = -1.0
    for t in tsyns:
        for a in anchor_syns:
            sim = t.wup_similarity(a)
            if sim is not None and sim > best:
                best = sim
    return best if best >= 0.0 else np.nan

def safe_corr_pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 2: return np.nan
    return float(np.corrcoef(x[m], y[m])[0,1])

def safe_corr_spearman(x, y):
    xs = pd.Series(x, dtype=float); ys = pd.Series(y, dtype=float)
    m = xs.notna() & ys.notna()
    if m.sum() < 2: return np.nan
    return float(xs[m].rank().corr(ys[m].rank()))

# --- LOAD DATA ---
df = pd.read_csv(CSV_PATH)
lower_map = {c.lower(): c for c in df.columns}

# find top-terms column
terms_col = None
for c in TOP_TERMS_COL_CANDIDATES:
    if c in lower_map:
        terms_col = lower_map[c]
        break
if terms_col is None:
    raise ValueError(f"Could not find any of {TOP_TERMS_COL_CANDIDATES} in the CSV.")

# find RGB columns (case-insensitive)
missing_rgb = [c for c in RGB_COLS if c not in lower_map]
if missing_rgb:
    raise ValueError("Expected R, G, B columns in the CSV (any case).")
R = df[lower_map["r"]].to_numpy(dtype=float)
G = df[lower_map["g"]].to_numpy(dtype=float)
B = df[lower_map["b"]].to_numpy(dtype=float)

# parse top terms into dicts
term_dicts = [parse_top_terms(v) for v in df[terms_col]]

# --- BUILD NLTK-EXPANDED ANCHORS ---
time_lex  = expand_with_wordnet(TIME_SEEDS,  depth=WN_EXPAND_DEPTH)
space_lex = expand_with_wordnet(SPACE_SEEDS, depth=WN_EXPAND_DEPTH)
light_lex = expand_with_wordnet(LIGHT_SEEDS, depth=WN_EXPAND_DEPTH)

# Anchor synsets for semantic similarity
time_syns  = [s for w in TIME_SEEDS  for s in synsets_for_term(w)] + [s for w in time_lex  for s in synsets_for_term(w)]
space_syns = [s for w in SPACE_SEEDS for s in synsets_for_term(w)] + [s for w in space_lex for s in synsets_for_term(w)]
light_syns = [s for w in LIGHT_SEEDS for s in synsets_for_term(w)] + [s for w in light_lex for s in synsets_for_term(w)]

# Deduplicate synsets
def dedup_syns(syns):
    seen, uniq = set(), []
    for s in syns:
        if s.name() not in seen:
            seen.add(s.name()); uniq.append(s)
    return uniq

time_syns  = dedup_syns(time_syns)
space_syns = dedup_syns(space_syns)
light_syns = dedup_syns(light_syns)

# --- COMPUTE PER-COLOR CONCEPT SCORES ---
# We do two scoring modes:
# (A) LEXICAL OVERLAP: lemmatized-term overlap with expanded lexicon (weighted by term score)
# (B) WORDNET SIMILARITY: max Wu-Palmer similarity from each term to anchor synsets (weighted)

wnl = WordNetLemmatizer()
stopset = set(stopwords.words("english"))

def lexical_score(term_dict, anchor_lex):
    """
    Weighted lexical overlap: sum(score * indicator(term in anchor_lex_normalized)) / sum(scores)
    Term normalization uses lemmatization and stopword removal.
    """
    if not term_dict:
        return np.nan
    num = 0.0
    den = 0.0
    for term, score in term_dict.items():
        den += float(score)
        norms = normalize_term(term, wnl, stopset)
        if norms & anchor_lex:  # set intersection non-empty
            num += float(score)
    return (num / den) if den > 0 else np.nan

def semantic_score(term_dict, anchor_syns):
    """
    Weighted semantic similarity via WordNet Wu-Palmer:
    For each term, take max_wup_similarity(term, anchor_syns) and average weighted by score.
    """
    if not term_dict:
        return np.nan
    vals = []
    weights = []
    for term, score in term_dict.items():
        sim = max_wup_similarity(term, anchor_syns)
        if not math.isnan(sim) and sim is not None:
            vals.append(float(sim))
            weights.append(float(score))
    if not weights:
        return np.nan
    vals = np.array(vals, dtype=float)
    weights = np.array(weights, dtype=float)
    return float(np.average(vals, weights=weights))

N = len(term_dicts)
time_lex_scores  = np.array([lexical_score(d, time_lex)  for d in term_dicts], dtype=float)
space_lex_scores = np.array([lexical_score(d, space_lex) for d in term_dicts], dtype=float)
light_lex_scores = np.array([lexical_score(d, light_lex) for d in term_dicts], dtype=float)

time_sem_scores  = np.array([semantic_score(d, time_syns)  for d in term_dicts], dtype=float)
space_sem_scores = np.array([semantic_score(d, space_syns) for d in term_dicts], dtype=float)
light_sem_scores = np.array([semantic_score(d, light_syns) for d in term_dicts], dtype=float)

# --- CORRELATIONS: R↔time, G↔space, B↔light ---
def corr_table(R, G, B, t, s, l, label):
    return pd.DataFrame([
        {"type": label, "pair": "R vs time",  "pearson": safe_corr_pearson(R, t), "spearman": safe_corr_spearman(R, t)},
        {"type": label, "pair": "G vs space", "pearson": safe_corr_pearson(G, s), "spearman": safe_corr_spearman(G, s)},
        {"type": label, "pair": "B vs light", "pearson": safe_corr_pearson(B, l), "spearman": safe_corr_spearman(B, l)},
    ])

corr_lex = corr_table(R, G, B, time_lex_scores,  space_lex_scores,  light_lex_scores,  "NLTK Lexical Overlap")
corr_sem = corr_table(R, G, B, time_sem_scores,  space_sem_scores,  light_sem_scores,  "NLTK WordNet Similarity")
corr_all = pd.concat([corr_lex, corr_sem], ignore_index=True)

print("Channel–Concept Correlations (NLTK-based):")
display(corr_all)

# --- DOMINANCE AGREEMENT (WordNet similarity mode) ---
rgb_stack = np.vstack([R, G, B]).T
anchor_stack_sem = np.vstack([time_sem_scores, space_sem_scores, light_sem_scores]).T
valid = np.any(~np.isnan(anchor_stack_sem), axis=1)

# safe argmax by filling NaNs with -inf, then only compute agreement on valid rows
safe = anchor_stack_sem.copy()
safe[np.isnan(safe)] = -np.inf
rgb_argmax  = np.argmax(rgb_stack, axis=1)          # 0=R,1=G,2=B
anch_argmax = np.argmax(safe, axis=1)               # 0=time,1=space,2=light

agreement = float(np.mean(rgb_argmax[valid] == anch_argmax[valid])) if valid.sum() > 0 else np.nan
print(f"\nDominance Agreement (WordNet similarity): {agreement:.3f} on {valid.sum()}/{N} valid rows")

# --- PER-COLOR SUMMARY TABLE ---
color_col = None
for cand in ["color", "name", "closest color", "closest_color", "hue"]:
    if cand in lower_map:
        color_col = lower_map[cand]
        break
if color_col is None:
    # fallback to an index label if no color name available
    df["__color__"] = [f"color_{i}" for i in range(len(df))]
    color_col = "__color__"

# --- PER-COLOR SUMMARY TABLE WITH COLOR NAME ---
summary = pd.DataFrame({
    "Color": df[color_col],
    "R": R, "G": G, "B": B,
    "time_lex":  time_lex_scores,
    "space_lex": space_lex_scores,
    "light_lex": light_lex_scores,
    "time_sem":  time_sem_scores,
    "space_sem": space_sem_scores,
    "light_sem": light_sem_scores,
})

print("\nPer-color NLTK similarity summary (first 15 rows):")
display(summary.head(15))

# --- SAVE OUTPUTS ---
summary.to_csv("nltk_color_similarity_summary.csv", index=False)
corr_all.to_csv("nltk_channel_concept_correlations.csv", index=False)
pd.DataFrame([{
    "agreement": agreement,
    "valid_rows": int(valid.sum()),
    "total_rows": N
}]).to_csv("nltk_dominance_agreement.csv", index=False)

print("\nSaved CSVs:")
print("  nltk_color_similarity_summary.csv")
print("  nltk_channel_concept_correlations.csv")
print("  nltk_dominance_agreement.csv")


# --- SCATTERS (one chart per figure= ---
# --- COLORIZED SCATTERS WITH ANNOTATIONS ---
def scatter_with_colors(x, y, rgb, labels, xlabel, ylabel, title):
    rgb_norm = rgb / 255.0  # scale to 0–1 for matplotlib
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, c=rgb_norm, s=60, edgecolors="k")

    # annotate each point with its label
    for xi, yi, lab in zip(x, y, labels):
        if not (np.isnan(xi) or np.isnan(yi)):
            plt.text(xi, yi, str(lab), fontsize=7, ha="left", va="bottom")

    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Build RGB array for coloring
RGB = np.vstack([R, G, B]).T

# Use the color name column for labels
labels = df[color_col].astype(str).to_numpy()

# Now plot annotated scatters
for (x, y, xl, yl, title) in [
    (R, time_sem_scores, "R", "time_sem", "R vs Time (WordNet sim)"),
    (G, space_sem_scores, "G", "space_sem", "G vs Space (WordNet sim)"),
    (B, light_sem_scores, "B", "light_sem", "B vs Light (WordNet sim)"),
]:
    scatter_with_colors(x, y, RGB, labels, xl, yl, title)

