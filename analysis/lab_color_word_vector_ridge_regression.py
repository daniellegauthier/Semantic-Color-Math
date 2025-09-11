"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Color ↔ Word similarity via color vectors with ridge regression to identify most predictable color sentiments using Lab color space

'''

import pandas as pd, numpy as np, re, json
import spacy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from skimage.color import rgb2lab

CSV_PATH = '/data/la matrice.csv'  # make sure this file exists in your runtime

# --- load data ---
df = pd.read_csv(CSV_PATH)
for col in ['color','r','g','b','matrice']:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# --- spacy model with vectors ---
nlp = spacy.load("en_core_web_md")  # has 300-d vectors

# --- helpers ---
def toks(s: str):
    if not isinstance(s,str): return []
    return [t for t in re.findall(r"[A-Za-z'][A-Za-z']+", s.lower()) if len(t) > 2]

def phrase_vec(phrase: str) -> np.ndarray:
    """Average token vectors (ignore OOV tokens with zero vectors)."""
    doc = nlp(phrase or "")
    vecs = [t.vector for t in doc if t.has_vector and t.vector_norm > 0]
    if not vecs:
        return np.zeros(nlp.vocab.vectors_length, dtype=np.float32)
    return np.mean(vecs, axis=0)

def rgb_to_lab_norm(r,g,b):
    arr = np.array([[[r/255.0, g/255.0, b/255.0]]], dtype=np.float32)
    L,a,b = rgb2lab(arr)[0,0]
    return np.array([L, a/128.0, b/128.0], dtype=np.float32)

def l2rows(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    return A / n

# --- preprocess ---
df['matrice_clean'] = df['matrice'].astype(str).str.strip().str.lower()
df['tokens'] = df['matrice_clean'].apply(toks)

# phrase vectors for each row
print("Embedding matrice phrases with spaCy…")
mat_vecs = np.stack([phrase_vec(p) for p in df['matrice_clean']])

# Lab features for each color
print("Converting RGB → Lab…")
lab_feats = np.stack([rgb_to_lab_norm(float(r), float(g), float(b))
                      for r,g,b in zip(df['r'], df['g'], df['b'])])

# --- learn mapping Lab -> text space ---
print("Fitting Lab→Text mapping…")
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0, random_state=42))
model.fit(lab_feats, mat_vecs)

# project colors into text space
pred_vecs = model.predict(lab_feats)

# self similarity (predicted vs own phrase vector)
mat_vecs_n = l2rows(mat_vecs)
pred_vecs_n = l2rows(pred_vecs)
self_sim = np.sum(mat_vecs_n * pred_vecs_n, axis=1)  # cosine

# build a vocab of unique matrice terms (single tokens) with vectors
print("Building term vocabulary…")
all_terms = sorted({w for row in df['tokens'] for w in row})
term_vecs = []
term_list = []
for w in all_terms:
    tok = nlp.vocab[w]
    if tok.has_vector and tok.vector_norm > 0:
        term_vecs.append(tok.vector)
        term_list.append(w)
if not term_list:
    # fallback to phrases if no single tokens had vectors
    term_list = df['matrice_clean'].tolist()
    term_vecs = [phrase_vec(p) for p in term_list]

term_mat = np.stack(term_vecs).astype(np.float32)
term_mat_n = l2rows(term_mat)

def top_k_terms_for_vec(vec, k=6):
    sims = cosine_similarity(vec.reshape(1,-1), term_mat_n)[0]
    idx = np.argsort(-sims)[:k]
    return [(term_list[i], float(sims[i])) for i in idx]

print("Scoring nearest terms…")
topk_terms = [top_k_terms_for_vec(pred_vecs_n[i], k=6) for i in range(len(df))]

# --- output ---
df_out = df.copy()
df_out['Lab'] = [tuple(x) for x in lab_feats]
df_out['color_word_similarity_self'] = np.round(self_sim, 4)
df_out['color_word_top_terms'] = [json.dumps([(w, round(s,4)) for w,s in lst]) for lst in topk_terms]
df_out['Top Terms (pretty)'] = [
    "; ".join([f"{w}:{s:.2f}" for w,s in lst]) for lst in topk_terms
]

save_path = "la_matrice_with_color_word_similarity.csv"
df_out.to_csv(save_path, index=False)
print(f"Saved: {save_path}")
print(df_out[['color','matrice_clean','color_word_similarity_self','Top Terms (pretty)']].head(10))
