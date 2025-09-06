"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Physical Magnetism to Color Sequence Engine

Needs the three CSVs below:
  - semantic_rgb_mapping_with_sentiment.csv
  - la matrice.csv
  - la matrice sequences.csv

Then, in Part 2) Core Generator, adjust "length, seed, mass, voltage, charge, and null_time_bias", 
and enter ""start_color" in the console, to explore the trajectory of a current feeling.
'''

import os
import io
import json
import math
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============
# 1) Load Data
# ===============

# Update these if needed
SEM_PATH = 'data/semantic_rgb_mapping_with_sentiment.csv'
MATRIX_PATH = '/data/la matrice.csv'
SEQ_PATH = '/data/la matrice sequences.csv'

sem = pd.read_csv(SEM_PATH)
mat = pd.read_csv(MATRIX_PATH)
seqs = pd.read_csv(SEQ_PATH)


# Basic cleaning / derived fields

def _parse_words(s: str) -> List[str]:
    if isinstance(s, str):
        parts = [p.strip().lower() for p in s.replace('/', ',').replace(';', ',').split(',')]
        return [p for p in parts if p]
    return []

sem['words'] = sem['Original Words'].apply(_parse_words)
sem['color_name'] = sem['Closest Color'].astype(str).str.strip().str.lower()
sem['sentiment_norm'] = (sem['Sentiment Score'] - sem['Sentiment Score'].mean()) / (sem['Sentiment Score'].std() + 1e-9)

# Origin = closest to neutral sentiment (≈ 0)
origin_idx = (sem['sentiment_norm'].abs()).idxmin()

# Small helpers

def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

# HSV utilities

def _rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx, mn = max(r, g, b), min(r, g, b)
    df = mx - mn
    if df == 0: h = 0
    elif mx == r: h = (60 * ((g - b) / df) + 360) % 360
    elif mx == g: h = (60 * ((b - r) / df) + 120) % 360
    else: h = (60 * ((r - g) / df) + 240) % 360
    s = 0 if mx == 0 else df / mx
    v = mx
    return h, s, v

def _hsv_to_rgb(h, s, v):
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if   0 <= h < 60:  rp, gp, bp = c, x, 0
    elif 60 <= h < 120: rp, gp, bp = x, c, 0
    elif 120 <= h < 180: rp, gp, bp = 0, c, x
    elif 180 <= h < 240: rp, gp, bp = 0, x, c
    elif 240 <= h < 300: rp, gp, bp = x, 0, c
    else:                rp, gp, bp = c, 0, x
    r, g, b = (rp + m), (gp + m), (bp + m)
    return int(round(r*255)), int(round(g*255)), int(round(b*255))

def _apply_mass(rgb: Tuple[int, int, int], mass: float) -> Tuple[int, int, int]:
    h, s, v = _rgb_to_hsv(*rgb)
    s = np.clip(s * (1 + 0.75*mass), 0, 1)
    v = np.clip(v * (1 + 0.50*mass), 0, 1)
    return _hsv_to_rgb(h, s, v)

def _similarity(i: int, j: int, alpha: float=0.6) -> float:
    ds = abs(sem.at[i, 'sentiment_norm'] - sem.at[j, 'sentiment_norm'])
    sent_sim = 1 - (ds / (abs(sem['sentiment_norm']).max() + 1e-9))
    word_sim = _jaccard(sem.at[i,'words'], sem.at[j,'words'])
    return float(alpha*sent_sim + (1-alpha)*word_sim)

def _choose_next(current_idx: int, charge: float, rng: np.random.Generator) -> int:
    sims = np.array([_similarity(current_idx, j) for j in range(len(sem))], dtype=float)
    dist = 1 - sims
    with np.errstate(divide='ignore'):
        attraction = charge / np.maximum(dist, 1e-3)
    attraction[current_idx] = 0
    w = np.maximum(attraction, 0)
    if w.sum() == 0:
        return int(rng.integers(0, len(sem)))
    probs = w / w.sum()
    return int(rng.choice(len(sem), p=probs))

def _neutral_indices():
    candidates = {'grey','gray','white','nude','beige'}
    idx = [i for i,c in enumerate(sem['color_name']) if isinstance(c,str) and c.lower() in candidates]
    if not idx:
        order = np.argsort(np.abs(sem['sentiment_norm'].values))
        idx = order[:5].tolist()
    return idx

def _narrative_for(name: str) -> str:
    row = mat[mat['color'].astype(str).str.lower()==str(name).lower()].head(1)
    if row.empty:
        return ''
    parts = [str(row['matrice1'].values[0]), str(row['matrice'].values[0]), str(row['english-words-code'].values[0])]
    return ", ".join([p for p in parts if p and p != 'nan'])

# =============================
# 2) Core Generator (algorithm for physical noise to color noise)
# =============================

def generate_sequence(length: int=9, seed: int=9, mass: float=0.6, voltage: float=0.8, charge: float=1.2,
                      null_time_bias: float=0.7, start_color: str|None=None) -> List[Dict[str,Any]]:
    rng = np.random.default_rng(seed)
    if start_color:
        cands = sem[sem['color_name'] == str(start_color).strip().lower()]
        start_idx = int(cands.index[0]) if len(cands) else int(origin_idx)
    else:
        start_idx = int(origin_idx)

    seq: List[Dict[str,Any]] = []
    current_idx = start_idx
    neutrals = _neutral_indices()

    for t in range(length):
        row = sem.loc[current_idx]
        rgb = (int(row['R']), int(row['G']), int(row['B']))
        rgb_mass = _apply_mass(rgb, mass)
        item = {
            't': t,
            'name': row['color_name'],
            'rgb': rgb_mass,
            'base_rgb': rgb,
            'sentiment': float(row['Sentiment Score']),
            'words': row['words'],
            'narrative': _narrative_for(row['color_name'])
        }
        seq.append(item)

        magnitude = abs(row['sentiment_norm'])
        p_rest = null_time_bias * (1 - magnitude)
        if rng.random() < p_rest:
            ni = int(rng.choice(neutrals))
            nrow = sem.loc[ni]
            nrgb = (int(nrow['R']), int(nrow['G']), int(nrow['B']))
            seq.append({
                't': t + 0.5,
                'name': nrow['color_name'],
                'rgb': _apply_mass(nrgb, mass*0.2),
                'base_rgb': nrgb,
                'sentiment': float(nrow['Sentiment Score']),
                'words': nrow['words'],
                'narrative': _narrative_for(nrow['color_name']) + ' (rest)'
            })

        current_idx = _choose_next(current_idx, charge*(1+0.4*voltage), rng)

    return seq

# =========================
# 3) Visualization + Export
# =========================

def visualize_sequence(seq: List[Dict[str,Any]], title: str='Magnetism‑to‑Color Sequence'):
    n = len(seq)
    h = 40
    img = np.zeros((h, n, 3), dtype=np.float32)
    for i, item in enumerate(seq):
        r, g, b = item['rgb']
        img[:, i, 0] = r/255.0
        img[:, i, 1] = g/255.0
        img[:, i, 2] = b/255.0
    plt.figure(figsize=(max(6, n/3), 2))
    plt.imshow(img, aspect='auto')
    plt.axis('off')
    plt.title(title)
    plt.show()

def sequence_dataframe(seq: List[Dict[str,Any]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'step': i,
            't': item['t'],
            'color': item['name'],
            'rgb': item['rgb'],
            'sentiment': item['sentiment'],
            'narrative': item['narrative']
        } for i, item in enumerate(seq)
    ])

def export_json(seq: List[Dict[str,Any]]) -> str:
    payload = {
        'version': 1,
        'engine': 'magnetism-to-color',
        'sequence': seq,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

# =====================
# 4) CLI Interaction
# =====================

if __name__ == "__main__":
    # Configurable CLI values
    mass = float(input("Mass: "))
    voltage = float(input("Voltage: "))
    charge = float(input("Charge (Magnetism): "))
    null_time_bias = float(input("Null Time: "))
    length = int(input("Sequence Length: "))
    seed = int(input("Seed: "))
    start_color = input("Start Color (or leave blank): ").strip() or None

    sequence = generate_sequence(length, seed, mass, voltage, charge, null_time_bias, start_color)
    visualize_sequence(sequence)
    df = sequence_dataframe(sequence)
    print(df.head(10))

    print("\nNarrative Overlay:")
    for i, s in enumerate(sequence[:20]):
        print(f"{i:02d} — {s['name']} — {s['narrative']}")

    # Export option
    save_json = input("\nExport sequence to JSON file? (y/n): ").strip().lower()
    if save_json == 'y':
        with open("magnetism_sequence_output.json", "w", encoding='utf-8') as f:
            f.write(export_json(sequence))
        print("Saved to magnetism_sequence_output.json")
