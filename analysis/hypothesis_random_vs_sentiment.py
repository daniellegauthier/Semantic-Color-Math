"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Null hypothesis: random and sentiment-weighted colors produce the same resonance.
Alternative: sentiment-weighted colors or random colors are more resonant.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import random

# --- Load data ---
color_sim = pd.read_csv('/data/color_similarity_rgb.csv')
sequences = pd.read_csv('/data/la matrice sequences.csv')

# Normalize column names and index
color_sim.columns = [str(c).lower() for c in color_sim.columns]
color_sim['color'] = color_sim['color'].str.lower()
color_sim.set_index('color', inplace=True)
similarity_lookup = color_sim.to_dict(orient='index')

# --- Parse and normalize sequences ---
def parse_sequence(seq):
    return [c.strip().lower() for c in str(seq).split(',') if c.strip().lower() in similarity_lookup]

sequences['parsed'] = sequences['sequence'].apply(parse_sequence)

# --- Compute coherence score ---
def sequence_coherence(seq):
    if len(seq) < 2:
        return np.nan
    scores = []
    for i in range(len(seq) - 1):
        c1, c2 = seq[i], seq[i + 1]
        sim = similarity_lookup.get(c1, {}).get(c2, None)
        if sim is not None:
            scores.append(sim)
    return np.mean(scores) if scores else np.nan

sequences['coherence'] = sequences['parsed'].apply(sequence_coherence)

# --- Generate matched random sequences ---
all_colors = list(similarity_lookup.keys())

def generate_random_sequence(length):
    return random.sample(all_colors, length) if length <= len(all_colors) else random.choices(all_colors, k=length)

random_coherences = []
for parsed in sequences['parsed']:
    if len(parsed) < 2:
        continue
    random_seq = generate_random_sequence(len(parsed))
    score = sequence_coherence(random_seq)
    if score is not None:
        random_coherences.append(score)

# --- Filter real scores and run t-test ---
real_coherences = sequences['coherence'].dropna().tolist()
t_stat, p_val = ttest_ind(real_coherences, random_coherences, equal_var=False)

# --- Output results ---
print(f"\n--- Sequence Coherence Test ---")
print(f"Mean Coherence (La Matriz): {np.mean(real_coherences):.3f}")
print(f"Mean Coherence (Random):    {np.mean(random_coherences):.3f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value:     {p_val:.3f}")

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.hist(real_coherences, bins=15, alpha=0.7, label='La Matriz Sequences')
plt.hist(random_coherences, bins=15, alpha=0.7, label='Random Sequences')
plt.axvline(np.mean(real_coherences), color='blue', linestyle='--', label='Mean (La Matriz)')
plt.axvline(np.mean(random_coherences), color='orange', linestyle='--', label='Mean (Random)')
plt.title('Sequence Coherence Distribution')
plt.xlabel('Average Pairwise Similarity')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()
