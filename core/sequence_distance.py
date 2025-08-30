"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Compute the overall distance for any sequence compared to all color combinations 
for that sequence length.
'''


import pandas as pd
import numpy as np
import networkx as nx
from ipywidgets import interact, Text
from IPython.display import display

# Load data
colors_df = pd.read_csv("/data/la matrice.csv")
sequences_df = pd.read_csv("/data/la matrice sequences.csv")

# Build color-to-RGB dictionary
rgb_map = {
    row['color'].strip().lower(): (int(row['r']), int(row['g']), int(row['b']))
    for _, row in colors_df.iterrows()
}

# Distance function in RGB space
def rgb_distance(c1, c2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

# Generate all valid sequences of 2–8 colors from the dataset
from itertools import permutations

all_colors = list(rgb_map.keys())
length_to_path_lengths = {n: [] for n in range(2, 9)}

print("Precomputing path lengths... (this might take a moment)")

for n in range(2, 9):
    for seq in permutations(all_colors, n):
        total = sum(rgb_distance(rgb_map[seq[i]], rgb_map[seq[i+1]]) for i in range(len(seq)-1))
        length_to_path_lengths[n].append(total)

print("Finished computing all path lengths.")

# Function to get percentile
def get_percentile(user_input):
    input_colors = [c.strip().lower() for c in user_input.split(',')]

    if not (2 <= len(input_colors) <= 8):
        print("Please enter between 2 and 8 colors.")
        return

    if any(c not in rgb_map for c in input_colors):
        print("One or more color names are invalid.")
        return

    path_len = sum(rgb_distance(rgb_map[input_colors[i]], rgb_map[input_colors[i+1]])
                   for i in range(len(input_colors)-1))

    all_lengths = length_to_path_lengths[len(input_colors)]
    percentile = (sum(1 for l in all_lengths if l <= path_len) / len(all_lengths)) * 100

    print(f"\n📊 Path length: {path_len:.2f}")
    print(f"📈 Percentile (lower = shorter path): {percentile:.2f}%")

# Interactive input
interact(get_percentile, user_input=Text(value="blue, purple, gold", description='Colors:'))
