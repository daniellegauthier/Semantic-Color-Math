"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Find the transitions between colors in an input  color sequence.
'''

import pandas as pd
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

# Load the dataset
file_path = "/data/semantic_rgb_mapping_with_sentiment.csv"
df = pd.read_csv(file_path)

# Create a mapping from RGB tuple to sentiment score
rgb_to_sentiment = {
    (int(row['R']), int(row['G']), int(row['B'])): row['Sentiment Score']
    for _, row in df.iterrows()
}

# Get RGB for a color name
def get_rgb_for_color(color_name):
    color_name = color_name.strip().lower()
    for _, row in df.iterrows():
        if str(row['Closest Color']).strip().lower() == color_name:
            return (int(row['R']), int(row['G']), int(row['B']))
    return None

# Estimate partial derivatives using central difference with fallback visualization
def estimate_partials(rgb):
    deltas = [-10, 10]
    partials = []
    base = rgb_to_sentiment.get(rgb, 0)

    for i in range(3):  # R, G, B
        scores = []
        skipped = 0
        for delta in deltas:
            offset = list(rgb)
            offset[i] = max(0, min(255, offset[i] + delta))
            offset = tuple(offset)
            neighbor = rgb_to_sentiment.get(offset)
            if neighbor is not None:
                scores.append((delta, neighbor))
                print(f"RGB offset {offset}: Sentiment Score = {neighbor}")
            else:
                print(f"RGB offset {offset}: No sentiment score available")
                skipped += 1

        if len(scores) == 2:
            dy = scores[1][1] - scores[0][1]
            dx = scores[1][0] - scores[0][0]
            partials.append(dy / dx)
        else:
            print(f"Skipped {skipped} offsets due to missing sentiment scores for channel index {i}.")
            partials.append(0.0)

    return partials

# Estimate sentiment change using total differential
def total_differential(rgb_start, delta_rgb):
    partials = estimate_partials(rgb_start)
    estimated_change = sum(partials[i] * delta_rgb[i] for i in range(3))
    return estimated_change

# Analyze function for a sequence of color names
def analyze_color_sequence(color_sequence):
    color_names = [c.strip().lower() for c in color_sequence.split(',') if c.strip()]
    rgbs = [get_rgb_for_color(name) for name in color_names]

    if None in rgbs:
        print("One or more color names not found.")
        return

    print("Analyzing sentiment change across sequence:")
    for i in range(len(rgbs) - 1):
        rgb1 = rgbs[i]
        rgb2 = rgbs[i+1]
        delta_rgb = tuple(c2 - c1 for c1, c2 in zip(rgb1, rgb2))
        estimated_change = total_differential(rgb1, delta_rgb)
        print(f"From '{color_names[i]}' RGB {rgb1} to '{color_names[i+1]}' RGB {rgb2}")
        print(f"\u0394R, \u0394G, \u0394B = {delta_rgb}")
        print(f"Estimated Sentiment Change (Total Differential): {estimated_change:.5f}\n")

# Heatmap visualization of sentiment scores with longer resolution
def plot_sentiment_heatmap(channel='R', fixed_g=128, fixed_b=128):
    resolution = 10
    size = 256 // resolution
    values = np.full((size, size), np.nan)

    for i, r in enumerate(range(0, 256, resolution)):
        for j, g in enumerate(range(0, 256, resolution)):
            rgb = (r, g, fixed_b) if channel == 'R' else (fixed_g, r, g)
            score = rgb_to_sentiment.get(rgb, None)
            if score is not None:
                values[j, i] = score

    plt.figure(figsize=(8, 6))
    plt.imshow(values, origin='lower', cmap='coolwarm', extent=[0, 255, 0, 255], aspect='auto')
    plt.title(f"Sentiment Heatmap - Varying {channel} with {'G='+str(fixed_g) if channel=='R' else 'R='+str(fixed_g)}, B={fixed_b}")
    plt.colorbar(label='Sentiment Score')
    plt.xlabel(f"{channel}-axis")
    plt.ylabel("Other Color Axis")
    plt.show()

# Interactive widgets
sequence_input = widgets.Textarea(value='red, orange, yellow, green, blue, indigo, violet', description='Color Sequence:', layout=widgets.Layout(width='500px', height='80px'))
run_button = widgets.Button(description="Analyze Sequence")
heatmap_button = widgets.Button(description="Show Heatmap")

# Run callbacks
def on_click_analyze_sequence(b):
    analyze_color_sequence(sequence_input.value)

def on_click_heatmap(b):
    plot_sentiment_heatmap(channel='R', fixed_g=128, fixed_b=128)

run_button.on_click(on_click_analyze_sequence)
heatmap_button.on_click(on_click_heatmap)

# Display the interactive interface
display(sequence_input, run_button, heatmap_button)
