"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Compute La Matriz color sequence integration (accumulation of color values).
'''

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Safe integral computation using numpy.trapezoid (modern replacement)
def compute_integral(y_vals, x_vals):
    if len(y_vals) != len(x_vals) or len(y_vals) < 2:
        return 0.0  # Not enough or mismatched points to integrate
    return np.trapezoid(y_vals, x_vals)

# Full color data extracted from your uploaded CSV
color_lookup = {
    'gold': {'r': [250], 'g': [200], 'b': [0]},
    'orange': {'r': [250], 'g': [110], 'b': [0]},
    'yellow': {'r': [255], 'g': [255], 'b': [0]},
    'green': {'r': [0], 'g': [255], 'b': [0]},
    'blue': {'r': [0], 'g': [0], 'b': [255]},
    'brown': {'r': [180], 'g': [50], 'b': [0]},
    'nude': {'r': [250], 'g': [180], 'b': [120]},
    'white': {'r': [255], 'g': [255], 'b': [255]},
    'purple': {'r': [180], 'g': [50], 'b': [255]},
    'grey': {'r': [170], 'g': [170], 'b': [170]},
    'red': {'r': [255], 'g': [0], 'b': [0]},
    'pink': {'r': [250], 'g': [0], 'b': [90]},
    'black': {'r': [0], 'g': [0], 'b': [0]}
}

sequences = {
    'plot': ['grey', 'pink', 'gold', 'nude', 'orange'],
    'knot': ['white', 'blue', 'green', 'red', 'black', 'brown', 'yellow', 'purple'],
    'pain': ['gold', 'orange'],
    'practical': ['yellow', 'green'],
    'spiritual': ['blue', 'brown'],
    'prayer': ['nude', 'white'],
    'sad': ['purple', 'grey', 'red'],
    'precise': ['pink', 'black'],
    'fem': ['brown', 'gold', 'orange', 'pink'],
    'masc': ['red', 'blue', 'orange'],
    'direct': ['red', 'orange']
}

# Function to process and plot a sequence
def process_sequence(seq_name):
    colors = sequences.get(seq_name, [])
    if not colors:
        print(f"Sequence '{seq_name}' not found.")
        return

    r_vals, g_vals, b_vals = [], [], []
    positions = list(range(len(colors)))

    for color in colors:
        color_lower = color.lower()
        rgb = color_lookup.get(color_lower)
        if rgb:
            r_vals.append(rgb['r'])
            g_vals.append(rgb['g'])
            b_vals.append(rgb['b'])
        else:
            print(f"Warning: Color '{color}' not found in lookup.")
            r_vals.append(0)
            g_vals.append(0)
            b_vals.append(0)

    if len(positions) != len(r_vals) or len(positions) < 2:
        print(f"Not enough valid data points in '{seq_name}' to compute an integral.")
        return

    # Plot RGB curves
    plt.figure()
    plt.plot(positions, r_vals, label='R')
    plt.plot(positions, g_vals, label='G')
    plt.plot(positions, b_vals, label='B')
    plt.title(f"Color Sequence: {seq_name}")
    plt.xlabel("Position")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

    # Compute integrals
    r_integral = compute_integral(r_vals, positions)
    g_integral = compute_integral(g_vals, positions)
    b_integral = compute_integral(b_vals, positions)

    print(f"\nIntegrals (Accumulated Weights) for '{seq_name}':")
    print(f"R: {r_integral:.2f}")
    print(f"G: {g_integral:.2f}")
    print(f"B: {b_integral:.2f}")

# Interactive widget
sequence_dropdown = widgets.Dropdown(
    options=list(sequences.keys()),
    description='Select Sequence:'
)

widgets.interact(process_sequence, seq_name=sequence_dropdown)
