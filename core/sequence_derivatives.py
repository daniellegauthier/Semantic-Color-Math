"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Compute partial derivatives for any color sequence to determine how hard the journey is.
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, Text

# Load data
colors_df = pd.read_csv("/data/la matrice.csv")

# Build RGB map
rgb_map = {
    row['color'].strip().lower(): (row['r'], row['g'], row['b'])
    for _, row in colors_df.iterrows()
}

# Compute partial derivatives using finite differences
# Partial derivative with respect to red (time)
def partial_derivative_r(f, r, g, b, h=1e-3):
    return (f(r + h, g, b) - f(r, g, b)) / h

# Partial derivative with respect to green (space)
def partial_derivative_g(f, r, g, b, h=1e-3):
    return (f(r, g + h, b) - f(r, g, b)) / h

# Partial derivative with respect to blue (light)
def partial_derivative_b(f, r, g, b, h=1e-3):
    return (f(r, g, b + h) - f(r, g, b)) / h

# Function to analyze a sequence of color names
def analyze_color_input(color_input):
    color_names = [c.strip().lower() for c in color_input.split(',')]
    rgb_values = [rgb_map[c] for c in color_names if c in rgb_map]

    if not rgb_values:
        print("No valid colors found in input.")
        return

    rgb_array = np.array(rgb_values, dtype=float)

    # Define a mock function to simulate semantic value from RGB
    def semantic_function(r, g, b):
        return r**2 + g**2 + b**2  # Replace with real semantic function if available

    # Compute partials
    partials_r = [partial_derivative_r(semantic_function, *rgb) for rgb in rgb_array]
    partials_g = [partial_derivative_g(semantic_function, *rgb) for rgb in rgb_array]
    partials_b = [partial_derivative_b(semantic_function, *rgb) for rgb in rgb_array]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = list(range(len(rgb_values)))
    ax.plot(x, partials_r, label='∂Time (Red)', color='red')
    ax.plot(x, partials_g, label='∂Space (Green)', color='green')
    ax.plot(x, partials_b, label='∂Light (Blue)', color='blue')
    ax.set_title(f"Partial Derivatives for Input Colors")
    ax.set_xlabel("Step")
    ax.set_ylabel("Change Rate")
    ax.legend()
    plt.grid(True)
    plt.show()

# Create interactive text input
interact(analyze_color_input, color_input=Text(value='red, blue, green', description='Colors:'))
