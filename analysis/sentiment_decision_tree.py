"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Decision tree of color sentiment with MSE.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from ipywidgets import interact, Dropdown, Output, VBox, HBox, Button, Label
from IPython.display import display

# Load data
semantic_df = pd.read_csv('/Users/daniellegauthier/Library/Mobile Documents/com~apple~CloudDocs/Document/premordial altar/architect 🧱📐🌐🔭/la_matriz_data/semantic_rgb_mapping_with_sentiment.csv')

# Preprocess
semantic_df.dropna(subset=['Sentiment Score'], inplace=True)
semantic_df['Color'] = semantic_df['Closest Color'].str.lower()

# Setup decision tree
X = semantic_df[['R', 'G', 'B']]
y = semantic_df['Sentiment Score']
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)

# Visualization Output
output = Output()

# Dropdown for color selection
color_dropdown = Dropdown(
    options=sorted(semantic_df['Color'].unique()),
    description='Color:',
    layout={'width': '300px'}
)

# Function to analyze color with tree prediction
@output.capture()
def on_color_selected(color):
    color_data = semantic_df[semantic_df['Color'] == color]
    if color_data.empty:
        print(f"❌ No data found for color: {color}")
        return

    print(f"\n🎨 Analyzing color: {color.title()}")
    for _, row in color_data.iterrows():
        r, g, b = row['R'], row['G'], row['B']
        predicted = tree.predict([[r, g, b]])[0]
        print(f"  RGB({int(r)}, {int(g)}, {int(b)}) → Predicted Sentiment: {predicted:.3f} | Actual: {row['Sentiment Score']:.3f}")

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_tree(tree, feature_names=['R', 'G', 'B'], filled=True, ax=ax)
    plt.title("Decision Tree - RGB to Sentiment")
    plt.show()

interact(on_color_selected, color=color_dropdown)
display(output)
