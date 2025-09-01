"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Variance in random vs. sentiment-weighted sampling.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import random

class LaMatrizCoherenceExploration:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.rgb_colors = self.df[['R', 'G', 'B']].values
        self.color_names = self.df['Closest Color']
        self.continuous_pdf = self._get_continuous_pdf()
        self.discrete_pdf = self._get_discrete_pdf()

    def _get_discrete_pdf(self):
        return self.color_names.value_counts(normalize=True).to_dict()

    def _get_continuous_pdf(self):
        sentiments = self.df.groupby('Closest Color')['Sentiment Score'].sum()
        s_min, s_max = sentiments.min(), sentiments.max()
        scaled = (sentiments - s_min) / (s_max - s_min)
        return (scaled / scaled.sum()).to_dict()

    def _sample_colors(self, strategy, n=50):
        if strategy == "random":
            sampled_indices = np.random.choice(len(self.rgb_colors), size=n)
        else:
            pdf = self.discrete_pdf if strategy == "discrete" else self.continuous_pdf
            color_indices = [i for i, name in enumerate(self.color_names) if name in pdf]
            probs = [pdf[self.color_names[i]] for i in color_indices]
            probs = np.array(probs) / np.sum(probs)
            sampled_indices = np.random.choice(color_indices, size=n, p=probs)

        return self.rgb_colors[sampled_indices]

    def _plot_color_bar(self, colors, ax, title):
        ax.imshow([colors.astype(np.uint8)], aspect='auto')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)

    def _color_variance(self, colors):
        return np.var(colors, axis=0).mean()  # mean variance across R, G, B

    def run_comparison(self, n=50):
        fig, axs = plt.subplots(3, 1, figsize=(10, 3), constrained_layout=True)

        strategies = ['random', 'discrete', 'continuous']
        variances = {}

        for ax, strat in zip(axs, strategies):
            colors = self._sample_colors(strat, n)
            self._plot_color_bar(colors, ax, f"{strat.capitalize()} Sampling")
            variances[strat] = self._color_variance(colors)

        plt.suptitle("Color Sampling Comparison: Random vs Frequency vs Sentiment-Coherent", fontsize=14)
        plt.show()

        print("Color Variance (lower = more coherence):")
        for k, v in variances.items():
            print(f"  {k.capitalize():<10}: {v:.4f}")

explorer = LaMatrizCoherenceExploration('/data/semantic_rgb_mapping_with_sentiment.csv')
explorer.run_comparison(n=50)
