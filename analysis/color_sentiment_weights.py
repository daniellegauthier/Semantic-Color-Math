"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Refine probability weights for each color sentiment (discrete + continuous models).
'''

import matplotlib.pyplot as plt
import pandas as pd

class ColorSentimentWeights:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.discrete_weights = self._calculate_discrete_weights()
        self.continuous_weights = self._calculate_continuous_weights()

    def _calculate_discrete_weights(self):
        color_counts = self.df['Closest Color'].value_counts(normalize=True)
        return color_counts.to_dict()

    def _calculate_continuous_weights(self):
        sentiment_sums = self.df.groupby('Closest Color')['Sentiment Score'].sum()
        total_sentiment = sentiment_sums.sum()
        normalized_weights = (sentiment_sums / total_sentiment).to_dict()
        return normalized_weights

    def get_discrete_weights(self):
        return self.discrete_weights

    def get_continuous_weights(self):
        return self.continuous_weights


# === Instantiate and get weights ===
csv_path = "/data/semantic_rgb_mapping_with_sentiment.csv"
weights = ColorSentimentWeights(csv_path)

discrete_weights = weights.get_discrete_weights()
continuous_weights = weights.get_continuous_weights()

# === Plotting ===
colors = sorted(set(discrete_weights.keys()) | set(continuous_weights.keys()))
discrete_values = [discrete_weights.get(color, 0) for color in colors]
continuous_values = [continuous_weights.get(color, 0) for color in colors]

x = range(len(colors))
bar_width = 0.4

plt.figure(figsize=(12, 6))
plt.bar([i - bar_width/2 for i in x], discrete_values, width=bar_width, label='Discrete Weights')
plt.bar([i + bar_width/2 for i in x], continuous_values, width=bar_width, label='Continuous Weights')

plt.xlabel('Color')
plt.ylabel('Probability Weight')
plt.title('Comparison of Discrete vs. Continuous Color Sentiment Weights')
plt.xticks(ticks=x, labels=colors, rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()
