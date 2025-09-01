"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Normalize color sentiment densities into proper probability density functions.
'''

import pandas as pd
import matplotlib.pyplot as plt

class NormalizedColorSentimentPDF:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.discrete_pdf = self._calculate_discrete_pdf()
        self.continuous_pdf = self._calculate_minmax_normalized_continuous_pdf()

    def _calculate_discrete_pdf(self):
        counts = self.df['Closest Color'].value_counts(normalize=True)
        return counts.to_dict()

    def _calculate_minmax_normalized_continuous_pdf(self):
        grouped = self.df.groupby('Closest Color')['Sentiment Score'].sum()
        s_min, s_max = grouped.min(), grouped.max()

        if s_max == s_min:
            normalized = pd.Series([1.0 / len(grouped)] * len(grouped), index=grouped.index)
        else:
            normalized = (grouped - s_min) / (s_max - s_min)

        pdf = normalized / normalized.sum()
        return pdf.to_dict()

    def get_discrete_pdf(self):
        return self.discrete_pdf

    def get_continuous_pdf(self):
        return self.continuous_pdf

    def plot_pdfs(self):
        all_colors = sorted(set(self.discrete_pdf.keys()) | set(self.continuous_pdf.keys()))
        discrete_vals = [self.discrete_pdf.get(color, 0) for color in all_colors]
        continuous_vals = [self.continuous_pdf.get(color, 0) for color in all_colors]

        x = range(len(all_colors))
        bar_width = 0.4

        plt.figure(figsize=(12, 6))
        plt.bar([i - bar_width/2 for i in x], discrete_vals, width=bar_width, label='Discrete PDF', color='orange')
        plt.bar([i + bar_width/2 for i in x], continuous_vals, width=bar_width, label='Continuous PDF (Min-Max Scaled)', color='orangered')

        plt.xlabel('Color')
        plt.ylabel('Probability Density')
        plt.title('Normalized Color Sentiment PDFs (Discrete vs. Scaled Continuous)')
        plt.xticks(ticks=x, labels=all_colors, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()


pdf_gen = NormalizedColorSentimentPDF('/data/semantic_rgb_mapping_with_sentiment.csv')
pdf_gen.plot_pdfs()