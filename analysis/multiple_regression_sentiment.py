"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Multiple regression of RGB to predict color sentiment.
'''

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Load your data ---
# Assumes you're using 'semantic_rgb_mapping_with_similarity.xlsx'
rgb_df = pd.read_excel('/data/semantic_rgb_mapping_with_similarity.xlsx')

# --- Step 2: Clean and select columns ---
df = rgb_df[['R', 'G', 'B', 'Sentiment Score']].dropna()
X = df[['R', 'G', 'B']]
y = df['Sentiment Score']

# --- Step 3: Fit multiple regression model ---
X = sm.add_constant(X)  # Add intercept term
model = sm.OLS(y, X).fit()
print(model.summary())

# --- Step 4: Visualize individual relationships ---
sns.set(style="whitegrid")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

for ax, color in zip(axs, ['R', 'G', 'B']):
    sns.regplot(x=color, y='Sentiment Score', data=df, ax=ax,
                scatter_kws={'alpha': 0.5}, line_kws={'color': 'black'})
    ax.set_title(f'Sentiment vs {color}')
    ax.set_ylabel('Sentiment Relevance')
    ax.set_xlabel(f'{color} Value')

plt.suptitle('Multiple Regression: RGB → Semantic Score', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
