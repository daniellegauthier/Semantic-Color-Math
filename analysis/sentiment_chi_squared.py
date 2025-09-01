"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Sentiment goodness of fit test.
'''

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re

# Load your taxonomy data
taxonomy_df = pd.read_csv('/data/la matrice.csv')

# Load BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare comparison between category names and their associated words
results = []
for _, row in taxonomy_df.iterrows():
    top_cat = str(row['matrice1']).strip().lower()
    sub_cat = str(row['matrice']).strip().lower()
    word_list = re.findall(r'\b\w+\b', str(row['english-words']).lower())

    if not word_list or not top_cat or not sub_cat:
        continue

    words_emb = model.encode(word_list)
    top_emb = model.encode([top_cat])[0]
    sub_emb = model.encode([sub_cat])[0]

    sim_to_top = cosine_similarity(words_emb, [top_emb]).mean()
    sim_to_sub = cosine_similarity(words_emb, [sub_emb]).mean()

    results.append({
        'color': row['color'],
        'matrice1': top_cat,
        'matrice': sub_cat,
        'avg_similarity_to_matrice1': sim_to_top,
        'avg_similarity_to_matrice': sim_to_sub,
        'word_count': len(word_list)
    })

# Convert to DataFrame
similarity_df = pd.DataFrame(results)
similarity_df.to_csv("category_similarity_scores.csv", index=False)
print(similarity_df.head())
