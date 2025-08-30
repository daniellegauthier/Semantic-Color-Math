"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Find the relevant sentiment words for 216 RGB vectors spaced by 50 units.
'''

import pandas as pd
import numpy as np
import requests
from io import StringIO
import math
import re
import time
import nltk
from nltk.corpus import wordnet
from nltk.sentiment import SentimentIntensityAnalyzer

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
print("NLTK data downloaded successfully!")

sia = SentimentIntensityAnalyzer()

def fetch_data():
    """Fetch data from URL and return as DataFrame."""
    base_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/"
    file_path = "la%20matrice-tAkQ2ShtbvUM4bx61GHVjOFgDYUcOQ.csv"
    url = base_url + file_path

    print(f"Fetching data from URL...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = StringIO(response.text)
            df = pd.read_csv(data)
            print("Data fetched successfully!")
            return df
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def rgb_distance(color1, color2):
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

def extract_words_from_dataset(df):
    all_words = set()
    word_columns = ['matrice1', 'matrice', 'english-words', 'english-words-code']
    for _, row in df.iterrows():
        for col in word_columns:
            if pd.notna(row[col]) and row[col]:
                words = re.findall(r'\b\w+\b', str(row[col]).lower())
                all_words.update([w for w in words if len(w) > 2])
    return list(all_words)

def sentiment_score(word):
    return sia.polarity_scores(word)['compound']

def find_closest_color(rgb_point, df):
    min_distance = float('inf')
    closest_color = None
    for _, row in df.iterrows():
        color_rgb = (int(row['r']), int(row['g']), int(row['b']))
        distance = rgb_distance(rgb_point, color_rgb)
        if distance < min_distance:
            min_distance = distance
            words = []
            for col in ['matrice', 'matrice1', 'english-words', 'english-words-code']:
                if pd.notna(row[col]) and row[col]:
                    col_words = re.findall(r'\b\w+\b', str(row[col]).lower())
                    words.extend([w for w in col_words if len(w) > 2])
            closest_color = {
                'name': row['color'],
                'rgb': color_rgb,
                'distance': distance,
                'words': list(set(words)),
                'hex': '#{:02x}{:02x}{:02x}'.format(*color_rgb)
            }
    return closest_color

def generate_new_words_for_rgb(rgb_point, df, all_dataset_words):
    closest_color = find_closest_color(rgb_point, df)
    color_words = closest_color['words']
    new_words = []
    sentiment_scores = [sentiment_score(word) for word in color_words]
    return {
        'rgb': rgb_point,
        'closest_color': closest_color['name'],
        'closest_color_rgb': closest_color['rgb'],
        'distance': closest_color['distance'],
        'original_words': color_words,
        'sentiment_score': np.mean(sentiment_scores) if sentiment_scores else 0.0
    }

def sample_rgb_space(step=50):
    print(f"Sampling RGB colorspace with step size {step}...")
    r_values = np.arange(0, 256, step)
    g_values = np.arange(0, 256, step)
    b_values = np.arange(0, 256, step)
    rgb_samples = [(r, g, b) for r in r_values for g in g_values for b in b_values]
    print(f"Generated {len(rgb_samples)} sample points")
    return rgb_samples

def create_semantic_rgb_table(rgb_results):
    results = []
    for result in rgb_results:
        original_words = ", ".join(result['original_words']) if result['original_words'] else ""
        results.append({
            'R': result['rgb'][0],
            'G': result['rgb'][1],
            'B': result['rgb'][2],
            'RGB': f"({result['rgb'][0]}, {result['rgb'][1]}, {result['rgb'][2]})",
            'Closest Color': result['closest_color'],
            'Sentiment Score': round(result['sentiment_score'], 3),
            'Original Words': original_words
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv('semantic_rgb_mapping_with_sentiment.csv', index=False)
    print("Saved enhanced semantic RGB mapping to semantic_rgb_mapping_with_sentiment.csv")
    return results_df

def main():
    df = fetch_data()
    if df is not None:
        print(f"\nDataset has {len(df)} entries.")
        all_dataset_words = extract_words_from_dataset(df)
        rgb_samples = sample_rgb_space(step=50)
        print("\nGenerating semantic mappings and sentiment scores...")
        rgb_results = [generate_new_words_for_rgb(rgb, df, all_dataset_words) for rgb in rgb_samples]
        results_df = create_semantic_rgb_table(rgb_results)
        print("\nSample Results:")
        print(results_df[['RGB', 'Closest Color', 'Sentiment Score', 'Original Words']].head())
    else:
        print("Failed to load dataset.")

if __name__ == "__main__":
    main()


