"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Create a more precise palette of colors for an input phrase/word.
'''


import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from english_words import get_english_words_set

def load_data(csv_url):
    """Load data from CSV URL and create word-to-color mapping."""
    print(f"Loading data from {csv_url}...")

    try:
        response = requests.get(csv_url)
        if response.status_code == 200:
            data = StringIO(response.text)
            df = pd.read_csv(data)
            print("Data loaded successfully!")

            # Create word-to-color mapping
            word_to_colors = {}
            all_words = set()

            # Process each row
            for _, row in df.iterrows():
                rgb = (row['R'], row['G'], row['B'])
                rgb_str = row['RGB']

                # Process original words
                if pd.notna(row['Original Words']):
                    original_words = [w.strip().lower() for w in str(row['Original Words']).split(',')]
                    for word in original_words:
                        if word:  # Skip empty strings
                            all_words.add(word)
                            if word not in word_to_colors:
                                word_to_colors[word] = []
                            word_to_colors[word].append({
                                'rgb': rgb,
                                'rgb_str': rgb_str,
                                'type': 'original'
                            })

                # Process new words
                if pd.notna(row['New Words']):
                    new_words = [w.strip().lower() for w in str(row['New Words']).split(',')]
                    for word in new_words:
                        if word:  # Skip empty strings
                            all_words.add(word)
                            if word not in word_to_colors:
                                word_to_colors[word] = []
                            word_to_colors[word].append({
                                'rgb': rgb,
                                'rgb_str': rgb_str,
                                'type': 'new'
                            })

            print(f"Processed {len(df)} RGB points")
            print(f"Created mappings for {len(word_to_colors)} words")
            print(f"Total unique words: {len(all_words)}")

            return df, word_to_colors, all_words
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return None, {}, set()
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, {}, set()

def load_dictionary():
    """Load English dictionary from english-words package."""
    print("Loading English dictionary from english-words package...")
    try:
        # Get all English words from english-words package using the correct method
        english_words = get_english_words_set(['web2'], lower=True)
        dictionary = set(english_words)
        print(f"Loaded {len(dictionary)} words from English dictionary")
        return dictionary
    except Exception as e:
        print(f"Error loading dictionary: {e}")
        print("Using an empty dictionary as fallback.")
        return set()

def find_similar_words(word, all_words, limit=5):
    """Find words similar to the given word."""
    word = word.lower()

    # Calculate similarity scores
    word_scores = []
    for dataset_word in all_words:
        # Character overlap score
        word_chars = set(word)
        dataset_word_chars = set(dataset_word)
        overlap_score = len(word_chars.intersection(dataset_word_chars)) / len(word_chars.union(dataset_word_chars))

        # Prefix matching
        prefix_len = 0
        for i in range(min(len(word), len(dataset_word))):
            if word[i] == dataset_word[i]:
                prefix_len += 1
            else:
                break

        prefix_score = prefix_len / max(len(word), len(dataset_word))

        # Combined score
        combined_score = (overlap_score * 0.6) + (prefix_score * 0.4)

        word_scores.append((dataset_word, combined_score))

    # Sort by score (descending)
    word_scores.sort(key=lambda x: x[1], reverse=True)

    # Return top matches
    return [w for w, _ in word_scores[:limit]]

def get_colors_for_word(word, word_to_colors, all_words, dictionary, limit=5):
    """Get colors for a word, using similar words if necessary."""
    word = word.lower()

    # Check if the word is in our dictionary
    if dictionary and word not in dictionary and len(word) > 2:
        print(f"Note: '{word}' is not in the English dictionary")

    # Case 1: Word is directly in our dataset
    if word in word_to_colors:
        return word_to_colors[word][:limit]

    # Case 2: Word is not in our dataset, find similar words
    similar_words = find_similar_words(word, all_words, limit=10)

    if not similar_words:
        print(f"No similar words found for '{word}'")
        return []

    print(f"Word '{word}' not found directly. Using similar words: {', '.join(similar_words[:3])}...")

    # Collect colors from similar words
    colors = []
    for similar_word in similar_words:
        if similar_word in word_to_colors:
            colors.extend(word_to_colors[similar_word])
            if len(colors) >= limit:
                break

    return colors[:limit]

def visualize_word_colors(word, colors):
    """Visualize the colors associated with a word."""
    if not colors:
        print(f"No colors to visualize for word: {word}")
        return

    plt.figure(figsize=(12, 6))

    # Create color swatches
    for i, color_info in enumerate(colors):
        rgb = color_info['rgb']
        normalized_rgb = (rgb[0]/255, rgb[1]/255, rgb[2]/255)

        plt.subplot(1, len(colors), i+1)
        plt.axhspan(0, 1, color=normalized_rgb)
        plt.title(f"RGB: {color_info['rgb_str']}\nType: {color_info['type']}")
        plt.axis('off')

    plt.suptitle(f"Colors associated with '{word}'", fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    # URL to the CSV file
    csv_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/semantic_rgb_mapping-h61ZapfAQib2wQhF5WKfnkR54rcbBF.csv"

    # Load dictionary
    dictionary = load_dictionary()

    # Load data
    df, word_to_colors, all_words = load_data(csv_url)

    if not word_to_colors:
        print("Failed to load data. Exiting.")
        return

    print("\n===== Minimal English Words Mapper =====")
    print("Type any word to see its colors")
    print("Type 'exit' to quit")

    while True:
        user_input = input("\n> ").strip().lower()

        if not user_input:
            continue

        if user_input == 'exit':
            print("Exiting...")
            break

        # Get colors for the word
        colors = get_colors_for_word(user_input, word_to_colors, all_words, dictionary)

        if colors:
            print(f"\nColors for word '{user_input}':")
            for color in colors:
                print(f"  RGB: {color['rgb_str']}, Type: {color['type']}")

            visualize = input("Visualize these colors? (y/n): ")
            if visualize.lower() == 'y':
                visualize_word_colors(user_input, colors)

if __name__ == "__main__":
    main()