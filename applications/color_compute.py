"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Create a color palette based on input phrase/word and constraints to color outputs.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets
from IPython.display import display, clear_output
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
import random
import math
import itertools
from collections import defaultdict

# Install required packages
#pip install colormath ipywidgets matplotlib scikit-learn

# Import colormath after installation
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# URLs for the CSV files
colors_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/la%20matrice-qhiF8W1MZiXnmjlkL6xJasXeG1FryC.csv"
sequences_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/la%20matrice%20sequences-2j9pbWxr7vXA22q5VUTWXYgyaSe9dO.csv"
semantic_mapping_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/semantic_rgb_mapping-HG6RA90fo5iiJpus92mQ9hrgmHZUgl.csv"

# Global variables to store data
colors_df = None
sequences_df = None
semantic_mapping_df = None
k = 5  # Number of clusters to preserve
kmeans = None
color_to_cluster = None
cluster_weights = None
sentiment_df = None

# Data structures for efficient lookups
word_to_colors_map = {}  # Hash map: word -> list of colors
color_to_words_map = {}  # Hash map: color -> list of words
word_set = set()         # Set of all words for fast membership testing
cluster_word_map = {}    # Hash map: cluster -> list of words
permutation_cache = {}   # Cache for word permutation results

# Function to fetch and parse CSV data
def fetch_csv(url):
    print(f"Fetching data from {url}...")
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

# Function to find words with a given prefix
def find_words_with_prefix(prefix, all_words, limit=10):
    """Find words that start with the given prefix"""
    if not prefix:
        return []

    prefix = prefix.lower()
    matching_words = []

    for word in all_words:
        if word.startswith(prefix):
            matching_words.append(word)
            if len(matching_words) >= limit:
                break

    return matching_words

# Function to build hash maps for O(1) lookups
def build_hash_maps(colors_df, semantic_mapping_df):
    print("Building hash maps for O(1) lookups...")
    word_to_colors = defaultdict(list)
    color_to_words = defaultdict(list)
    cluster_words = defaultdict(set)
    all_words = set()

    # Process semantic mapping data
    for _, row in semantic_mapping_df.iterrows():
        rgb = (row['R'], row['G'], row['B'])
        rgb_str = row['RGB']
        closest_color = row['Closest Color'].lower() if pd.notna(row['Closest Color']) else None

        # Process original words
        if pd.notna(row['Original Words']):
            original_words = [w.strip().lower() for w in str(row['Original Words']).split(',')]
            for word in original_words:
                if word:  # Skip empty strings
                    all_words.add(word)
                    word_to_colors[word].append({
                        'rgb': rgb,
                        'rgb_str': rgb_str,
                        'type': 'original',
                        'closest_color': closest_color
                    })
                    if closest_color:
                        color_to_words[closest_color].append(word)

        # Process new words
        if pd.notna(row['New Words']):
            new_words = [w.strip().lower() for w in str(row['New Words']).split(',')]
            for word in new_words:
                if word:  # Skip empty strings
                    all_words.add(word)
                    word_to_colors[word].append({
                        'rgb': rgb,
                        'rgb_str': rgb_str,
                        'type': 'new',
                        'closest_color': closest_color
                    })
                    if closest_color:
                        color_to_words[closest_color].append(word)

    # Process cluster words from colors_df
    for _, row in colors_df.iterrows():
        if pd.notna(row['cluster']) and pd.notna(row['clean_words']):
            cluster = int(row['cluster'])
            words = [w.strip().lower() for w in str(row['clean_words']).split(',')]
            for word in words:
                if word:
                    all_words.add(word)
                    cluster_words[cluster].add(word)

    return dict(word_to_colors), dict(color_to_words), dict(cluster_words), all_words

# Function to load all data and build efficient data structures
def load_all_data():
    global colors_df, sequences_df, semantic_mapping_df, kmeans, color_to_cluster, cluster_weights, sentiment_df
    global word_to_colors_map, color_to_words_map, word_set, cluster_word_map

    # Fetch the data
    print("Loading all data...")
    colors_df = fetch_csv(colors_url)
    sequences_df = fetch_csv(sequences_url)
    semantic_mapping_df = fetch_csv(semantic_mapping_url)

    # Make sure RGB values are numeric
    for col in ['r', 'g', 'b']:
        colors_df[col] = pd.to_numeric(colors_df[col], errors='coerce')

    # Convert RGB columns to numeric in the semantic mapping
    for col in ['R', 'G', 'B']:
        semantic_mapping_df[col] = pd.to_numeric(semantic_mapping_df[col], errors='coerce')

    # Extract sentiment features using text vectorization
    # First, clean the english-words column
    colors_df['clean_words'] = colors_df['english-words'].fillna('').astype(str)

    # Use CountVectorizer to create binary features for sentiment words
    vectorizer = CountVectorizer(binary=True, max_features=100)
    sentiment_features = vectorizer.fit_transform(colors_df['clean_words'])
    sentiment_df = pd.DataFrame(
        sentiment_features.toarray(),
        columns=[f'sentiment_{word}' for word in vectorizer.get_feature_names_out()],
        index=colors_df.index
    )

    # Combine RGB values with sentiment features
    X_rgb = colors_df[['r', 'g', 'b']].fillna(0)
    X = pd.concat([X_rgb, sentiment_df], axis=1)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform k-means clustering with k=5 as specified
    print(f"Performing k-means clustering with k={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    colors_df['cluster'] = kmeans.fit_predict(X_scaled)

    # Create a color-to-cluster mapping
    color_to_cluster = dict(zip(colors_df['color'], colors_df['cluster']))

    # Define cluster transition weights based on sentiment similarity
    # Calculate sentiment similarity between clusters
    cluster_centers = kmeans.cluster_centers_
    # Extract just the sentiment part of the cluster centers (not RGB)
    sentiment_centers = cluster_centers[:, 3:]
    # Calculate cosine similarity between sentiment centers
    sentiment_similarity = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            # Dot product of normalized vectors = cosine similarity
            norm_i = np.linalg.norm(sentiment_centers[i])
            norm_j = np.linalg.norm(sentiment_centers[j])
            if norm_i > 0 and norm_j > 0:
                sentiment_similarity[i, j] = np.dot(sentiment_centers[i], sentiment_centers[j]) / (norm_i * norm_j)
            else:
                sentiment_similarity[i, j] = 0

    # Convert similarity to weights (higher similarity = lower momentum)
    cluster_weights = 1 - sentiment_similarity

    # Enhance the colors_df with semantic information
    print("Enhancing color data with semantic information...")
    colors_df['semantic_words'] = None
    colors_df['semantic_closest_color'] = None

    for idx, row in colors_df.iterrows():
        if pd.notna(row['r']) and pd.notna(row['g']) and pd.notna(row['b']):
            closest = find_closest_semantic(row['r'], row['g'], row['b'], semantic_mapping_df)
            if closest is not None:
                colors_df.at[idx, 'semantic_words'] = closest['New Words'] if pd.notna(closest['New Words']) else closest['Original Words']
                colors_df.at[idx, 'semantic_closest_color'] = closest['Closest Color']

    # Build efficient data structures for lookups
    word_to_colors_map, color_to_words_map, cluster_word_map, word_set = build_hash_maps(colors_df, semantic_mapping_df)

    print("Data loading and preprocessing complete!")
    print(f"Built word set with {len(word_set)} words")
    print(f"Built word-to-colors map with {len(word_to_colors_map)} entries")
    print(f"Built color-to-words map with {len(color_to_words_map)} entries")
    print(f"Built cluster-word map with {len(cluster_word_map)} entries")

# Function to find the closest semantic mapping for a given RGB value
def find_closest_semantic(r, g, b, semantic_df):
    min_distance = float('inf')
    closest_row = None

    for _, row in semantic_df.iterrows():
        # Calculate Euclidean distance in RGB space
        distance = np.sqrt((r - row['R'])**2 + (g - row['G'])**2 + (b - row['B'])**2)

        if distance < min_distance:
            min_distance = distance
            closest_row = row

    return closest_row

# Function to filter colors based on RGB constraints
def filter_colors_by_rgb(r_min=0, r_max=255, g_min=0, g_max=255, b_min=0, b_max=255):
    if colors_df is None:
        print("Data not loaded. Please load data first.")
        return None

    filtered = colors_df[
        (colors_df['r'] >= r_min) & (colors_df['r'] <= r_max) &
        (colors_df['g'] >= g_min) & (colors_df['g'] <= g_max) &
        (colors_df['b'] >= b_min) & (colors_df['b'] <= b_max)
    ]

    print(f"Found {len(filtered)} colors that meet the constraints")
    return filtered

# Function to find colors associated with a word using the hash map (O(1) lookup)
def find_colors_by_word(word, limit=10):
    if word_to_colors_map is None:
        print("Data structures not built. Please load data first.")
        return []

    word = word.lower().strip()

    # Direct lookup in hash map
    if word in word_to_colors_map:
        return word_to_colors_map[word][:limit]

    # If not found, try prefix search
    prefix_matches = []
    prefix_words = find_words_with_prefix(word, word_set)
    for prefix_word in prefix_words:
        if prefix_word in word_to_colors_map:
            prefix_matches.extend(word_to_colors_map[prefix_word])
            if len(prefix_matches) >= limit:
                break

    # If still not enough, find similar words
    if len(prefix_matches) < limit:
        similar_words = find_similar_words(word, word_set, limit=5)
        for similar_word in similar_words:
            if similar_word in word_to_colors_map:
                prefix_matches.extend(word_to_colors_map[similar_word])
                if len(prefix_matches) >= limit:
                    break

    return prefix_matches[:limit]

# Function to find similar words
def find_similar_words(word, all_words, limit=5):
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

# Function to find colors associated with a combination of words
def find_colors_by_word_combination(words, limit=10, method='intersection'):
    """
    Find colors associated with a combination of words.

    Args:
        words: List of words to search for
        limit: Maximum number of results to return
        method: How to combine results - 'intersection' (AND), 'union' (OR), or 'weighted' (score-based)

    Returns:
        List of colors associated with the word combination
    """
    if not words:
        return []

    # Check if we have this combination in cache
    cache_key = (tuple(sorted(words)), method, limit)
    if cache_key in permutation_cache:
        return permutation_cache[cache_key]

    # Get colors for each word
    word_colors = []
    for word in words:
        colors = find_colors_by_word(word)
        if colors:
            word_colors.append(colors)

    if not word_colors:
        return []

    # Combine results based on method
    if method == 'intersection':
        # Find colors that appear for all words (AND)
        result = []
        if word_colors:
            # Convert to sets of RGB strings for comparison
            color_sets = []
            for colors in word_colors:
                color_sets.append(set(c['rgb_str'] for c in colors))

            # Find intersection
            common_rgb_strs = set.intersection(*color_sets)

            # Convert back to color objects
            for rgb_str in common_rgb_strs:
                for colors in word_colors:
                    for color in colors:
                        if color['rgb_str'] == rgb_str:
                            result.append(color)
                            break
                    break

    elif method == 'union':
        # Combine all colors (OR)
        result = []
        seen_rgb_strs = set()
        for colors in word_colors:
            for color in colors:
                if color['rgb_str'] not in seen_rgb_strs:
                    result.append(color)
                    seen_rgb_strs.add(color['rgb_str'])

    else:  # weighted
        # Score colors based on how many words they match
        color_scores = defaultdict(int)
        color_objects = {}

        for colors in word_colors:
            for color in colors:
                rgb_str = color['rgb_str']
                color_scores[rgb_str] += 1
                if rgb_str not in color_objects:
                    color_objects[rgb_str] = color

        # Sort by score
        sorted_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)
        result = [color_objects[rgb_str] for rgb_str, _ in sorted_colors]

    # Cache the result
    permutation_cache[cache_key] = result[:limit]
    return result[:limit]

# Function to convert RGB to hex color
def rgb_to_hex(r, g, b):
    return f'#{int(r):02x}{int(g):02x}{int(b):02x}'

# Function to visualize colors
def visualize_colors(colors_list, title="Color Visualization"):
    if not colors_list:
        print("No colors to visualize")
        return

    plt.figure(figsize=(12, 3))
    for i, color in enumerate(colors_list):
        plt.subplot(1, len(colors_list), i + 1)

        # Get RGB values
        if 'r' in color and 'g' in color and 'b' in color:
            r, g, b = color['r'], color['g'], color['b']
        elif 'rgb' in color:
            r, g, b = color['rgb']
        else:
            continue

        plt.fill([0, 1, 1, 0], [0, 0, 1, 1], color=rgb_to_hex(r, g, b))
        plt.axis('off')

        # Get color name
        if 'color' in color:
            color_name = color['color']
        elif 'closest_color' in color:
            color_name = color['closest_color']
        else:
            color_name = f"RGB: {r},{g},{b}"

        plt.title(color_name, fontsize=10)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.show()

# Function to visualize clusters
def visualize_clusters(filtered_colors):
    if filtered_colors is None or len(filtered_colors) == 0:
        print("No colors to visualize")
        return

    # Define a list of distinct colors for visualization
    viz_colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for cluster in range(k):
        cluster_colors = filtered_colors[filtered_colors['cluster'] == cluster]
        if len(cluster_colors) > 0:
            ax.scatter(
                cluster_colors['r'],
                cluster_colors['g'],
                cluster_colors['b'],
                color=viz_colors[cluster % len(viz_colors)],
                label=f'Cluster {cluster}',
                alpha=0.7
            )

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title('Colors in RGB Space by Cluster')
    plt.legend()
    plt.show()

    # Create cluster swatches
    plt.figure(figsize=(15, 10))
    for cluster in range(k):
        cluster_colors = filtered_colors[filtered_colors['cluster'] == cluster]
        if len(cluster_colors) > 0:
            plt.subplot(k, 1, cluster + 1)

            # Get cluster sentiment profile
            cluster_sentiments = sentiment_df.loc[cluster_colors.index].sum().sort_values(ascending=False)
            top_sentiments = cluster_sentiments.head(5).index.str.replace('sentiment_', '')

            # Display color swatches
            for i, (_, color) in enumerate(cluster_colors.iterrows()):
                if i < 10:  # Limit to 10 colors per cluster
                    plt.fill([i, i+0.9, i+0.9, i], [0, 0, 1, 1],
                             color=rgb_to_hex(color['r'], color['g'], color['b']))
                    plt.text(i+0.45, 0.5, color['color'],
                             ha='center', va='center', rotation=90,
                             fontsize=8, color='white' if sum([color['r'], color['g'], color['b']]) < 380 else 'black')

            plt.title(f"Cluster {cluster}: {', '.join(top_sentiments)}")
            plt.axis('off')

    plt.tight_layout()
    plt.show()

# Function to find words in a specific cluster
def find_words_in_cluster(cluster_id, limit=20):
    """Find the most common words associated with a specific cluster"""
    if cluster_id not in cluster_word_map:
        return []

    # Get words from the cluster
    words = list(cluster_word_map[cluster_id])

    # Sort by frequency in the cluster
    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1

    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:limit]]

# Function to find clusters for a word
def find_clusters_for_word(word):
    """Find which clusters a word is associated with"""
    word = word.lower().strip()

    # Get colors for the word
    colors = find_colors_by_word(word)

    # Map colors to clusters
    cluster_counts = defaultdict(int)
    for color in colors:
        if 'closest_color' in color and color['closest_color'] in color_to_cluster:
            cluster = color_to_cluster[color['closest_color']]
            cluster_counts[cluster] += 1

    # Sort clusters by count
    sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_clusters

# Function to generate word permutations and find associated colors
def analyze_word_permutations(words, max_length=3):
    """
    Generate permutations of words and find associated colors

    Args:
        words: List of words to permute
        max_length: Maximum permutation length

    Returns:
        Dictionary mapping permutations to colors
    """
    results = {}

    # Generate permutations of different lengths
    for length in range(1, min(len(words) + 1, max_length + 1)):
        for perm in itertools.combinations(words, length):
            # Find colors for this permutation
            perm_str = " + ".join(perm)
            colors = find_colors_by_word_combination(perm, method='weighted')

            if colors:
                results[perm_str] = colors

    return results

# Function to analyze a word and its associated colors
def analyze_word(word, r_min=0, r_max=255, g_min=0, g_max=255, b_min=0, b_max=255):
    if colors_df is None or word_to_colors_map is None:
        print("Data not loaded. Please load data first.")
        return

    word = word.lower().strip()

    # Filter colors by RGB constraints
    filtered_colors = filter_colors_by_rgb(r_min, r_max, g_min, g_max, b_min, b_max)

    if filtered_colors is None or len(filtered_colors) == 0:
        print("No colors match the RGB constraints. Try relaxing the constraints.")
        return

    print(f"\n===== Analysis for word '{word}' =====")

    # Find colors associated with the word using efficient lookup
    word_colors = find_colors_by_word(word)

    if not word_colors:
        print(f"No colors found directly for word '{word}'")

        # Try to find similar words
        similar_words = find_similar_words(word, word_set, limit=5)
        print(f"Similar words: {', '.join(similar_words)}")

        # Get colors for similar words
        for similar_word in similar_words:
            similar_colors = find_colors_by_word(similar_word)
            if similar_colors:
                print(f"\nColors for similar word '{similar_word}':")
                visualize_colors(similar_colors[:5], f"Colors for '{similar_word}'")

        return

    # Filter word colors by RGB constraints
    filtered_word_colors = []
    for color in word_colors:
        rgb = color['rgb']
        if (r_min <= rgb[0] <= r_max and
            g_min <= rgb[1] <= g_max and
            b_min <= rgb[2] <= b_max):
            filtered_word_colors.append(color)

    print(f"Found {len(filtered_word_colors)} colors associated with '{word}' that meet the RGB constraints")

    # Visualize the colors
    if filtered_word_colors:
        print("\nColors associated with the word:")
        visualize_colors(filtered_word_colors, f"Colors for '{word}'")

    # Find which clusters the word is associated with
    word_clusters = find_clusters_for_word(word)
    if word_clusters:
        print("\nClusters associated with the word:")
        for cluster, count in word_clusters:
            print(f"Cluster {cluster}: {count} colors")

            # Show top words in this cluster
            cluster_words = find_words_in_cluster(cluster, limit=10)
            if cluster_words:
                print(f"  Top words in cluster {cluster}: {', '.join(cluster_words)}")

    # Generate a color palette recommendation
    print("\nRecommended Color Palette:")

    # Get colors from different clusters
    palette = []
    for cluster in range(k):
        cluster_colors = filtered_colors[filtered_colors['cluster'] == cluster]
        if len(cluster_colors) > 0:
            # Try to find a color in this cluster that matches the word
            word_match = False
            for color in filtered_word_colors:
                rgb = color['rgb']
                cluster_match = cluster_colors[
                    (cluster_colors['r'] == rgb[0]) &
                    (cluster_colors['g'] == rgb[1]) &
                    (cluster_colors['b'] == rgb[2])
                ]
                if not cluster_match.empty:
                    palette.append(cluster_match.iloc[0])
                    word_match = True
                    break

            # If no word match in this cluster, add a random color from the cluster
            if not word_match and len(cluster_colors) > 0:
                palette.append(cluster_colors.iloc[np.random.randint(0, len(cluster_colors))])

    if palette:
        visualize_colors(palette, f"Recommended Palette for '{word}'")

        print("Palette Colors:")
        for color in palette:
            semantic_info = color['semantic_words'] if pd.notna(color['semantic_words']) else "No semantic data"
            print(f"  {color['color']} - RGB: ({color['r']}, {color['g']}, {color['b']}) - Semantics: {semantic_info}")
    else:
        print("Could not generate a palette with the available colors")

# Function to analyze multiple words and their combinations
def analyze_word_combination(words, r_min=0, r_max=255, g_min=0, g_max=255, b_min=0, b_max=255):
    if colors_df is None or word_to_colors_map is None:
        print("Data not loaded. Please load data first.")
        return

    # Clean and split the input
    if isinstance(words, str):
        words = [w.strip().lower() for w in words.split(',')]
    else:
        words = [w.strip().lower() for w in words]

    words = [w for w in words if w]  # Remove empty strings

    if not words:
        print("No valid words provided")
        return

    # Filter colors by RGB constraints
    filtered_colors = filter_colors_by_rgb(r_min, r_max, g_min, g_max, b_min, b_max)

    if filtered_colors is None or len(filtered_colors) == 0:
        print("No colors match the RGB constraints. Try relaxing the constraints.")
        return

    print(f"\n===== Analysis for word combination: {', '.join(words)} =====")

    # Find colors for each individual word
    for word in words:
        word_colors = find_colors_by_word(word)
        if word_colors:
            print(f"\nColors for '{word}':")
            visualize_colors(word_colors[:5], f"Colors for '{word}'")

    # Find colors for the combination (AND)
    intersection_colors = find_colors_by_word_combination(words, method='intersection')
    if intersection_colors:
        print(f"\nColors matching ALL words ({' AND '.join(words)}):")
        visualize_colors(intersection_colors[:5], f"Colors for {' AND '.join(words)}")
    else:
        print(f"\nNo colors found matching ALL words ({' AND '.join(words)})")

    # Find colors for the combination (OR)
    union_colors = find_colors_by_word_combination(words, method='union')
    if union_colors:
        print(f"\nColors matching ANY word ({' OR '.join(words)}):")
        visualize_colors(union_colors[:5], f"Colors for {' OR '.join(words)}")

    # Find colors for the combination (weighted)
    weighted_colors = find_colors_by_word_combination(words, method='weighted')
    if weighted_colors:
        print(f"\nColors weighted by word matches:")
        visualize_colors(weighted_colors[:5], f"Weighted colors for {', '.join(words)}")

    # Generate permutations and analyze
    if len(words) > 1:
        print("\nAnalyzing word permutations...")
        permutations = analyze_word_permutations(words)

        # Show the best permutation results
        best_perms = sorted(permutations.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        for perm_str, colors in best_perms:
            print(f"\nBest colors for permutation '{perm_str}':")
            visualize_colors(colors[:5], f"Colors for {perm_str}")

    # Find clusters associated with these words
    print("\nClusters associated with these words:")
    cluster_scores = defaultdict(int)

    for word in words:
        word_clusters = find_clusters_for_word(word)
        for cluster, count in word_clusters:
            cluster_scores[cluster] += count

    # Sort clusters by score
    sorted_clusters = sorted(cluster_scores.items(), key=lambda x: x[1], reverse=True)

    for cluster, score in sorted_clusters:
        print(f"Cluster {cluster}: Score {score}")

        # Show top words in this cluster
        cluster_words = find_words_in_cluster(cluster, limit=10)
        if cluster_words:
            print(f"  Top words in cluster {cluster}: {', '.join(cluster_words)}")

    # Generate a color palette recommendation based on the word combination
    print("\nRecommended Color Palette for Word Combination:")

    # Use weighted colors for the palette
    if weighted_colors:
        # Try to get one color from each cluster
        palette = []
        cluster_used = set()

        for color in weighted_colors:
            rgb = color['rgb']
            for _, row in filtered_colors.iterrows():
                if (row['r'] == rgb[0] and row['g'] == rgb[1] and row['b'] == rgb[2] and
                    row['cluster'] not in cluster_used and len(palette) < 5):
                    palette.append(row)
                    cluster_used.add(row['cluster'])
                    break

        # Fill in any missing clusters
        for cluster in range(k):
            if cluster not in cluster_used and len(palette) < 5:
                cluster_colors = filtered_colors[filtered_colors['cluster'] == cluster]
                if len(cluster_colors) > 0:
                    palette.append(cluster_colors.iloc[np.random.randint(0, len(cluster_colors))])

        if palette:
            visualize_colors(palette, f"Recommended Palette for {', '.join(words)}")

            print("Palette Colors:")
            for color in palette:
                semantic_info = color['semantic_words'] if pd.notna(color['semantic_words']) else "No semantic data"
                print(f"  {color['color']} - RGB: ({color['r']}, {color['g']}, {color['b']}) - Semantics: {semantic_info}")
        else:
            print("Could not generate a palette with the available colors")

# Create the dashboard UI with enhanced search capabilities
def create_dashboard():
    # Load data first
    load_all_data()

    # Create widgets
    word_input = widgets.Text(
        value='',
        placeholder='Enter a word (e.g., hope, energy, calm)',
        description='Word:',
        disabled=False
    )

    word_combination_input = widgets.Text(
        value='',
        placeholder='Enter multiple words separated by commas',
        description='Word Combination:',
        disabled=False
    )

    r_min_slider = widgets.IntSlider(value=0, min=0, max=255, step=1, description='R Min:')
    r_max_slider = widgets.IntSlider(value=255, min=0, max=255, step=1, description='R Max:')
    g_min_slider = widgets.IntSlider(value=0, min=0, max=255, step=1, description='G Min:')
    g_max_slider = widgets.IntSlider(value=255, min=0, max=255, step=1, description='G Max:')
    b_min_slider = widgets.IntSlider(value=0, min=0, max=255, step=1, description='B Min:')
    b_max_slider = widgets.IntSlider(value=255, min=0, max=255, step=1, description='B Max:')

    # Preset buttons for common constraints
    preset_buttons = widgets.ToggleButtons(
        options=['All Colors', 'Reds (R>128)', 'Greens (G>128)', 'Blues (B>128)', 'Warm Colors', 'Cool Colors'],
        description='Presets:',
        disabled=False,
        button_style='',
    )

    # Search type selector
    search_type = widgets.RadioButtons(
        options=['Single Word', 'Word Combination'],
        value='Single Word',
        description='Search Type:',
        disabled=False
    )

    # Analyze button
    analyze_button = widgets.Button(
        description='Analyze',
        disabled=False,
        button_style='success',
        tooltip='Click to analyze the word and colors',
        icon='search'
    )

    # Word suggestions dropdown (will be populated dynamically)
    word_suggestions = widgets.Dropdown(
        options=[],
        description='Suggestions:',
        disabled=False,
    )

    # Cluster explorer
    cluster_selector = widgets.Dropdown(
        options=[(f'Cluster {i}', i) for i in range(k)],
        value=0,
        description='Explore Cluster:',
        disabled=False
    )

    cluster_explore_button = widgets.Button(
        description='Explore Cluster',
        disabled=False,
        button_style='info',
        tooltip='Click to explore the selected cluster',
        icon='folder-open'
    )

    # Output area
    output = widgets.Output()

    # Function to handle preset selection
    def on_preset_change(change):
        if change['new'] == 'All Colors':
            r_min_slider.value = 0
            r_max_slider.value = 255
            g_min_slider.value = 0
            g_max_slider.value = 255
            b_min_slider.value = 0
            b_max_slider.value = 255
        elif change['new'] == 'Reds (R>128)':
            r_min_slider.value = 128
            r_max_slider.value = 255
            g_min_slider.value = 0
            g_max_slider.value = 255
            b_min_slider.value = 0
            b_max_slider.value = 255
        elif change['new'] == 'Greens (G>128)':
            r_min_slider.value = 0
            r_max_slider.value = 255
            g_min_slider.value = 128
            g_max_slider.value = 255
            b_min_slider.value = 0
            b_max_slider.value = 255
        elif change['new'] == 'Blues (B>128)':
            r_min_slider.value = 0
            r_max_slider.value = 255
            g_min_slider.value = 0
            g_max_slider.value = 255
            b_min_slider.value = 128
            b_max_slider.value = 255
        elif change['new'] == 'Warm Colors':
            r_min_slider.value = 128
            r_max_slider.value = 255
            g_min_slider.value = 50
            g_max_slider.value = 255
            b_min_slider.value = 0
            b_max_slider.value = 128
        elif change['new'] == 'Cool Colors':
            r_min_slider.value = 0
            r_max_slider.value = 128
            g_min_slider.value = 50
            g_max_slider.value = 255
            b_min_slider.value = 128
            b_max_slider.value = 255

    preset_buttons.observe(on_preset_change, names='value')

    # Function to update word suggestions based on input
    def update_word_suggestions(change):
        prefix = change['new'].lower().strip()
        if prefix:
            # Find words with this prefix
            suggestions = find_words_with_prefix(prefix, word_set, limit=10)

            if suggestions:
                word_suggestions.options = suggestions
            else:
                # If no exact prefix matches, try similar words
                similar_words = find_similar_words(prefix, word_set, limit=10)
                if similar_words:
                    word_suggestions.options = similar_words
                else:
                    word_suggestions.options = ["No suggestions"]
        else:
            word_suggestions.options = []

    word_input.observe(update_word_suggestions, names='value')

    # Function to handle word suggestion selection
    def on_suggestion_select(change):
        if change['new'] and change['new'] != "No suggestions":
            word_input.value = change['new']

    word_suggestions.observe(on_suggestion_select, names='value')

    # Function to handle search type change
    def on_search_type_change(change):
        if change['new'] == 'Single Word':
            word_input.disabled = False
            word_suggestions.disabled = False
            word_combination_input.disabled = True
        else:
            word_input.disabled = True
            word_suggestions.disabled = True
            word_combination_input.disabled = False

    search_type.observe(on_search_type_change, names='value')

    # Function to handle analyze button click
    def on_analyze_button_clicked(b):
        with output:
            clear_output()

            if search_type.value == 'Single Word':
                word = word_input.value.strip()
                if not word:
                    print("Please enter a word to analyze")
                    return

                analyze_word(
                    word,
                    r_min_slider.value, r_max_slider.value,
                    g_min_slider.value, g_max_slider.value,
                    b_min_slider.value, b_max_slider.value
                )
            else:
                words = word_combination_input.value.strip()
                if not words:
                    print("Please enter words to analyze")
                    return

                analyze_word_combination(
                    words,
                    r_min_slider.value, r_max_slider.value,
                    g_min_slider.value, g_max_slider.value,
                    b_min_slider.value, b_max_slider.value
                )

    analyze_button.on_click(on_analyze_button_clicked)

    # Function to handle cluster exploration
    def on_explore_cluster_clicked(b):
        with output:
            clear_output()
            cluster_id = cluster_selector.value

            print(f"Exploring Cluster {cluster_id}")

            # Get colors in this cluster
            cluster_colors = colors_df[colors_df['cluster'] == cluster_id]

            if len(cluster_colors) == 0:
                print(f"No colors found in Cluster {cluster_id}")
                return

            print(f"Found {len(cluster_colors)} colors in Cluster {cluster_id}")

            # Show top words in this cluster
            cluster_words = find_words_in_cluster(cluster_id, limit=20)
            if cluster_words:
                print(f"Top words in Cluster {cluster_id}: {', '.join(cluster_words)}")

            # Show sample colors from this cluster
            sample_size = min(10, len(cluster_colors))
            sample_colors = cluster_colors.sample(sample_size)

            plt.figure(figsize=(12, 3))
            for i, (_, color) in enumerate(sample_colors.iterrows()):
                plt.subplot(1, sample_size, i + 1)
                plt.fill([0, 1, 1, 0], [0, 0, 1, 1], color=rgb_to_hex(color['r'], color['g'], color['b']))
                plt.axis('off')
                plt.title(color['color'], fontsize=10)
            plt.suptitle(f"Sample Colors from Cluster {cluster_id}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.9])
            plt.show()

            # Show cluster sentiment profile
            cluster_sentiments = sentiment_df.loc[cluster_colors.index].mean()
            top_sentiments = cluster_sentiments.sort_values(ascending=False).head(10)

            plt.figure(figsize=(10, 6))
            top_sentiments.plot(kind='bar')
            plt.title(f'Sentiment Profile for Cluster {cluster_id}')
            plt.xlabel('Sentiment')
            plt.ylabel('Average Presence')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()

    cluster_explore_button.on_click(on_explore_cluster_clicked)

    # Layout the widgets
    header = widgets.HTML(value="<h1>Enhanced Color Semantic Dashboard</h1><p>Efficiently search and analyze colors by words, combinations, and clusters</p>")

    rgb_constraints = widgets.VBox([
        widgets.HBox([r_min_slider, r_max_slider]),
        widgets.HBox([g_min_slider, g_max_slider]),
        widgets.HBox([b_min_slider, b_max_slider]),
    ])

    search_controls = widgets.VBox([
        search_type,
        widgets.HBox([word_input, word_suggestions]),
        word_combination_input,
    ])

    cluster_controls = widgets.HBox([
        cluster_selector,
        cluster_explore_button
    ])

    controls = widgets.VBox([
        search_controls,
        widgets.HTML(value="<h3>RGB Constraints</h3>"),
        preset_buttons,
        rgb_constraints,
        analyze_button,
        widgets.HTML(value="<h3>Cluster Explorer</h3>"),
        cluster_controls
    ])

    # Display the dashboard
    display(header)
    display(widgets.HBox([controls, output]))

# Run the dashboard
if __name__ == "__main__":
    create_dashboard()