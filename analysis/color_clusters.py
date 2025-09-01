"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Color clustering.
'''

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import requests
from io import StringIO
import re
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D

# URLs for the CSV files
colors_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/la%20matrice-qhiF8W1MZiXnmjlkL6xJasXeG1FryC.csv"
sequences_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/la%20matrice%20sequences-2j9pbWxr7vXA22q5VUTWXYgyaSe9dO.csv"

# Function to fetch and parse CSV data
def fetch_csv(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

# Fetch the data
print("Fetching color data...")
colors_df = fetch_csv(colors_url)
print("Fetching sequence data...")
sequences_df = fetch_csv(sequences_url)

# Display the first few rows of each dataset
print("\nColor Data Sample:")
print(colors_df.head())
print("\nSequence Data Sample:")
print(sequences_df.head())

# Clean and prepare the color sentiment data
print("\nPreparing color sentiment data using 'matrice' column...")

# Make sure RGB values are numeric
for col in ['r', 'g', 'b']:
    colors_df[col] = pd.to_numeric(colors_df[col], errors='coerce')

# Create one-hot encoding for matrice values
colors_df['matrice'] = colors_df['matrice'].fillna('Unknown').astype(str)
matrice_dummies = pd.get_dummies(colors_df['matrice'], prefix='matrice')

# Combine RGB values with matrice features
X_rgb = colors_df[['r', 'g', 'b']].fillna(0)
X = pd.concat([X_rgb, matrice_dummies], axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using the elbow method
inertia = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, 'o-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
print("Determining optimal number of clusters...")

# Calculate silhouette scores
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, 10), silhouette_scores, 'o-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.grid(True)

# Choose k=5 based on analysis
k = 5
print(f"\nPerforming k-means clustering with k={k}...")
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
colors_df['cluster'] = kmeans.fit_predict(X_scaled)

# Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
for cluster in range(k):
    plt.scatter(
        X_pca[colors_df['cluster'] == cluster, 0],
        X_pca[colors_df['cluster'] == cluster, 1],
        label=f'Cluster {cluster}',
        alpha=0.7
    )
plt.title('Color Sentiment Clusters (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)

# Analyze the clusters
print("\nCluster Analysis:")
for cluster in range(k):
    cluster_colors = colors_df[colors_df['cluster'] == cluster]
    print(f"\nCluster {cluster} ({len(cluster_colors)} colors):")
    print(f"  Common colors: {', '.join(cluster_colors['color'].value_counts().head(5).index.tolist())}")

    # Find most common matrice values in this cluster
    matrice_counts = cluster_colors['matrice'].value_counts().head(5)
    print(f"  Top matrice values: {', '.join(matrice_counts.index.tolist())}")

    # Average RGB values for this cluster
    avg_rgb = cluster_colors[['r', 'g', 'b']].mean()
    print(f"  Average RGB: ({avg_rgb['r']:.1f}, {avg_rgb['g']:.1f}, {avg_rgb['b']:.1f})")

# Create a color-to-cluster mapping
color_to_cluster = dict(zip(colors_df['color'], colors_df['cluster']))

# Parse the sequences
print("\nAnalyzing color sequences...")
def parse_sequence(seq_str):
    if pd.isna(seq_str):
        return []
    return [color.strip().lower() for color in seq_str.split(',')]

sequences_df['parsed_sequence'] = sequences_df['sequence'].apply(parse_sequence)

# Calculate sequence momentum based on cluster transitions
def calculate_momentum(sequence, color_to_cluster, cluster_weights=None):
    if cluster_weights is None:
        # Default weights: transitions between different clusters have higher momentum
        cluster_weights = np.ones((k, k)) - np.eye(k)

    momentum = 0
    valid_transitions = 0

    for i in range(len(sequence) - 1):
        color1 = sequence[i]
        color2 = sequence[i + 1]

        # Skip if colors not in our dataset
        if color1 not in color_to_cluster or color2 not in color_to_cluster:
            continue

        cluster1 = color_to_cluster[color1]
        cluster2 = color_to_cluster[color2]

        # Add momentum based on cluster transition
        momentum += cluster_weights[cluster1, cluster2]
        valid_transitions += 1

    # Return average momentum per transition to normalize for sequence length
    return momentum / max(1, valid_transitions)

# Define cluster transition weights based on matrice similarity
# Calculate matrice similarity between clusters
cluster_centers = kmeans.cluster_centers_
# Extract just the matrice part of the cluster centers (not RGB)
matrice_centers = cluster_centers[:, 3:]
# Calculate cosine similarity between matrice centers
matrice_similarity = np.zeros((k, k))
for i in range(k):
    for j in range(k):
        # Dot product of normalized vectors = cosine similarity
        norm_i = np.linalg.norm(matrice_centers[i])
        norm_j = np.linalg.norm(matrice_centers[j])
        if norm_i > 0 and norm_j > 0:
            matrice_similarity[i, j] = np.dot(matrice_centers[i], matrice_centers[j]) / (norm_i * norm_j)
        else:
            matrice_similarity[i, j] = 0

# Convert similarity to weights (higher similarity = lower momentum)
cluster_weights = 1 - matrice_similarity

# Visualize the cluster transition weights as a heatmap
plt.figure(figsize=(10, 8))
plt.imshow(cluster_weights, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Momentum Weight')
plt.title('Cluster Transition Momentum Weights')
plt.xlabel('To Cluster')
plt.ylabel('From Cluster')
for i in range(k):
    for j in range(k):
        plt.text(j, i, f'{cluster_weights[i, j]:.2f}',
                 ha='center', va='center',
                 color='white' if cluster_weights[i, j] > 0.5 else 'black')
plt.xticks(range(k))
plt.yticks(range(k))

# Calculate momentum for each sequence
sequences_df['momentum'] = sequences_df['parsed_sequence'].apply(
    lambda seq: calculate_momentum(seq, color_to_cluster, cluster_weights)
)

# Find sequences with highest and lowest momentum
print("\nSequences with Highest Momentum:")
high_momentum = sequences_df.sort_values('momentum', ascending=False).head(5)
for _, row in high_momentum.iterrows():
    print(f"  {row['ï»¿name']}: {row['sequence']} (Momentum: {row['momentum']:.2f})")

print("\nSequences with Lowest Momentum:")
low_momentum = sequences_df.sort_values('momentum').head(5)
for _, row in low_momentum.iterrows():
    print(f"  {row['ï»¿name']}: {row['sequence']} (Momentum: {row['momentum']:.2f})")

# Function to optimize a sequence by maximizing or minimizing momentum
def optimize_sequence(sequence, color_to_cluster, cluster_weights, maximize=True, iterations=100):
    colors_in_clusters = {}
    for color, cluster in color_to_cluster.items():
        if cluster not in colors_in_clusters:
            colors_in_clusters[cluster] = []
        colors_in_clusters[cluster].append(color)

    best_sequence = sequence.copy()
    best_momentum = calculate_momentum(best_sequence, color_to_cluster, cluster_weights)

    for _ in range(iterations):
        # Randomly select a position to modify
        if len(sequence) == 0:
            continue
        pos = np.random.randint(0, len(sequence))

        # Get current color and its cluster
        if sequence[pos] not in color_to_cluster:
            continue
        current_cluster = color_to_cluster[sequence[pos]]

        # Choose a random different cluster and a random color from it
        other_clusters = [c for c in range(k) if c != current_cluster and c in colors_in_clusters]
        if not other_clusters:
            continue
        new_cluster = np.random.choice(other_clusters)
        new_color = np.random.choice(colors_in_clusters[new_cluster])

        # Create a new sequence with the modified color
        new_sequence = sequence.copy()
        new_sequence[pos] = new_color

        # Calculate new momentum
        new_momentum = calculate_momentum(new_sequence, color_to_cluster, cluster_weights)

        # Update if better (higher momentum if maximizing, lower if minimizing)
        if (maximize and new_momentum > best_momentum) or (not maximize and new_momentum < best_momentum):
            best_sequence = new_sequence
            best_momentum = new_momentum

    return best_sequence, best_momentum

# Example: Optimize a sequence to maximize momentum
print("\nOptimizing Sequences:")
sample_sequence = sequences_df.iloc[0]['parsed_sequence']
if len(sample_sequence) > 0:
    print(f"Original sequence: {', '.join(sample_sequence)}")

    # Maximize momentum
    optimized_max, max_momentum = optimize_sequence(
        sample_sequence, color_to_cluster, cluster_weights, maximize=True
    )
    print(f"Optimized for maximum momentum: {', '.join(optimized_max)} (Momentum: {max_momentum:.2f})")

    # Minimize momentum
    optimized_min, min_momentum = optimize_sequence(
        sample_sequence, color_to_cluster, cluster_weights, maximize=False
    )
    print(f"Optimized for minimum momentum: {', '.join(optimized_min)} (Momentum: {min_momentum:.2f})")

# Create a function to generate new sequences based on matrice clusters
def generate_sequence_from_clusters(length, color_to_cluster, colors_in_clusters, target_clusters=None):
    if target_clusters is None:
        target_clusters = list(range(k))

    sequence = []
    for _ in range(length):
        # Choose a cluster from target clusters
        cluster = np.random.choice(target_clusters)

        # Choose a random color from that cluster
        if cluster in colors_in_clusters and colors_in_clusters[cluster]:
            color = np.random.choice(colors_in_clusters[cluster])
            sequence.append(color)

    return sequence

# Group colors by cluster
colors_in_clusters = {}
for color, cluster in color_to_cluster.items():
    if cluster not in colors_in_clusters:
        colors_in_clusters[cluster] = []
    colors_in_clusters[cluster].append(color)

# Generate new sequences based on specific matrice clusters
print("\nGenerating New Sequences Based on Matrice Clusters:")
for cluster in range(k):
    new_sequence = generate_sequence_from_clusters(5, color_to_cluster, colors_in_clusters, [cluster])
    momentum = calculate_momentum(new_sequence, color_to_cluster, cluster_weights)
    print(f"Sequence from Cluster {cluster}: {', '.join(new_sequence)} (Momentum: {momentum:.2f})")

# Generate a sequence with maximum expected momentum
print("\nGenerating a Sequence with Maximum Expected Momentum:")
# Find cluster pairs with highest weights
flat_weights = cluster_weights.flatten()
top_indices = np.argsort(flat_weights)[-5:]  # Top 5 transitions
top_pairs = [(idx // k, idx % k) for idx in top_indices]

# Create a sequence alternating between these clusters
high_momentum_sequence = []
for _ in range(5):  # Generate a sequence of length 10
    pair = top_pairs[np.random.randint(0, len(top_pairs))]
    if pair[0] in colors_in_clusters and colors_in_clusters[pair[0]]:
        high_momentum_sequence.append(np.random.choice(colors_in_clusters[pair[0]]))
    if pair[1] in colors_in_clusters and colors_in_clusters[pair[1]]:
        high_momentum_sequence.append(np.random.choice(colors_in_clusters[pair[1]]))

momentum = calculate_momentum(high_momentum_sequence, color_to_cluster, cluster_weights)
print(f"High momentum sequence: {', '.join(high_momentum_sequence)} (Momentum: {momentum:.2f})")

# Visualize the clusters in RGB space
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define a list of distinct colors for visualization
viz_colors = ['red', 'blue', 'green', 'purple', 'orange']

for cluster in range(k):
    cluster_colors = colors_df[colors_df['cluster'] == cluster]
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
ax.set_title('Color Clusters in RGB Space')
plt.legend()

# Create a visualization of cluster matrice profiles
# Get all unique matrice values
unique_matrice = colors_df['matrice'].unique()
matrice_counts = np.zeros((k, len(unique_matrice)))

for cluster in range(k):
    cluster_colors = colors_df[colors_df['cluster'] == cluster]
    for i, matrice in enumerate(unique_matrice):
        matrice_counts[cluster, i] = (cluster_colors['matrice'] == matrice).sum() / len(cluster_colors)

# Plot matrice profiles
plt.figure(figsize=(14, 8))
bar_width = 0.15
index = np.arange(len(unique_matrice))

for i in range(k):
    plt.bar(index + i * bar_width, matrice_counts[i], bar_width,
            label=f'Cluster {i}', color=viz_colors[i % len(viz_colors)])

plt.xlabel('Matrice Values')
plt.ylabel('Proportion in Cluster')
plt.title('Matrice Profiles by Cluster')
plt.xticks(index + bar_width * (k-1)/2, unique_matrice, rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Create a visualization of the emotional trajectory of sequences
def plot_sequence_trajectory(sequence_name, sequence, color_to_cluster):
    valid_colors = [color for color in sequence if color in color_to_cluster]
    if len(valid_colors) < 2:
        return

    clusters = [color_to_cluster[color] for color in valid_colors]

    # Get RGB values for each color
    rgb_values = []
    matrice_values = []
    for color in valid_colors:
        color_row = colors_df[colors_df['color'] == color].iloc[0]
        rgb_values.append((color_row['r'], color_row['g'], color_row['b']))
        matrice_values.append(color_row['matrice'])

    # Plot the trajectory in RGB space
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    for i, (r, g, b) in enumerate(rgb_values):
        ax.scatter(r, g, b, color=viz_colors[clusters[i]], s=100, label=f'{valid_colors[i]} (Cluster {clusters[i]})' if i == 0 else "")

    # Plot lines connecting points
    for i in range(len(rgb_values) - 1):
        ax.plot([rgb_values[i][0], rgb_values[i+1][0]],
                [rgb_values[i][1], rgb_values[i+1][1]],
                [rgb_values[i][2], rgb_values[i+1][2]],
                'k-', alpha=0.5)

    # Add arrows to show direction
    for i in range(len(rgb_values) - 1):
        ax.quiver(rgb_values[i][0], rgb_values[i][1], rgb_values[i][2],
                 rgb_values[i+1][0] - rgb_values[i][0],
                 rgb_values[i+1][1] - rgb_values[i][1],
                 rgb_values[i+1][2] - rgb_values[i][2],
                 color='black', arrow_length_ratio=0.1)

    ax.set_xlabel('Red')
    ax.set_ylabel('Green')
    ax.set_zlabel('Blue')
    ax.set_title(f'Color Trajectory: {sequence_name}')

    # Add color names and matrice values as annotations
    for i, (r, g, b) in enumerate(rgb_values):
        ax.text(r, g, b, f'  {valid_colors[i]} ({matrice_values[i]})', size=10)

    plt.legend()
    plt.tight_layout()

    # Create a second plot showing the matrice transitions
    plt.figure(figsize=(12, 4))
    for i in range(len(valid_colors)):
        plt.scatter(i, 0, s=200, color=viz_colors[clusters[i]])
        plt.text(i, 0.1, valid_colors[i], ha='center')
        plt.text(i, -0.1, matrice_values[i], ha='center', fontweight='bold')

        if i < len(valid_colors) - 1:
            plt.arrow(i + 0.1, 0, 0.8, 0, head_width=0.05, head_length=0.1, fc='black', ec='black')

    plt.ylim(-0.5, 0.5)
    plt.xlim(-0.5, len(valid_colors) - 0.5)
    plt.axis('off')
    plt.title(f'Matrice Transition Sequence: {sequence_name}')
    plt.tight_layout()

# Plot trajectories for a few example sequences
print("\nPlotting color trajectories for example sequences...")
for _, row in sequences_df.head(3).iterrows():
    plot_sequence_trajectory(row['ï»¿name'], row['parsed_sequence'], color_to_cluster)

# Create a correlation matrix between RGB values and matrice values
# First, create dummy variables for matrice
matrice_dummies = pd.get_dummies(colors_df['matrice'])

# Combine with RGB values
rgb_matrice_df = pd.concat([colors_df[['r', 'g', 'b']], matrice_dummies], axis=1)

# Calculate correlation matrix
corr_matrix = rgb_matrice_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.title('Correlation between RGB Values and Matrice Values')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

# Add correlation values
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        if i < 3 and j >= 3:  # Only show RGB to matrice correlations
            plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                    fontsize=8)

plt.tight_layout()

print("\nAnalysis complete!")