"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Analyze color sequence momentum for input phrase.
'''

import pandas as pd
import numpy as np
import requests
from io import StringIO
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import random
import re
from collections import Counter, defaultdict
import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class SemanticColorPhraseGenerator:
    def __init__(self):
        self.semantic_df = None
        self.word_to_rgb_map = {}
        self.rgb_to_words_map = {}
        self.all_semantic_words = set()
        self.color_clusters = None
        self.use_nltk = False

        # Try to initialize NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)

            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.word_tokenize = word_tokenize
            self.use_nltk = True
            print("NLTK initialized successfully")
        except Exception as e:
            print(f"NLTK initialization failed: {e}. Using fallback tokenization.")
            self.use_nltk = False

    def fetch_semantic_data(self, url):
        """Fetch the semantic mapping data from the provided URL"""
        print("Fetching semantic mapping data...")
        try:
            response = requests.get(url)
            if response.status_code == 200:
                self.semantic_df = pd.read_csv(StringIO(response.text))
                # Convert RGB columns to numeric
                for col in ['R', 'G', 'B']:
                    self.semantic_df[col] = pd.to_numeric(self.semantic_df[col], errors='coerce')
                print(f"Loaded semantic data with {len(self.semantic_df)} entries")

                # Build word to RGB mapping for faster lookups
                self._build_word_to_rgb_map()

                # Create color clusters for semantic grouping
                self._create_color_clusters()

                return True
            else:
                print(f"Failed to fetch data: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def _build_word_to_rgb_map(self):
        """Build bidirectional mappings between words and RGB values"""
        self.word_to_rgb_map = {}
        self.rgb_to_words_map = defaultdict(list)
        self.all_semantic_words = set()

        for _, row in self.semantic_df.iterrows():
            rgb = (int(row['R']), int(row['G']), int(row['B']))
            rgb_key = f"{rgb[0]}_{rgb[1]}_{rgb[2]}"

            # Process original words
            if pd.notna(row['Original Words']):
                words = [w.strip().lower() for w in str(row['Original Words']).split(',')]
                for word in words:
                    if word:
                        self.word_to_rgb_map[word] = rgb
                        self.rgb_to_words_map[rgb_key].append(word)
                        self.all_semantic_words.add(word)

            # Process new words
            if pd.notna(row['New Words']):
                words = [w.strip().lower() for w in str(row['New Words']).split(',')]
                for word in words:
                    if word:
                        self.word_to_rgb_map[word] = rgb
                        self.rgb_to_words_map[rgb_key].append(word)
                        self.all_semantic_words.add(word)

        print(f"Built word-to-RGB map with {len(self.word_to_rgb_map)} entries")
        print(f"Built RGB-to-words map with {len(self.rgb_to_words_map)} entries")

    def _create_color_clusters(self, n_clusters=8):
        """Create clusters of colors for semantic grouping"""
        if self.semantic_df is None:
            print("Semantic data not loaded. Please fetch data first.")
            return

        # Extract RGB values
        rgb_values = self.semantic_df[['R', 'G', 'B']].values

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.semantic_df['cluster'] = kmeans.fit_predict(rgb_values)

        # Store cluster centers
        self.color_clusters = kmeans.cluster_centers_

        # Create a mapping of clusters to semantic themes
        cluster_words = defaultdict(list)
        for _, row in self.semantic_df.iterrows():
            cluster = row['cluster']

            if pd.notna(row['Original Words']):
                words = [w.strip().lower() for w in str(row['Original Words']).split(',')]
                cluster_words[cluster].extend(words)

            if pd.notna(row['New Words']):
                words = [w.strip().lower() for w in str(row['New Words']).split(',')]
                cluster_words[cluster].extend(words)

        # Find most common words for each cluster
        self.cluster_themes = {}
        for cluster, words in cluster_words.items():
            word_counts = Counter(words)
            most_common = word_counts.most_common(5)
            self.cluster_themes[cluster] = [word for word, _ in most_common if word]

        print(f"Created {n_clusters} color clusters with semantic themes")

    def preprocess_text(self, text):
        """Preprocess text by tokenizing, removing stopwords, and lemmatizing"""
        if self.use_nltk:
            # Use NLTK for preprocessing
            try:
                # Tokenize
                tokens = self.word_tokenize(text.lower())

                # Remove stopwords and non-alphabetic tokens
                filtered_tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]

                # Lemmatize
                lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in filtered_tokens]

                return lemmatized_tokens
            except Exception as e:
                print(f"NLTK preprocessing failed: {e}. Using fallback preprocessing.")
                # Fall through to the fallback method

        # Fallback simple preprocessing
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())

        # Split into words
        words = text.split()

        # Simple stopwords list
        simple_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                           'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                           'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                           'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                           'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                           'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                           'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                           'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                           'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                           'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
                           'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won',
                           'wouldn', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                           'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                           'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
                           'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who',
                           'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
                           'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                           'does', 'did', 'doing'}

        # Filter out stopwords
        filtered_words = [word for word in words if word not in simple_stopwords and len(word) > 1]

        return filtered_words

    def analyze_phrase(self, phrase):
        """Analyze a user phrase to extract key concepts and sentiment"""
        # Preprocess the phrase
        tokens = self.preprocess_text(phrase)

        # Count token frequency
        token_counts = Counter(tokens)

        # Simple sentiment analysis
        positive_words = {'love', 'happy', 'joy', 'good', 'great', 'excellent', 'wonderful',
                         'amazing', 'fantastic', 'beautiful', 'nice', 'best', 'better',
                         'awesome', 'positive', 'success', 'successful', 'win', 'winning'}

        negative_words = {'hate', 'sad', 'unhappy', 'bad', 'terrible', 'awful', 'horrible',
                         'worst', 'worse', 'negative', 'fail', 'failure', 'lose', 'losing',
                         'lost', 'difficult', 'hard', 'trouble', 'problem', 'worry', 'worried',
                         'fear', 'afraid', 'scary', 'angry', 'mad', 'upset', 'hurt', 'pain',
                         'painful', 'sick', 'ill', 'disease', 'death', 'dead', 'dying', 'cry',
                         'crying', 'tears', 'alone', 'lonely', 'abandoned', 'reject', 'rejected'}

        # Count positive and negative words
        pos_count = sum(1 for token in tokens if token.lower() in positive_words)
        neg_count = sum(1 for token in tokens if token.lower() in negative_words)

        if pos_count > neg_count:
            sentiment = "POSITIVE"
            sentiment_score = min(0.9, 0.5 + (pos_count - neg_count) * 0.1)
        elif neg_count > pos_count:
            sentiment = "NEGATIVE"
            sentiment_score = min(0.9, 0.5 + (neg_count - pos_count) * 0.1)
        else:
            sentiment = "NEUTRAL"
            sentiment_score = 0.5

            # Check for specific negative phrases
            if any(neg_phrase in phrase.lower() for neg_phrase in
                  ["doesn't love", "don't love", "not love", "no love",
                   "hate", "dislike", "abandon", "reject", "alone"]):
                sentiment = "NEGATIVE"
                sentiment_score = 0.7

        # Find similar words in our semantic dataset
        similar_words = []
        for token in tokens:
            if token in self.word_to_rgb_map:
                similar_words.append((token, 1.0))  # Exact match
            else:
                # Find similar words
                matches = self._find_similar_words(token, limit=3)
                similar_words.extend(matches)

        return {
            'tokens': tokens,
            'token_counts': token_counts,
            'sentiment': sentiment,
            'sentiment_score': sentiment_score,
            'similar_words': similar_words
        }

    def _find_similar_words(self, word, limit=3):
        """Find words in our semantic dataset that are similar to the given word"""
        if not self.all_semantic_words:
            return []

        # Simple character-based similarity
        word_scores = []
        for semantic_word in self.all_semantic_words:
            # Skip very short words
            if len(semantic_word) < 3:
                continue

            # Character overlap score
            word_chars = set(word)
            semantic_chars = set(semantic_word)
            if not word_chars or not semantic_chars:
                continue

            overlap_score = len(word_chars.intersection(semantic_chars)) / len(word_chars.union(semantic_chars))

            # Prefix matching
            prefix_len = 0
            for i in range(min(len(word), len(semantic_word))):
                if word[i] == semantic_word[i]:
                    prefix_len += 1
                else:
                    break

            prefix_score = prefix_len / max(len(word), len(semantic_word))

            # Combined score
            combined_score = (overlap_score * 0.6) + (prefix_score * 0.4)

            if combined_score > 0.3:  # Threshold for similarity
                word_scores.append((semantic_word, combined_score))

        # Sort by score (descending)
        word_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top matches
        return word_scores[:limit]

    def find_words_for_rgb(self, r, g, b, max_words=5):
        """Find words associated with a specific RGB value or the closest match"""
        rgb_key = f"{r}_{g}_{b}"

        # Direct match
        if rgb_key in self.rgb_to_words_map and self.rgb_to_words_map[rgb_key]:
            words = self.rgb_to_words_map[rgb_key]
            return random.sample(words, min(max_words, len(words)))

        # Find closest RGB
        min_distance = float('inf')
        closest_rgb = None

        for row_idx, row in self.semantic_df.iterrows():
            distance = np.sqrt((row['R'] - r)**2 + (row['G'] - g)**2 + (row['B'] - b)**2)
            if distance < min_distance:
                min_distance = distance
                closest_rgb = (int(row['R']), int(row['G']), int(row['B']))

        if closest_rgb:
            closest_key = f"{closest_rgb[0]}_{closest_rgb[1]}_{closest_rgb[2]}"
            if closest_key in self.rgb_to_words_map and self.rgb_to_words_map[closest_key]:
                words = self.rgb_to_words_map[closest_key]
                return random.sample(words, min(max_words, len(words)))

        return []

    def generate_color_sequence(self, seed_words=None, length=5, coherence=0.7):
        """Generate a sequence of colors based on seed words or random selection"""
        if self.semantic_df is None:
            print("Semantic data not loaded. Please fetch data first.")
            return []

        rgb_sequence = []

        # Start with seed words if provided
        if seed_words and isinstance(seed_words, list) and len(seed_words) > 0:
            # Find RGB values for seed words
            seed_rgbs = []
            for word in seed_words:
                word = word.lower().strip()
                if word in self.word_to_rgb_map:
                    seed_rgbs.append(self.word_to_rgb_map[word])
                else:
                    # Try to find similar words
                    similar_words = self._find_similar_words(word, limit=3)
                    for similar_word, _ in similar_words:
                        if similar_word in self.word_to_rgb_map:
                            seed_rgbs.append(self.word_to_rgb_map[similar_word])
                            break

            # If we found RGB values, use them as starting points
            if seed_rgbs:
                # Start with a random seed RGB
                current_rgb = random.choice(seed_rgbs)
                rgb_sequence.append(current_rgb)
            else:
                # Start with a random RGB from the dataset
                random_idx = random.randint(0, len(self.semantic_df) - 1)
                current_rgb = (
                    int(self.semantic_df.iloc[random_idx]['R']),
                    int(self.semantic_df.iloc[random_idx]['G']),
                    int(self.semantic_df.iloc[random_idx]['B'])
                )
                rgb_sequence.append(current_rgb)
        else:
            # Start with a random RGB from the dataset
            random_idx = random.randint(0, len(self.semantic_df) - 1)
            current_rgb = (
                int(self.semantic_df.iloc[random_idx]['R']),
                int(self.semantic_df.iloc[random_idx]['G']),
                int(self.semantic_df.iloc[random_idx]['B'])
            )
            rgb_sequence.append(current_rgb)

        # Generate the rest of the sequence
        for i in range(1, length):
            # Decide whether to follow coherence or make a jump
            if random.random() < coherence:
                # Follow coherence - find a color that's similar but with some variation
                variation = random.randint(10, 50)  # Amount of variation

                # Add variation to current RGB
                new_r = max(0, min(255, current_rgb[0] + random.randint(-variation, variation)))
                new_g = max(0, min(255, current_rgb[1] + random.randint(-variation, variation)))
                new_b = max(0, min(255, current_rgb[2] + random.randint(-variation, variation)))

                # Find the closest RGB in our dataset
                min_distance = float('inf')
                closest_rgb = None

                for _, row in self.semantic_df.iterrows():
                    distance = np.sqrt((row['R'] - new_r)**2 + (row['G'] - new_g)**2 + (row['B'] - new_b)**2)
                    if distance < min_distance:
                        min_distance = distance
                        closest_rgb = (int(row['R']), int(row['G']), int(row['B']))

                if closest_rgb:
                    current_rgb = closest_rgb
                    rgb_sequence.append(current_rgb)
            else:
                # Make a jump to a different color cluster
                current_cluster = None

                # Find current cluster
                for _, row in self.semantic_df.iterrows():
                    if (int(row['R']), int(row['G']), int(row['B'])) == current_rgb:
                        current_cluster = row['cluster']
                        break

                # Choose a different cluster
                available_clusters = list(range(len(self.color_clusters)))
                if current_cluster is not None and len(available_clusters) > 1:
                    available_clusters.remove(current_cluster)

                if available_clusters:
                    new_cluster = random.choice(available_clusters)

                    # Find a random RGB in the new cluster
                    cluster_rgbs = []
                    for _, row in self.semantic_df.iterrows():
                        if row['cluster'] == new_cluster:
                            cluster_rgbs.append((int(row['R']), int(row['G']), int(row['B'])))

                    if cluster_rgbs:
                        current_rgb = random.choice(cluster_rgbs)
                        rgb_sequence.append(current_rgb)
                    else:
                        # Fallback to a random RGB
                        random_idx = random.randint(0, len(self.semantic_df) - 1)
                        current_rgb = (
                            int(self.semantic_df.iloc[random_idx]['R']),
                            int(self.semantic_df.iloc[random_idx]['G']),
                            int(self.semantic_df.iloc[random_idx]['B'])
                        )
                        rgb_sequence.append(current_rgb)
                else:
                    # Fallback to a random RGB
                    random_idx = random.randint(0, len(self.semantic_df) - 1)
                    current_rgb = (
                        int(self.semantic_df.iloc[random_idx]['R']),
                        int(self.semantic_df.iloc[random_idx]['G']),
                        int(self.semantic_df.iloc[random_idx]['B'])
                    )
                    rgb_sequence.append(current_rgb)

        return rgb_sequence

    def calculate_momentum(self, rgb_sequence):
        """Calculate the momentum (rate of change) between consecutive colors in the sequence"""
        if len(rgb_sequence) < 2:
            return []

        momentum = []

        for i in range(1, len(rgb_sequence)):
            prev_rgb = rgb_sequence[i-1]
            curr_rgb = rgb_sequence[i]

            # Calculate Euclidean distance between colors
            distance = np.sqrt(
                (curr_rgb[0] - prev_rgb[0])**2 +
                (curr_rgb[1] - prev_rgb[1])**2 +
                (curr_rgb[2] - prev_rgb[2])**2
            )

            # Calculate direction vector
            direction = [
                curr_rgb[0] - prev_rgb[0],
                curr_rgb[1] - prev_rgb[1],
                curr_rgb[2] - prev_rgb[2]
            ]

            # Normalize direction vector
            magnitude = np.sqrt(sum(d**2 for d in direction))
            if magnitude > 0:
                normalized_direction = [d/magnitude for d in direction]
            else:
                normalized_direction = [0, 0, 0]

            momentum.append({
                'distance': distance,
                'direction': direction,
                'normalized_direction': normalized_direction,
                'magnitude': magnitude
            })

        return momentum

    def generate_phrase_from_colors(self, rgb_sequence, momentum=None, max_words_per_color=2):
        """Generate a phrase based on a sequence of colors and their momentum"""
        if not rgb_sequence:
            return ""

        phrase_parts = []

        # Generate words for each color in the sequence
        for i, rgb in enumerate(rgb_sequence):
            r, g, b = rgb

            # Get words for this color
            words = self.find_words_for_rgb(r, g, b, max_words=max_words_per_color)

            if words:
                # If we have momentum data, use it to modify the words or add transitions
                if momentum and i > 0:
                    m = momentum[i-1]

                    # High momentum = strong transition
                    if m['magnitude'] > 100:
                        transition_words = ["suddenly", "dramatically", "powerfully", "intensely"]
                        phrase_parts.append(random.choice(transition_words))
                    # Medium momentum = moderate transition
                    elif m['magnitude'] > 50:
                        transition_words = ["shifting to", "moving toward", "transforming into", "evolving into"]
                        phrase_parts.append(random.choice(transition_words))
                    # Low momentum = subtle transition
                    else:
                        transition_words = ["gently becoming", "subtly shifting to", "quietly blending with", "softly merging into"]
                        phrase_parts.append(random.choice(transition_words))

                # Add the color words
                phrase_parts.append(" and ".join(words))

        # Join all parts into a coherent phrase
        phrase = " ".join(phrase_parts)

        # Capitalize first letter and add period
        phrase = phrase[0].upper() + phrase[1:] + "."

        return phrase

    def visualize_color_sequence(self, rgb_sequence, momentum=None, title="Color Sequence", input_phrase=None):
        """Visualize a sequence of colors in 2D and 3D"""
        if not rgb_sequence:
            return

        # Create a figure with 2 subplots (2D and 3D)
        fig = plt.figure(figsize=(15, 10))

        # Add input phrase to title if provided
        if input_phrase:
            full_title = f"{title}\nInput: \"{input_phrase}\""
        else:
            full_title = title

        plt.suptitle(full_title, fontsize=16, y=0.98)

        # 3D plot of the color sequence
        ax1 = fig.add_subplot(211, projection='3d')

        # Extract RGB components
        r_values = [rgb[0] for rgb in rgb_sequence]
        g_values = [rgb[1] for rgb in rgb_sequence]
        b_values = [rgb[2] for rgb in rgb_sequence]

        # Plot the path
        ax1.plot(r_values, g_values, b_values, 'o-', linewidth=2, markersize=10)

        # Add color to each point
        for i, (r, g, b) in enumerate(rgb_sequence):
            ax1.scatter(r, g, b, color=[r/255, g/255, b/255], s=100)
            ax1.text(r, g, b, f"{i+1}", fontsize=12)

        # If we have momentum data, visualize it with arrows
        if momentum:
            for i in range(len(momentum)):
                r, g, b = rgb_sequence[i]
                next_r, next_g, next_b = rgb_sequence[i+1]
                m = momentum[i]

                # Scale the arrow based on momentum magnitude
                scale = min(50, max(10, m['magnitude'] / 5))

                # Draw an arrow showing the direction and magnitude of momentum
                ax1.quiver(
                    r, g, b,  # Start point
                    m['direction'][0], m['direction'][1], m['direction'][2],  # Direction
                    color='red', alpha=0.6, arrow_length_ratio=0.2,
                    length=scale/100  # Scale the arrow length
                )

        # Set labels and title
        ax1.set_xlabel('Red (R)')
        ax1.set_ylabel('Green (G)')
        ax1.set_zlabel('Blue (B)')
        ax1.set_title('3D Color Trajectory in RGB Space')

        # Set axis limits
        ax1.set_xlim(0, 255)
        ax1.set_ylim(0, 255)
        ax1.set_zlim(0, 255)

        # 2D visualization of the color sequence
        ax2 = fig.add_subplot(212)

        # Create a horizontal bar for each color
        bar_height = 1
        for i, (r, g, b) in enumerate(rgb_sequence):
            ax2.add_patch(plt.Rectangle((i, 0), 1, bar_height, color=[r/255, g/255, b/255]))

            # Add words for this color
            words = self.find_words_for_rgb(r, g, b, max_words=2)
            if words:
                ax2.text(i + 0.5, bar_height + 0.1, ", ".join(words),
                         ha='center', va='bottom', fontsize=10, rotation=45)

            # If we have momentum data, visualize it
            if momentum and i < len(momentum):
                m = momentum[i]
                # Add a marker for momentum magnitude
                marker_height = m['magnitude'] / 255 * bar_height
                ax2.add_patch(plt.Rectangle((i+0.9, 0), 0.1, marker_height, color='red', alpha=0.7))

        # Set labels and title
        ax2.set_xlim(0, len(rgb_sequence))
        ax2.set_ylim(0, bar_height * 2)
        ax2.set_title('Color Sequence with Semantic Words and Momentum')
        ax2.set_xlabel('Sequence Position')
        ax2.set_yticks([])

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()

    def analyze_and_optimize_sequence(self, rgb_sequence, target_momentum='balanced'):
        """Analyze a color sequence and optimize it for a target momentum profile"""
        if not rgb_sequence or len(rgb_sequence) < 2:
            return rgb_sequence

        # Calculate current momentum
        current_momentum = self.calculate_momentum(rgb_sequence)

        # Calculate average and standard deviation of momentum
        magnitudes = [m['magnitude'] for m in current_momentum]
        avg_magnitude = sum(magnitudes) / len(magnitudes)
        std_magnitude = np.std(magnitudes)

        print(f"Current sequence - Avg momentum: {avg_magnitude:.2f}, Std dev: {std_magnitude:.2f}")

        # Create a new optimized sequence
        optimized_sequence = [rgb_sequence[0]]  # Start with the first color

        # Optimize based on target momentum profile
        if target_momentum == 'balanced':
            # Aim for consistent, moderate momentum throughout
            target_mag = 50
            for i in range(1, len(rgb_sequence)):
                prev_rgb = optimized_sequence[-1]

                # Find a color that gives us the target momentum
                best_rgb = None
                best_diff = float('inf')

                # Try 20 candidate colors from the dataset
                candidates = self.semantic_df.sample(min(20, len(self.semantic_df))).iterrows()
                for _, row in candidates:
                    candidate_rgb = (int(row['R']), int(row['G']), int(row['B']))

                    # Calculate momentum if we choose this color
                    distance = np.sqrt(
                        (candidate_rgb[0] - prev_rgb[0])**2 +
                        (candidate_rgb[1] - prev_rgb[1])**2 +
                        (candidate_rgb[2] - prev_rgb[2])**2
                    )

                    # How close is this to our target?
                    diff = abs(distance - target_mag)

                    if diff < best_diff:
                        best_diff = diff
                        best_rgb = candidate_rgb

                if best_rgb:
                    optimized_sequence.append(best_rgb)
                else:
                    # Fallback to original
                    optimized_sequence.append(rgb_sequence[i])

        elif target_momentum == 'increasing':
            # Aim for gradually increasing momentum
            for i in range(1, len(rgb_sequence)):
                prev_rgb = optimized_sequence[-1]

                # Target momentum increases with position
                target_mag = 20 + (i / (len(rgb_sequence) - 1)) * 100

                # Find a color that gives us the target momentum
                best_rgb = None
                best_diff = float('inf')

                # Try 20 candidate colors from the dataset
                candidates = self.semantic_df.sample(min(20, len(self.semantic_df))).iterrows()
                for _, row in candidates:
                    candidate_rgb = (int(row['R']), int(row['G']), int(row['B']))

                    # Calculate momentum if we choose this color
                    distance = np.sqrt(
                        (candidate_rgb[0] - prev_rgb[0])**2 +
                        (candidate_rgb[1] - prev_rgb[1])**2 +
                        (candidate_rgb[2] - prev_rgb[2])**2
                    )

                    # How close is this to our target?
                    diff = abs(distance - target_mag)

                    if diff < best_diff:
                        best_diff = diff
                        best_rgb = candidate_rgb

                if best_rgb:
                    optimized_sequence.append(best_rgb)
                else:
                    # Fallback to original
                    optimized_sequence.append(rgb_sequence[i])

        elif target_momentum == 'decreasing':
            # Aim for gradually decreasing momentum
            for i in range(1, len(rgb_sequence)):
                prev_rgb = optimized_sequence[-1]

                # Target momentum decreases with position
                target_mag = 120 - (i / (len(rgb_sequence) - 1)) * 100

                # Find a color that gives us the target momentum
                best_rgb = None
                best_diff = float('inf')

                # Try 20 candidate colors from the dataset
                candidates = self.semantic_df.sample(min(20, len(self.semantic_df))).iterrows()
                for _, row in candidates:
                    candidate_rgb = (int(row['R']), int(row['G']), int(row['B']))

                    # Calculate momentum if we choose this color
                    distance = np.sqrt(
                        (candidate_rgb[0] - prev_rgb[0])**2 +
                        (candidate_rgb[1] - prev_rgb[1])**2 +
                        (candidate_rgb[2] - prev_rgb[2])**2
                    )

                    # How close is this to our target?
                    diff = abs(distance - target_mag)

                    if diff < best_diff:
                        best_diff = diff
                        best_rgb = candidate_rgb

                if best_rgb:
                    optimized_sequence.append(best_rgb)
                else:
                    # Fallback to original
                    optimized_sequence.append(rgb_sequence[i])

        elif target_momentum == 'wave':
            # Aim for a wave pattern of momentum
            for i in range(1, len(rgb_sequence)):
                prev_rgb = optimized_sequence[-1]

                # Target momentum follows a sine wave
                wave_position = i / (len(rgb_sequence) - 1) * 2 * math.pi
                target_mag = 50 + 50 * math.sin(wave_position)

                # Find a color that gives us the target momentum
                best_rgb = None
                best_diff = float('inf')

                # Try 20 candidate colors from the dataset
                candidates = self.semantic_df.sample(min(20, len(self.semantic_df))).iterrows()
                for _, row in candidates:
                    candidate_rgb = (int(row['R']), int(row['G']), int(row['B']))

                    # Calculate momentum if we choose this color
                    distance = np.sqrt(
                        (candidate_rgb[0] - prev_rgb[0])**2 +
                        (candidate_rgb[1] - prev_rgb[1])**2 +
                        (candidate_rgb[2] - prev_rgb[2])**2
                    )

                    # How close is this to our target?
                    diff = abs(distance - target_mag)

                    if diff < best_diff:
                        best_diff = diff
                        best_rgb = candidate_rgb

                if best_rgb:
                    optimized_sequence.append(best_rgb)
                else:
                    # Fallback to original
                    optimized_sequence.append(rgb_sequence[i])

        else:
            # Unknown target, return original
            return rgb_sequence

        # Calculate new momentum
        new_momentum = self.calculate_momentum(optimized_sequence)
        new_magnitudes = [m['magnitude'] for m in new_momentum]
        new_avg_magnitude = sum(new_magnitudes) / len(new_magnitudes)
        new_std_magnitude = np.std(new_magnitudes)

        print(f"Optimized sequence - Avg momentum: {new_avg_magnitude:.2f}, Std dev: {new_std_magnitude:.2f}")

        return optimized_sequence, new_momentum

    def process_user_input(self, input_phrase, sequence_length=6, momentum_pattern='balanced'):
        """Process user input to generate a color sequence and phrase"""
        if not input_phrase:
            print("Please provide a phrase to process.")
            return None, None, None

        print(f"Processing input phrase: '{input_phrase}'")

        # Analyze the phrase
        analysis = self.analyze_phrase(input_phrase)

        # Extract key words from the phrase
        seed_words = []

        # First, try to use exact matches from our semantic dataset
        for token, count in analysis['token_counts'].items():
            if token in self.word_to_rgb_map:
                seed_words.append(token)

        # If we don't have enough exact matches, use similar words
        if len(seed_words) < 3:
            for word, score in analysis['similar_words']:
                if word not in seed_words:
                    seed_words.append(word)
                if len(seed_words) >= 3:
                    break

        print(f"Extracted seed words: {seed_words}")

        # Generate a color sequence based on the seed words
        rgb_sequence = self.generate_color_sequence(
            seed_words=seed_words,
            length=sequence_length,
            coherence=0.7
        )

        # Calculate momentum
        momentum = self.calculate_momentum(rgb_sequence)

        # Optimize the sequence based on the momentum pattern
        if momentum_pattern != 'original':
            print(f"Optimizing sequence for {momentum_pattern} momentum pattern...")
            rgb_sequence, momentum = self.analyze_and_optimize_sequence(
                rgb_sequence, target_momentum=momentum_pattern)

        # Generate a phrase from the colors
        color_phrase = self.generate_phrase_from_colors(rgb_sequence, momentum)

        return rgb_sequence, momentum, color_phrase

# Main execution
def main():
    # URL for the semantic mapping CSV
    semantic_mapping_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/semantic_rgb_mapping-7Fyy0MQQFX3s6KmXIryYY5kH3cG2qk.csv"

    # Initialize the generator
    generator = SemanticColorPhraseGenerator()

    # Fetch semantic data
    if not generator.fetch_semantic_data(semantic_mapping_url):
        print("Failed to fetch semantic data. Exiting.")
        return

    print("\n" + "="*50)
    print("Welcome to the Semantic Color Phrase Generator!")
    print("This program converts your input phrase into a color sequence")
    print("and generates a new phrase based on the semantic meanings of those colors.")
    print("="*50 + "\n")

    while True:
        # Get user input
        input_phrase = input("\nEnter a phrase (or 'quit' to exit): ")

        if input_phrase.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        # Get sequence length
        try:
            length_input = input("Enter sequence length (3-10, default 6): ")
            sequence_length = int(length_input) if length_input.strip() else 6
            sequence_length = max(3, min(10, sequence_length))  # Ensure within range
        except ValueError:
            sequence_length = 6
            print("Using default sequence length of 6.")

        # Get momentum pattern
        print("\nMomentum patterns:")
        print("1. original - Keep the natural momentum")
        print("2. balanced - Consistent, moderate changes")
        print("3. increasing - Gradually intensifying changes")
        print("4. decreasing - Gradually diminishing changes")
        print("5. wave - Oscillating between subtle and dramatic changes")

        pattern_input = input("Choose a momentum pattern (1-5, default 1): ")

        pattern_map = {
            '1': 'original',
            '2': 'balanced',
            '3': 'increasing',
            '4': 'decreasing',
            '5': 'wave'
        }

        momentum_pattern = pattern_map.get(pattern_input.strip(), 'original')

        # Process the input
        rgb_sequence, momentum, color_phrase = generator.process_user_input(
            input_phrase, sequence_length, momentum_pattern)

        if rgb_sequence and color_phrase:
            print("\n" + "="*50)
            print(f"Input phrase: '{input_phrase}'")
            print(f"Generated color phrase: '{color_phrase}'")
            print("="*50 + "\n")

            # Visualize the sequence
            generator.visualize_color_sequence(
                rgb_sequence,
                momentum,
                title=f"Color Sequence: {momentum_pattern.capitalize()} Momentum",
                input_phrase=input_phrase
            )
        else:
            print("Failed to generate a color sequence. Please try a different phrase.")

if __name__ == "__main__":
    main()