"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
Find summaries of the trajectories of the 11 La Matriz sequences.
'''


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import wordnet as wn
import requests
from io import StringIO
from textblob import TextBlob
import random
import string
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

print("Loading sentence transformer model...")
# Load a sentence transformer model (free alternative to OpenAI)
# This is a smaller model that works well for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully!")

def fetch_csv(url):
    """Fetch CSV data from a URL."""
    print(f"Fetching data from: {url}")
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.content.decode('utf-8')))
    else:
        raise Exception(f"Failed to fetch data: {response.status_code}")

# Function to perform semantic analysis on color descriptions using TextBlob
def analyze_color_semantics(color_data):
    print("\nAnalyzing color semantics...")
    results = {}
    for _, row in color_data.iterrows():
        color = str(row['color'])
        english_words = row['english-words']

        # TextBlob analysis for sentiment and subjectivity
        blob = TextBlob(english_words)
        sentiment = blob.sentiment

        # Store analysis results
        results[color] = {
            'rgb': (float(row['r']), float(row['g']), float(row['b'])),
            'english_words': english_words,
            'english_words_code': row.get('english-words-code', ''),
            'sentiment_polarity': sentiment.polarity,
            'sentiment_subjectivity': sentiment.subjectivity,
            'matrice': row.get('matrice', ''),
            'matrice1': row.get('matrice1', '')
        }
        print(f"  Analyzed '{color}': sentiment={sentiment.polarity:.2f}, subjectivity={sentiment.subjectivity:.2f}")

    return results

# Function to analyze transitions between colors in a sequence
def analyze_color_transitions(colors, color_semantics):
    transitions = []

    for i in range(len(colors) - 1):
        color1 = colors[i]
        color2 = colors[i+1]

        if color1 in color_semantics and color2 in color_semantics:
            # Get semantic data for both colors
            sem1 = color_semantics[color1]
            sem2 = color_semantics[color2]

            # Calculate sentiment shift
            sentiment_shift = sem2['sentiment_polarity'] - sem1['sentiment_polarity']

            # Determine transition type based on sentiment shift
            transition_type = "neutral"
            if sentiment_shift > 0.3:
                transition_type = "strong positive shift"
            elif sentiment_shift > 0.1:
                transition_type = "positive shift"
            elif sentiment_shift < -0.3:
                transition_type = "strong negative shift"
            elif sentiment_shift < -0.1:
                transition_type = "negative shift"
            else:
                transition_type = "subtle shift"

            # Get embeddings for the English words descriptions
            text1 = sem1['english_words']
            text2 = sem2['english_words']

            # Calculate semantic similarity using sentence transformers
            embedding1 = model.encode(text1)
            embedding2 = model.encode(text2)
            semantic_similarity = cosine_similarity([embedding1], [embedding2])[0][0]

            # Store transition data
            transitions.append({
                'from': color1,
                'to': color2,
                'sentiment_shift': sentiment_shift,
                'transition_type': transition_type,
                'semantic_similarity': semantic_similarity,
                'conjoining_interpretation': f"The transition from {color1} to {color2} represents a {transition_type}."
            })

    return transitions

# Function to analyze color sequences with transitions
def analyze_sequences_with_transitions(sequences_data, color_semantics):
    print("\nAnalyzing color sequences and transitions...")
    sequence_analysis = {}

    for _, row in sequences_data.iterrows():
        name = row['name']
        sequence = row['sequence']

        print(f"\n  Analyzing pathway: {name}")
        print(f"  Sequence: {sequence}")

        # Split the sequence into individual colors
        colors = [color.strip() for color in sequence.split(',')]

        # Analyze transitions between colors
        transitions = analyze_color_transitions(colors, color_semantics)

        # Collect semantic data for each color in the sequence
        color_data = []
        missing_colors = []
        for color in colors:
            if color in color_semantics:
                color_data.append(color_semantics[color])
            else:
                missing_colors.append(color)
                print(f"  Warning: Color '{color}' not found in color semantics data")

        if missing_colors:
            print(f"  Note: {len(missing_colors)} colors in this sequence were not found in the color data: {', '.join(missing_colors)}")

        sequence_analysis[name] = {
            'colors': colors,
            'color_data': color_data,
            'transitions': transitions,
            'missing_colors': missing_colors
        }

    return sequence_analysis

# Function to analyze RGB vectors in relation to word semantics
def analyze_rgb_word_semantics(colors, color_semantics):
    results = []

    for color in colors:
        if color in color_semantics:
            sem = color_semantics[color]
            rgb = sem['rgb']

            # Normalize RGB to [0,1]
            normalized_rgb = (rgb[0]/255, rgb[1]/255, rgb[2]/255)

            # Create a simple mapping from RGB to semantic properties
            # This is a simplified approach - in reality, the relationship is more complex
            brightness = sum(normalized_rgb) / 3
            saturation = max(normalized_rgb) - min(normalized_rgb)

            # Associate RGB properties with semantic concepts
            semantic_properties = {
                'brightness': brightness,
                'saturation': saturation,
                'red_dominance': normalized_rgb[0] / sum(normalized_rgb) if sum(normalized_rgb) > 0 else 0,
                'green_dominance': normalized_rgb[1] / sum(normalized_rgb) if sum(normalized_rgb) > 0 else 0,
                'blue_dominance': normalized_rgb[2] / sum(normalized_rgb) if sum(normalized_rgb) > 0 else 0
            }

            # Map RGB properties to semantic concepts
            semantic_mapping = {}

            if semantic_properties['brightness'] > 0.7:
                semantic_mapping['brightness'] = "light, bright, clear, open"
            elif semantic_properties['brightness'] < 0.3:
                semantic_mapping['brightness'] = "dark, deep, mysterious, closed"
            else:
                semantic_mapping['brightness'] = "balanced, moderate, neutral"

            if semantic_properties['saturation'] > 0.5:
                semantic_mapping['saturation'] = "vibrant, intense, energetic"
            else:
                semantic_mapping['saturation'] = "subtle, calm, reserved"

            if semantic_properties['red_dominance'] > 0.5:
                semantic_mapping['dominance'] = "passionate, energetic, aggressive"
            elif semantic_properties['green_dominance'] > 0.5:
                semantic_mapping['dominance'] = "natural, balanced, growth-oriented"
            elif semantic_properties['blue_dominance'] > 0.5:
                semantic_mapping['dominance'] = "calm, trustworthy, deep"
            else:
                semantic_mapping['dominance'] = "balanced, complex"

            results.append({
                'color': color,
                'rgb': rgb,
                'normalized_rgb': normalized_rgb,
                'semantic_properties': semantic_properties,
                'semantic_mapping': semantic_mapping
            })

    return results

# Function to generate summaries using a template-based approach (free alternative to OpenAI)
def generate_template_summary(name, colors, transitions, color_semantics, rgb_semantics, missing_colors):
    # Get sentiment values for all colors in the sequence
    sentiment_values = [color_semantics[color]['sentiment_polarity'] for color in colors if color in color_semantics]
    avg_sentiment = sum(sentiment_values) / len(sentiment_values) if sentiment_values else 0

    # Determine overall sentiment trend
    if len(sentiment_values) >= 2:
        first_sentiment = sentiment_values[0]
        last_sentiment = sentiment_values[-1]

        if last_sentiment > first_sentiment + 0.2:
            sentiment_trend = "positive progression (improving sentiment)"
        elif first_sentiment > last_sentiment + 0.2:
            sentiment_trend = "negative progression (declining sentiment)"
        else:
            sentiment_trend = "stable progression (consistent sentiment)"
    else:
        sentiment_trend = "singular emotional state"

    # Get key words for each color
    color_words = []
    for color in colors:
        if color in color_semantics:
            color_words.append({
                'color': color,
                'code_word': color_semantics[color]['english_words_code'],
                'description': color_semantics[color]['english_words']
            })

    # Get RGB semantic mappings
    rgb_mappings = []
    for item in rgb_semantics:
        rgb_mappings.append({
            'color': item['color'],
            'brightness': item['semantic_mapping']['brightness'],
            'saturation': item['semantic_mapping']['saturation'],
            'dominance': item['semantic_mapping']['dominance']
        })

    # Generate summary using a template
    summary = f"# Analysis of the '{name}' Color Pathway\n\n"

    # Overview section
    summary += "## Overview\n\n"

    # Note about missing colors
    if missing_colors:
        summary += f"**Note:** {len(missing_colors)} colors in this sequence were not found in the color data: {', '.join(missing_colors)}. "
        summary += "The analysis below is based only on the colors that were found in the dataset.\n\n"

    # Continue with overview
    available_colors = [c for c in colors if c in color_semantics]
    summary += f"The '{name}' pathway consists of {len(colors)} colors: {', '.join(colors)}. "

    if sentiment_values:
        summary += f"Overall, this pathway exhibits a {sentiment_trend} with an average sentiment of {avg_sentiment:.2f} "
        summary += "(positive)" if avg_sentiment > 0 else "(negative)" if avg_sentiment < 0 else "(neutral)"
        summary += ".\n\n"
    else:
        summary += "No sentiment analysis could be performed as none of the colors in this sequence were found in the dataset.\n\n"
        return summary  # Return early if no colors were found

    # Color analysis section
    summary += "## Individual Color Analysis\n\n"
    for word_data in color_words:
        color = word_data['color']
        rgb_data = next((item for item in rgb_mappings if item['color'] == color), None)

        summary += f"### {color.capitalize()}\n"
        summary += f"- **Key concept**: {word_data['code_word']}\n"
        summary += f"- **Description**: {word_data['description']}\n"

        if rgb_data:
            summary += f"- **RGB characteristics**: {rgb_data['brightness']}; {rgb_data['saturation']}; {rgb_data['dominance']}\n"

        if color in color_semantics:
            sem = color_semantics[color]
            summary += f"- **Sentiment**: {sem['sentiment_polarity']:.2f} "
            summary += "(positive)" if sem['sentiment_polarity'] > 0 else "(negative)" if sem['sentiment_polarity'] < 0 else "(neutral)"
            summary += f", Subjectivity: {sem['sentiment_subjectivity']:.2f}\n"

        summary += "\n"

    # Transitions analysis section
    if transitions:
        summary += "## Transitions Analysis\n\n"
        for i, t in enumerate(transitions):
            summary += f"### {t['from'].capitalize()} → {t['to'].capitalize()}\n"
            summary += f"- **Type**: {t['transition_type']}\n"
            summary += f"- **Sentiment shift**: {t['sentiment_shift']:.2f}\n"
            summary += f"- **Semantic similarity**: {t['semantic_similarity']:.2f}\n"
            summary += f"- **Interpretation**: {t['conjoining_interpretation']}\n\n"

    # Overall interpretation section
    summary += "## Overall Interpretation\n\n"

    # Generate interpretation based on the analysis
    if sentiment_trend == "positive progression (improving sentiment)":
        summary += "This pathway represents a journey of improvement, growth, or resolution. "
        summary += "It begins with more challenging or negative emotions and progresses toward more positive states. "
        summary += "This type of progression is often associated with narratives of overcoming obstacles, personal growth, or resolution of conflicts.\n\n"
    elif sentiment_trend == "negative progression (declining sentiment)":
        summary += "This pathway represents a journey of increasing challenge, decline, or complexity. "
        summary += "It begins with more positive or stable emotions and moves toward more difficult or negative states. "
        summary += "This type of progression can be associated with narratives of increasing tension, confronting difficulties, or exploring deeper emotional complexities.\n\n"
    else:
        summary += "This pathway maintains a relatively consistent emotional tone throughout. "
        summary += "This stability suggests a steady state or persistent emotional quality. "
        summary += "Such pathways can represent enduring emotional states, established environments, or consistent character traits.\n\n"

    # Add specific interpretation based on the colors
    if len(sentiment_values) >= 2:
        first_color = next((c for c in colors if c in color_semantics), None)
        last_color = next((c for c in reversed(colors) if c in color_semantics), None)

        if first_color and last_color and first_color in color_semantics and last_color in color_semantics:
            first_code = color_semantics[first_color]['english_words_code']
            last_code = color_semantics[last_color]['english_words_code']

            summary += f"The journey from '{first_code}' to '{last_code}' suggests a narrative arc that "

            # Add specific interpretation based on the first and last colors
            first_sentiment = color_semantics[first_color]['sentiment_polarity']
            last_sentiment = color_semantics[last_color]['sentiment_polarity']

            if first_sentiment > 0 and last_sentiment > 0:
                summary += "maintains positive qualities while transforming their expression or focus.\n\n"
            elif first_sentiment < 0 and last_sentiment > 0:
                summary += "transforms challenging or negative aspects into positive outcomes or realizations.\n\n"
            elif first_sentiment > 0 and last_sentiment < 0:
                summary += "explores how initially positive situations can develop complexity or challenges.\n\n"
            else:
                summary += "navigates through different forms of challenge or complexity.\n\n"

    # Applications section
    summary += "## Potential Applications\n\n"

    # Suggest applications based on the analysis
    applications = []

    if avg_sentiment > 0.3:
        applications.append("Marketing campaigns focused on positive emotions and aspirational messaging")
        applications.append("Wellness and personal development contexts")
    elif avg_sentiment < -0.3:
        applications.append("Dramatic or intense storytelling")
        applications.append("Art that explores complex or challenging emotions")
    else:
        applications.append("Balanced communication that acknowledges multiple perspectives")
        applications.append("Educational contexts where neutral presentation is valued")

    if sentiment_trend == "positive progression (improving sentiment)":
        applications.append("Narratives of personal growth or transformation")
        applications.append("Product journeys that show improvement or problem-solving")
    elif sentiment_trend == "negative progression (declining sentiment)":
        applications.append("Suspense or thriller narratives")
        applications.append("Awareness campaigns that highlight developing problems")

    # Add the applications to the summary
    for app in applications:
        summary += f"- {app}\n"

    return summary

# Main execution
def main():
    # URLs for the CSV files - using the correct URLs provided by the user
    matrice_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/la%20matrice-lhndo9QxXACLKNX51xseKuGB4otmGq.csv"
    sequences_url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/la%20matrice%20sequences-CXBDEiMweV8QV70yHZWb5wWRwugMW3.csv"

    # Load the datasets
    print("Fetching color data from 'la matrice.csv'...")
    matrice_df = fetch_csv(matrice_url)
    print(f"Successfully loaded color data with {len(matrice_df)} rows")

    print("Fetching sequence data from 'la matrice sequences.csv'...")
    sequences_df = fetch_csv(sequences_url)
    print(f"Successfully loaded sequence data with {len(sequences_df)} rows")

    # Print column names to help with debugging
    print("\nColor data columns:", matrice_df.columns.tolist())
    print("Sequence data columns:", sequences_df.columns.tolist())

    # Print a sample of each dataset
    print("\nSample of color data:")
    print(matrice_df.head(2))
    print("\nSample of sequence data:")
    print(sequences_df.head(2))

    # Analyze color semantics
    color_semantics = analyze_color_semantics(matrice_df)
    print(f"Analyzed semantics for {len(color_semantics)} colors")

    # Analyze sequences with transitions
    sequence_analysis = analyze_sequences_with_transitions(sequences_df, color_semantics)
    print(f"Analyzed {len(sequence_analysis)} color sequences with transitions")

    # Generate summaries for each pathway
    print("\nGenerating summaries for color pathways...")
    summaries = {}

    for name, data in sequence_analysis.items():
        colors = data['colors']
        transitions = data['transitions']
        missing_colors = data['missing_colors']

        # Analyze RGB vectors in relation to word semantics
        rgb_semantics = analyze_rgb_word_semantics(colors, color_semantics)

        # Generate summary using template-based approach
        summary = generate_template_summary(name, colors, transitions, color_semantics, rgb_semantics, missing_colors)
        summaries[name] = summary

        print(f"Generated summary for '{name}' pathway")

    return color_semantics, sequence_analysis, summaries

if __name__ == "__main__":
    color_semantics, sequence_analysis, summaries = main()

    # Print detailed analysis for each pathway
    for name, summary in summaries.items():
        print("\n" + "="*80)
        print(f"SUMMARY FOR '{name.upper()}' PATHWAY")
        print("="*80)
        print(summary)