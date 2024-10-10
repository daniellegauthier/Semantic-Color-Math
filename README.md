## Color Wave Analysis and Semantic Mapping
This Jupyter notebook contains two primary analysis modules for color-based data processing and visualization. The project explores the relationship between colors, semantic meaning, and wave-like properties in spatiotemporal sequences.<br>
<br>
# Table of Contents
Dependencies
Module 1: Color Wave Momentum Analysis
Module 2: Color-Word Semantic Analysis
Usage
Data Requirements

# Dependencies
Module 1<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
<br>
Module 2<br>
import pandas as pd<br>
from sklearn.metrics.pairwise import cosine_similarity<br>
import nltk<br>
from nltk.corpus import wordnet as wn<br>
import spacy<br>
from textblob import TextBlob<br>
from english_words import get_english_words_set<br>
<br>
<br>
Ensure you have spaCy's English model installed:<br>
!python -m spacy download en_core_web_sm<br>
<br>
# Module 1: Color Wave Momentum Analysis
Overview<br>
This module treats color sequences as wave functions, calculating and visualizing their momentum properties in 2D space.<br>
<br>
Features<br>
Momentum calculation using wave physics principles<br>
Multiple predefined pathways mapped to color sequences<br>
Comprehensive visualization including:<br>
Real and imaginary parts of momentum<br>
Magnitude and phase<br>
Trajectory visualization with time encoding<br>

# Module 2: Color-Word Semantic Analysis
Overview<br>
This module analyzes the semantic relationship between colors and words, using NLP techniques to find meaningful associations.<br>
<br>
Key Components<br>
RGB normalization
Word-color similarity calculation using spaCy
Sentiment analysis of word-color relationships
<br>
Features<br>
Processes CSV input containing color and word data<br>
Calculates semantic similarity between words and colors<br>
Generates replacement word suggestions based on color properties<br>
Provides detailed scoring including sentiment and color similarity<br>
<br>
# Usage
Running the Wave Analysis<br>
#Choose a predefined pathway<br>
chosen_pathway = 'knot'  # Options: knot, plot, pain, practical, spiritual, etc.<br>
<br>
# Generate and display analysis
plot_momentum_analysis(x_coords, y_coords, t_coords)<br>
Running the Semantic Analysis<br>
pythonCopy# Load your color-word dataset<br>
results_df, color_similarity = main()<br>
<br>
# Data Requirements
For Wave Analysis<br>
No external data file required; uses predefined color sequences.<br>
For Semantic Analysisv
Requires a CSV file ('la matrice.csv') with columns:<br>
'color': Color name<br>
'r', 'g', 'b': RGB values<br>
'matrice': Original word<br>
'matrice1': Additional word column<br>
<br>
# Notes
The wave analysis assumes non-commutative properties in color sequences for conformal time values.<br>
Semantic analysis results are influenced by the spaCy model's training data.<br>
Performance may vary based on the size of your word dataset<br>
<br>
<br>
For questions or discussions about the non-commutative color wave theory, please contact danielle.gauthier6@gmail.com.
