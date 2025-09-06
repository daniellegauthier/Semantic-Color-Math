"""
Core Semantic Color Math Computations

MSDS Thesis Research by Danielle Gauthier
danielle.gauthier@lamatriz.ai
https://lamatriz.ai/

"""

'''
GNH Analyzer

Transform an input phrase to GNH weighted scores, sorted by RGB sentiment color.
'''
# --- Imports ---
import re
import nltk
import spacy
import torch
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any

# --- Load Models ---
nltk.download('punkt')
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
bert_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Emotion model
emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_model_name)

# --- GNH Domain Descriptions ---
GNH_DOMAINS = {
    "Mental Wellness": "mental health, emotional clarity, peace of mind",
    "Social Wellness": "relationships, community, friendship, social harmony",
    "Economic Wellness": "income, savings, financial stability, cost of living",
    "Workplace Wellness": "career, work-life balance, promotion, productivity",
    "Physical Wellness": "physical health, sleep, fitness, exercise",
    "Environmental Wellness": "green space, nature, environmental care",
    "Health": "healthcare, medical care, recovery, well-being",
    "Education Value": "learning, education, school, knowledge, wisdom",
    "Good Governance": "freedom, justice, fairness, democratic participation",
    "Living Standards": "housing, wealth, basic needs, affordability",
    "Cultural Diversity": "tradition, language, cultural expression, heritage",
    "Political Wellness": "rights, law, free speech, civic participation",
    "Ecological Diversity": "biodiversity, forest, ecosystem, wildlife"
}

# GNH Colors (used for plotting)
GNH_COLORS = {
    'Economic Wellness': '#808080',
    'Mental Wellness': '#ffc0cb',
    'Workplace Wellness': '#ffd700',
    'Physical Wellness': '#f5deb3',
    'Social Wellness': '#ffa500',
    'Political Wellness': '#ffffff',
    'Environmental Wellness': '#87ceeb',
    'Ecological Diversity': '#228B22',
    'Health': '#ff6347',
    'Good Governance': '#000000',
    'Education Value': '#8b4513',
    'Living Standards': '#ffff00',
    'Cultural Diversity': '#9370db'
}

# --- Emotion Classification ---
def classify_emotion(text: str) -> Tuple[str, float]:
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()

    labels = emotion_model.config.id2label
    top_idx = torch.argmax(probs).item()
    emotion = labels[top_idx]
    confidence = round(probs[top_idx].item(), 3)
    return emotion, confidence

# --- BERT Sentiment Scoring ---
def score_sentiment(text: str) -> float:
    try:
        result = bert_sentiment(text[:512])[0]
        label = result['label']
        score = result['score']
        if label == 'POSITIVE':
            scaled = 5 + 5 * score
        else:
            scaled = 1 + 4 * (1 - score)
        return round(min(max(scaled, 1), 10), 2)
    except Exception as e:
        print(f"Error: {e}")
        return 5.0

# --- Accomplishment Scoring ---
def score_accomplishment(text: str) -> float:
    doc = nlp(text)
    score = 5.0
    key_phrases = ["finally", "told", "decided", "quit", "refused", "stood", "walked away"]
    for token in doc:
        if token.text.lower() in key_phrases:
            score += 1.5
        if token.tag_ in ["VBD", "VBN"]:
            score += 0.5
    return round(min(max(score, 1), 10), 2)

# --- Semantic Mapping ---
def semantic_indicator_mapping(text: str, sentiment_score: float, sentiment_weight: float = 0.3) -> Dict[str, float]:
    input_vec = sbert_model.encode(text, convert_to_tensor=True)
    result = {}
    for label, desc in GNH_DOMAINS.items():
        desc_vec = sbert_model.encode(desc, convert_to_tensor=True)
        sim = util.cos_sim(input_vec, desc_vec).item()
        norm_sim = max(0, min(sim, 1))
        weighted = (1 - sentiment_weight) * norm_sim + sentiment_weight * (sentiment_score / 10.0)
        result[label] = round(weighted, 3)
    return dict(sorted(result.items(), key=lambda x: -x[1]))

# --- Console Analyzer ---
def analyze_text(text: str, show_chart: bool = True):
    sentiment = score_sentiment(text)
    accomplishment = score_accomplishment(text)
    emotion, emotion_conf = classify_emotion(text)
    indicators = semantic_indicator_mapping(text, sentiment)

    print("\n🧿 Input:")
    print(f"  {text}")
    print(f"\n💠 Sentiment Score (1–10): {sentiment}")
    print(f"📈 Accomplishment Score (1–10): {accomplishment}")
    print(f"🎭 Emotion: {emotion} (confidence: {emotion_conf})")

    print("\n🎨 GNH Indicator Mapping (Top 5):")
    for i, (label, score) in enumerate(list(indicators.items())[:5]):
        print(f"  {i+1}. {label}: {score}")

    if show_chart:
        labels = list(indicators.keys())
        values = list(indicators.values())
        colors = [GNH_COLORS.get(label, "#cccccc") for label in labels]

        plt.figure(figsize=(10, 5))
        plt.barh(labels, values, color=colors)
        plt.gca().invert_yaxis()
        plt.title("GNH Indicator Similarity (Sentiment Weighted)")
        plt.xlabel("Score")
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.show()

# --- Run in Console ---
if __name__ == "__main__":
    print("\n💬 GNH Analyzer (BERT + Emotion + SBERT Mapping)")
    while True:
        user_input = input("\nEnter a phrase (or type 'exit'): ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Exiting. Stay well! 🌱")
            break
        analyze_text(user_input, show_chart=True)
