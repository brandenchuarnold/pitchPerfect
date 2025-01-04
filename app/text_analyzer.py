# app/text_analyzer.py

import re
import spacy
from textblob import TextBlob

nlp = spacy.load("en_core_web_sm")


def clean_text(profile_text: str) -> str:
    """Simple cleanup of whitespace and special characters."""
    text = re.sub(r"[^a-zA-Z0-9\s.,!?']", " ", profile_text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(profile_text: str) -> list:
    """Extract named entities or keywords using spaCy."""
    doc = nlp(profile_text)
    keywords = [ent.text for ent in doc.ents]
    return keywords


def analyze_sentiment(profile_text: str) -> str:
    """Return 'positive', 'negative', or 'neutral' based on TextBlob polarity."""
    blob = TextBlob(profile_text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"
