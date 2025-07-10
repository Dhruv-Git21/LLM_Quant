# src/llm_sentiment.py
# LLM/NLP pipeline for extracting sentiment from text.

from transformers import pipeline
import pandas as pd

def get_sentiment_pipeline():
    """Initializes FinBERT sentiment analysis pipeline."""
    return pipeline("sentiment-analysis", model="ProsusAI/finbert")

def extract_sentiment_from_headline(headline, nlp=None):
    if nlp is None:
        nlp = get_sentiment_pipeline()
    try:
        result = nlp(headline)[0]
        label = result['label']
        score = result['score']
    except Exception:
        label, score = "neutral", 0.0
    return label, score

def sentiment_df(news_df, text_col='headline'):
    nlp = get_sentiment_pipeline()
    sentiments = news_df[text_col].apply(lambda x: extract_sentiment_from_headline(x, nlp))
    news_df['sentiment_label'] = [s[0] for s in sentiments]
    news_df['sentiment_score'] = [s[1] for s in sentiments]
    return news_df
