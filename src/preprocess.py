# ============================================================
# FILE: src/preprocess.py
# PURPOSE: Text cleaning and preprocessing for sentiment analysis
# ============================================================

import re
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


# ------------------------------------------------------------
# Download required NLTK resources (safe to run multiple times)
# ------------------------------------------------------------
def download_nltk_data():
    resources = [
        ('corpora/stopwords', 'stopwords'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/wordnet', 'wordnet'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name)


download_nltk_data()


# ------------------------------------------------------------
# Global objects
# ------------------------------------------------------------
STOP_WORDS = set(stopwords.words("english"))

NEGATION_WORDS = {
    "no", "not", "nor", "never", "neither",
    "nobody", "nothing", "nowhere", "hardly",
    "barely", "scarcely"
}

STOP_WORDS -= NEGATION_WORDS

lemmatizer = WordNetLemmatizer()


# ------------------------------------------------------------
# POS tag mapping for lemmatization
# ------------------------------------------------------------
def get_wordnet_pos(word):

    tag = nltk.pos_tag([word])[0][1][0].upper()

    tag_map = {
        "J": wordnet.ADJ,
        "V": wordnet.VERB,
        "N": wordnet.NOUN,
        "R": wordnet.ADV,
    }

    return tag_map.get(tag, wordnet.NOUN)


# ------------------------------------------------------------
# Clean raw text
# ------------------------------------------------------------
def clean_text(text):

    text = str(text)
    text = text.lower()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------------
# Tokenize text
# ------------------------------------------------------------
def tokenize_text(text):

    tokens = word_tokenize(text)

    tokens = [token for token in tokens if token.isalpha()]

    return tokens


# ------------------------------------------------------------
# Remove stopwords
# ------------------------------------------------------------
def remove_stopwords(tokens):

    filtered = [
        token for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    return filtered


# ------------------------------------------------------------
# Lemmatize tokens
# ------------------------------------------------------------
def lemmatize_tokens(tokens):

    lemmatized = []

    for token in tokens:

        pos = get_wordnet_pos(token)

        lemma = lemmatizer.lemmatize(token, pos)

        lemmatized.append(lemma)

    return lemmatized


# ------------------------------------------------------------
# Full preprocessing pipeline
# ------------------------------------------------------------
def preprocess_text(text):

    text = clean_text(text)

    tokens = tokenize_text(text)

    tokens = remove_stopwords(tokens)

    tokens = lemmatize_tokens(tokens)

    result = " ".join(tokens)

    return result


# ------------------------------------------------------------
# Apply preprocessing to dataset
# ------------------------------------------------------------
def preprocess_dataset(df, text_column="review", show_progress=True):

    df_clean = df.copy()

    total = len(df_clean)

    if show_progress:
        print(f"Preprocessing {total} reviews...\n")

    cleaned_reviews = []

    for idx, text in enumerate(df_clean[text_column]):

        if show_progress and idx % 500 == 0 and idx > 0:
            pct = (idx / total) * 100
            print(f"Progress: {idx}/{total} ({pct:.0f}%)")

        cleaned_reviews.append(preprocess_text(text))

    df_clean["clean_review"] = cleaned_reviews

    before = len(df_clean)

    df_clean = df_clean[df_clean["clean_review"].str.strip() != ""]
    df_clean = df_clean.reset_index(drop=True)

    after = len(df_clean)

    if show_progress:
        print("\nPreprocessing complete")
        print("Original rows:", before)
        print("Rows after cleaning:", after)
        print("Removed rows:", before - after)

    return df_clean
if __name__ == "__main__":

    print("Loading dataset...")

    df = pd.read_csv("data/raw/reviews.csv")

    print("Starting preprocessing...\n")

    df_clean = preprocess_dataset(df)

    import os
    os.makedirs("data/processed", exist_ok=True)

    output_path = "data/processed/clean_reviews.csv"

    df_clean.to_csv(output_path, index=False)

    print("\nSaved processed dataset to:", output_path)

    print("\nSample rows:")
    print(df_clean.head())