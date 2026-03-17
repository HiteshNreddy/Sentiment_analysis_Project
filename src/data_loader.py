# ============================================================
# FILE: src/data_loader.py
# PURPOSE: Load and prepare the dataset for sentiment analysis
# ============================================================

import pandas as pd
import nltk
from nltk.corpus import movie_reviews

# ------------------------------------------------------------
# STEP 1 — Download required NLTK resources
# ------------------------------------------------------------

print("Downloading required NLTK resources...")

nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

print("NLTK setup complete.\n")

# ------------------------------------------------------------
# STEP 2 — Build 3-class dataset
# ------------------------------------------------------------

def build_three_class_dataset():

    pos_reviews = []
    neg_reviews = []

    # Load positive reviews
    for fileid in movie_reviews.fileids('pos'):
        pos_reviews.append(' '.join(movie_reviews.words(fileid)))

    # Load negative reviews
    for fileid in movie_reviews.fileids('neg'):
        neg_reviews.append(' '.join(movie_reviews.words(fileid)))

    # Create neutral reviews by mixing pos + neg text
    neutral_reviews = []

    for i in range(min(len(pos_reviews), len(neg_reviews))):

        pos_words = pos_reviews[i].split()[:60]
        neg_words = neg_reviews[i].split()[:60]

        neutral_text = ' '.join(pos_words + neg_words)
        neutral_reviews.append(neutral_text)

    neutral_reviews = neutral_reviews[:1000]

    # Build dataset
    reviews = pos_reviews + neg_reviews + neutral_reviews

    sentiments = (
        ['positive'] * len(pos_reviews) +
        ['negative'] * len(neg_reviews) +
        ['neutral'] * len(neutral_reviews)
    )

    df = pd.DataFrame({
        "review": reviews,
        "sentiment": sentiments
    })

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


# ------------------------------------------------------------
# STEP 3 — Load dataset
# ------------------------------------------------------------

df = build_three_class_dataset()

print("Dataset created successfully.")
print("Total samples:", len(df))

print("\nSentiment distribution:")
print(df['sentiment'].value_counts())

print("\nSample rows:")
print(df.head())