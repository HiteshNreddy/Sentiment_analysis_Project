# ============================================================
# FILE: src/features.py
# PURPOSE: Convert clean text into numeric feature vectors
#          using TF-IDF (Term Frequency - Inverse Document Frequency)
#
# WHY TF-IDF?
#   ML models need numbers, not words.
#   TF-IDF gives each word a score based on:
#     - TF  : how often the word appears in THIS review
#     - IDF : how rare the word is across ALL reviews
#   Result: rare but meaningful words like "amazing" or "terrible"
#   get higher scores than common words like "movie" or "film".
# ============================================================

import numpy  as np
import pandas as pd
import joblib                          # For saving/loading the vectorizer
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection         import train_test_split

# ---------------------------------------------------------------
# FUNCTION 1: Build and fit the TF-IDF Vectorizer
# ---------------------------------------------------------------
def build_tfidf_vectorizer(
    max_features  = 10000,
    ngram_range   = (1, 2),
    min_df        = 2,
    max_df        = 0.95,
    sublinear_tf  = True
):
    """
    Creates a configured TF-IDF Vectorizer object.

    Parameters explained — each one is a tuning knob:

    max_features (int):
        Keep only the top N most important words by TF-IDF score.
        10,000 is a good balance for a ~3000 review dataset.
        More features = slower training but potentially better accuracy.
        Fewer features = faster but might miss important words.

    ngram_range (tuple):
        (1, 1) = unigrams only  → ["great", "movie"]
        (1, 2) = unigrams + bigrams → ["great", "movie", "great movie"]
        Bigrams capture phrases like "not good", "very bad", "highly recommend"
        which have different meaning than individual words.
        This SIGNIFICANTLY improves accuracy for sentiment analysis.

    min_df (int or float):
        Minimum document frequency.
        min_df=2 means: ignore words that appear in fewer than 2 reviews.
        This removes typos, rare names, one-off words that don't generalize.

    max_df (float):
        Maximum document frequency (as fraction).
        max_df=0.95 means: ignore words appearing in more than 95% of reviews.
        These ultra-common words add noise, not signal.

    sublinear_tf (bool):
        If True, applies log(1 + tf) instead of raw tf.
        This prevents very long reviews from dominating just because
        they repeat words more often.
        Example: a word appearing 100 times is NOT 100x more important
        than one appearing 1 time — log scaling makes it more realistic.

    Returns:
        TfidfVectorizer: A configured but NOT yet trained vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = ngram_range,
        min_df       = min_df,
        max_df       = max_df,
        sublinear_tf = sublinear_tf,
        # strip_accents removes accented chars like é → e (handles international text)
        strip_accents = 'unicode',
        # analyzer='word' means we work at the word level (not character level)
        analyzer      = 'word',
    )
    return vectorizer


# ---------------------------------------------------------------
# FUNCTION 2: Split data into Train and Test sets
# ---------------------------------------------------------------
def split_data(df, text_column='clean_review',
               label_column='sentiment', test_size=0.2):
    """
    Splits the dataset into training and testing portions.

    WHY WE SPLIT:
      We train the model on one portion of data and evaluate it
      on a completely separate portion it has NEVER seen.
      This simulates real-world performance — the model won't
      just memorize the training data.

    Standard split: 80% train, 20% test
      With 3000 samples: 2400 for training, 600 for testing.

    stratify=y:
      Ensures EACH SPLIT has the same class proportions as the full dataset.
      Without stratify, you might accidentally get all positives in train
      and all negatives in test — that would be a terrible evaluation!

    Args:
        df           (pd.DataFrame): Preprocessed dataframe
        text_column  (str): Column with clean review text
        label_column (str): Column with sentiment labels
        test_size    (float): Fraction for test set (0.2 = 20%)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
               X = text data, y = labels
    """
    # X = features (the text we'll train on)
    # y = target labels (what we want to predict)
    X = df[text_column].values    # numpy array of clean review strings
    y = df[label_column].values   # numpy array of sentiment labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = test_size,
        random_state = 42,     # Fixed seed = reproducible results every run
        stratify     = y       # Keep class balance in both splits
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------
# FUNCTION 3: Fit vectorizer on training data, transform both sets
# ---------------------------------------------------------------
def fit_transform_tfidf(vectorizer, X_train, X_test):
    """
    Fits the TF-IDF vectorizer on training data ONLY,
    then transforms both training and test data.

    CRITICAL RULE — fit on TRAIN only, never on TEST:
      If we fit on ALL data including test, the vectorizer learns
      statistics (word frequencies) from the test set.
      This is called DATA LEAKAGE — it inflates accuracy scores
      and makes your model look better than it really is.

      Correct workflow:
        1. fit_transform(X_train) → learns vocabulary from training data
                                  → transforms training data
        2. transform(X_test)     → uses the SAME learned vocabulary
                                  → transforms test data

    Args:
        vectorizer (TfidfVectorizer): Configured but unfitted vectorizer
        X_train (array): Training text strings
        X_test  (array): Test text strings

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, fitted_vectorizer)
               Both X matrices are scipy sparse matrices of shape
               (n_samples, max_features)
    """
    print("Fitting TF-IDF vectorizer on training data...")

    # fit_transform does two things in one call:
    #   fit      → learns the vocabulary and IDF weights from X_train
    #   transform → converts X_train into a TF-IDF matrix
    X_train_tfidf = vectorizer.fit_transform(X_train)

    print("Transforming test data using fitted vocabulary...")

    # transform ONLY — no fitting — uses the vocabulary learned above
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, vectorizer


# ---------------------------------------------------------------
# FUNCTION 4: Save the fitted vectorizer to disk
# ---------------------------------------------------------------
def save_vectorizer(vectorizer, path='models/tfidf_vectorizer.pkl'):
    """
    Saves the fitted TF-IDF vectorizer to a file.

    WHY SAVE IT?
      When a user submits a new review for prediction, we need to
      convert it to TF-IDF using the EXACT SAME vocabulary that
      was used during training. If we refit on new data, the
      feature columns will be different and the model will break.

      Save once after training → load at prediction time → consistent.

    Args:
        vectorizer: Fitted TfidfVectorizer object
        path (str): Where to save the .pkl file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved to: {path}")


# ---------------------------------------------------------------
# FUNCTION 5: Load a previously saved vectorizer
# ---------------------------------------------------------------
def load_vectorizer(path='models/tfidf_vectorizer.pkl'):
    """
    Loads a saved TF-IDF vectorizer from disk.

    Args:
        path (str): Path to the saved .pkl file

    Returns:
        TfidfVectorizer: The fitted vectorizer, ready to use
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Vectorizer not found at {path}. "
            f"Run training first to create it."
        )
    vectorizer = joblib.load(path)
    print(f"Vectorizer loaded from: {path}")
    return vectorizer


# ---------------------------------------------------------------
# FUNCTION 6: Inspect the learned vocabulary (great for reports!)
# ---------------------------------------------------------------
def inspect_vocabulary(vectorizer, top_n=20):
    """
    Displays the most important TF-IDF features (words/bigrams).
    Useful for understanding what the model is learning and for
    including in your project report.

    Args:
        vectorizer (TfidfVectorizer): A FITTED vectorizer
        top_n      (int): How many top features to show
    """
    # Get all feature names (words/bigrams the vectorizer learned)
    feature_names = vectorizer.get_feature_names_out()

    # Get IDF scores — higher IDF = rarer word = more informative
    idf_scores = vectorizer.idf_

    # Sort by IDF score (highest first = rarest/most informative)
    sorted_indices = np.argsort(idf_scores)[::-1]

    print(f"\nTop {top_n} most informative features (highest IDF scores):")
    print(f"{'Rank':<6} {'Feature':<25} {'IDF Score':<12}")
    print("-" * 45)
    for rank, idx in enumerate(sorted_indices[:top_n], 1):
        print(f"{rank:<6} {feature_names[idx]:<25} {idf_scores[idx]:.4f}")

    print(f"\nTotal vocabulary size: {len(feature_names):,} features")

    # Also show lowest IDF = most common words (these are near-stopwords)
    print(f"\nBottom {top_n} least informative features (lowest IDF — appear everywhere):")
    print(f"{'Rank':<6} {'Feature':<25} {'IDF Score':<12}")
    print("-" * 45)
    for rank, idx in enumerate(sorted_indices[-top_n:], 1):
        print(f"{rank:<6} {feature_names[idx]:<25} {idf_scores[idx]:.4f}")