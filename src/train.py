# ============================================================
# FILE: src/train.py
# PURPOSE: Train three ML models on TF-IDF features
#
# MODELS:
#   1. Multinomial Naive Bayes  — fast probabilistic baseline
#   2. Logistic Regression      — linear model, learns word weights
#   3. Linear SVM               — margin-based, great for sparse text
# ============================================================

import numpy  as np
import pandas as pd
import joblib
import os
import time                        # To measure how long training takes

# Scikit-learn model classes
from sklearn.naive_bayes    import MultinomialNB
from sklearn.linear_model   import LogisticRegression
from sklearn.svm            import LinearSVC
from sklearn.pipeline       import Pipeline

# Cross-validation for robust evaluation during training
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ---------------------------------------------------------------
# HELPER: Print a formatted section header (looks good in output)
# ---------------------------------------------------------------
def print_header(title):
    line = "=" * 55
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")


# ---------------------------------------------------------------
# MODEL 1: Multinomial Naive Bayes
# ---------------------------------------------------------------
def train_naive_bayes(X_train, y_train, alpha=0.1):
    """
    Trains a Multinomial Naive Bayes classifier.

    HOW IT WORKS:
      Uses Bayes' theorem to calculate the probability of each
      sentiment class given the words in the review.

      P(positive | "amazing movie") ∝ P("amazing"|positive)
                                     × P("movie"|positive)
                                     × P(positive)

      It multiplies the probability of each word appearing in
      positive reviews, then compares across all classes.

    WHY "Multinomial"?
      Because TF-IDF gives us count-like scores (not binary 0/1).
      MultinomialNB handles positive numeric features — perfect for TF-IDF.

    Parameter:
      alpha (float): Laplace smoothing parameter.
        Smoothing prevents the "zero probability problem":
        if the word "extraordinary" never appeared in training data,
        its probability would be 0, making the ENTIRE product 0.
        alpha=0.1 adds a tiny count to every word so nothing is ever 0.
        Lower alpha = less smoothing = model trusts training data more.
        Typical range: 0.01 to 1.0

    Args:
        X_train: TF-IDF sparse matrix of training data
        y_train: Array of training labels
        alpha:   Smoothing parameter

    Returns:
        Trained MultinomialNB model
    """
    print_header("Training Model 1: Naive Bayes")

    # Record training start time
    start_time = time.time()

    # Create and train the model
    # fit() is the scikit-learn method that actually trains the model
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)

    # Calculate training time
    train_time = time.time() - start_time

    print(f"\n  Algorithm   : Multinomial Naive Bayes")
    print(f"  Alpha       : {alpha} (Laplace smoothing)")
    print(f"  Training time: {train_time:.3f} seconds")
    print(f"  Status      : Training complete!")

    # Cross-validation: evaluate stability of the model
    # This splits training data into 5 folds and trains/tests on each
    # Gives us a more reliable accuracy estimate than a single split
    print(f"\n  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring = 'accuracy',
        n_jobs  = -1    # Use all CPU cores
    )
    print(f"  CV Accuracy scores : {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  CV Mean Accuracy   : {cv_scores.mean():.4f} "
          f"(+/- {cv_scores.std():.4f})")

    return model


# ---------------------------------------------------------------
# MODEL 2: Logistic Regression
# ---------------------------------------------------------------
def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000):
    """
    Trains a Logistic Regression classifier.

    HOW IT WORKS:
      Learns a weight (coefficient) for every feature in the vocabulary.
      For each review:
        score = w₁×tfidf("amazing") + w₂×tfidf("terrible") + ... + bias

      The weights are learned during training to MINIMIZE prediction errors.
      After training, positive words have positive weights,
      negative words have negative weights.

    WHY IT WORKS WELL FOR SENTIMENT:
      Sentiment is largely additive — a review with many positive words
      is likely positive. Logistic Regression captures this naturally.

    Parameters:
      C (float): Regularization strength (inverse).
        HIGH C (e.g., 10)  → model fits training data closely
                           → risk of overfitting (memorizing training data)
        LOW  C (e.g., 0.1) → model is more constrained/general
                           → risk of underfitting
        C=1.0 is a good default starting point.

      max_iter (int):
        Maximum iterations for the solver to converge.
        1000 is usually enough. Increase if you see a ConvergenceWarning.

      solver='lbfgs':
        The optimization algorithm used to find weights.
        lbfgs works well for multi-class problems.

      multi_class='multinomial':
        Handles 3 classes (positive/negative/neutral) properly.
        Uses softmax to produce probabilities across all 3 classes.

    Args:
        X_train  : TF-IDF sparse matrix
        y_train  : Training labels
        C        : Regularization parameter
        max_iter : Maximum iterations

    Returns:
        Trained LogisticRegression model
    """
    print_header("Training Model 2: Logistic Regression")

    start_time = time.time()

    model = LogisticRegression(
        C           = C,
        max_iter    = max_iter,
        solver      = 'lbfgs',
        multi_class = 'multinomial',
        random_state= 42,
        n_jobs      = -1        # Parallel processing
    )
    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    print(f"\n  Algorithm    : Logistic Regression")
    print(f"  C (regulariz): {C}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Training time : {train_time:.3f} seconds")
    print(f"  Status        : Training complete!")

    # Cross-validation
    print(f"\n  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring = 'accuracy',
        n_jobs  = -1
    )
    print(f"  CV Accuracy scores : {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  CV Mean Accuracy   : {cv_scores.mean():.4f} "
          f"(+/- {cv_scores.std():.4f})")

    # Show the most influential words for each class
    # (This is great to include in your report!)
    _show_top_features_lr(model, top_n=8)

    return model


def _show_top_features_lr(model, top_n=8):
    """
    Shows the words with highest positive/negative weights
    for each sentiment class in the Logistic Regression model.
    Internal helper — called automatically during training.
    """
    # We need the vectorizer's feature names
    # Load it from disk if available
    vec_path = 'models/tfidf_vectorizer.pkl'
    if not os.path.exists(vec_path):
        return   # Skip if vectorizer not saved yet

    vectorizer    = joblib.load(vec_path)
    feature_names = vectorizer.get_feature_names_out()
    class_labels  = model.classes_

    print(f"\n  Top {top_n} most influential words per class:")
    print(f"  {'Class':<12} {'Top positive words':<45} {'Top negative words'}")
    print(f"  {'-'*95}")

    for i, label in enumerate(class_labels):
        # model.coef_[i] = weight array for class i
        # Higher weight = word strongly predicts this class
        coef        = model.coef_[i]
        top_pos_idx = np.argsort(coef)[-top_n:][::-1]   # Highest weights
        top_neg_idx = np.argsort(coef)[:top_n]           # Lowest weights

        pos_words = ', '.join([feature_names[j] for j in top_pos_idx])
        neg_words = ', '.join([feature_names[j] for j in top_neg_idx])

        print(f"  {label:<12} {pos_words[:44]:<45} {neg_words[:44]}")


# ---------------------------------------------------------------
# MODEL 3: Linear SVM
# ---------------------------------------------------------------
def train_svm(X_train, y_train, C=1.0):
    """
    Trains a Linear Support Vector Machine classifier.

    HOW IT WORKS:
      SVM finds the hyperplane (decision boundary) that separates
      classes with the MAXIMUM MARGIN.

      Imagine plotting all reviews in 10,000-dimensional space
      (one dimension per TF-IDF feature). SVM finds the plane
      that puts positive reviews on one side and negative on the other,
      with as much space as possible between the boundary and the
      nearest data points (called Support Vectors).

      WHY MAXIMUM MARGIN?
        A wider margin means the model is more confident in its decisions
        and generalizes better to new, unseen reviews.

    WHY LinearSVC (not SVC)?
      Regular SVC with linear kernel is O(n²) or O(n³) in training time.
      LinearSVC is O(n) — much faster for large datasets.
      For text classification, LinearSVC almost always gives the same
      accuracy as SVC with linear kernel, but trains 10-100x faster.

    Parameter:
      C (float): Controls the trade-off between:
        - Having a wider margin (more generalizable)
        - Correctly classifying all training points
        HIGH C → tries harder to classify everything correctly
               → narrower margin → might overfit
        LOW  C → allows some misclassification
               → wider margin → better generalization
        C=1.0 is a solid default for text.

    Args:
        X_train : TF-IDF sparse matrix
        y_train : Training labels
        C       : Regularization parameter

    Returns:
        Trained LinearSVC model
    """
    print_header("Training Model 3: Linear SVM")

    start_time = time.time()

    model = LinearSVC(
        C            = C,
        max_iter     = 2000,
        random_state = 42,
        # dual=False is recommended when n_samples > n_features
        # Our case: 2397 samples, 10000 features → use dual=True (default)
    )
    model.fit(X_train, y_train)

    train_time = time.time() - start_time

    print(f"\n  Algorithm    : Linear SVM (LinearSVC)")
    print(f"  C (regulariz): {C}")
    print(f"  Training time : {train_time:.3f} seconds")
    print(f"  Status        : Training complete!")

    # Cross-validation
    print(f"\n  Running 5-fold cross-validation...")
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring = 'accuracy',
        n_jobs  = -1
    )
    print(f"  CV Accuracy scores : {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  CV Mean Accuracy   : {cv_scores.mean():.4f} "
          f"(+/- {cv_scores.std():.4f})")

    return model


# ---------------------------------------------------------------
# FUNCTION: Save a trained model to disk
# ---------------------------------------------------------------
def save_model(model, model_name, folder='models'):
    """
    Saves a trained model to a .pkl file using joblib.

    WHY SAVE MODELS?
      Training can take seconds to minutes. Once trained, we save
      the model so we can:
        - Load it instantly for predictions
        - Share it with others
        - Avoid retraining every time the program restarts

    joblib is better than pickle for scikit-learn models because
    it handles large numpy arrays more efficiently.

    Args:
        model      : Trained scikit-learn model
        model_name : String name for the file (e.g., 'naive_bayes')
        folder     : Directory to save in (default: 'models/')
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{model_name}_model.pkl")
    joblib.dump(model, path)
    print(f"\n  Model saved to: {path}")
    return path


# ---------------------------------------------------------------
# FUNCTION: Load a saved model from disk
# ---------------------------------------------------------------
def load_model(model_name, folder='models'):
    """
    Loads a previously saved model from disk.

    Args:
        model_name : String name (e.g., 'naive_bayes')
        folder     : Directory where model is saved

    Returns:
        Loaded scikit-learn model ready for prediction
    """
    path = os.path.join(folder, f"{model_name}_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model '{model_name}' not found at {path}. "
            f"Run training first."
        )
    model = joblib.load(path)
    print(f"Model loaded from: {path}")
    return model


# ---------------------------------------------------------------
# MASTER FUNCTION: Train all three models in one call
# ---------------------------------------------------------------
def train_all_models(X_train, y_train):
    """
    Trains all three models sequentially and saves each one.

    This is the main function you call from main.py or notebook.
    It handles training, cross-validation, and saving automatically.

    Args:
        X_train : TF-IDF sparse matrix (training data)
        y_train : Array of training labels

    Returns:
        dict: {'naive_bayes': model1,
               'logistic_regression': model2,
               'svm': model3}
    """
    print("\n" + "=" * 55)
    print("  STARTING MODEL TRAINING")
    print("  Training data shape:", X_train.shape)
    print("  Labels distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"    {label}: {count} samples")
    print("=" * 55)

    models = {}

    # --- Train Naive Bayes ---
    nb_model = train_naive_bayes(X_train, y_train, alpha=0.1)
    save_model(nb_model, 'naive_bayes')
    models['naive_bayes'] = nb_model

    # --- Train Logistic Regression ---
    lr_model = train_logistic_regression(
        X_train, y_train, C=1.0, max_iter=1000
    )
    save_model(lr_model, 'logistic_regression')
    models['logistic_regression'] = lr_model

    # --- Train SVM ---
    svm_model = train_svm(X_train, y_train, C=1.0)
    save_model(svm_model, 'svm')
    models['svm'] = svm_model

    # --- Summary ---
    print("\n" + "=" * 55)
    print("  ALL MODELS TRAINED SUCCESSFULLY")
    print("  Saved files:")
    print("    models/naive_bayes_model.pkl")
    print("    models/logistic_regression_model.pkl")
    print("    models/svm_model.pkl")
    print("=" * 55)

    return models