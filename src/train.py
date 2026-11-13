"""Sentiment analysis on IMDB movie reviews (Part A of this project).

Usage:  python src/train.py

Data: the Large Movie Review Dataset (Maas et al., 2011) — 50,000 reviews,
25,000 positive / 25,000 negative. The full CSV (~66 MB) is too large to
bundle in the repo, so this script downloads it once from a public mirror
(primary: rasbt/pattern_classification; fallback: Ankit152) into
data/imdb_full.csv, then trains on a stratified 20,000-review subsample.
If the download is unavailable it falls back to the bundled
data/imdb_sample.csv (2,000 reviews) and says so in the output.

Models: TF-IDF (20k features, word uni+bi-grams) + Logistic Regression,
compared with a Multinomial Naive Bayes baseline and a majority-class
baseline. Stratified 80/20 train/test split, random_state=42.
Writes metrics.json and plots/*.svg.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FULL_URLS = [
    "https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/50k_imdb_movie_reviews.csv",
    "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
]
SAMPLE_N = 20_000
RANDOM_STATE = 42


def clean(text: str) -> str:
    """Strip HTML line breaks; TF-IDF lowercases and tokenises the rest."""
    return re.sub(r"<br\s*/?>", " ", str(text))


def load_data() -> tuple[pd.DataFrame, str]:
    full = DATA / "imdb_full.csv"
    if not full.exists():
        import urllib.request

        for url in FULL_URLS:
            try:
                print(f"Downloading dataset from {url} ...")
                urllib.request.urlretrieve(url, full)
                break
            except Exception as exc:  # noqa: BLE001 - try next mirror
                print(f"  failed ({exc}); trying next mirror", file=sys.stderr)
        else:
            full = None
    if full is not None and full.exists():
        df = pd.read_csv(full)
        df = df.rename(columns={"sentiment": "label"})
        # rasbt file: label is 1/0 ints; Ankit152 file: 'positive'/'negative'
        if df["label"].dtype == object:
            df["label"] = (df["label"].str.lower() == "positive").astype(int)
        return df, "full 50,000-review file"
    sample = DATA / "imdb_sample.csv"
    print("WARNING: full dataset unavailable; using bundled 2,000-row sample",
          file=sys.stderr)
    df = pd.read_csv(sample)
    return df, "bundled 2,000-row sample"


def main() -> None:
    df, source_note = load_data()
    print(f"Loaded {len(df):,} rows ({source_note}); "
          f"labels: {df['label'].value_counts().to_dict()}")
    df = df.dropna(subset=["review"]).reset_index(drop=True)

    # Stratified subsample of at most SAMPLE_N rows (balanced by construction).
    if len(df) > SAMPLE_N:
        df, _ = train_test_split(df, train_size=SAMPLE_N,
                                 stratify=df["label"],
                                 random_state=RANDOM_STATE)
    print(f"Using {len(df):,} rows; "
          f"positive={int((df['label'] == 1).sum()):,}, "
          f"negative={int((df['label'] == 0).sum()):,}")
    df["text"] = df["review"].map(clean)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"],
        random_state=RANDOM_STATE)
    print(f"Train {len(X_train):,} / test {len(X_test):,}")

    vectorizer = TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=2)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    results: dict = {
        "dataset": {
            "name": "Large Movie Review Dataset (IMDB, Maas et al. 2011)",
            "source_urls": FULL_URLS,
            "full_rows": 50_000,
            "full_label_balance": {"positive": 25_000, "negative": 25_000},
            "rows_used": int(len(df)),
            "used_label_balance": {
                "positive": int((df["label"] == 1).sum()),
                "negative": int((df["label"] == 0).sum()),
            },
            "split": "stratified 80/20, random_state=42",
            "test_rows": int(len(X_test)),
            "source_note": source_note,
        }
    }

    def evaluate(name: str, y_pred, y_proba) -> dict:
        m = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_positive": round(
                float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall_positive": round(
                float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "f1_positive": round(
                float(f1_score(y_test, y_pred, zero_division=0)), 4),
            "roc_auc_positive": (round(float(roc_auc_score(y_test, y_proba)), 4)
                                 if y_proba is not None else None),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }
        results[name] = m
        print(f"{name}: acc={m['accuracy']} P={m['precision_positive']} "
              f"R={m['recall_positive']} F1={m['f1_positive']} "
              f"AUC={m['roc_auc_positive']}")
        return m

    # Baseline 1: majority class (always predict the majority label).
    majority = int(y_train.mode().iloc[0])
    evaluate("Majority class baseline",
             np.full(len(y_test), majority), None)

    # Baseline 2: Multinomial Naive Bayes on the same TF-IDF features.
    nb = MultinomialNB(alpha=0.1)
    nb.fit(Xtr, y_train)
    evaluate("TF-IDF + MultinomialNB",
             nb.predict(Xte), nb.predict_proba(Xte)[:, 1])

    # Main model: TF-IDF + Logistic Regression.
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(Xtr, y_train)
    lr_pred = lr.predict(Xte)
    lr_proba = lr.predict_proba(Xte)[:, 1]
    evaluate("TF-IDF + Logistic Regression", lr_pred, lr_proba)

    # Most informative words for each class (Logistic Regression weights).
    feats = np.array(vectorizer.get_feature_names_out())
    coef = lr.coef_[0]
    top_pos = feats[np.argsort(coef)[-10:][::-1]].tolist()
    top_neg = feats[np.argsort(coef)[:10]].tolist()
    results["top_words"] = {"positive": top_pos, "negative": top_neg}
    print("Top positive words:", top_pos)
    print("Top negative words:", top_neg)

    (ROOT / "metrics.json").write_text(json.dumps(results, indent=2))

    # Plots (SVG).
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    cm = np.array(results["TF-IDF + Logistic Regression"]["confusion_matrix"])
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix — TF-IDF + Logistic Regression (test set)")
    ax.set_xlabel("Predicted (0 = negative, 1 = positive)")
    ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(plots / "confusion_matrix.svg")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, lr_proba)
    fpr_nb, tpr_nb, _ = roc_curve(y_test, nb.predict_proba(Xte)[:, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"LogReg (AUC {results['TF-IDF + Logistic Regression']['roc_auc_positive']})")
    ax.plot(fpr_nb, tpr_nb, label=f"Naive Bayes (AUC {results['TF-IDF + MultinomialNB']['roc_auc_positive']})")
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — positive class (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "roc_curves.svg")
    plt.close(fig)
    print("Wrote metrics.json and plots/*.svg")


if __name__ == "__main__":
    main()
