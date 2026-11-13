"""Streamlit demo: IMDB sentiment analysis + simple RAG Q&A.

Run locally:  streamlit run app.py

Tab 1 — Sentiment: type a movie review, get a positive/negative
prediction with probability from a TF-IDF + Logistic Regression model.
Tab 2 — Ask a question: TF-IDF retrieval over 50 review passages
(data/rag_corpus.jsonl); the top-3 passages are shown with scores and
source ids. No generative LLM is involved — the "answer" is the
retrieved passages themselves (see src/rag.py).

The demo trains the small sentiment model in memory on launch from the
bundled data/imdb_sample.csv (2,000 reviews) so it starts fast without
the 66 MB full download. Full training on 20,000 reviews is src/train.py;
its measured test-set metrics are in README.md / metrics.json.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from rag import answer as rag_answer  # noqa: E402

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_sentiment_model():
    df = pd.read_csv(ROOT / "data" / "imdb_sample.csv")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20_000, ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])
    model.fit(df["review"], df["label"])
    return model


st.set_page_config(page_title="Sentiment + RAG — IMDB reviews",
                   layout="centered")
st.title("Movie Review Sentiment & RAG Q&A")
st.write("TF-IDF + Logistic Regression sentiment classifier, plus TF-IDF "
         "retrieval over 50 review passages. Trained on the IMDB Large "
         "Movie Review Dataset (Maas et al., 2011).")

tab_sentiment, tab_rag = st.tabs(["😊 Sentiment", "🔎 Ask a question"])

with tab_sentiment:
    st.subheader("Is this review positive or negative?")
    text = st.text_area(
        "Review text",
        "This movie was a wonderful surprise — great acting and a story "
        "that kept me watching until the very end.",
        height=120)
    if st.button("Classify"):
        model = load_sentiment_model()
        proba = float(model.predict_proba([text])[0, 1])
        label = "Positive 😊" if proba >= 0.5 else "Negative 😞"
        st.metric("Prediction", label, f"{proba:.1%} positive")
        st.caption("Demo model trained in memory on the bundled 2,000-review "
                   "sample. Full 20,000-review model: see README.md — "
                   "test accuracy 0.892, ROC-AUC 0.962.")

with tab_rag:
    st.subheader("Ask about the reviews")
    st.write("Retrieval only: the top-3 passages below are shown with "
             "their cosine-similarity scores and source document ids. "
             "No answer is generated — what you see is what was retrieved.")
    question = st.text_input(
        "Question",
        "Was the plot boring or exciting?")
    if st.button("Search passages"):
        out = rag_answer(question)
        for r in out["results"]:
            st.markdown(f"**[{r['rank']}] {r['id']}** — score "
                        f"`{r['score']}` · {r['sentiment']} · source "
                        f"review #{r['source_review_index']}")
            st.write(r["text"])
