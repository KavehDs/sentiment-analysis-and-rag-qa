"""Simple RAG Q&A over IMDB review excerpts (Part B of this project).

What this is — and what it is not
----------------------------------
This is TF-IDF retrieval + extractive passage display: a question is
matched against 50 short review passages (data/rag_corpus.jsonl, drawn
from the same IMDB dataset used in Part A) with cosine similarity, and
the top-3 passages are shown with their scores and source document ids.
No generative LLM is involved: nothing here writes a new answer from its
own knowledge, and no large transformer model is downloaded. The
"answer" is the retrieved passages themselves.

Usage:
    python src/rag.py "What do reviewers say about the acting?"
    python src/rag.py            # runs three worked example questions
"""
import json
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "data" / "rag_corpus.jsonl"

EXAMPLE_QUESTIONS = [
    "What do reviewers say about the acting?",
    "Was the plot boring or exciting?",
    "Did people enjoy the movie overall?",
]


def load_corpus() -> list[dict]:
    docs = []
    with open(CORPUS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def build_index(docs: list[dict]):
    vectorizer = TfidfVectorizer(max_features=5_000, ngram_range=(1, 2),
                                 sublinear_tf=True, min_df=1)
    matrix = vectorizer.fit_transform([d["text"] for d in docs])
    return vectorizer, matrix


def retrieve(question: str, docs, vectorizer, matrix, top_k: int = 3):
    """Return the top-k passages for a question, with cosine scores."""
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix)[0]
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    return [{
        "rank": r + 1,
        "id": docs[i]["id"],
        "score": round(float(scores[i]), 4),
        "sentiment": docs[i]["sentiment"],
        "source_review_index": docs[i]["source_review_index"],
        "text": docs[i]["text"],
    } for r, i in enumerate(ranked[:top_k])]


def answer(question: str, docs=None, vectorizer=None, matrix=None):
    if docs is None:
        docs = load_corpus()
        vectorizer, matrix = build_index(docs)
    results = retrieve(question, docs, vectorizer, matrix)
    return {"question": question, "method": "TF-IDF cosine retrieval, top-3",
            "results": results}


def main() -> None:
    docs = load_corpus()
    vectorizer, matrix = build_index(docs)
    print(f"Corpus: {len(docs)} passages "
          f"({sum(d['sentiment'] == 'positive' for d in docs)} positive, "
          f"{sum(d['sentiment'] == 'negative' for d in docs)} negative)")
    questions = sys.argv[1:] or EXAMPLE_QUESTIONS
    for q in questions:
        out = answer(q, docs, vectorizer, matrix)
        print(f"\nQ: {out['question']}")
        for r in out["results"]:
            print(f"  [{r['rank']}] {r['id']} score={r['score']} "
                  f"({r['sentiment']}, source review #{r['source_review_index']})")
            print(f"      {r['text'][:220]}")


if __name__ == "__main__":
    main()
