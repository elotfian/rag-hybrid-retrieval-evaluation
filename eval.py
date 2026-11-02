"""
Retrieval evaluation: compares four retrievers on the same questions and
documents, scoring retrieval directly rather than by the final generated
answer.

No answers are generated here, so neither FLAN-T5 nor the OpenAI model is
called. For each question the script asks one thing: where does the correct
document (the F1 document) appear in the retriever's ranking? Each retriever
returns a full ranking of all documents, the rank of the F1 document is
recorded, and three metrics are computed from those ranks:

  - Top-1 accuracy: how often the F1 document is ranked first
  - Recall@2:       how often it appears in the top two
  - MRR:            mean of 1/rank, which rewards ranking it higher

Retrievers compared:

  - Dense (manual):      cosine similarity over embeddings (manual_rag.py)
  - Dense (LangChain):   same embedding model via FAISS (langchain_rrf.py)
  - BM25 only:           keyword retrieval (langchain_rrf.py)
  - BM25 + dense (RRF):  rank fusion of the two (langchain_rrf.py)

Run with `python eval.py`. No API key is needed.
"""

import contextlib
import io

from sentence_transformers import SentenceTransformer

from manual_rag import DOCUMENTS, retrieve as dense_retrieve
from langchain_rrf import build_retrievers, hybrid_retrieve

EXPECTED_DOC = DOCUMENTS[2]  # the F1 document is the right answer for every question

# Keep in sync with the questions in manual_rag.py and langchain_rrf.py.
QUESTIONS = [
    "What metric balances the trade-off between precision and recall?",
    "Which metric combines precision and recall?",
    "What measure takes both precision and recall into account?",
    "What is the harmonic mean of precision and recall?",
]


def rank_of(expected: str, ranked_docs: list[str]):
    """1-based rank of the expected document, or None if it was not retrieved."""
    for i, doc in enumerate(ranked_docs, start=1):
        if doc == expected:
            return i
    return None


def manual_dense_ranking(question, embedder, doc_embeddings):
    # retrieve() prints similarity scores; silence that here
    with contextlib.redirect_stdout(io.StringIO()):
        return dense_retrieve(question, DOCUMENTS, doc_embeddings, embedder, top_k=len(DOCUMENTS))


def single_ranking(retriever, question):
    return [d.page_content for d in retriever.invoke(question)]


def rrf_ranking(question, bm25_retriever, semantic_retriever):
    return hybrid_retrieve(question, bm25_retriever, semantic_retriever, top_n=len(DOCUMENTS))


def summarize(name: str, ranks: list):
    n = len(ranks)
    top1 = sum(r == 1 for r in ranks) / n
    recall_at_2 = sum(r is not None and r <= 2 for r in ranks) / n
    mrr = sum(1 / r if r else 0 for r in ranks) / n
    print(f"{name:<22} Top-1: {top1:>4.0%}   Recall@2: {recall_at_2:>4.0%}   MRR: {mrr:.2f}")


if __name__ == "__main__":
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    doc_embeddings = embedder.encode(DOCUMENTS, convert_to_tensor=True)

    # RRF uses the default retrievers (top-2 candidates from each)
    bm25_retriever, semantic_retriever = build_retrievers(DOCUMENTS)

    # standalone retrievers return all documents, so each one ranks the full set
    bm25_full, sem_full = build_retrievers(DOCUMENTS)
    bm25_full.k = len(DOCUMENTS)
    sem_full.search_kwargs["k"] = len(DOCUMENTS)

    ranks = {
        "Dense (manual)": [],
        "Dense (LangChain)": [],
        "BM25 only": [],
        "BM25 + dense (RRF)": [],
    }

    print("Rank of the F1 document per question (1 = best, None = not retrieved)\n")
    print(f"{'manual':>7} {'lc':>4} {'bm25':>5} {'rrf':>4}   question")
    for q in QUESTIONS:
        r_manual = rank_of(EXPECTED_DOC, manual_dense_ranking(q, embedder, doc_embeddings))
        r_lc = rank_of(EXPECTED_DOC, single_ranking(sem_full, q))
        r_bm25 = rank_of(EXPECTED_DOC, single_ranking(bm25_full, q))
        r_rrf = rank_of(EXPECTED_DOC, rrf_ranking(q, bm25_retriever, semantic_retriever))

        ranks["Dense (manual)"].append(r_manual)
        ranks["Dense (LangChain)"].append(r_lc)
        ranks["BM25 only"].append(r_bm25)
        ranks["BM25 + dense (RRF)"].append(r_rrf)

        print(f"{str(r_manual):>7} {str(r_lc):>4} {str(r_bm25):>5} {str(r_rrf):>4}   {q}")

    print()
    for name, r in ranks.items():
        summarize(name, r)


# ---------------------------------------------------------------------------
# FINDINGS
#
# Four retrieval methods were evaluated on the same 4 paraphrased questions
# and 3-document corpus. The F1 document is the expected relevant document
# for every question.
#
# Results (run `python eval.py`):
#
#   Dense (manual)       Top-1  25%   Recall@2  50%   MRR 0.54
#   Dense (LangChain)    Top-1  25%   Recall@2  50%   MRR 0.54
#   BM25 only            Top-1 100%   Recall@2 100%   MRR 1.00
#   BM25 + dense (RRF)   Top-1  50%   Recall@2 100%   MRR 0.75
#
# The manual dense retriever and the LangChain/FAISS dense retriever give
# identical rankings on this test set. This suggests that the weak dense
# results here are not caused by the difference between the manual
# cosine-similarity implementation and the LangChain/FAISS implementation.
#
# RRF improves over dense-only retrieval: Top-1 increases from 25% to 50%,
# Recall@2 from 50% to 100%, and MRR from 0.54 to 0.75. However, RRF still
# performs worse than BM25 alone on this dataset.
#
# A closer look at the RRF rankings where F1 was not ranked first suggests
# that the top-2 candidate restriction may be contributing to the behavior.
# For two questions, F1 is ranked third by the dense retriever, so it is
# excluded from the dense candidate list and receives an RRF contribution
# only from BM25. Other documents may receive contributions from both
# retrievers and therefore outrank F1. This interpretation is consistent
# with the observed ranks, but it has not been verified by inspecting the
# individual top-2 rankings and fused RRF scores.
#
# This small test set strongly favors keyword retrieval: every question
# contains "precision" and "recall", and F1 is the only document containing
# both terms. These results therefore do not show that BM25 is generally
# better than dense or hybrid retrieval. A larger and more varied query set,
# including paraphrases with less lexical overlap, is needed for a more
# meaningful comparison.
# ---------------------------------------------------------------------------