"""Tests for retrieval: RRF logic, BM25 ranking, and the hybrid pipeline."""
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from manual_rag import DOCUMENTS
from langchain_rrf import build_retrievers, hybrid_retrieve

F1_DOC = next(d for d in DOCUMENTS if d.startswith("F1"))
QUESTION = "What is the harmonic mean of precision and recall?"


class StubRetriever:
    """Returns a fixed ranking, so RRF can be tested without any model."""

    def __init__(self, docs):
        self.docs = [Document(page_content=d) for d in docs]

    def invoke(self, question):
        return self.docs


def test_rrf_ranks_document_in_both_lists_first():
    bm25 = StubRetriever(["A", "B"])
    dense = StubRetriever(["B", "C"])
    # B: 1/62 + 1/61, A: 1/61, C: 1/62
    assert hybrid_retrieve("q", bm25, dense, top_n=3) == ["B", "A", "C"]


def test_bm25_ranks_f1_document_first():
    bm25 = BM25Retriever.from_texts(DOCUMENTS)
    bm25.k = len(DOCUMENTS)
    results = [d.page_content for d in bm25.invoke(QUESTION)]
    assert results[0] == F1_DOC


def test_hybrid_retrieval_returns_f1_in_top_2():
    bm25, dense = build_retrievers(DOCUMENTS)
    results = hybrid_retrieve(QUESTION, bm25, dense, top_n=2)
    assert F1_DOC in results