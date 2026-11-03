"""
RAG pipeline built with LangChain — hybrid retrieval using Reciprocal Rank
Fusion (RRF) to combine BM25 keyword search with dense embedding search,
generation via an OpenAI chat model.
"""

from dotenv import load_dotenv

load_dotenv()

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler

DOCUMENTS = [
    "Precision measures the proportion of positive predictions that are actually correct, penalizing false positives.",
    "Recall measures the proportion of actual positive cases correctly identified, penalizing false negatives.",
    "F1 score is the harmonic mean of precision and recall, balancing the trade-off between the two metrics.",
]


def build_retrievers(documents: list[str]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(documents, embeddings)
    semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    bm25_retriever = BM25Retriever.from_texts(documents)
    bm25_retriever.k = 2

    return bm25_retriever, semantic_retriever


def hybrid_retrieve(question: str, bm25_retriever, semantic_retriever, k0: int = 60, top_n: int = 2) -> list[str]:
    """
    Combine BM25 and semantic retrieval results using Reciprocal Rank
    Fusion (RRF). Each document's score is the sum of 1 / (k0 + rank)
    across both retrievers, so a document that ranks well in either list
    scores higher — and documents are fused by rank position, which avoids
    needing to calibrate BM25 and cosine-similarity scores onto the same
    scale, since the two are not naturally comparable.
    """
    bm25_docs = bm25_retriever.invoke(question)
    semantic_docs = semantic_retriever.invoke(question)

    scores = {}
    for rank, doc in enumerate(bm25_docs, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (k0 + rank)
    for rank, doc in enumerate(semantic_docs, start=1):
        scores[doc.page_content] = scores.get(doc.page_content, 0) + 1 / (k0 + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]


def generate(question: str, contexts: list[str], llm, callbacks=None) -> str:
    context_text = "\n".join(f"- {c}" for c in contexts)
    prompt = f"""Use only the context below to answer the question.
Give the shortest direct answer possible. Do not add outside knowledge.

Context:
{context_text}

Question: {question}
Answer:"""
    return llm.invoke(prompt, config={"callbacks": callbacks or []}).content


if __name__ == "__main__":
    langfuse_handler = CallbackHandler()
    bm25_retriever, semantic_retriever = build_retrievers(DOCUMENTS)
    llm = ChatOpenAI(model="gpt-6-luna")

    QUESTIONS = [
        "What metric balances the trade-off between precision and recall?",
        "Which metric combines precision and recall?",
        "What measure takes both precision and recall into account?",
        "What is the harmonic mean of precision and recall?",
    ]

    @observe(name="rag-question")
    def answer_question(question):
        contexts = hybrid_retrieve(question, bm25_retriever, semantic_retriever)
        answer = generate(question, contexts, llm, callbacks=[langfuse_handler])
        return contexts, answer

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print("Question:", question)
        contexts, answer = answer_question(question)
        print("Retrieved (RRF):", contexts)
        print("Answer:", answer)

    get_client().flush()

# ---------------------------------------------------------------------------
# FINDINGS
#
# Used the same 4 paraphrased questions and 3-document corpus as
# manual_rag.py.
#
# Retrieval combines BM25 keyword search with dense semantic retrieval using
# Reciprocal Rank Fusion (RRF). Each document receives a score based on its
# rank in each result list rather than its raw retrieval score. This avoids
# directly combining BM25 scores with FAISS vector-search scores, which are not
# directly comparable.
#
# I first tried concatenating the BM25 and dense results and removing
# duplicates. F1 appeared first for all 4 questions, but this was not a
# principled combined ranking: the result depended on the order in which the
# two retrieval lists were concatenated. I therefore replaced this approach
# with RRF.
#
# Final-answer accuracy was 4/4. This is not a controlled comparison with
# manual_rag.py (1/4): that version used a different generator and gave it
# one retrieved document, while this one gives it two.
#
# RRF did not completely solve the ranking problem: for 2 of the 4
# questions the Recall document still ranked above the F1 document. The
# retrieval comparison in eval.py shows this is not only an RRF issue: BM25
# alone outperformed RRF on this question set. See the findings in eval.py.
#
# Because a generator can answer correctly whenever the relevant document is
# somewhere in the retrieved context, final-answer accuracy can look perfect
# while ranking is still weak. Retrieval is therefore evaluated separately:
# Top-1 accuracy, Recall@2, and MRR for dense-only retrieval and RRF are
# computed in eval.py (run `python eval.py` to reproduce).
#
# Possible next steps:
#   - retrieve a larger candidate pool before fusion
#   - apply cross-encoder reranking to the fused candidates
#   - evaluate on a larger and less ambiguous query set
#   - inspect RRF scores to understand ranking ties and disagreements
# ---------------------------------------------------------------------------