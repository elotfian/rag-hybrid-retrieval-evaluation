"""
Hand-built RAG pipeline — retrieval via sentence-transformers,
generation via flan-t5 (seq2seq), no framework.
Built to understand the RAG mechanism before wrapping it in LangChain.
"""

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DOCUMENTS = [
    "Precision measures the proportion of positive predictions that are actually correct, penalizing false positives.",
    "Recall measures the proportion of actual positive cases correctly identified, penalizing false negatives.",
    "F1 score is the harmonic mean of precision and recall, balancing the trade-off between the two metrics.",
]


def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    gen_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embedder, gen_tokenizer, gen_model


def retrieve(question: str, documents: list[str], doc_embeddings, embedder, top_k: int = 1) -> list[str]:
    """Return the top_k most relevant documents, printing similarity scores for diagnosis."""
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    similarities = util.cos_sim(question_embedding, doc_embeddings)[0]
    for doc, score in zip(documents, similarities):
        print(f"{score.item():.4f} | {doc}")
    top_indices = similarities.argsort(descending=True)[:top_k]
    return [documents[i] for i in top_indices]


def generate(question: str, contexts: list[str], gen_tokenizer, gen_model) -> str:
    """Generate a short, strictly extractive answer grounded in the given context(s)."""
    context_text = "\n".join(f"- {c}" for c in contexts)
    prompt = f"""Use only the context below to answer the question.
If the answer is explicitly stated in the context, give the shortest direct answer possible.
Do not add outside knowledge.
If the answer cannot be determined from the context, say "I don't know based on the provided context."

Context:
{context_text}

Question: {question}
Answer:"""
    inputs = gen_tokenizer(prompt, return_tensors="pt")
    output_ids = gen_model.generate(**inputs, max_new_tokens=30, do_sample=False)
    return gen_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def rag_answer(question: str, documents: list[str], doc_embeddings, embedder, gen_tokenizer, gen_model, top_k: int = 1) -> tuple[list[str], str]:
    """doc_embeddings is computed ONCE outside this function and reused across calls."""
    retrieved_docs = retrieve(question, documents, doc_embeddings, embedder, top_k)
    answer = generate(question, retrieved_docs, gen_tokenizer, gen_model)
    return retrieved_docs, answer


if __name__ == "__main__":
    embedder, gen_tokenizer, gen_model = load_models()
    doc_embeddings = embedder.encode(DOCUMENTS, convert_to_tensor=True)  # computed once

    QUESTIONS = [
        "What metric balances the trade-off between precision and recall?",
        "Which metric combines precision and recall?",
        "What measure takes both precision and recall into account?",
        "What is the harmonic mean of precision and recall?",
    ]

    for question in QUESTIONS:
        print("\n" + "=" * 80)
        print("Question:", question)
        retrieved, answer = rag_answer(question, DOCUMENTS, doc_embeddings, embedder, gen_tokenizer, gen_model)
        print("Retrieved:", retrieved)
        print("Answer:", answer)

# ---------------------------------------------------------------------------
# FINDINGS
#
# Tested 4 paraphrased questions against this 3-document corpus.
#
# Generation was relatively stable. With a strict extractive prompt and
# greedy decoding (do_sample=False), the generated answer consistently
# followed the retrieved context in these tests. The observed errors therefore
# appeared to originate mainly from retrieval rather than generation.
#
# Retrieval was sensitive to query phrasing. Similarity scores for Precision,
# Recall and F1 were relatively close (0.48-0.62), and for 3 of the 4
# paraphrases the Recall document ranked above F1, even though F1 was the
# intended answer.
#
# The only query that retrieved F1 first contained the phrase "harmonic mean",
# which also appears directly in the F1 document. This suggests that lexical
# overlap may still influence ranking even when semantic embeddings are used.
#
# This is not primarily a manual-vs-framework issue. With the same corpus,
# embedding model, similarity metric and top_k, a LangChain implementation
# would be expected to show similar retrieval behavior. LangChain organizes
# the pipeline but does not automatically improve embedding quality.
#
# Possible improvements:
#   - combine BM25 and semantic retrieval
#   - retrieve more candidates and rerank them
#   - test a stronger or domain-specific embedding model
#   - improve chunk/document distinctiveness
#   - evaluate retrieval on a larger paraphrase set
#
# Next step:
# Rebuild the pipeline in LangChain under the same retrieval settings, then
# compare it with hybrid retrieval and reranking approaches.
# ---------------------------------------------------------------------------