# rag-retrieval-study

A small study of retrieval quality in RAG, built by hand first and then with
LangChain. The point is to measure retrieval separately from generation,
because a correct final answer can hide a weak ranking.

## The problem

The corpus has three near-identical documents (precision, recall, F1). A
question like "Which metric combines precision and recall?" overlaps with all
three, so the retriever has to rank the F1 document above the other two. It
is a deliberately hard case for embeddings, which place all three documents
close together.

## What is in the repo

| File | What it does |
|---|---|
| `manual_rag.py` | RAG written by hand: MiniLM embeddings, cosine similarity, FLAN-T5 generator |
| `langchain_rrf.py` | LangChain version: BM25 + FAISS retrieval fused with Reciprocal Rank Fusion, OpenAI generator |
| `eval.py` | Scores the retrievers directly on the rank of the correct document. No generation, no API key |

## Pipeline

1. **Manual baseline.** Embed the documents with `all-MiniLM-L6-v2`, retrieve
   by cosine similarity, answer with `flan-t5-base` using a strict extractive
   prompt.
2. **Hybrid retrieval.** BM25 (keyword) and FAISS (dense) retrievers, combined
   with Reciprocal Rank Fusion: each document scores the sum of
   `1 / (60 + rank)` over the lists it appears in.

An intermediate experiment: I first tested a simple hybrid that concatenated
the BM25 results with the dense results and removed duplicates. F1 came first
for all four test questions, but this was partly an artifact of placing the
BM25 list before the semantic list, not a principled ranking. I replaced it
with RRF.

## Results

Rank of the F1 document for four paraphrases of the same question
(1 = best):

| Question | Dense (manual) | Dense (LangChain) | BM25 | RRF |
|---|---|---|---|---|
| What metric balances the trade-off between precision and recall? | 3 | 3 | 1 | 2 |
| Which metric combines precision and recall? | 2 | 2 | 1 | 1 |
| What measure takes both precision and recall into account? | 3 | 3 | 1 | 2 |
| What is the harmonic mean of precision and recall? | 1 | 1 | 1 | 1 |

| Retriever | Top-1 | Recall@2 | MRR |
|---|---|---|---|
| Dense (manual) | 25% | 50% | 0.54 |
| Dense (LangChain) | 25% | 50% | 0.54 |
| BM25 only | 100% | 100% | 1.00 |
| BM25 + dense (RRF) | 50% | 100% | 0.75 |

Reproduce with `python eval.py`. Output from my run is in `results.txt`.

## What the results show

- The two dense retrievers rank F1 identically on every question, so the
  manual cosine implementation and LangChain/FAISS make no difference here.
  The sensitivity to phrasing comes from the embedding approach, not the
  framework.
- RRF improves on dense-only retrieval, but BM25 alone does better than both.
- A likely reason: on two questions the dense retrievers rank F1 third, which
  is outside the top-2 candidates used for fusion. F1 then gets a score from
  BM25 only, while the Recall document appears in both lists and can outrank
  it. This fits the observed ranks, but I have not checked it by inspecting
  the individual top-2 lists and fused scores.

## Limitations

- This is an illustrative experiment, not a benchmark: 3 documents and 4
  questions.
- The test set favors keyword matching. Every question contains "precision"
  and "recall", and F1 is the only document containing both words. The
  results do not show that BM25 beats hybrid retrieval in general.
- Final-answer accuracy in the two pipelines is not a controlled comparison.
  The generator (FLAN-T5 vs an OpenAI model) and the context size (top-1 vs
  top-2) both differ.

## Possible next steps

- Larger candidate pool before fusion
- Cross-encoder reranking
- A larger question set with less word overlap, where dense retrieval should
  help
- Inspect the per-retriever rankings and RRF scores to verify the explanation
  above

## Setup

```bash
pip install -r requirements.txt
```

Create your `.env` file from the example, then add your OpenAI key to it:

```bash
cp .env.example .env        # macOS / Linux
copy .env.example .env      # Windows CMD
```

Run the scripts:

```bash
python eval.py              # no API key needed
python manual_rag.py
python langchain_rrf.py     # needs the key in .env
```