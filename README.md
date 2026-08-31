# Mini Semantic Search Engine (FAISS + Sentence-Transformers)

A small semantic search engine that embeds a knowledge base with
`all-MiniLM-L6-v2`, indexes the embeddings in a FAISS `IndexFlatL2`, and
retrieves the top-3 most relevant sentences for any user query — the same
core pattern used in production RAG (Retrieval-Augmented Generation) systems.

## Project Structure

```
.
├── semantic_search.py   # Tasks 1–4: embeddings, FAISS index, search, CLI
├── REFLECTION.md         # Task 5: theory question answers
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

(First run will download the `all-MiniLM-L6-v2` model, ~90MB, from
Hugging Face — an internet connection is required for that one-time
download; after that it's cached locally.)

## Run

```bash
python semantic_search.py
```

This will:
1. Load the model and embed the 12-sentence knowledge base, printing the
   embedding matrix shape (`(12, 384)`).
2. Build a `faiss.IndexFlatL2(384)` index, normalise the embeddings for
   cosine-similarity behaviour, and print `index.ntotal`.
3. Run 3 sample queries and print Rank | Score | Matched Sentence tables.
4. Drop into an interactive CLI — type any query, get the top-3 matches.
   Type `exit` to quit.

### Example

```
Query> How do I get my money back?
Rank  Score     Matched Sentence
--------------------------------------------------------------------------------
1     0.6123    Refunds are processed within 5-7 business days back to the original payment method.
2     0.4118    If your card payment fails, please check with your bank or try a different payment method.
3     0.3722    Subscription plans can be upgraded or downgraded anytime from the Billing > Plans page.
```

## Notes

- Embeddings are normalised with `faiss.normalize_L2()` before indexing and
  before every query, so L2 distance ranks results identically to cosine
  similarity. The displayed "Score" column converts L2 distance back to a
  cosine-similarity value (`1 − L2²/2`) for a more intuitive 0–1 reading.
- See `REFLECTION.md` for the Task 5 theory answers (`IndexFlatL2` vs.
  `IndexFlatIP`, why normalise for cosine similarity, and what "approximate"
  means in ANN search).
