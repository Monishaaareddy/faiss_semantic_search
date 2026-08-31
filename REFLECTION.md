# Bonus Reflection Questions

### Q1: Difference between `IndexFlatL2` and `IndexFlatIP` — when to use each?

`IndexFlatL2` ranks vectors by **Euclidean (L2) distance** — it finds vectors that
are physically closest in the embedding space. Smaller distance = more similar.

`IndexFlatIP` ranks vectors by **inner product (dot product)** — it finds vectors
that maximise `a · b`. Larger inner product = more similar.

Both do an exact, brute-force ("Flat") comparison against every stored vector,
so neither is approximate — the difference is purely the distance metric.

**When to use which:**
- Use `IndexFlatIP` when your embeddings are already normalised (unit length)
  and you want **cosine similarity** — for normalised vectors, inner product
  and cosine similarity are mathematically identical.
- Use `IndexFlatL2` when you care about actual geometric distance, or when
  working with embeddings that are *not* normalised (e.g. some vision
  embeddings where magnitude carries meaning).
- In practice, for text semantic search with sentence-transformers, both work
  once vectors are L2-normalised, because L2 distance on unit vectors is a
  monotonic transform of cosine similarity: `L2² = 2 − 2·cosine_sim`. This
  project uses `IndexFlatL2` on normalised vectors and converts the resulting
  distance back to a cosine-similarity score for display, but swapping in
  `IndexFlatIP` on the same normalised vectors would rank results identically.

### Q2: Why normalise embeddings before adding to FAISS for cosine similarity?

Cosine similarity measures the **angle** between two vectors, ignoring their
magnitude:

```
cosine_sim(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

If every vector is normalised to unit length (‖a‖ = ‖b‖ = 1), the denominator
becomes 1, so:

```
cosine_sim(a, b) = a · b
```

That means the **inner product of normalised vectors is exactly the cosine
similarity** — no division needed at search time. This lets us use FAISS's
fast, hardware-optimised distance kernels (`IndexFlatIP`, or `IndexFlatL2` via
the identity above) to get cosine-similarity-based ranking without having to
implement or compute cosine similarity manually for every comparison.
`faiss.normalize_L2()` does this normalisation in place before the vectors go
into the index, so every stored vector — and every query vector at search
time — has unit length and comparisons are angle-only, not magnitude-biased.

### Q3: What does "approximate" mean in ANN search, and why is it acceptable?

FAISS's `IndexFlatL2` / `IndexFlatIP` (used in this project) are **exact**
search — they compare a query against every single stored vector, guaranteeing
the true top-k nearest neighbours. That's fine at small scale (our 10–12
sentence knowledge base) but becomes slow at millions/billions of vectors
because it's O(n) per query.

For large-scale production systems, FAISS offers **ANN (Approximate Nearest
Neighbour)** index types — e.g. `IndexIVFFlat`, `IndexHNSWFlat`, `IndexPQ`.
These build a data structure (inverted lists, graphs, quantised codes) that
lets a query skip the vast majority of vectors and only compare against a
promising subset. "Approximate" means the returned top-k results are **very
likely, but not mathematically guaranteed,** to be the exact true nearest
neighbours — occasionally a genuinely close vector might be missed in favour
of one that's marginally further away.

This trade-off is acceptable in practice because:
- The accuracy loss is usually tiny (often 95–99%+ recall relative to exact
  search) and tunable via index parameters (e.g. `nprobe`, graph connectivity).
- The speed gain is enormous — sub-linear or near-constant-time lookups
  instead of scanning millions of vectors, which is essential for real-time
  RAG/search systems.
- In semantic search, the "true" #1 vs. #2 nearest neighbour are usually
  semantically similar anyway, so a near-miss rarely changes the practical
  quality of results the user sees.
