"""TF-IDF character n-gram blocking.

The rare-token blocker requires two records to share an *exact* token. When a
typo corrupts every shared token the pair becomes invisible to it - which is
most of the 15.6% of true pairs it loses.

Character n-grams degrade gracefully instead. `enterprises` and `enrtprmises`
share most of their 3-grams despite no whole token matching, so cosine
similarity over char n-grams still ranks the pair highly.

`sparse_dot_topn.sp_matmul_topn` keeps only the top-k entries per row during
multiplication, so the (n1 x n2) product is never materialised - without it
this is a 883k x 2M dense matrix and completely infeasible.
"""

from __future__ import annotations

import gc
import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

# Use every core. The matmul is the dominant cost and scales close to linearly.
N_THREADS = os.cpu_count() or 8


def build_vectorizer(corpus, analyzer: str = "char_wb",
                     ngram_range: tuple[int, int] = (3, 4),
                     max_features: int = 300_000,
                     min_df: int = 2,
                     fit_sample: int = 400_000,
                     seed: int = 42) -> TfidfVectorizer:
    """Fit a TF-IDF vectoriser on the pooled corpus.

    Fitted on S2+S3 (the pool being searched) so IDF reflects the space we are
    querying into. L2 normalisation is on by default, which makes the dot
    product exactly cosine similarity.

    `fit_sample` caps how many documents the *fit* sees. sklearn's vectoriser
    fit is single-threaded, and on 4.1M documents with char n-grams it becomes
    the dominant cost of the whole blocking stage. IDF is a log of a document
    ratio, so a 400k-document sample estimates it to well within the precision
    that matters for ranking candidates - and `transform` still runs over every
    document. Pass 0 to fit on everything.
    """
    if fit_sample and len(corpus) > fit_sample:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(corpus), fit_sample, replace=False)
        fit_on = [corpus[int(i)] for i in idx]
    else:
        fit_on = corpus

    vec = TfidfVectorizer(
        analyzer=analyzer, ngram_range=ngram_range,
        max_features=max_features, min_df=min_df,
        lowercase=False,          # normalisation already lowercased
        dtype=np.float32,
    )
    vec.fit(fit_on)
    return vec


def topk_neighbours(vec: TfidfVectorizer, query_text, index_text,
                    top_k: int = 30, threshold: float = 0.20,
                    chunk: int = 40_000, n_threads: int = N_THREADS,
                    verbose: bool = True):
    """Top-k cosine neighbours in `index_text` for every row of `query_text`.

    Returns (pairs (n,2) int32, sims (n,) float32).

    Queries are chunked because the intermediate product for a chunk is
    (chunk x n_index) before pruning; 40k rows keeps the peak bounded.
    """
    Xi = vec.transform(index_text)
    Xi = sp.csr_matrix(Xi.T)          # (vocab, n_index) for A @ B
    n_q = len(query_text)

    all_pairs, all_sims = [], []
    for start in range(0, n_q, chunk):
        stop = min(start + chunk, n_q)
        Xq = vec.transform(query_text[start:stop])

        C = sp_matmul_topn(Xq, Xi, top_n=top_k, threshold=threshold,
                           sort=True, n_threads=n_threads)
        C = C.tocoo()
        if C.nnz:
            all_pairs.append(np.stack([C.row.astype(np.int32) + start,
                                       C.col.astype(np.int32)], axis=1))
            all_sims.append(C.data.astype(np.float32))
        del Xq, C
        if verbose:
            print(f"      tfidf {stop:,}/{n_q:,}  ({n_threads} threads)", flush=True)
        gc.collect()

    del Xi
    gc.collect()
    if not all_pairs:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    return np.concatenate(all_pairs), np.concatenate(all_sims)


def union_pairs(*pair_arrays) -> np.ndarray:
    """Union several (n,2) candidate arrays, de-duplicated."""
    arrs = [p for p in pair_arrays if len(p)]
    if not arrs:
        return np.empty((0, 2), np.int32)
    return np.unique(np.concatenate(arrs), axis=0).astype(np.int32)
