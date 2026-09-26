"""Candidate generation (blocking).

Why exact-key blocking fails here - measured, not assumed. On the India shard a
union of six exact keys reached only 64% recall. Inspecting the misses showed
four recurring patterns:

    32/33                  vs  032/33                    leading zeros
    LIG 40, Shakar Nagar   vs  PLOT 944/5 LIG 40, ...    inserted component
    no 12 prem nagar gwalior vs no 12 gwalior            dropped components
    Tamil Nadu             vs  தமிழ்நாடு -> tmilllnaattu  romanised state

Any single fixed key breaks under insertion, deletion or reordering. But in
every one of those examples the two records still share a **rare token**
(`gwalior`, `siyaram`, `shakar`, `seaward`). So we block on rare tokens instead:

  * each record emits SEVERAL keys, not one
  * two records are candidates if they share ANY key
  * keys pair a rare alphabetic token with a digit token, which keeps blocks
    small while surviving insertion and deletion
  * very rare tokens also emit a key on their own

Rarity is computed from the corpus itself (document frequency over the country
shard), not from any external resource.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:                       # progress bars are optional
    def tqdm(x=None, **kw):
        return x if x is not None else _NullBar()

class _NullBar:
    def update(self, *a): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass

_DIGIT_RE = re.compile(r"\d+")


def strip_zeros(tok: str) -> str:
    """'032' -> '32'. Leading zeros are a pure formatting artefact."""
    t = tok.lstrip("0")
    return t if t else "0"


def token_document_frequency(series: pd.Series) -> Counter:
    """Document frequency of each alphabetic token across the shard."""
    df = Counter()
    for s in series:
        df.update({t for t in s.split() if t.isalpha() and len(t) > 2})
    return df


def rare_tokens(text: str, dfreq: Counter, k: int = 3, max_df: int = 100_000) -> list[str]:
    """The k rarest alphabetic tokens in `text`.

    Tokens above `max_df` are dropped as too generic to be useful keys
    (city names, 'nagar', 'road' and similar).
    """
    toks = {t for t in text.split() if t.isalpha() and len(t) > 2}
    scored = [(dfreq.get(t, 1), t) for t in toks]
    scored = [(d, t) for d, t in scored if d <= max_df]
    scored.sort()
    return [t for _, t in scored[:k]]


def digit_tokens(text: str, k: int = 3) -> list[str]:
    """Digit runs with leading zeros stripped, longest first.

    Longest first because a 5-digit PIN discriminates far better than a
    1-digit floor number.
    """
    ds = {strip_zeros(d) for d in _DIGIT_RE.findall(text)}
    ds = [d for d in ds if d != "0"]
    ds.sort(key=lambda x: (-len(x), x))
    return ds[:k]


def make_keys(df: pd.DataFrame, dfreq_addr: Counter, dfreq_name: Counter,
              n_rare: int = 3, n_digit: int = 3,
              solo_df: int = 400, max_df: int = 100_000) -> pd.DataFrame:
    """Explode each record into its blocking keys.

    Returns a long frame of (row index, key). Key families:

      d|<digit>|<rare addr token>   main key: survives insertion and deletion
      a|<rare addr token>           solo, only for tokens rarer than `solo_df`
      n|<rare name token>           name-side key, from the romanised name
      p|<postal>|<rare addr token>  postal code paired with a rare token
    """
    idx, keys = [], []

    addr = df["addr_n"].to_numpy()
    name = df["name_n"].to_numpy()
    pin = df["addr_pin"].to_numpy()

    rng = tqdm(range(len(df)), desc="  keys   ", unit="rec", unit_scale=True,
               leave=False, mininterval=0.3, ascii=True, ncols=78)
    for i in rng:
        a, nm = addr[i], name[i]
        rare_a = rare_tokens(a, dfreq_addr, n_rare, max_df)
        digs = digit_tokens(a, n_digit)

        for t in rare_a:
            for d in digs:
                idx.append(i); keys.append(f"d|{d}|{t}")
            if dfreq_addr.get(t, 1) <= solo_df:
                idx.append(i); keys.append(f"a|{t}")
            if pin[i]:
                idx.append(i); keys.append(f"p|{strip_zeros(pin[i])}|{t}")

        for t in rare_tokens(nm, dfreq_name, 2, max_df):
            if dfreq_name.get(t, 1) <= solo_df:
                idx.append(i); keys.append(f"n|{t}")

        # --- families added after diagnosing missed pairs -------------------
        # (a) Full normalised name. Records whose address is EMPTY are
        #     otherwise unreachable, and empty addresses are common in S2/S3.
        #     'mumbai security co' matches exactly across sources but every
        #     token is too frequent to qualify as a rare key.
        if nm:
            idx.append(i); keys.append(f"N|{nm}")

        # (b) Digit signature. Digits survive transliteration and rewriting,
        #     and two or more of them together are highly discriminating even
        #     when no alphabetic token is shared.
        if len(digs) >= 2:
            idx.append(i); keys.append("D|" + "-".join(sorted(digs)))

        # (c) Digit x name token. Catches transliterated names with a partly
        #     rewritten address, where the only stable anchors are a house
        #     number plus one surviving name token.
        for t in rare_tokens(nm, dfreq_name, 3, max_df):
            for d in digs[:2]:
                idx.append(i); keys.append(f"dn|{d}|{t}")

    return pd.DataFrame({"i": np.asarray(idx, dtype=np.int32), "k": keys})


def prepare_index(s23: pd.DataFrame, dfreq_addr: Counter, dfreq_name: Counter,
                  max_block: int = 300, **kw) -> pd.DataFrame:
    """Build the S2/S3 key index once, for reuse across many S1 chunks.

    Key generation over a 2M-record source takes about a minute; at inference
    we query it with ~17 chunks of S1, so building it once instead of per chunk
    saves the bulk of the blocking time.
    """
    k2 = make_keys(s23, dfreq_addr, dfreq_name, **kw)
    sz = k2.groupby("k", sort=False)["i"].transform("size")
    return k2[sz <= max_block].reset_index(drop=True)


def candidates(s1: pd.DataFrame, s23: pd.DataFrame,
               dfreq_addr: Counter, dfreq_name: Counter,
               max_block: int = 300, max_per_s1: int = 0,
               k2: pd.DataFrame | None = None,
               **kw) -> np.ndarray:
    """Return ((n,2) int32 pairs, (n,) int16 shared-key counts).

    Pass `k2` from `prepare_index` to skip rebuilding the S2/S3 key index.
    """
    k1 = make_keys(s1, dfreq_addr, dfreq_name, **kw)
    if k2 is None:
        k2 = make_keys(s23, dfreq_addr, dfreq_name, **kw)
        sz = k2.groupby("k", sort=False)["i"].transform("size")
        k2 = k2[sz <= max_block]
    k1 = k1[k1["k"].isin(k2["k"].unique())]
    if k1.empty or k2.empty:
        return np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.int16)

    m = k1.merge(k2, on="k", suffixes=("1", "2"), copy=False)
    pairs = m[["i1", "i2"]].to_numpy(dtype=np.int32)
    del m, k1, k2

    # A pair found via several keys appears several times. Collapse to unique
    # pairs and keep the multiplicity: the number of distinct keys two records
    # share is a strong, free relevance signal - pairs agreeing on four keys
    # are far more likely to be true than pairs agreeing on one.
    pairs, nkeys = np.unique(pairs, axis=0, return_counts=True)

    if max_per_s1:
        # Keep the top `max_per_s1` candidates per S1 by shared-key count.
        # Sorting by (s1, -nkeys) puts the best candidates first within each
        # S1 group, so a per-group head slice is the top-k.
        order = np.lexsort((-nkeys, pairs[:, 0]))
        pairs, nkeys = pairs[order], nkeys[order]
        _, starts, counts = np.unique(pairs[:, 0], return_index=True, return_counts=True)
        keep = np.concatenate([np.arange(s, s + min(c, max_per_s1))
                               for s, c in zip(starts, counts)])
        pairs, nkeys = pairs[keep], nkeys[keep]

    return pairs, nkeys.astype(np.int16)


def rescore_and_cut(s1: pd.DataFrame, s23: pd.DataFrame,
                    pairs: np.ndarray, nkeys: np.ndarray,
                    top_k: int = 50, workers: int = -1,
                    chunk: int = 4_000_000):
    """Re-rank raw candidates by string similarity, then keep the top k per S1.

    Why this exists: measurement showed the rare-token blocker *finds* 93.2% of
    true S2 pairs across its raw ~300 candidates per entity, but keeping the
    top 50 by shared-key count retains only 88.9%. The 4.3-point gap is a
    ranking failure, not a candidate-generation failure - the true pairs are
    already in the set, just sorted below the cut.

    A full TF-IDF nearest-neighbour *search* would fix the ranking but costs
    roughly nine hours per source at full scale (measured). Scoring pairs we
    already have is orders of magnitude cheaper: `rapidfuzz.process.cpdist`
    runs elementwise over two aligned lists in parallel C++, so this is seconds
    per million pairs rather than hours.

    Returns (pairs, nkeys, score) filtered to the top k per S1 entity.
    """
    from rapidfuzz import fuzz, process

    A1 = s1["addr_n"].to_numpy()
    A2 = s23["addr_n"].to_numpy()
    N1 = s1["name_n"].to_numpy()
    N2 = s23["name_n"].to_numpy()

    # Scored in chunks. The generous key families produce ~2100 candidates per
    # entity, so a 25k-entity shard is ~50M pairs; materialising four Python
    # string lists over all of them needs ~14 GB and OOMs. Chunking keeps peak
    # memory flat regardless of shard size.
    score = np.empty(len(pairs), dtype=np.float32)
    bar = tqdm(total=len(pairs), desc="  rescore", unit="pair",
               unit_scale=True, leave=False, mininterval=0.3, ascii=True, ncols=78)
    for lo in range(0, len(pairs), chunk):
        hi = min(lo + chunk, len(pairs))
        i1, i2 = pairs[lo:hi, 0], pairs[lo:hi, 1]
        a1 = A1[i1].tolist(); a2 = A2[i2].tolist()
        n1 = N1[i1].tolist(); n2 = N2[i2].tolist()
        c1 = [x + " " + y for x, y in zip(n1, a1)]
        c2 = [x + " " + y for x, y in zip(n2, a2)]

        # token_set_ratio is insensitive to inserted/dropped components and to
        # word order - the noise patterns that broke the exact keys.
        # partial_ratio adds the case where one record is a truncated fragment
        # of the other, which is what happens when an address is shortened or
        # emptied; it was worth +2.9pp of top-50 recall on its own (measured).
        sa = np.asarray(process.cpdist(a1, a2, scorer=fuzz.token_set_ratio,
                                       workers=workers, dtype=np.float32))
        sn = np.asarray(process.cpdist(n1, n2, scorer=fuzz.token_set_ratio,
                                       workers=workers, dtype=np.float32))
        sc = np.asarray(process.cpdist(c1, c2, scorer=fuzz.token_set_ratio,
                                       workers=workers, dtype=np.float32))
        sp = np.asarray(process.cpdist(c1, c2, scorer=fuzz.partial_ratio,
                                       workers=workers, dtype=np.float32))
        # Weights from a sweep over ranker variants:
        #   addr+name 92.33 | combined 93.91 | max 94.20 | this blend 95.18
        score[lo:hi] = 0.3 * sa + 0.2 * sn + 0.2 * sc + 0.3 * sp
        del a1, a2, n1, n2, c1, c2, sa, sn, sc, sp
        bar.update(hi - lo)
    bar.close()

    order = np.lexsort((-score, pairs[:, 0]))
    pairs, nkeys, score = pairs[order], nkeys[order], score[order]
    _, starts, counts = np.unique(pairs[:, 0], return_index=True, return_counts=True)
    keep = np.concatenate([np.arange(s, s + min(c, top_k))
                           for s, c in zip(starts, counts)])
    return pairs[keep], nkeys[keep], score[keep]


def build_dfreq(*frames: pd.DataFrame) -> tuple[Counter, Counter]:
    """Document frequencies over the pooled shard (all three sources)."""
    da, dn = Counter(), Counter()
    for f in frames:
        da += token_document_frequency(f["addr_n"])
        dn += token_document_frequency(f["name_n"])
    return da, dn
