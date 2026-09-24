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

    for i in range(len(df)):
        a = addr[i]
        rare_a = rare_tokens(a, dfreq_addr, n_rare, max_df)
        digs = digit_tokens(a, n_digit)

        for t in rare_a:
            for d in digs:
                idx.append(i); keys.append(f"d|{d}|{t}")
            if dfreq_addr.get(t, 1) <= solo_df:
                idx.append(i); keys.append(f"a|{t}")
            if pin[i]:
                idx.append(i); keys.append(f"p|{strip_zeros(pin[i])}|{t}")

        for t in rare_tokens(name[i], dfreq_name, 2, max_df):
            if dfreq_name.get(t, 1) <= solo_df:
                idx.append(i); keys.append(f"n|{t}")

    return pd.DataFrame({"i": np.asarray(idx, dtype=np.int32), "k": keys})


def candidates(s1: pd.DataFrame, s23: pd.DataFrame,
               dfreq_addr: Counter, dfreq_name: Counter,
               max_block: int = 300, max_per_s1: int = 0,
               **kw) -> np.ndarray:
    """Return ((n,2) int32 pairs, (n,) int16 shared-key counts)."""
    k1 = make_keys(s1, dfreq_addr, dfreq_name, **kw)
    k2 = make_keys(s23, dfreq_addr, dfreq_name, **kw)

    # Drop keys whose block on the S2/S3 side is too large to be informative.
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


def build_dfreq(*frames: pd.DataFrame) -> tuple[Counter, Counter]:
    """Document frequencies over the pooled shard (all three sources)."""
    da, dn = Counter(), Counter()
    for f in frames:
        da += token_document_frequency(f["addr_n"])
        dn += token_document_frequency(f["name_n"])
    return da, dn
