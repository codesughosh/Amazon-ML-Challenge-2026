"""Pairwise features for candidate (S1, S2/S3) pairs.

Three families, in rough order of value:

1. **String similarity** on romanised name and address. `rapidfuzz.process.cpdist`
   computes these elementwise over two aligned lists in parallel C++, which is
   the only way this is affordable at tens of millions of pairs.
2. **Structured overlap** - digit tokens, house number, postal code, IDF-weighted
   rare-token overlap. Digits survive transliteration, so these carry more
   signal than the name similarities on the India shard.
3. **Relational** - how a candidate ranks *among the other candidates of the same
   S1 entity*. A candidate that is merely good matters far less than one that is
   clearly the best available, and these features are consistently among the
   strongest in entity resolution.

No country feature is ever emitted: the test set contains France, unseen in
training, and a country-conditioned model would fail on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

_WORKERS = -1  # all cores


def _cp(a: list[str], b: list[str], scorer) -> np.ndarray:
    """Elementwise similarity over two aligned lists, in [0, 1]."""
    out = process.cpdist(a, b, scorer=scorer, workers=_WORKERS, dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _tokset(s: str) -> set[str]:
    return set(s.split())


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def build_features(s1: pd.DataFrame, s23: pd.DataFrame,
                   pairs: np.ndarray, nkeys: np.ndarray,
                   idf: dict[str, float] | None = None) -> pd.DataFrame:
    """Return a float32 feature frame, one row per candidate pair."""
    i1 = pairs[:, 0]
    i2 = pairs[:, 1]

    n1 = s1["name_n"].to_numpy()[i1].tolist()
    n2 = s23["name_n"].to_numpy()[i2].tolist()
    a1 = s1["addr_n"].to_numpy()[i1].tolist()
    a2 = s23["addr_n"].to_numpy()[i2].tolist()

    f: dict[str, np.ndarray] = {}

    # ---- 1. string similarity -------------------------------------------
    for tag, x, y in (("nm", n1, n2), ("ad", a1, a2)):
        f[f"{tag}_ratio"] = _cp(x, y, fuzz.ratio) / 100.0
        f[f"{tag}_tsort"] = _cp(x, y, fuzz.token_sort_ratio) / 100.0
        f[f"{tag}_tset"] = _cp(x, y, fuzz.token_set_ratio) / 100.0
        f[f"{tag}_part"] = _cp(x, y, fuzz.partial_ratio) / 100.0
        f[f"{tag}_jw"] = _cp(x, y, JaroWinkler.normalized_similarity)

    # Cross-field: name of one against address of the other, which catches
    # records where the fields were swapped or merged.
    f["x_nm1_ad2"] = _cp(n1, a2, fuzz.partial_ratio) / 100.0
    f["x_nm2_ad1"] = _cp(n2, a1, fuzz.partial_ratio) / 100.0

    # ---- 2. structured overlap ------------------------------------------
    f["nkeys"] = nkeys.astype(np.float32)

    for col, tag in (("addr_house", "house"), ("addr_pin", "pin")):
        v1 = s1[col].to_numpy()[i1]
        v2 = s23[col].to_numpy()[i2]
        both = (v1 != "") & (v2 != "")
        f[f"{tag}_eq"] = np.where(both, (v1 == v2).astype(np.float32), -1.0).astype(np.float32)

    d1 = s1["addr_digits"].to_numpy()[i1]
    d2 = s23["addr_digits"].to_numpy()[i2]
    f["dig_jac"] = np.fromiter(
        (_jaccard(set(x.split("-")) - {""}, set(y.split("-")) - {""}) for x, y in zip(d1, d2)),
        dtype=np.float32, count=len(i1))

    # IDF-weighted token overlap: sharing a rare token is worth far more than
    # sharing 'road'. This is the structured analogue of TF-IDF cosine.
    if idf is not None:
        def widf(xs, ys):
            out = np.empty(len(xs), dtype=np.float32)
            for j, (x, y) in enumerate(zip(xs, ys)):
                sx, sy = _tokset(x), _tokset(y)
                inter = sx & sy
                union = sx | sy
                num = sum(idf.get(t, 8.0) for t in inter)
                den = sum(idf.get(t, 8.0) for t in union)
                out[j] = num / den if den else 0.0
            return out
        f["ad_idf_ov"] = widf(a1, a2)
        f["nm_idf_ov"] = widf(n1, n2)

    # Length signals: a much shorter address usually means dropped components.
    l1 = np.fromiter((len(x) for x in a1), dtype=np.float32, count=len(a1))
    l2 = np.fromiter((len(x) for x in a2), dtype=np.float32, count=len(a2))
    f["ad_len_ratio"] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1.0)
    m1 = np.fromiter((len(x) for x in n1), dtype=np.float32, count=len(n1))
    m2 = np.fromiter((len(x) for x in n2), dtype=np.float32, count=len(n2))
    f["nm_len_ratio"] = np.minimum(m1, m2) / np.maximum(np.maximum(m1, m2), 1.0)

    # Script mismatch: was one side transliterated? Name similarity is much
    # less trustworthy when it was.
    f["nonascii_xor"] = (s1["was_nonascii"].to_numpy()[i1]
                         ^ s23["was_nonascii"].to_numpy()[i2]).astype(np.float32)

    X = pd.DataFrame(f)

    # ---- 3. relational: rank within the S1 entity's candidate group -------
    # A high absolute similarity means little if every candidate scores high;
    # what matters is being clearly the best of the group.
    base = (X["ad_tset"] + X["nm_tset"] + X["ad_idf_ov"] if "ad_idf_ov" in X
            else X["ad_tset"] + X["nm_tset"]).to_numpy()
    X["base"] = base.astype(np.float32)

    g = pd.DataFrame({"i1": i1, "b": base})
    grp = g.groupby("i1", sort=False)["b"]
    X["grp_size"] = grp.transform("size").to_numpy(dtype=np.float32)
    gmax = grp.transform("max").to_numpy(dtype=np.float32)
    gmean = grp.transform("mean").to_numpy(dtype=np.float32)
    gstd = grp.transform("std").fillna(0).to_numpy(dtype=np.float32)
    X["rank"] = grp.rank(ascending=False, method="first").to_numpy(dtype=np.float32)
    X["ratio_to_best"] = (base / np.maximum(gmax, 1e-6)).astype(np.float32)
    X["margin_to_best"] = (gmax - base).astype(np.float32)
    X["z_in_grp"] = ((base - gmean) / np.maximum(gstd, 1e-6)).astype(np.float32)

    # Second-best margin: how far clear is the leader? Large gap means the top
    # candidate is unambiguous.
    srt = np.lexsort((-base, i1))
    b_s, i_s = base[srt], i1[srt]
    second = np.zeros(len(base), dtype=np.float32)
    _, starts, counts = np.unique(i_s, return_index=True, return_counts=True)
    for s, c in zip(starts, counts):
        if c > 1:
            second[s:s + c] = b_s[s + 1]
    inv = np.empty_like(srt)
    inv[srt] = np.arange(len(srt))
    X["gap_to_second"] = (gmax - second[inv]).astype(np.float32)

    return X.astype(np.float32)


def compute_idf(*series: pd.Series) -> dict[str, float]:
    """Smoothed IDF over the pooled shard."""
    from collections import Counter
    df = Counter()
    n = 0
    for s in series:
        n += len(s)
        for x in s:
            df.update(set(x.split()))
    return {t: float(np.log(n / (1 + c))) for t, c in df.items()}
