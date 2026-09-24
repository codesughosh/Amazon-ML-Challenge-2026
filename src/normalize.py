"""Normalisation for business names and addresses.

Design rules, driven by the EDA:

* **Language-agnostic.** No country conditioning anywhere. The test set contains
  France, which never appears in training, so any country-specific branch is a
  latent failure on 15% of the test set.
* **Romanise everything.** Names are transliterated into Tamil/Devanagari in the
  India shard and share zero character n-grams with their Latin original.
  `unidecode` maps them back phonetically, after which ordinary string
  similarity works. We keep both the raw and romanised forms - the difference
  between them is itself a useful signal.
* **Addresses are the reliable field.** Names get destroyed by typos and
  transliteration; addresses keep their digits and street tokens. Extract the
  structured parts (house number, digit runs, postal code) explicitly.

Everything here is general string knowledge written in code. Nothing is looked
up from an external database or API, so it is within the fair-play rules.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd
from unidecode import unidecode

# --------------------------------------------------------------------------
# Vocabularies. France is included although absent from training - these are
# ordinary linguistic facts, not external data.
# --------------------------------------------------------------------------

LEGAL_SUFFIXES = {
    # US / general
    "corporation": "corp", "incorporated": "inc", "company": "co",
    "limited": "ltd", "llc": "llc", "llp": "llp", "lp": "lp",
    "plc": "plc", "holdings": "hold", "group": "grp",
    # India
    "private": "pvt", "pvt": "pvt", "prvt": "pvt",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "eurl": "eurl",
    "snc": "snc", "societe": "ste", "ste": "ste",
    # German / misc, cheap to include
    "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
}

STREET_TYPES = {
    "road": "rd", "rd": "rd",
    "street": "st", "st": "st", "saint": "st",   # 'St'->'SAINT' corruption seen in train
    "avenue": "ave", "ave": "ave", "av": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd",
    "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr",
    "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl",
    "square": "sq", "sq": "sq",
    "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy",
    "suite": "ste", "apartment": "apt", "apt": "apt", "floor": "flr",
    # India
    "marg": "marg", "nagar": "nagar", "colony": "colony", "sector": "sector",
    "cross": "cross", "main": "main", "phase": "phase", "block": "blk",
    # France
    "rue": "rue", "allee": "allee", "impasse": "imp", "chemin": "chem",
    "quai": "quai", "cours": "cours",
}

US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "ohio": "oh", "oklahoma": "ok", "oregon": "or",
    "pennsylvania": "pa", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "wisconsin": "wi",
    "wyoming": "wy",
}

IN_STATES = {
    "tamil nadu": "tn", "tamilnadu": "tn",
    "karnataka": "ka", "kerala": "kl", "maharashtra": "mh",
    "gujarat": "gj", "rajasthan": "rj", "punjab": "pb", "haryana": "hr",
    "west bengal": "wb", "uttar pradesh": "up", "madhya pradesh": "mp",
    "andhra pradesh": "ap", "telangana": "tg", "bihar": "br",
    "odisha": "od", "orissa": "od", "assam": "as", "jharkhand": "jh",
    "chhattisgarh": "cg", "uttarakhand": "uk", "goa": "ga",
    "delhi": "dl", "new delhi": "dl",
}

# Longest-first so multi-word states match before their first word.
_STATE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted((*US_STATES, *IN_STATES), key=len, reverse=True)) + r")\b"
)
_STATE_MAP = {**US_STATES, **IN_STATES}

_TOKEN_MAP = {**LEGAL_SUFFIXES, **STREET_TYPES}
_TOKEN_PATTERN = re.compile(r"\b(" + "|".join(sorted(_TOKEN_MAP, key=len, reverse=True)) + r")\b")

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")
_WS = re.compile(r"\s+")
_ASCII_ONLY = re.compile(r"^[\x00-\x7F]*$")


# --------------------------------------------------------------------------
# Core
# --------------------------------------------------------------------------

def romanise(s: pd.Series) -> pd.Series:
    """Map non-Latin scripts to Latin phonetically.

    `unidecode` is pure Python and slow, so only run it on rows that actually
    contain non-ASCII characters - typically a minority, which makes this
    roughly an order of magnitude cheaper than applying it to everything.
    """
    s = s.fillna("").astype(str)
    # NFKD + drop combining marks handles accents (Énterprises -> Enterprises)
    needs = ~s.str.match(_ASCII_ONLY)
    if not needs.any():
        return s
    out = s.copy()
    sub = s[needs].map(lambda x: unidecode(unicodedata.normalize("NFKD", x)))
    out.loc[needs] = sub
    return out


def basic_clean(s: pd.Series) -> pd.Series:
    """Lowercase, strip punctuation, collapse whitespace."""
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(_NON_ALNUM, " ", regex=True)
    return s.str.replace(_WS, " ", regex=True).str.strip()


def canon_tokens(s: pd.Series) -> pd.Series:
    """Canonicalise legal suffixes and street types to one spelling each."""
    return s.str.replace(_TOKEN_PATTERN, lambda m: _TOKEN_MAP[m.group(0)], regex=True)


def canon_states(s: pd.Series) -> pd.Series:
    """Collapse state names to their abbreviation (Illinois -> il, Tamil Nadu -> tn)."""
    return s.str.replace(_STATE_PATTERN, lambda m: _STATE_MAP[m.group(0)], regex=True)


def normalize_name(s: pd.Series) -> pd.Series:
    return canon_tokens(basic_clean(romanise(s)))


def normalize_address(s: pd.Series) -> pd.Series:
    return canon_states(canon_tokens(basic_clean(romanise(s))))


# --------------------------------------------------------------------------
# Structured extraction from the address - the transliteration-proof signal
# --------------------------------------------------------------------------

_DIGITS = re.compile(r"\d+")


def digit_signature(s: pd.Series) -> pd.Series:
    """All digit runs, sorted and joined: '3315' or '2-6-29'.

    Digits survive transliteration untouched - a Tamil-script address keeps its
    house number - which makes this the single most robust blocking key we have.
    """
    return s.str.findall(_DIGITS).map(lambda xs: "-".join(sorted(xs)) if xs else "")


def house_number(s: pd.Series) -> pd.Series:
    """The first digit run, which is usually the building number."""
    return s.str.extract(r"(\d+)", expand=False).fillna("")


def postal_code(s: pd.Series) -> pd.Series:
    """First 5-6 digit run: US ZIP or Indian PIN. French codes are 5 digits."""
    return s.str.extract(r"\b(\d{5,6})\b", expand=False).fillna("")


# Function words that must never become a blocking key. French articles matter
# here: 'rue de la paix' would otherwise key on 'de', which is useless.
STOPWORDS = frozenset({
    "de", "la", "le", "du", "des", "les", "un", "une", "et", "aux", "au",
    "the", "of", "and", "at", "in", "on", "for", "near", "opp", "opposite",
    "no", "nos", "new", "old",
})


def first_alpha_token(s: pd.Series, skip: frozenset = frozenset(STREET_TYPES)) -> pd.Series:
    """First meaningful alphabetic token - usually the street or locality name.

    Street-type words are skipped: 'st fremont' and 'fremont st' should both
    yield 'fremont'.
    """
    def pick(x: str) -> str:
        for t in x.split():
            if t.isalpha() and len(t) > 1 and t not in skip and t not in STOPWORDS:
                return t
        return ""
    return s.map(pick)


def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach every derived column a downstream stage needs. Modifies a copy."""
    df = df.copy()
    df["name_n"] = normalize_name(df["business_name"])
    df["addr_n"] = normalize_address(df["business_address"])
    df["addr_digits"] = digit_signature(df["addr_n"])
    df["addr_house"] = house_number(df["addr_n"])
    df["addr_pin"] = postal_code(df["addr_n"])
    df["addr_tok1"] = first_alpha_token(df["addr_n"])
    df["name_tok1"] = first_alpha_token(df["name_n"], skip=frozenset(LEGAL_SUFFIXES))
    # Did this record arrive in a non-Latin script? A useful pair feature later.
    df["was_nonascii"] = (~df["business_name"].fillna("").str.match(_ASCII_ONLY)).astype("int8")
    return df


if __name__ == "__main__":
    demo = pd.DataFrame({
        "business_name": [
            "Raj Investments LLP",
            "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
            "Payne Énterprises",
            "PAYNE-ENRTPRMISES",
            "Boulangerie Dupont SARL",
        ],
        "business_address": [
            "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu",
            "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu",
            "3315 Fremont Street, Peoria, IL",
            "3315 FREMONT SAINT, PEORIA, IL",
            "12 Rue de la Paix, 75002 Paris",
        ],
    })
    out = add_normalized_columns(demo)
    for r in out.itertuples(index=False):
        print(f"\nname  : {r.name_n}")
        print(f"addr  : {r.addr_n}")
        print(f"keys  : digits={r.addr_digits!r} house={r.addr_house!r} "
              f"pin={r.addr_pin!r} tok1={r.addr_tok1!r}")
