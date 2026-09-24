"""Bulk image downloader for product-image URLs.

The organisers' own `download_images` (2025 edition) used a 100-process pool
around `urllib.request.urlretrieve` with no retries, no resume, no timeout and
no connection reuse. On 75k train + 75k test URLs that reliably burns hours and
still leaves gaps. This version:

  * uses threads, not processes - the work is I/O bound, so threads reuse one
    connection pool instead of opening 100 independent ones
  * retries with backoff on 429/5xx (Amazon's CDN throttles aggressively)
  * resumes - already-downloaded files are skipped, so re-running is cheap
  * optionally downscales on write, which is the difference between ~8 GB and
    ~1 GB on disk and makes every later epoch faster
  * records failures to a file so you can re-run just those

Typical use on Day 1:

    from src.images import download_all
    download_all(train["image_link"], "data/train_images", max_side=256)
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional
    def tqdm(x, **kw):
        return x


_thread_local = threading.local()

DEFAULT_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 " \
             "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _session() -> requests.Session:
    """One pooled Session per worker thread."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        retry = Retry(
            total=4,
            backoff_factor=0.6,          # 0.6s, 1.2s, 2.4s, 4.8s
            status_forcelist=(408, 429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=64, pool_maxsize=64)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        s.headers["User-Agent"] = USER_AGENT
        _thread_local.session = s
    return s


def filename_for(url: str) -> str:
    """Match the organisers' convention: the basename of the URL path.

    e.g. .../images/I/71XfHPR36-L.jpg -> 71XfHPR36-L.jpg
    Keeping this identical to theirs means a dataframe can join on it directly.
    """
    return os.path.basename(urlparse(str(url)).path)


def _download_one(url: str, out_dir: Path, max_side: int | None, quality: int):
    if not isinstance(url, str) or not url.strip():
        return ("skip", url, "empty url")

    name = filename_for(url)
    if not name:
        return ("fail", url, "no filename in url")

    # Downscaling re-encodes to JPEG, so normalise the extension.
    if max_side:
        name = str(Path(name).with_suffix(".jpg"))
    dest = out_dir / name

    if dest.exists() and dest.stat().st_size > 0:
        return ("cached", url, str(dest))

    try:
        resp = _session().get(url, timeout=DEFAULT_TIMEOUT, stream=True)
        if resp.status_code != 200:
            return ("fail", url, f"HTTP {resp.status_code}")
        payload = resp.content
        if not payload:
            return ("fail", url, "empty body")

        if max_side:
            from PIL import Image

            img = Image.open(BytesIO(payload))
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_side:
                scale = max_side / max(w, h)
                img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                 Image.BILINEAR)
            # Write to a temp name then rename, so an interrupted run never
            # leaves a truncated file that a later resume would treat as done.
            tmp = dest.with_suffix(".part")
            img.save(tmp, "JPEG", quality=quality, optimize=True)
            tmp.replace(dest)
        else:
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)

        return ("ok", url, str(dest))
    except Exception as e:
        return ("fail", url, f"{type(e).__name__}: {e}")


def download_all(
    urls,
    out_dir: str | Path,
    max_workers: int = 32,
    max_side: int | None = 256,
    quality: int = 88,
    failed_log: str | Path | None = None,
):
    """Download every URL into `out_dir`. Safe to re-run; skips what exists.

    max_side  - longest edge in pixels, re-encoded as JPEG. 256 suits CLIP/
                ConvNeXT at 224. Pass None to keep originals (much more disk).
    max_workers - 32 is a good balance. Push to 64 if the CDN tolerates it;
                back off to 16 if you start seeing 429s in the failure log.

    Returns (n_ok, n_cached, n_failed, failed_urls).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = [u for u in list(urls)]
    counts = {"ok": 0, "cached": 0, "fail": 0, "skip": 0}
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_download_one, u, out_dir, max_side, quality)
                   for u in urls]
        bar = tqdm(as_completed(futures), total=len(futures), desc="images",
                   unit="img", smoothing=0.05)
        for fut in bar:
            status, url, info = fut.result()
            counts[status] += 1
            if status == "fail":
                failed.append((url, info))

    print(f"\ndownloaded {counts['ok']:,} | cached {counts['cached']:,} | "
          f"failed {counts['fail']:,} | skipped {counts['skip']:,}")

    if failed:
        log = Path(failed_log or (out_dir.parent / f"failed_{out_dir.name}.txt"))
        log.write_text("\n".join(f"{u}\t{why}" for u, why in failed), encoding="utf-8")
        print(f"failure log -> {log}")
        print("Re-run download_all with the same arguments to retry only these.")

    if max_side:
        total = sum(f.stat().st_size for f in out_dir.glob("*.jpg")) / 1024**3
        print(f"on disk: {total:.2f} GB in {out_dir}")

    return counts["ok"], counts["cached"], counts["fail"], [u for u, _ in failed]


def attach_paths(df, url_col: str, out_dir: str | Path,
                 path_col: str = "image_path", max_side: int | None = 256):
    """Add a local-path column and report how many images are actually present.

    Run this after download_all. Any row with a missing image needs an explicit
    decision - usually a zero vector for the image branch, never a silent drop.
    """
    out_dir = Path(out_dir)

    def resolve(u):
        if not isinstance(u, str) or not u.strip():
            return None
        name = filename_for(u)
        if max_side:
            name = str(Path(name).with_suffix(".jpg"))
        p = out_dir / name
        return str(p) if p.exists() else None

    df = df.copy()
    df[path_col] = [resolve(u) for u in df[url_col]]
    n_missing = int(df[path_col].isnull().sum())
    print(f"{len(df) - n_missing:,}/{len(df):,} images present "
          f"({n_missing:,} missing)")
    if n_missing:
        print("Decide explicitly how the model handles missing images "
              "(zero vector is usually right). Do not drop the rows - the test "
              "set needs a prediction for every id.")
    return df
