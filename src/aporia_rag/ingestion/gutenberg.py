"""Downloading plain-text ebooks from a Project Gutenberg mirror, with a local cache.

Project Gutenberg forbids automated access to www.gutenberg.org and asks
scripts to use a mirror, waiting 2 seconds between requests
(https://www.gutenberg.org/policy/robot_access.html). Everything here talks
to a mirror only, and `GutenbergMirror` enforces the wait itself rather than
trusting every caller to remember it.
"""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Final, NamedTuple, Protocol

import httpx

from aporia_rag.domain.errors import CorpusError

DEFAULT_MIRROR: Final = "https://gutenberg.pglaf.org"
POLITE_DELAY_SECONDS: Final = 2.0
USER_AGENT: Final = "aporia-rag/0.1 (portfolio research project; python-httpx)"

# Plain-text files in order of preference: "-0" is UTF-8 and "-8" ISO-8859-1.
# The bare file is US-ASCII, which is a subset of UTF-8; decoding is strict,
# so a mislabelled file fails loudly instead of being silently garbled.
_CANDIDATE_FILES: Final = (("-0.txt", "utf-8"), ("-8.txt", "latin-1"), (".txt", "utf-8"))


class EbookNotFoundError(CorpusError, LookupError):
    def __init__(self, gutenberg_id: int) -> None:
        super().__init__(f"no plain-text file for Project Gutenberg ebook #{gutenberg_id}")
        self.gutenberg_id = gutenberg_id


class EbookSource(Protocol):
    """Anything that can produce the raw text of a Gutenberg ebook."""

    def download(self, gutenberg_id: int) -> str: ...


def mirror_directory(gutenberg_id: int) -> str:
    """Mirror folder of an ebook: one folder per digit except the last (`0` for 1-digit ids)."""
    digits = str(gutenberg_id)
    parents = list(digits[:-1]) or ["0"]
    return "/".join([*parents, digits])


class GutenbergMirror:
    """An `EbookSource` backed by a Project Gutenberg mirror over HTTP."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        base_url: str = DEFAULT_MIRROR,
        delay_seconds: float = POLITE_DELAY_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._delay_seconds = delay_seconds
        self._sleep = sleep
        self._has_requested = False

    def download(self, gutenberg_id: int) -> str:
        directory = mirror_directory(gutenberg_id)
        for suffix, encoding in _CANDIDATE_FILES:
            response = self._get(f"{self._base_url}/{directory}/{gutenberg_id}{suffix}")
            if response.status_code == httpx.codes.NOT_FOUND:
                continue
            response.raise_for_status()
            return response.content.decode(encoding)
        raise EbookNotFoundError(gutenberg_id)

    def _get(self, url: str) -> httpx.Response:
        if self._has_requested:
            self._sleep(self._delay_seconds)
        self._has_requested = True
        return self._client.get(url, headers={"User-Agent": USER_AGENT})


class CachedText(NamedTuple):
    text: str
    path: Path
    from_cache: bool


def cache_path(cache_dir: Path, gutenberg_id: int) -> Path:
    """Where an ebook's raw text lives in the local cache."""
    return cache_dir / f"pg{gutenberg_id}.txt"


def fetch_cached(gutenberg_id: int, cache_dir: Path, source: EbookSource) -> CachedText:
    """Return an ebook's raw text, downloading it into `cache_dir` only if absent.

    The cache stores exactly what was downloaded (as UTF-8), so cleaning can be
    changed and re-run without touching the network. Bytes are read and written
    directly to avoid any newline translation. The file is written under a
    temporary name and then renamed, so an interrupted write never leaves a
    truncated file that a later run would mistake for a complete download.
    """
    path = cache_path(cache_dir, gutenberg_id)
    if path.is_file():
        return CachedText(path.read_bytes().decode("utf-8"), path, from_cache=True)

    text = source.download(gutenberg_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_bytes(text.encode("utf-8"))
    partial.replace(path)
    return CachedText(text, path, from_cache=False)
