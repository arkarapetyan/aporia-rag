"""Tests for aporia_rag.ingestion.gutenberg (test plan: Component 4).

No test here touches the network. The mirror client is exercised through
`httpx.MockTransport` (an in-memory fake server), and the cache through a
fake `EbookSource`, both implementing the same interfaces as the real thing.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Protocol

import httpx
import pytest

from aporia_rag.ingestion.gutenberg import (
    USER_AGENT,
    EbookNotFoundError,
    GutenbergMirror,
    fetch_cached,
    mirror_directory,
)

BASE_URL = "https://mirror.test"


class FakeMirrorServer:
    """Serves `files` by URL path, answers `errors` with a status code, 404s everything else."""

    def __init__(self, files: dict[str, bytes], errors: dict[str, int] | None = None) -> None:
        self.files = files
        self.errors = errors or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path in self.errors:
            return httpx.Response(self.errors[path])
        if path in self.files:
            return httpx.Response(200, content=self.files[path])
        return httpx.Response(404)

    @property
    def paths(self) -> list[str]:
        return [request.url.path for request in self.requests]


class MirrorFactory(Protocol):
    def __call__(
        self,
        server: FakeMirrorServer,
        *,
        delay_seconds: float = ...,
        sleep: Callable[[float], None] = ...,
    ) -> GutenbergMirror: ...


def _no_sleep(_: float) -> None:
    pass


@pytest.fixture
def make_mirror() -> Iterator[MirrorFactory]:
    clients: list[httpx.Client] = []

    def make(
        server: FakeMirrorServer,
        *,
        delay_seconds: float = 0.0,
        sleep: Callable[[float], None] = _no_sleep,
    ) -> GutenbergMirror:
        client = httpx.Client(transport=httpx.MockTransport(server))
        clients.append(client)
        return GutenbergMirror(client, base_url=BASE_URL, delay_seconds=delay_seconds, sleep=sleep)

    yield make
    for client in clients:
        client.close()


class FakeSource:
    """An EbookSource that serves fixed texts and records what was asked for."""

    def __init__(self, texts: dict[int, str]) -> None:
        self.texts = texts
        self.calls: list[int] = []

    def download(self, gutenberg_id: int) -> str:
        self.calls.append(gutenberg_id)
        if gutenberg_id not in self.texts:
            raise EbookNotFoundError(gutenberg_id)
        return self.texts[gutenberg_id]


class TestMirrorLayout:
    @pytest.mark.parametrize(
        ("gutenberg_id", "expected"),
        [(1497, "1/4/9/1497"), (64576, "6/4/5/7/64576"), (59, "5/59"), (5, "0/5")],
    )
    def test_directory_has_one_folder_per_digit_but_the_last(
        self, gutenberg_id: int, expected: str
    ) -> None:  # 4.1
        assert mirror_directory(gutenberg_id) == expected


class TestDownload:
    def test_downloads_the_utf8_file(self, make_mirror: MirrorFactory) -> None:  # 4.2
        text = "\u201cWhat is justice?\u201d asked Socrates."
        server = FakeMirrorServer({"/1/4/9/1497/1497-0.txt": text.encode("utf-8")})

        assert make_mirror(server).download(1497) == text
        assert server.paths == ["/1/4/9/1497/1497-0.txt"]

    def test_falls_back_to_the_latin1_file(self, make_mirror: MirrorFactory) -> None:  # 4.3
        server = FakeMirrorServer({"/9/6/6/9662/9662-8.txt": "Hume's caf\u00e9".encode("latin-1")})

        assert make_mirror(server).download(9662) == "Hume's caf\u00e9"
        assert server.paths == ["/9/6/6/9662/9662-0.txt", "/9/6/6/9662/9662-8.txt"]

    def test_falls_back_to_the_plain_file(self, make_mirror: MirrorFactory) -> None:  # 4.4
        server = FakeMirrorServer({"/3/4/9/0/34901/34901.txt": b"On Liberty"})

        assert make_mirror(server).download(34901) == "On Liberty"
        assert server.paths[-1] == "/3/4/9/0/34901/34901.txt"

    def test_raises_when_no_text_file_exists(self, make_mirror: MirrorFactory) -> None:  # 4.5
        with pytest.raises(EbookNotFoundError, match="#59"):
            make_mirror(FakeMirrorServer({})).download(59)

    def test_server_errors_are_not_mistaken_for_missing_files(
        self, make_mirror: MirrorFactory
    ) -> None:  # 4.6
        server = FakeMirrorServer({}, errors={"/5/59/59-0.txt": 503})

        with pytest.raises(httpx.HTTPStatusError):
            make_mirror(server).download(59)
        assert len(server.requests) == 1

    def test_identifies_itself(self, make_mirror: MirrorFactory) -> None:  # 4.8
        server = FakeMirrorServer({"/5/59/59-0.txt": b"Discourse"})

        make_mirror(server).download(59)

        assert server.requests[0].headers["User-Agent"] == USER_AGENT
        assert "aporia-rag" in USER_AGENT


class TestPoliteness:
    def test_waits_between_fallback_probes(self, make_mirror: MirrorFactory) -> None:  # 4.7
        sleeps: list[float] = []
        server = FakeMirrorServer({"/3/4/9/0/34901/34901.txt": b"On Liberty"})

        make_mirror(server, delay_seconds=2.0, sleep=sleeps.append).download(34901)

        assert sleeps == [2.0, 2.0]  # 3 requests: no wait before the first

    def test_waits_between_downloads(self, make_mirror: MirrorFactory) -> None:  # 4.7
        sleeps: list[float] = []
        server = FakeMirrorServer({"/5/59/59-0.txt": b"a", "/1/4/9/1497/1497-0.txt": b"b"})
        mirror = make_mirror(server, delay_seconds=2.0, sleep=sleeps.append)

        mirror.download(59)
        mirror.download(1497)

        assert sleeps == [2.0]


class TestCache:
    def test_hit_makes_no_request(self, tmp_path: Path) -> None:  # 4.9
        (tmp_path / "pg1497.txt").write_text("cached Republic", encoding="utf-8")
        source = FakeSource({})

        result = fetch_cached(1497, tmp_path, source)

        assert result == (tmp_path / "pg1497.txt", True)  # path, from_cache
        assert source.calls == []

    def test_miss_downloads_and_stores_utf8(self, tmp_path: Path) -> None:  # 4.10
        text = "\u201cJustice\u201d, said Thrasymachus."
        source = FakeSource({1497: text})

        result = fetch_cached(1497, tmp_path, source)

        assert result == (tmp_path / "pg1497.txt", False)  # path, from_cache
        assert result.path.read_bytes() == text.encode("utf-8")
        assert source.calls == [1497]

    def test_creates_the_cache_directory(self, tmp_path: Path) -> None:  # 4.11
        cache_dir = tmp_path / "data" / "raw"

        fetch_cached(59, cache_dir, FakeSource({59: "Discourse"}))

        assert (cache_dir / "pg59.txt").is_file()

    def test_failed_download_writes_nothing(self, tmp_path: Path) -> None:  # 4.12
        with pytest.raises(EbookNotFoundError):
            fetch_cached(59, tmp_path, FakeSource({}))

        assert list(tmp_path.iterdir()) == []

    def test_successful_download_leaves_no_temporary_files(self, tmp_path: Path) -> None:  # 4.12
        fetch_cached(59, tmp_path, FakeSource({59: "Discourse"}))

        assert list(tmp_path.iterdir()) == [tmp_path / "pg59.txt"]
