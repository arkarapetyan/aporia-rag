"""Suite-wide test configuration.

Hypothesis profiles: `dev` keeps the local feedback loop fast; CI sets
HYPOTHESIS_PROFILE=ci to search harder for counterexamples.
"""

import os
from typing import Any

import pytest
from ddgs.ddgs import DDGS  # the class itself: `ddgs.DDGS` is a proxy that loads it lazily
from hypothesis import settings

settings.register_profile("dev", max_examples=100)
settings.register_profile("ci", max_examples=1000, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture(autouse=True)
def _isolate_from_local_configuration(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a developer's own APORIA_* variables out of the tests.

    Tests that talk to the real local model (`-m llm`) are left alone, so
    they use the same configuration as the application itself.
    """
    if request.node.get_closest_marker("llm") is None:
        for name in [name for name in os.environ if name.startswith("APORIA_")]:
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _no_real_web_search(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail any unit test that would search the real web.

    ddgs does not go through httpx, so the injected transport cannot stop it:
    a test that reaches the web must give the container a `ScriptedWebSearch`.
    """
    if request.node.get_closest_marker("integration") is None:

        def refuse(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            raise AssertionError("a unit test searched the real web: use a ScriptedWebSearch")

        monkeypatch.setattr(DDGS, "text", refuse)
