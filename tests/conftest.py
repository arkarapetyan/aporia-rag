"""Suite-wide test configuration.

Hypothesis profiles: `dev` keeps the local feedback loop fast; CI sets
HYPOTHESIS_PROFILE=ci to search harder for counterexamples.
"""

import os

import pytest
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
