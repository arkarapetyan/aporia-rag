"""Suite-wide test configuration.

Hypothesis profiles: `dev` keeps the local feedback loop fast; CI sets
HYPOTHESIS_PROFILE=ci to search harder for counterexamples.
"""

import os

from hypothesis import settings

settings.register_profile("dev", max_examples=100)
settings.register_profile("ci", max_examples=1000, deadline=None)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
