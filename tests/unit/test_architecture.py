"""The architecture's dependency rules, enforced (test plan: A.1-A.2).

The rules are import-linter contracts in pyproject.toml. import-linter
builds the graph of every import in the package (direct and indirect), so
"the indexer never touches Chroma" is checked on every run rather than
remembered in code review.
"""

from pathlib import Path

import pytest
from importlinter.cli import lint_imports

PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


@pytest.mark.parametrize(
    "contract",
    [
        "layers",  # A.1
        "provider-libraries",  # A.2
    ],
)
def test_import_contract_holds(contract: str) -> None:
    exit_code = lint_imports(
        config_filename=str(PYPROJECT),
        limit_to_contracts=(contract,),
        no_cache=True,
        no_logo=True,
    )

    assert exit_code == 0, "see the import-linter report above"
