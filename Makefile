.DEFAULT_GOAL := help
.PHONY: help install hooks lint format typecheck test cov test-integration test-llm check

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Create .venv and install runtime + dev dependencies from the lockfile
	uv sync --locked

hooks: ## Install git pre-commit hooks (requires a git repository)
	uv run pre-commit install

lint: ## Lint and verify formatting
	uv run ruff check .
	uv run ruff format --check .

format: ## Auto-fix lint findings and format the code
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Static type checking (mypy --strict)
	uv run mypy

test: ## Fast suite: unit + property-based tests (no network, no LLM)
	uv run pytest

cov: ## Fast suite with branch coverage (fails under 90%)
	uv run pytest --cov --cov-report=term-missing

test-integration: ## Tests that need real local infrastructure
	uv run pytest -m integration

test-llm: ## Tests that call the real local Ollama model (slow, run by hand)
	uv run pytest -m llm

check: lint typecheck cov ## Everything CI runs
