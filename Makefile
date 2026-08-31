# agentic-reward-engineering — developer entry points.
# The deterministic core (inference, measurement, sandbox, baselines, env) runs on a light
# scientific stack; the agent-training / LLM paths additionally need torch + SB3 + an API key.

PY ?= python
VENV ?= .venv
VPY := $(VENV)/bin/python
# A venv puts its interpreter under Scripts/ on Windows and bin/ everywhere else.
ifeq ($(OS),Windows_NT)
VPY := $(VENV)/Scripts/python.exe
endif

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev test test-fast test-pipeline lint format typecheck audit mutation \
        smoke freeze freeze-check reproduce power campaign analyze figures lock clean prereg-bundle

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	 awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv:  ## Create a venv that inherits system scientific libs + adds gymnasium
	$(PY) -m venv $(VENV) --system-site-packages
	$(VPY) -m pip install -U pip
	$(VPY) -m pip install gymnasium pytest pytest-cov ruff mypy

install:  ## Install the package (full deps incl. torch/SB3 — for training)
	$(VPY) -m pip install -e .

install-dev:  ## Install with dev extras
	$(VPY) -m pip install -e ".[dev]"

test:  ## Run the full test suite (engine + data_pipeline)
	$(VPY) -m pytest
	$(MAKE) test-pipeline

test-fast:  ## Run tests, skipping agent-training ('slow') tests
	$(VPY) -m pytest -m "not slow"

test-pipeline:  ## Run the data_pipeline tests (isolated: 'src' = data_pipeline's package, not the engine's)
	PYTHONPATH=data_pipeline $(VPY) -m pytest data_pipeline/tests --noconftest -o addopts=""

lint:  ## Ruff lint
	$(VPY) -m ruff check src tests scripts

format:  ## Ruff format
	$(VPY) -m ruff format src tests scripts

typecheck:  ## mypy
	$(VPY) -m mypy src

audit:  ## Supply-chain CVE scan of the pinned dependencies (pip-audit; install: pip install pip-audit)
	$(VPY) -m pip_audit || echo "pip-audit not installed: $(VPY) -m pip install pip-audit"

mutation:  ## Mutation-testing exhibit on the core numeric modules (deterministic)
	$(VPY) scripts/mutation_probe.py --module src/backtest/metrics.py
	$(VPY) scripts/mutation_probe.py --module src/feedback/measurement.py

smoke:  ## Phase 0 gate — SB3 SAC + TQC on the RTX 4090, online, timed
	$(VPY) scripts/smoke_test.py

power:  ## Phase 1.C — power analysis
	$(VPY) scripts/power_analysis.py

freeze:  ## Phase 1.E — hash + freeze the pre-registration (WRITE: user-only, run once)
	$(VPY) scripts/freeze.py

freeze-check:  ## CI drift guard — verify prose<->yaml + Phase-0 + hash (no writes)
	$(VPY) scripts/freeze.py --check

reproduce:  ## Reproducibility — keyless SYNTHETIC golden reproduction (no API key, no licensed data)
	$(VPY) scripts/reproduce_synthetic.py --check

campaign:  ## Phase 3 — run the campaign (config/campaign.yaml)
	$(VPY) scripts/run_campaign.py

prereg-bundle:  ## Package the frozen pre-registration for external (OSF) deposit
	$(VPY) scripts/make_prereg_bundle.py

analyze:  ## Phase 4 — hypothesis tests + tables
	$(VPY) scripts/analyze_results.py

figures:  ## Render the headline figure suite (report-only)
	$(VPY) scripts/make_figures.py

# Rank 14 — hash-pinned lockfile for reproducibility. GATED: the lockfile MUST be generated on the
# Linux GPU box, because the pinned CUDA wheels (torch 2.6.0+cu124 etc., ADR-030/032) only resolve
# against the cu124 index on Linux — a lock produced on this Windows/dev machine would pin the wrong
# (CPU / win) wheels. Prefer `uv pip compile ... --generate-hashes` (true hash-pinning); fall back to
# `pip freeze` if uv is absent. DO NOT commit a lockfile produced off the GPU box.
LOCKFILE ?= requirements.lock
lock:  ## [GATED: Linux GPU box] Generate the hash-pinned $(LOCKFILE) (uv compile; pip-freeze fallback)
	@echo "[lock] GATED — run this ONLY on the Linux RTX-4090 box (cu124 wheels); see ADR-030/032."
	@if command -v uv >/dev/null 2>&1; then \
		echo "[lock] uv pip compile pyproject.toml --all-extras --generate-hashes -> $(LOCKFILE)"; \
		uv pip compile pyproject.toml --all-extras --generate-hashes -o $(LOCKFILE); \
	else \
		echo "[lock] uv not found — FALLBACK: $(VPY) -m pip freeze (NOT hash-pinned) -> $(LOCKFILE)"; \
		$(VPY) -m pip freeze > $(LOCKFILE); \
	fi
	@echo "[lock] wrote $(LOCKFILE). Verify the torch build is the cu124 GPU wheel before committing."

clean:  ## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
