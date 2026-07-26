"""Shared pytest fixtures.

These back the behaviour tests across the suite: a reproducible RNG, a small synthetic market
panel, closed-form-checkable return samples (normal, Student-t), an equal-Sharpe / equal-CVaR pair
for null-calibration tests, and an isolated results directory.
"""
from __future__ import annotations

import gc
import sys

# H1: import pyarrow BEFORE torch (order-independent SIGSEGV fix). pytest-randomly shuffles test /
# module order, so a test that imports torch before any pyarrow-backed gold load can trigger the
# torch/pyarrow first-loader-wins ABI segfault (src/utils/preload). Preloading here — at the very top
# of conftest, which pytest imports before any test module — pins pyarrow first regardless of order.
from src.utils.preload import preload

preload()

# E402 is intentional here: pyarrow MUST be imported (via preload above) BEFORE any module that pulls
# in torch, so these imports deliberately follow the preload call.
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.data.panel import Panel  # noqa: E402
from src.data.synthetic import make_synthetic_panel  # noqa: E402

SEED = 12345


@pytest.fixture(autouse=True)
def _release_cuda_between_tests():
    """Return retained GPU memory after any test that touched CUDA (2026-07-26 deep review).

    SAME ORDER-DEPENDENCE CLASS as the pyarrow-before-torch fix at the top of this file. A test that
    trains IN-PROCESS (``tests/test_reproduce_synthetic.py`` runs a real SB3-SAC fit) leaves PyTorch's
    caching allocator holding the GPU. A later test that SPAWNS cuda workers then OOMs —
    ``tests/test_cluster_pack_integration.py`` deliberately packs TWO 'cuda' tokens onto the one
    physical GPU (that IS what it verifies: Myriad-style device-token routing), and on the 6 GB laptop
    GPU the parent's retained cache leaves the children nothing:
    ``RuntimeError: CUDA error: out of memory``.

    Because ``pytest-randomly`` shuffles module order, this made the WHOLE SUITE green or red
    depending on the run's random seed. Reproduced deterministically with
    ``pytest -p no:randomly tests/test_reproduce_synthetic.py tests/test_cluster_pack_integration.py``.

    Cost is ~zero for the ~2,000 non-GPU tests: ``is_initialized()`` is false unless a CUDA context
    actually exists. ``gc.collect()`` runs first so model objects still referencing CUDA tensors are
    dropped before the cache is handed back to the driver.
    """
    yield
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture
def seed() -> int:
    return SEED


@pytest.fixture
def rng() -> np.random.Generator:
    """A fresh, isolated NumPy Generator."""
    return np.random.default_rng(SEED)


@pytest.fixture
def synthetic_panel() -> Panel:
    """A small synthetic panel (fast) with heavy tails and volatility clustering."""
    return make_synthetic_panel(n_assets=8, n_days=600, seed=SEED)


@pytest.fixture
def normal_returns(rng: np.random.Generator) -> np.ndarray:
    """A large standard-normal sample for closed-form CVaR/quantile checks."""
    return rng.standard_normal(200_000)


@pytest.fixture
def heavy_tail_returns(rng: np.random.Generator) -> np.ndarray:
    """A heavy-tailed (Student-t, df=3) daily-return-scaled sample."""
    return 0.01 * rng.standard_t(3, size=50_000)


@pytest.fixture
def equal_sharpe_pair(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two iid return series drawn from the SAME distribution (true Sharpe difference = 0).

    Used to check that difference-test p-values are ~uniform under the null.
    """
    a = 0.0005 + 0.01 * rng.standard_normal(2_000)
    b = 0.0005 + 0.01 * rng.standard_normal(2_000)
    return a, b


@pytest.fixture
def results_dir(tmp_path):
    """An isolated results directory for IO round-trip tests."""
    d = tmp_path / "runs"
    d.mkdir()
    return d


# ----------------------------------------------------------------------------- #
# Hypothesis profile — DETERMINISTIC property-based testing                       #
# ----------------------------------------------------------------------------- #
# The property/metamorphic suite (tests/test_properties.py) must be REPRODUCIBLE:
# the whole project's credibility rests on determinism, so Hypothesis is
# derandomised (a fixed, example-derived seed) rather than allowed to explore
# randomly each run — a property failure is then a stable, replayable counter-
# example, not a flaky one. This composes cleanly with pytest-randomly (which
# reseeds the stdlib/NumPy RNGs per test). Guarded so the suite still collects if
# Hypothesis is absent (the property module itself importorskips it).
try:  # pragma: no cover - trivial import guard
    from hypothesis import HealthCheck, settings

    settings.register_profile(
        "deterministic",
        derandomize=True,
        max_examples=200,
        deadline=None,  # numpy property checks vary in cost; no per-example wall-clock deadline
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
        print_blob=True,
    )
    settings.load_profile("deterministic")
except ImportError:  # Hypothesis not installed (minimal env) — property tests skip.
    pass
