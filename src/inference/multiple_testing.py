"""Multiple-testing corrections across the arm x metric family (FINAL_PLAN F.11, audit B-8).

Purpose
-------
Control the family-wise error rate / false-discovery rate over the family of
hypotheses formed by the cross-product of arms and held-out metrics.

Methods
-------
- ``benjamini_hochberg`` -- the Benjamini-Hochberg (1995) linear step-up
  procedure controlling the false-discovery rate at level ``q``.
- ``romano_wolf`` -- the Romano & Wolf (2005) stepdown procedure for strong
  control of the family-wise error rate, using a bootstrap null distribution of
  the test statistics (via the max statistic) to account for dependence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


import numpy as np

#: NOTE: 2026-07-27 (deep review, finding #90): this list omitted the module's most important members.
#: `graphical_alpha_propagation` + `registered_alpha_graph` implement the R108-RATIFIED PRIMARY
#: confirmatory decision rule (`primary_rule: bonferroni_weighted_graph`), which SUPERSEDED R31 — yet the
#: declared public API still advertised only the two multiplicity helpers that the ratified rule replaced
#: as primary. Runtime was unaffected (`validity_tier.py:43` imports both explicitly, bypassing `__all__`),
#: but a star-import, an API-doc generator, or a re-export audit would all have concluded the primary rule
#: is not public. Same stale-index class as #88.
__all__ = [
    "benjamini_hochberg",
    "romano_wolf",
    "graphical_alpha_propagation",
    "registered_alpha_graph",
]


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR control at level ``q``.

    Parameters
    ----------
    pvals:
        One p-value per hypothesis.
    q:
        Target false-discovery rate. The default MIRRORS the registered value
        (``config/inference.yaml: multiplicity.q = 0.05``) — it was ``0.1`` until
        2026-07-26, i.e. TWICE the pre-registered FDR, so any caller that omitted ``q``
        would silently have run a more permissive correction than the design registers.
        The change is behaviour-free today (every call site in ``src/``, ``scripts/`` and
        ``tests/`` passes ``q`` explicitly, so the default was unreachable) and removes the
        trap for future callers. ``config/inference.yaml`` remains the single source of
        truth; production code must keep passing it explicitly rather than lean on this.

    Returns
    -------
    numpy.ndarray
        Boolean array of rejections (``True`` = reject the null), aligned with
        the input order.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    reject = np.zeros(m, dtype=bool)
    if m == 0:
        return reject
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    # Largest k with p_(k) <= (k/m) * q.
    thresholds = (np.arange(1, m + 1) / m) * q
    below = sorted_p <= thresholds
    if not below.any():
        return reject
    k_max = np.max(np.nonzero(below)[0])  # 0-based index of largest passing rank
    reject[order[: k_max + 1]] = True
    return reject


def romano_wolf(
    stats: np.ndarray,
    boot_stats: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Romano-Wolf stepdown FWER control via a bootstrap max-statistic null.

    Parameters
    ----------
    stats:
        Observed test statistics, one per hypothesis (larger = more evidence
        against the null; e.g. ``|t|`` or a one-sided statistic).
    boot_stats:
        Bootstrap draws of the (centred, null) statistics, shape
        ``(n_boot, n_hypotheses)``.
    alpha:
        Target family-wise error rate.

    Returns
    -------
    numpy.ndarray
        Boolean array of rejections aligned with ``stats``.

    Notes
    -----
    Stepdown: order hypotheses by observed statistic (descending). At each step,
    over the *remaining* (not-yet-rejected) hypotheses, compute the bootstrap
    distribution of the maximum statistic and its ``1 - alpha`` quantile. Reject
    the leading hypothesis if its statistic exceeds this critical value; repeat
    on the remainder. Stop at the first non-rejection.
    """
    s = np.asarray(stats, dtype=float)
    boot = np.asarray(boot_stats, dtype=float)
    m = s.size
    reject = np.zeros(m, dtype=bool)
    if m == 0:
        return reject
    if boot.ndim != 2 or boot.shape[1] != m:
        raise ValueError("boot_stats must have shape (n_boot, n_hypotheses)")

    order = np.argsort(s, kind="mergesort")[::-1]  # descending observed stats
    remaining = list(order)

    while remaining:
        rem_idx = np.array(remaining)
        # Bootstrap max over the remaining hypotheses.
        max_boot = boot[:, rem_idx].max(axis=1)
        crit = np.quantile(max_boot, 1.0 - alpha)
        lead = remaining[0]
        if s[lead] > crit:
            reject[lead] = True
            remaining.pop(0)
        else:
            break
    return reject


# --------------------------------------------------------------------------- #
# The RATIFIED primary rule: graphical alpha-propagation (Bretz et al. 2009)   #
# --------------------------------------------------------------------------- #
def graphical_alpha_propagation(
    pvalues: dict[str, float | None],
    weights: dict[str, float],
    edges: dict[str, dict[str, float]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    r"""Sequentially-rejective graphical multiple test (Bretz, Maurer, Brannath & Posch 2009).

    THE REGISTERED PRIMARY DECISION RULE. ``config/preregistration.yaml:
    inference.validity_tier`` declares ``method: graphical_bretz_maurer_brannath_posch_2009`` and
    ``primary_rule: bonferroni_weighted_graph``, ratified 2026-07-26 (R108) — which SUPERSEDED R31's
    separate-estimands stance. Until this function existed the pipeline could not execute the rule it
    is registered to run (deep review 2026-07-26, write-time registry row 36): the analysis computed
    only the Bonferroni-over-4 mirror, i.e. the *superseded* stance.

    Because a graphical procedure is a shortcut for a CLOSED test (Marcus, Peritz & Gabriel 1976), the
    whole family enjoys **strong FWER control at ``alpha`` under arbitrary dependence** — which is why
    the H2 co-primaries may be tested at ``w_i·alpha`` without any further leg correction inside each
    IUT (Berger 1982 already makes each node's max-p a valid level-alpha p-value).

    Algorithm (eq. (2)-(3) of Bretz et al. 2009). With index set ``J`` of not-yet-rejected nodes:

      1. reject any ``j in J`` with ``p_j <= w_j * alpha``; if none, stop;
      2. propagate the rejected node's weight   ``w_l += w_j * g_jl``            for ``l in J\{j}``;
      3. re-wire                                ``g_lk = (g_lk + g_lj*g_jk) / (1 - g_lj*g_jl)``
         (``0`` when ``g_lj*g_jl >= 1``; ``g_ll = 0``), drop ``j`` from ``J``, and repeat.

    **The rejected SET is invariant to the order** in which simultaneously-rejectable nodes are taken —
    a property of the closed-test shortcut, exercised directly by
    ``tests/test_graphical_alpha.py::test_rejected_set_is_order_invariant``. This implementation
    nevertheless picks a DETERMINISTIC order (smallest p first, ties broken by the caller's node order)
    so a run is byte-reproducible.

    Parameters
    ----------
    pvalues:
        Node p-value per node id. ``None``/NaN means the node was NOT TESTABLE (e.g. an arm with too
        few shared seeds). Such a node can never reject and is reported under ``untestable`` — it is
        NOT silently treated as a passing test.
    weights, edges:
        The graph. **Callers must READ these from the registered config, never hardcode them** — the
        graph is declared FROZEN by ``validity_tier.forking_path_guard``, and a hardcoded copy is
        exactly the executed-vs-registered drift the arm-roster / ``h1_baselines`` /
        ``confirmatory_author`` freeze guards exist to prevent. Use :func:`registered_alpha_graph`.
    alpha:
        Family-wise error rate. The registered value is 0.05.

    Returns
    -------
    dict
        ``{"alpha", "rejected" (in rejection order), "not_rejected", "untestable",
        "local_alpha" (the level each node was finally tested at), "steps" (the audit trail),
        "any_rejected"}``.
    """
    nodes = list(weights)
    if set(nodes) != set(edges) or not set(pvalues) >= set(nodes):
        raise ValueError(
            f"graph/p-value node sets disagree: weights={sorted(weights)}, edges={sorted(edges)}, "
            f"pvalues={sorted(pvalues)}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    w = {n: float(weights[n]) for n in nodes}
    if any(v < -1e-12 for v in w.values()):
        raise ValueError(f"negative initial weight in {w}")
    if sum(w.values()) > 1.0 + 1e-9:
        raise ValueError(f"initial weights must sum to <= 1; got {sum(w.values())}")
    g = {a: {b: float(edges.get(a, {}).get(b, 0.0)) for b in nodes} for a in nodes}
    for a in nodes:
        g[a][a] = 0.0
        if sum(g[a].values()) > 1.0 + 1e-9:
            raise ValueError(f"out-edges of {a!r} must sum to <= 1; got {sum(g[a].values())}")

    def _p(n: str) -> float:
        v = pvalues.get(n)
        if v is None:
            return float("nan")
        return float(v)

    untestable = [n for n in nodes if not np.isfinite(_p(n))]
    live = [n for n in nodes if n not in untestable]
    rejected: list[str] = []
    steps: list[dict[str, Any]] = []
    local_alpha = {n: w[n] * alpha for n in nodes}

    while True:
        # Deterministic choice among ALL simultaneously-rejectable nodes (the final set is invariant).
        candidates = [n for n in live if _p(n) <= w[n] * alpha + 1e-15]
        if not candidates:
            break
        j = min(candidates, key=lambda n: (_p(n), nodes.index(n)))
        steps.append({
            "rejected": j, "p": _p(j), "local_alpha": w[j] * alpha,
            "weights_before": {n: w[n] for n in live},
        })
        rejected.append(j)
        live.remove(j)
        # (2) propagate the rejected node's weight along its out-edges
        for leaf in live:
            w[leaf] += w[j] * g[j][leaf]
        # (3) re-wire the surviving graph
        new_g = {a: {b: 0.0 for b in live} for a in live}
        for a in live:
            for b in live:
                if a == b:
                    continue
                denom = 1.0 - g[a][j] * g[j][a]
                new_g[a][b] = (g[a][b] + g[a][j] * g[j][b]) / denom if denom > 1e-15 else 0.0
        for a in live:
            for b in live:
                g[a][b] = new_g[a][b]
        w[j] = 0.0
        for n in live:
            local_alpha[n] = w[n] * alpha

    # 2026-08-01 (record §100.33) — A NODE CAN BE UNTESTABLE FOR TWO REASONS, AND ONLY ONE WAS
    # REPORTED. ``untestable`` above catches a node with NO P-VALUE (too few shared seeds). It does NOT
    # catch a node whose FINAL LOCAL ALPHA IS ZERO — which, under the registered graph, is the case
    # that actually arises: ``initial_weights`` starts N3/N4/N5/N6 at 0.0, so if no upstream node
    # rejects, those four are tested at w*alpha = EXACTLY 0.0 and cannot reject at ANY attainable
    # p-value. They were then reported under ``not_rejected`` — indistinguishable from a hypothesis
    # that WAS tested and failed.
    #
    # WHY THAT WAS THE MOST DANGEROUS SENTENCE THIS FUNCTION COULD EMIT. Executed on the design's OWN
    # PRE-REGISTERED PREDICTION (H2 null on both co-primaries), the verdict read
    # ``not_rejected: [... N6_h1 ...]`` for an H1 p-value of 0.0001 — a result that rejects
    # comfortably under the very rule R105/R108 replaced. Writing "H1 was not rejected" into a chapter
    # graded on faultless presentation of data, when H1 was never TESTABLE, misstates the result in
    # the direction of a null we predicted. The docstring above already promised the right behaviour
    # for the other case: such a node "can never reject and is reported under untestable — it is NOT
    # silently treated as a passing test." This makes the code honour that promise for BOTH reasons.
    #
    # NOTE: THIS CHANGES NO INFERENCE. ``rejected`` is computed exactly as before and is untouched; only
    # the CATEGORISATION of the non-rejections changes. The two reasons are kept in SEPARATE keys
    # because they are different facts about the experiment: ``untestable`` = we had no data;
    # ``structurally_untestable`` = we had data but the graph allotted the node no alpha.
    #
    # A node that finished at zero alpha and STILL rejected (possible only for p <= 1e-15, the
    # numerical tolerance in the rejection test) is deliberately excluded — it rejected, so it was
    # testable in the only sense that matters.
    structurally_untestable = [n for n in nodes
                               if n not in rejected and n not in untestable
                               and local_alpha[n] <= 0.0]
    out: dict[str, Any] = {
        "alpha": float(alpha),
        "rejected": rejected,
        "not_rejected": [n for n in nodes if n not in rejected and n not in untestable
                         and n not in structurally_untestable],
        "untestable": untestable,
        "structurally_untestable": structurally_untestable,
        "local_alpha": local_alpha,
        "steps": steps,
        "any_rejected": bool(rejected),
    }
    if structurally_untestable:
        # Carried IN the artifact so it cannot be lost between here and the chapter.
        out["structurally_untestable_note"] = (
            f"{len(structurally_untestable)} node(s) were allotted local alpha EXACTLY 0.0 by the "
            f"graph and could not have rejected at any attainable p-value: "
            f"{', '.join(structurally_untestable)}. They are NOT 'not rejected' — they were never "
            f"tested. Report them as untestable-by-construction, never as failed tests."
        )
    return out


def registered_alpha_graph(root: Any = None) -> tuple[dict[str, float], dict[str, dict[str, float]], float]:
    """Read the FROZEN alpha-graph from ``config/preregistration.yaml`` — never hardcode it.

    Returns ``(initial_weights, edges, alpha)`` exactly as registered under
    ``inference.validity_tier``. Reading rather than copying is the point: the graph is declared FROZEN
    by ``forking_path_guard``, so a hardcoded duplicate would be executed-vs-registered drift of the
    kind the freeze gate's cross-file guards exist to catch.
    """
    import yaml

    if root is None:
        root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((Path(root) / "config" / "preregistration.yaml").read_text(encoding="utf-8"))
    tier = cfg["inference"]["validity_tier"]
    return (
        {str(k): float(v) for k, v in tier["initial_weights"].items()},
        {str(a): {str(b): float(v) for b, v in (d or {}).items()} for a, d in tier["edges"].items()},
        float(tier["alpha"]),
    )
