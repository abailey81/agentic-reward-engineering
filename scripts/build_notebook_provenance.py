"""Builder for ``notebooks/data_provenance_walkthrough.ipynb`` — the trust-but-verify showcase.

The licensed Refinitiv data cannot ship; what ships is the MACHINERY OF PROOF. This notebook lets
an examiner *run* the integrity claims and watch them verify themselves live: every code cell
recomputes a claim from the frozen artifacts and **asserts** it (fail-loud — the notebook IS a
verification artifact; a red cell means an integrity claim no longer holds). Regenerate with::

    python scripts/build_notebook_provenance.py

Section map:
  0  setup                 repo-root resolution + the fail-loud ``verify`` helper
  1  panel identity        gold_suffix() resolved from the hash-bound config (assert univ5)
  2  live checksums        SHA-256 of every gold univ5 artifact recomputed vs the frozen manifest,
                           then the loader's own gate + full data-contract validation
  3  rebuild byte-diff     panel_overlap_diff(univ5, univ3): 0 changed cells, +123 rows, +10 names
  4  membership splice     both PIT artifacts diffed month-by-month; the EVHC.N^L16 vendor-drift
                           story re-verified from the recorded overlap diagnostic + PIT sanity
  5  delisting integrity   the Shumway audit log (333 observed terminals, zero surcharges), the
                           double-counting finding, and the univ5s == univ5 byte-identity
  6  window integrity      resolve_windows on the real date axis == expected_windows.univ5, then
                           the production drift guard itself
  7  verdict               the full check ledger, asserted green

CPU-only, read-only, no network: cells read frozen parquet/JSON/YAML and hash local files.
Deterministic: double-build byte-identity is asserted at the end of ``main``.
"""

from __future__ import annotations

from pathlib import Path

from notebook_builder import build_notebook, code, md, sha256_bytes, write_notebook

_REPO = Path(__file__).resolve().parents[1]
_OUT = _REPO / "notebooks" / "data_provenance_walkthrough.ipynb"


def cells() -> list[dict]:
    """The notebook, cell by cell."""
    out: list[dict] = []

    # ------------------------------------------------------------------ title
    out.append(md(
        "# Data Provenance Walkthrough — *trust, but verify*\n"
        "\n"
        "**Every integrity claim in the Data chapter, re-verified live in front of you.**\n"
        "\n"
        "The gold panel is licensed Refinitiv/LSEG data and cannot be redistributed; what the "
        "repository ships instead is the **machinery of proof**: a write-once vault with a "
        "SHA-256 manifest, byte-diff validators, splice diagnostics, delisting audit logs, and "
        "window-drift guards. This notebook does not *describe* that machinery — it **runs** it. "
        "Each code cell recomputes one claim from the frozen artifacts and `assert`s it, so the "
        "notebook doubles as a verification artifact: **if any cell goes red, an integrity claim "
        "has stopped being true on this checkout.**\n"
        "\n"
        "Run top-to-bottom (Kernel → Restart & Run All). Read-only and CPU-only: nothing here "
        "trains, pulls, or writes — cells read frozen parquet/JSON/YAML and hash local files.\n"
        "\n"
        "| # | claim verified live |\n"
        "|---|---|\n"
        "| 1 | the active panel identity (`univ5`) resolves from the **hash-bound** config |\n"
        "| 2 | every gold `univ5` artifact's SHA-256 **recomputed** == the frozen manifest |\n"
        "| 3 | the 2026 rebuild changed **zero** shared historical cells vs frozen `univ3` |\n"
        "| 4 | the membership splice: one allowlisted vendor revision (`EVHC.N^L16`), zero unexplained |\n"
        "| 5 | delisting: 333 observed terminals kept, **zero** surcharges → `univ5s` ≡ `univ5` |\n"
        "| 6 | the Split-C windows resolve to the frozen `expected_windows.univ5` exactly |\n"
        "\n"
        "*Companion:* `results_walkthrough.ipynb` consumes what this notebook certifies — it "
        "loads the panel these hashes pin and quotes this walkthrough wherever an integrity "
        "fact is load-bearing."
    ))

    # ------------------------------------------------------------------ 0 setup
    out.append(md("## 0 · Setup — and the fail-loud contract"))
    out.append(code(
        "import json, sys, platform\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "\n"
        "# Resolve the repo root whether the notebook is opened from notebooks/ or the repo root.\n"
        "ROOT = Path.cwd()\n"
        "if not (ROOT / 'src').exists() and (ROOT.parent / 'src').exists():\n"
        "    ROOT = ROOT.parent\n"
        "for p in (str(ROOT), str(ROOT / 'scripts')):\n"
        "    if p not in sys.path:\n"
        "        sys.path.insert(0, p)\n"
        "\n"
        "CHECKS = []  # (name, detail) of every assertion that passed -- the sec. 7 ledger\n"
        "\n"
        "def verify(name, condition, detail=''):\n"
        "    \"\"\"Fail-loud check: assert `condition`, then log the pass to the ledger.\"\"\"\n"
        "    assert condition, f'INTEGRITY CHECK FAILED: {name} -- {detail}'\n"
        "    CHECKS.append((name, detail))\n"
        "    print(f'[OK] {name}' + (f'  ({detail})' if detail else ''))\n"
        "\n"
        "print('repo root :', ROOT)\n"
        "print('python    :', platform.python_version(), '| numpy', np.__version__,\n"
        "      '| pandas', pd.__version__)"
    ))

    # ------------------------------------------------------------------ 1 identity
    out.append(md(
        "## 1 · Panel identity — the suffix is hash-bound, not a habit\n"
        "\n"
        "Which panel is \"the\" panel? `src.data.loaders.gold_suffix()` resolves it with an "
        "explicit precedence: the `LLM_RP_GOLD_SUFFIX` env var (an *explicit sensitivity-band "
        "override only*), else **`config/data.yaml: gold.suffix`** — a file bound into the "
        "design freeze hash by `scripts/freeze.py` — else the last-resort default. Because the "
        "config is hash-bound, the headline panel **cannot be swapped silently**: changing it "
        "changes the freeze hash. Here we assert the resolution end-to-end and that no override "
        "is active while verifying."
    ))
    out.append(code(
        "import os, yaml\n"
        "from src.data import loaders\n"
        "\n"
        "override = os.environ.get('LLM_RP_GOLD_SUFFIX', '')\n"
        "verify('no suffix override active', override.strip() == '',\n"
        "       'LLM_RP_GOLD_SUFFIX is unset -- the hash-bound config governs')\n"
        "cfg = yaml.safe_load((ROOT / 'config' / 'data.yaml').read_text(encoding='utf-8'))\n"
        "cfg_suffix = cfg['gold']['suffix']\n"
        "live_suffix = loaders.gold_suffix()\n"
        "verify('config/data.yaml gold.suffix == univ5', cfg_suffix == 'univ5',\n"
        "       f'config says {cfg_suffix!r}')\n"
        "verify('gold_suffix() resolves the config value', live_suffix == cfg_suffix,\n"
        "       f'loader resolves {live_suffix!r}')\n"
        "print(f'\\nactive headline panel: returns_panel_{live_suffix}.parquet '\n"
        "      '(Split C, ADR-044/051)')"
    ))

    # ------------------------------------------------------------------ 2 checksums
    out.append(md(
        "## 2 · Live checksums — recompute, never trust\n"
        "\n"
        "The vault freeze (`config/data.yaml: freeze.checksum: sha256`) recorded every artifact's "
        "SHA-256 in `data/manifest/manifest.jsonl` at build time. Below, each gold `univ5` "
        "artifact is **re-hashed from its bytes in this cell** and compared to the frozen value — "
        "using the *same two functions the production loader runs* "
        "(`loaders._file_sha256` / `loaders._expected_sha256`), so this exhibit and the loader "
        "gate cannot drift apart. A mismatch would mean the frozen file changed on disk: tamper, "
        "corruption, or a wrong build.\n"
        "\n"
        "Then the loader's own gate runs end-to-end: `load_gold_panel(verify_checksum=True, "
        "validate=True)` re-verifies the checksums *and* asserts the full data contract "
        "(`src.data.validation.validate_panel`: strictly increasing dates — the look-ahead "
        "invariant; returns ≥ −100% and plausible; non-negative finite VIX; unique integer "
        "anonymised ids; no all-zero column).\n"
        "\n"
        "> *Disclosed cosmetic drift:* the 2026 rebuild wrote the univ5 manifest `relpath`s with "
        "Windows separators (`data\\gold\\...`) where the manifest convention is POSIX. The "
        "loader's documented basename fallback resolves them, and the hashes below prove it — "
        "flagged here rather than hidden."
    ))
    out.append(code(
        "gold = ROOT / 'data' / 'gold'\n"
        "for name in ('returns_panel_univ5.parquet', 'cash_features_univ5.parquet',\n"
        "             'top30_selection_univ5.parquet', 'splits_univ5.parquet'):\n"
        "    path = gold / name\n"
        "    frozen = loaders._expected_sha256(path)\n"
        "    verify(f'{name}: manifest entry exists', frozen is not None,\n"
        "           'frozen SHA-256 found in data/manifest/manifest.jsonl')\n"
        "    live = loaders._file_sha256(path)\n"
        "    verify(f'{name}: recomputed sha256 == frozen', live == frozen,\n"
        "           f'{live[:16]}... == {frozen[:16]}...')"
    ))
    out.append(code(
        "# The vault at a glance — shape, span, size, and the (just re-verified) hash prefix of\n"
        "# every gold univ5 artifact, read live from disk. A reviewer can quote this table.\n"
        "rows = []\n"
        "for name in ('returns_panel_univ5.parquet', 'cash_features_univ5.parquet',\n"
        "             'top30_selection_univ5.parquet', 'splits_univ5.parquet'):\n"
        "    path = gold / name\n"
        "    df = pd.read_parquet(path)\n"
        "    span = (f'{df.index[0].date()} -> {df.index[-1].date()}'\n"
        "            if isinstance(df.index, pd.DatetimeIndex) else '-')\n"
        "    rows.append({'artifact': name, 'rows x cols': f'{df.shape[0]:,} x {df.shape[1]}',\n"
        "                 'span': span, 'MiB': round(path.stat().st_size / 2**20, 1),\n"
        "                 'sha256 (frozen==live)': loaders._file_sha256(path)[:16] + '…'})\n"
        "vault = pd.DataFrame(rows).set_index('artifact')\n"
        "vault"
    ))
    out.append(code(
        "from src.data.loaders import load_gold_panel\n"
        "\n"
        "dev = load_gold_panel('development', verify_checksum=True, validate=True)\n"
        "p = dev.panel\n"
        "verify('loader gate: checksum + full data contract',\n"
        "       p.returns.shape == (3021, 30),\n"
        "       f'train window {str(p.dates[0])[:10]} -> {str(p.dates[-1])[:10]}, '\n"
        "       f'{p.returns.shape[0]} sessions x {p.returns.shape[1]} anonymised names')\n"
        "verify('anonymisation contract: integer ids only',\n"
        "       p.asset_ids.dtype.kind == 'i' and len(set(p.asset_ids)) == 30,\n"
        "       'no RIC / ticker / date ever reaches a reward or the LLM')"
    ))

    # ------------------------------------------------------------------ 3 byte-diff
    out.append(md(
        "## 3 · The rebuild byte-diff — history is append-only\n"
        "\n"
        "The 2026 rebuild (`univ3` → `univ5`, ADR-051) extended the panel to the settled "
        "2026-06-30 cutoff. The contract: a rebuild may **append** (new sessions, newly listed "
        "names) but must not perturb **one** shared historical cell — a rebuild that silently "
        "changed 2005-2025 returns would move the headline tail. "
        "`src.data.validation.panel_overlap_diff` aligns the two panels on their (date × RIC) "
        "intersection and diffs **every** cell, NaN-aware (aligned NaN == NaN is equal; "
        "NaN-vs-number is a change). Expected and asserted: **0 changed cells** over the full "
        "5,283 × 953 overlap, **+123 sessions** (2026H1), **+10 names** (2026 index joiners)."
    ))
    out.append(code(
        "from src.data.validation import panel_overlap_diff\n"
        "\n"
        "diff = panel_overlap_diff(gold / 'returns_panel_univ5.parquet',   # candidate (active)\n"
        "                          gold / 'returns_panel_univ3.parquet')   # frozen reference\n"
        "print(diff.summary(), '\\n')\n"
        "verify('overlap == full frozen panel (5,283 x 953)',\n"
        "       (diff.n_overlap_rows, diff.n_overlap_cols) == (5283, 953))\n"
        "verify('ZERO changed cells over the shared history',\n"
        "       diff.identical_over_overlap and diff.max_abs_delta == 0.0,\n"
        "       f'{diff.n_overlap_rows * diff.n_overlap_cols:,} overlap cells compared')\n"
        "verify('nothing removed', diff.ref_only_rows == 0 and diff.ref_only_cols == [],\n"
        "       'every frozen session and name survives in univ5')\n"
        "verify('+123 new sessions (2026H1)', diff.cand_only_rows == 123)\n"
        "verify('+10 new names', len(diff.cand_only_cols) == 10,\n"
        "       ', '.join(diff.cand_only_cols))"
    ))

    # ------------------------------------------------------------------ 4 membership splice
    out.append(md(
        "## 4 · The membership splice — one vendor revision, caught, explained, excluded\n"
        "\n"
        "Point-in-time S&P 500 membership is reconstructed by **reverse event replay** from "
        "today's chain through the vendor's joiner/leaver streams. The 2026 extension re-pulled "
        "those streams — and the fresh replay **disagreed with the frozen record**. The pull-time "
        "gate (ADR-051 gate 1, `data_pipeline/scripts/extend_universe_2026.py::overlap_check`) "
        "hard-fails on any unexplained difference; what it found was investigated and allowlisted "
        "as exactly **one** RIC:\n"
        "\n"
        "> **`EVHC.N^L16`** — old Envision Healthcare Holdings. Between the frozen pull "
        "(2026-06-12) and the extension pull (2026-07-01) the vendor **backfilled the Dec-2016 "
        "leaver event**; its join counterpart stayed missing/re-keyed, so reverse replay "
        "over-extends the name's membership back to the grid start. That over-extension is "
        "**provably wrong** — the company IPO'd in Aug-2013, merged into AMSURG on 2016-12-01, "
        "and its NYSE listing was removed 2016-12-13 (the `^L16` suffix; SEC Form 25-NSE) — and "
        "**provably immaterial** here: at a ~$7B peak market cap it was never remotely a "
        "top-30 megacap, so the selection is invariant either way.\n"
        "\n"
        "The **splice rule** (ADR-051 addendum) keeps the *frozen* record authoritative for its "
        "own span; the fresh replay contributes **only** the six 2026 month-ends. Two different "
        "diffs are therefore verified below — read them as a pair:\n"
        "1. **shipped-vs-frozen** (the two parquet artifacts): must be **identical** over every "
        "common month — the vendor revision was *not allowed in*;\n"
        "2. **pull-time fresh-vs-frozen** (recorded in the artifact's provenance sidecar): 145 of "
        "254 months differed, **all** explained by `EVHC.N^L16`, **zero** unexplained."
    ))
    out.append(code(
        "staged = ROOT / 'data' / 'staged'\n"
        "frozen_pit = pd.read_parquet(staged / 'pit_membership_200411_202512.parquet')\n"
        "spliced_pit = pd.read_parquet(staged / 'pit_membership_200411_202606.parquet')\n"
        "verify('frozen PIT shape', frozen_pit.shape == (127563, 2))\n"
        "verify('spliced PIT shape', spliced_pit.shape == (130581, 2))\n"
        "\n"
        "def member_sets(df):\n"
        "    return {pd.Timestamp(m): frozenset(g['ric'].astype(str))\n"
        "            for m, g in df.groupby('month_end')}\n"
        "\n"
        "old_m, new_m = member_sets(frozen_pit), member_sets(spliced_pit)\n"
        "common = sorted(set(old_m) & set(new_m))\n"
        "differing = [m for m in common if old_m[m] != new_m[m]]\n"
        "verify('254 common month-ends', len(common) == 254)\n"
        "verify('shipped-vs-frozen: ZERO differing common months', differing == [],\n"
        "       'the splice kept the frozen record authoritative for its whole span')\n"
        "extension = sorted(set(new_m) - set(old_m))\n"
        "verify('exactly 6 new month-ends (2026H1)',\n"
        "       [str(m.date()) for m in extension] ==\n"
        "       ['2026-01-31', '2026-02-28', '2026-03-31', '2026-04-30',\n"
        "        '2026-05-31', '2026-06-30'])\n"
        "counts = spliced_pit.groupby('month_end')['ric'].nunique()\n"
        "# 495..510 is the pipeline's own sanity band (extend_universe_2026): the S&P 500\n"
        "# constituent count legitimately hovers around 500 (multi-class listings, transition\n"
        "# days), so a hard ==500 would be wrong, not strict.\n"
        "verify('member counts sane on every month', 495 <= counts.min() <= counts.max() <= 510,\n"
        "       f'min {counts.min()} / max {counts.max()} members per month-end')\n"
        "\n"
        "# The drifted RIC entered NEITHER artifact; the SUCCESSOR company (post-AMSURG-merger\n"
        "# Envision, delisted Oct-2018) is a real member and must be IDENTICAL in both records.\n"
        "rics_frozen = set(frozen_pit['ric'].astype(str))\n"
        "rics_spliced = set(spliced_pit['ric'].astype(str))\n"
        "verify('EVHC.N^L16 excluded from BOTH records',\n"
        "       'EVHC.N^L16' not in rics_frozen | rics_spliced,\n"
        "       'the provably-wrong over-extension never entered a shipped artifact')\n"
        "ev_frozen = frozen_pit[frozen_pit['ric'] == 'EVHC.N^J18']['month_end'].tolist()\n"
        "ev_spliced = spliced_pit[spliced_pit['ric'] == 'EVHC.N^J18']['month_end'].tolist()\n"
        "verify('EVHC.N^J18 (successor) identical in both', ev_frozen == ev_spliced,\n"
        "       f'{len(ev_frozen)} month-ends, 2016-12 -> 2018-09')"
    ))
    out.append(code(
        "# The recorded PULL-TIME diagnostic (fresh replay vs frozen), from the provenance\n"
        "# sidecar frozen alongside the artifact -- re-asserted, not paraphrased.\n"
        "sidecar = json.loads((staged / 'pit_membership_200411_202606.parquet.provenance.json')\n"
        "                     .read_text(encoding='utf-8'))\n"
        "ov = sidecar['provenance']['overlap_diagnostic']\n"
        "print('recorded splice method :', sidecar['provenance']['method'])\n"
        "print('recorded diagnostic    :', json.dumps(ov))\n"
        "verify('pull-time diagnostic: 145/254 months differed',\n"
        "       (ov['months_checked'], ov['months_differing']) == (254, 145),\n"
        "       'the vendor revision was VISIBLE, not glossed over')\n"
        "verify('every difference allowlisted as EVHC.N^L16',\n"
        "       ov['allowlisted'] == ['EVHC.N^L16'] and ov['unexplained'] == [],\n"
        "       'zero unexplained rics -- anything else would have hard-failed the pull')\n"
        "\n"
        "# PIT sanity, recomputed live from the shipped artifact (the sidecar records these too):\n"
        "ms = member_sets(spliced_pit)\n"
        "verify('PIT sanity: Lehman in 2008H1, out 2008Q4',\n"
        "       'LEH.N^I08' in ms[pd.Timestamp('2008-06-30')]\n"
        "       and 'LEH.N^I08' not in ms[pd.Timestamp('2008-12-31')])\n"
        "verify('PIT sanity: Tesla out 2019, in 2021',\n"
        "       'TSLA.OQ' not in ms[pd.Timestamp('2019-12-31')]\n"
        "       and 'TSLA.OQ' in ms[pd.Timestamp('2021-12-31')])"
    ))

    # ------------------------------------------------------------------ 5 delisting
    out.append(md(
        "## 5 · Delisting integrity — 333 observed terminals and a double-counting finding\n"
        "\n"
        "A survivorship-free panel keeps its dead names, so *how a death enters the returns* is a "
        "tail-integrity question. The classical fix (Shumway 1997; Shumway & Warther 1999) "
        "surcharges performance delistings −30% (NYSE/AMEX) / −55% (Nasdaq) because CRSP's daily "
        "series often *missed* the terminal loss.\n"
        "\n"
        "**The finding:** on this vendor's data that premise fails — the Refinitiv total-return "
        "series **already embeds the terminal return** for every dead name in the panel. "
        "Applying the surcharge on top would **double-count** the delisting loss; worse, with no "
        "delisting *reason* in the vault it would hit premium M&A cash-outs (Dell, Time Warner, "
        "Abiomed …) — fabricating crashes in a tail-risk study (that is exactly the `univ4` "
        "panel's flaw, which is why it was demoted from headline to sensitivity band-end).\n"
        "\n"
        "The ADR-051 rebuild therefore runs **observed-terminal recovery**: keep the vendor's "
        "realised terminal wherever it exists; surcharge only genuinely terminal-less names. The "
        "audit log below shows the result — **all 333** dead names had observed terminals "
        "(`vendor_terminal_kept`), **zero** surcharges fired, and the \"corrected\" Shumway "
        "band-end `univ5s` is therefore **byte-identical** to the headline `univ5` (asserted from "
        "the manifest *and* re-hashed from disk). The delisting-treatment band "
        "`{0, −30, −55, −100}%` remains a report-only sensitivity surface."
    ))
    out.append(code(
        "audit = pd.read_parquet(ROOT / 'data' / 'clean' / 'shumway_audit_log_univ5s.parquet')\n"
        "verify('audit log covers 333 delisted names', len(audit) == 333)\n"
        "verify('every action is vendor_terminal_kept',\n"
        "       set(audit['action']) == {'vendor_terminal_kept'},\n"
        "       'zero shumway_correction_applied, zero skips -- the surcharge set is EMPTY')\n"
        "term = audit['value'].astype(float)\n"
        "verify('terminal returns are sane simple returns',\n"
        "       bool((term >= -1.0).all() and np.isfinite(term).all()))\n"
        "print(f'recovered terminal daily returns: min {term.min():+.1%}, '\n"
        "      f'median {term.median():+.2%}, max {term.max():+.1%}')\n"
        "print(f'  <= -30% (would-be NYSE surcharge zone): {(term <= -0.30).sum()} names;'\n"
        "      f'  exactly 0.0%: {(term == 0.0).sum()} names')\n"
        "\n"
        "# univ5s (Shumway band-end) == univ5 (headline): manifest AND disk agree.\n"
        "manifest = [json.loads(ln) for ln in\n"
        "            (ROOT / 'data' / 'manifest' / 'manifest.jsonl')\n"
        "            .read_text(encoding='utf-8').splitlines() if ln.strip()]\n"
        "sha = {e['name']: e['sha256'] for e in manifest if e.get('name')}\n"
        "verify('manifest: univ5s returns sha == univ5 returns sha',\n"
        "       sha['returns_panel_univ5s.parquet'] == sha['returns_panel_univ5.parquet'],\n"
        "       sha['returns_panel_univ5.parquet'][:16] + '...')\n"
        "live5s = loaders._file_sha256(gold / 'returns_panel_univ5s.parquet')\n"
        "verify('disk: univ5s re-hashed == frozen univ5 hash',\n"
        "       live5s == sha['returns_panel_univ5.parquet'],\n"
        "       'the corrected Shumway panel IS the headline panel, byte for byte')"
    ))
    out.append(code(
        "import matplotlib.pyplot as plt\n"
        "from src.viz.style import OKABE_ITO, apply_house_style\n"
        "apply_house_style()\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7.4, 3.4))\n"
        "ax.hist(term * 100.0, bins=60, color=OKABE_ITO['blue'], alpha=0.75,\n"
        "        edgecolor='white', linewidth=0.3)\n"
        "for dl, lab in ((-30.0, 'NYSE/AMEX surcharge (-30%)'),\n"
        "                (-55.0, 'Nasdaq surcharge (-55%)')):\n"
        "    ax.axvline(dl, color=OKABE_ITO['vermillion'], lw=1.1, ls='--')\n"
        "    ax.text(dl, ax.get_ylim()[1] * 0.95, ' ' + lab, rotation=90,\n"
        "            va='top', ha='left', fontsize=6.5, color=OKABE_ITO['vermillion'])\n"
        "ax.set_xlabel('recovered terminal daily return (%)')\n"
        "ax.set_ylabel('names')\n"
        "ax.set_title('The 333 recovered vendor terminal returns -- the fixed surcharges the\\n'\n"
        "             'panel did NOT need (every terminal was already observed)',\n"
        "             loc='left', fontsize=9)\n"
        "fig"
    ))

    # ------------------------------------------------------------------ 6 windows
    out.append(md(
        "## 6 · Window integrity — the frozen calendar on the real date axis\n"
        "\n"
        "The Split-C calendar (train 2005-2016 / val 2017-2019 / sealed test 2020-2026H1) is "
        "frozen as *dates*; the engine consumes integer `[start, end)` **positions** resolved by "
        "`scripts/run_campaign.py::resolve_windows` via `searchsorted` with clamps. The failure "
        "mode this guards: a rebuilt panel whose session axis shifts could slide those integers "
        "*through the clamps* silently. So `config/inference.yaml: splits.expected_windows.univ5` "
        "records the intended tuples, and the production guard `_assert_expected_windows` "
        "hard-fails the campaign on any drift (M1). Both are exercised here on the real axis, "
        "with the R18 purge — `max(embargo 21, feature-lookback 60) = 60` sessions carved out at "
        "each boundary so no 60-day observation window reaches across a split."
    ))
    out.append(code(
        "from types import SimpleNamespace\n"
        "import run_campaign as RC\n"
        "\n"
        "returns5 = pd.read_parquet(gold / 'returns_panel_univ5.parquet')\n"
        "shim = SimpleNamespace(dates=returns5.index.to_numpy(), T=len(returns5.index))\n"
        "inf_cfg = yaml.safe_load((ROOT / 'config' / 'inference.yaml').read_text(encoding='utf-8'))\n"
        "env_cfg = yaml.safe_load((ROOT / 'config' / 'environment.yaml')"
        ".read_text(encoding='utf-8'))\n"
        "lookback = int(env_cfg['state']['lookback_days'])\n"
        "embargo = int(inf_cfg['splits']['embargo_trading_days'])\n"
        "verify('production purge inputs', (lookback, embargo) == (60, 21),\n"
        "       'effective inter-split purge = max(21, 60) = 60 sessions (R18)')\n"
        "\n"
        "tr, va, te = RC.resolve_windows(shim, lookback, inf_cfg['splits'], embargo=embargo)\n"
        "want = inf_cfg['splits']['expected_windows']['univ5']\n"
        "verify('resolved windows == frozen expected_windows.univ5',\n"
        "       [list(tr), list(va), list(te)] == [want['train'], want['val'], want['test']],\n"
        "       f'train={list(tr)} val={list(va)} test={list(te)}')\n"
        "RC._assert_expected_windows('univ5', tr, va, te, inf_cfg['splits'])  # the M1 guard itself\n"
        "verify('production drift guard passes', True,\n"
        "       'run_campaign._assert_expected_windows raised nothing')\n"
        "\n"
        "d = returns5.index\n"
        "for name, (a, b) in (('train', tr), ('val', va), ('test', te)):\n"
        "    print(f'{name:>5} [{a:>4}, {b:>4})  =  {d[a].date()} -> {d[b - 1].date()}'\n"
        "          f'  ({b - a} sessions)')\n"
        "print(f'purge gaps: val starts {va[0] - tr[1]} sessions after train ends; '\n"
        "      f'test starts {te[0] - va[1]} after val ends')\n"
        "verify('60-session purge at both boundaries',\n"
        "       va[0] - tr[1] == 60 and te[0] - va[1] == 60)"
    ))

    # ------------------------------------------------------------------ 7 verdict
    out.append(md(
        "## 7 · Verdict\n"
        "\n"
        "Every integrity claim above was recomputed in this kernel, in this checkout, just now. "
        "The ledger below is the complete list of what was verified — reaching this cell at all "
        "means **nothing failed**."
    ))
    out.append(code(
        "print(f'ALL INTEGRITY CHECKS PASSED  (n={len(CHECKS)})\\n')\n"
        "for i, (name, _detail) in enumerate(CHECKS, 1):\n"
        "    print(f'  {i:>2}. {name}')\n"
        "assert len(CHECKS) >= 25, 'the ledger is unexpectedly short -- were cells skipped?'\n"
        "print(f'\\nsession: python {platform.python_version()} | numpy {np.__version__} | '\n"
        "      f'pandas {pd.__version__}')\n"
        "print('companion: results_walkthrough.ipynb (the analysis these certifications license)')\n"
        "\n"
        "# A styled verdict card: every recomputed integrity claim, all green (version-robust Styler\n"
        "# so it renders on nbconvert across pandas 1.x/2.x).\n"
        "verdict = pd.DataFrame({'integrity claim (recomputed live in this kernel)':\n"
        "                        [n for n, _ in CHECKS], 'status': 'PASSED'})\n"
        "verdict.index = range(1, len(verdict) + 1)\n"
        "(verdict.style\n"
        "        .apply(lambda col: ['background-color: #d7f0d7; font-weight: 600'] * len(col),\n"
        "               subset=['status'])\n"
        "        .set_caption(f'{len(CHECKS)} data-integrity claims, recomputed and asserted "
        "in this kernel -- a red cell would mean an integrity claim no longer holds.'))"
    ))
    return out


def main() -> None:
    nb = build_notebook(cells(), id_prefix="dp")
    path = write_notebook(nb, _OUT)
    h1 = sha256_bytes(path)
    write_notebook(nb, path)  # double-build: regeneration must be byte-identical
    h2 = sha256_bytes(path)
    if h1 != h2:  # pragma: no cover - determinism guard
        raise RuntimeError(f"non-deterministic notebook build: {h1[:12]} != {h2[:12]}")
    print(f"[build_notebook_provenance] wrote {path} ({len(nb['cells'])} cells, sha256 {h1[:12]})")


if __name__ == "__main__":
    main()
