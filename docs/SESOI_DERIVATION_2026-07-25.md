# SESOI derivation (2026-07-25, amendment R104)

The equivalence backdrop of this study rests on a smallest effect size of interest of
**0.05 validation-DSR**. Here we derive that value from the real panel and from economic
primitives, turning a threshold we had asserted into one we can defend. Amendment R104 then
registers the derivation itself as frozen data, in the `inference.sesoi_derivation` block of
`config/preregistration.yaml`. The justification therefore travels inside the pre-registration and
inside the design hash.

The derivation leaves the SESOI **value unchanged**, and adds only the account of where that value
comes from.

## What the derivation does not touch

Three properties of the inference machinery were already in place, and we alter none of them.

- **The TOST is already DSR-native.** We evaluate the equivalence verdict in per-seed
  validation-DSR units against a margin of ±0.05 DSR, and the selection metric is
  `validation_deflated_sharpe`. There is no Sharpe-against-DSR mismatch to repair.
- **The Sharpe-to-DSR gap is already reconciled** by a conservative delta-method ceiling,
  `ΔDSR_max = φ(0)·√(T−1)/√252·ΔSR_ann`, with `k = 0.6616` DSR per annualised Sharpe at `T = 694`
  scored validation sessions. The Sharpe minimum detectable effect of 0.181 therefore maps to
  approximately **0.120 validation-DSR** at that ceiling. We may read the mapped figure as an upper
bound on the detectable effect.
- **The inconclusive branch already exists** (Lakens 2017). Because that 0.120-DSR ceiling exceeds
  the 0.05 SESOI, a non-rejection in Sharpe units alone does not license an equivalence claim. Only
  the DSR-unit TOST interval can do that.

The tail statement is a registered bootstrap interval (amendment R86) and not a fixed margin. We
report a pooled 90 per cent seed-block-bootstrap interval on the distributional-minus-scalar
CVaR-5% difference, in daily-return units and as a percentage of the scalar arm's CVaR level.

## Three anchors, and the band they bracket

The SESOI is the smallest difference between the distributional-fed and scalar-fed arms that is
materially of interest. We bracket it from two directions and corroborate it from a third: below by
what a difference must exceed before it can be exploited after transaction costs, above by the
scale at which a practitioner would call a difference material, and alongside by the absolute-Sharpe
hurdle that multiple-testing correction imposes.

**1. Lower bound: transaction-cost breakeven.** The environment charges a one-way turnover cost of
`c = 10 bps`. Measured on the equal-weight book, which is the registered benchmark, annualised
volatility runs at **20.2 per cent** and one-way turnover at **112 per cent**. The annual cost drag
is therefore `10 bps × 112% = 0.112%/yr`, putting the breakeven Sharpe floor at
`0.112% / 20.2% = 0.0055` annualised Sharpe. A difference below 0.0055 cannot be exploited after
costs. We read such a difference as economically null and correctly of no interest.

**2. Upper bound: practitioner-material scale.** We take **0.10 annualised Sharpe** as a
deliberately conservative practitioner-material threshold. DeMiguel, Garlappi and Uppal (2009)
establish the qualitative result that the naive 1/N rule is remarkably hard to beat, because no
optimising strategy in their study delivers a Sharpe ratio statistically superior to it across
their seven datasets. Their tabulated 0.10 to 0.20 Sharpe differences are **monthly** figures,
worth roughly 0.35 to 0.69 annualised, and Lo (2002) notes that even a √12 conversion understates
the annualised scale. We therefore read 0.10 *annualised* as a conservative practitioner floor. It appears to
lie well below the scale their monthly figures imply.
Since no canonical "0.10 to 0.20 annualised" citation exists, we state this as our own
conservative synthesis. Correcting the unit only strengthens the argument, because the true practitioner scale
lies well above the SESOI. Since the SESOI falls at or below 0.10, we can never declare a
practitioner-material effect equivalent.

**3. Statistical corroboration: the Harvey-Liu-Zhu hurdle.** The absolute-Sharpe significance
hurdle under multiple testing (Harvey, Liu and Zhu 2016) lies far above the SESOI, because `t > 3`
implies a Sharpe ratio of roughly 0.87 or more over a twelve-year track, and more on the shorter
scored window. That confirms our SESOI as deliberately sub-significance, as an equivalence
threshold should be. The band itself is carried by the first two anchors, with this third one
corroborating.

## The verdict

The registered SESOI of **0.05 DSR ≈ 0.0756 annualised Sharpe**, through the conservative
`k = 0.6616` at `T = 694`, falls inside the derived band `[0.0055, 0.10]` on both sides.

- It is **13.7 times above** the cost-breakeven floor of 0.0055. We therefore never demand
  equivalence tighter than what could possibly be exploited after costs.
- It is **below** the practitioner-material floor of 0.10, which makes our equivalence bar stricter
  than practitioner-material. A bounded-effect verdict is therefore conservative, unable to
  absorb a 0.10-Sharpe effect that a practitioner would care about.

We read 0.05 DSR as the smallest edge that is at once exploitable after costs and
sub-practitioner. The three anchors above put a measurement behind it.

## What amendment R104 registers

R104 adds a frozen `inference.sesoi_derivation` block to `config/preregistration.yaml`, carrying
the economic anchors as hash-bound data. The justification then travels inside the
pre-registration. `tests/test_sesoi_derivation.py` re-derives the band from the registered inputs,
taking cost breakeven as `bps × turnover / σ` and the DSR-to-Sharpe map as `sesoi / k`, and asserts
both `0.0055 < 0.0756 < 0.10` and `sesoi == inference.sesoi`. The derivation is therefore verified
on every test run.

## Power consequence

The DSR-unit minimum detectable effect at `n = 30` seeds is approximately 0.120 DSR at the
conservative ceiling. Since the minimum detectable effect scales as `1/√n`, equivalence becomes
achievable at **n\* ≤ 173 seeds**, where the DSR-unit MDE falls at or below the 0.05 SESOI. That
figure is itself a conservative ceiling, because the true DSR-unit MDE lies below the at-the-money
bound, which suggests that fewer seeds may suffice in practice. The position is borderline at the
fair-share seed floor and comfortably achievable at the CPU-lane reach. Below `n\*` a Sharpe
non-rejection does not license an equivalence claim. The deliverable there is the DSR-unit
bounded-effect interval, whose limit comes from our own registered rule.

## Reproducing this derivation

```bash
pytest tests/test_sesoi_derivation.py      # re-derives the band from the frozen registered block
make power                                 # regenerates the power report, including the DSR-unit MDE
```

The economic inputs are measured from the frozen `returns_panel_univ5` equal-weight book, and the
DSR-to-Sharpe map (`k`, `T`) comes from the power analysis. Both are recorded in the frozen
`inference.sesoi_derivation` block, which lets a reader check the numbers above without re-running
anything.

## References

- DeMiguel, V., Garlappi, L. and Uppal, R. (2009) 'Optimal versus naive diversification: how
  inefficient is the 1/N portfolio strategy?', *Review of Financial Studies*, 22(5), pp. 1915-1953.
- Harvey, C. R., Liu, Y. and Zhu, H. (2016) '... and the cross-section of expected returns',
  *Review of Financial Studies*, 29(1), pp. 5-68.
- Lakens, D. (2017) 'Equivalence tests: a practical primer for t tests, correlations, and
  meta-analyses', *Social Psychological and Personality Science*, 8(4), pp. 355-362.
- Lo, A. W. (2002) 'The statistics of Sharpe ratios', *Financial Analysts Journal*, 58(4), pp. 36-52.
