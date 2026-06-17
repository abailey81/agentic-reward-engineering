"""Research-grade data platform (medallion architecture) — ADR-012.

Layers (config data.platform.layers):
    raw     vendor payloads exactly as received; WRITE-ONCE, checksummed, provenance-
            sidecar'd (R4). Nothing downstream ever edits raw.
    staged  structurally validated & calendar-aligned tables (schema-enforced).
    clean   integrity-checked, reconciled, authority-merged series; quarantine queue
            holds excluded rows WITH reason codes (nothing silently dropped).
    gold    model-ready artifacts: panels, features, splits — leakage-proof by
            construction (as-of joins only) and tested by leakage ASSERTIONS.

Every artifact at every layer is manifested with SHA-256 and linked in a lineage
graph, so any gold artifact traces byte-exact to the API calls that produced it.

Integrity rails: R3 leakage laws (features declare availability lags;
embargoed splits), R4 raw immutability (no fabrication / interpolation / silent
imputation — the missing-data engine COUNTS every decision), R7 citations in
docstrings. Stage modules:

    vault             layers, manifests, provenance, lineage, verified IO   (stage 2, 12)
    acquire           rate governor, backoff, resumable journal, vendors    (stage 1)
    probes            automated entitlement checklist -> evidence report    (stage 1)
    security_master   RIC<->ticker symbology with history                   (stage 3)
    validate          schema, calendar, duplicates, gaps, missing engine    (stages 4, 8)
    integrity         corporate actions + outlier/error taxonomy            (stages 5, 6)
    membership        PIT membership, joiners/leavers, Shumway log          (stage 7)
    reconcile_full    two-vendor reconciliation + clustering + quarantine   (stage 9)
    panel             gold construction + embargo-aware splits              (stage 10)
    eda               statistical profiling that motivates the method       (stage 11)
    quality           scoreboard, lineage map, datasheet                    (stage 12)
    cli               probe/pull/build/validate/reconcile/eda/status        (stage 13)
"""
