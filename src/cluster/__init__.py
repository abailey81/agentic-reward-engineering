"""Myriad cluster adapter (PLAN_IF_WE_USE_UCL_MYRIAD.md §12.4 B-A1 / §14 / §15).

The archive IS the message queue: jobs write the SAME atomic records the local pipeline writes
(``src.io.results.write_run`` via ``parallel._archive`` / the test-leg workers), so run_id
idempotency and the hash-verified replay transfer to the cluster UNCHANGED. This package only
adds the submission/packaging layer:

* :mod:`spec_io`     — task_<i>.json batches (+ content sha index) consumed by ``$SGE_TASK_ID``;
* :mod:`jobscript`   — renders the §14.6 template verbatim (arrays, -p ladder, tmpfs staging,
                       epilogue ledger line, ``-r y`` native resume);
* :mod:`submit`      — tar-over-ssh push, qsub arrays, marker hold-chains (no rsync on the driver);
* :mod:`poll`        — exact-incremental tar-over-ssh pull + the compacted-resume diff;
* :mod:`ledger`      — qacct forensics + bounded requeue -> permanent ledger;
* :mod:`run_one`     — the on-node entrypoint: one spec (or a pack of N on the job's
                       cgroup-EXCLUSIVE GPU, §15) through the EXISTING certified LEG workers;
* :mod:`driver`      — the batch-driver kernel (submit -> poll -> requeue -> done, archive-as-truth);
* :mod:`campaign`    — the generation-level composition: per-arm search->test pipelines run
                       concurrently, reusing every authoring/reflection/selection primitive
                       (laptop<->cluster PARITY) so the two substrates cannot run different science.

Everything here is testable WITHOUT Myriad (fake-ssh/golden-file + fake-cluster/stub-author tests);
nothing talks to the network at import time.
"""
from src.cluster.campaign import (
    ClusterRun,
    build_cluster_run,
    run_campaign_on_cluster,
    run_search_arm,
    run_test_leg,
)
from src.cluster.jobscript import render_jobscript, write_jobscript
from src.cluster.spec_io import read_spec, write_specs

__all__ = [
    "write_specs", "read_spec", "render_jobscript", "write_jobscript",
    "ClusterRun", "build_cluster_run", "run_search_arm", "run_test_leg",
    "run_campaign_on_cluster",
]
