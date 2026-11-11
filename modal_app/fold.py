"""Step 2: uniform re-prediction of every design on Modal.

Verified against the installed modal 1.5.5 API, not from memory. Note that the
old names are gone: ``container_idle_timeout`` is now ``scaledown_window`` and
``concurrency_limit`` is now ``max_containers``.

### Money discipline, per Nina's standing requirement

Modal bills container lifetime, not just compute, so an idle container with a
GPU attached costs real money while doing nothing. Every one of these is
deliberate:

- ``scaledown_window=5``: a container exits about five seconds after its last
  input instead of loitering for the 60-second default.
- no ``min_containers`` and ``buffer_containers=0``: nothing is kept warm and
  nothing is pre-spawned. A warm GPU pool is the most expensive way to wait.
- ``timeout=3600`` set EXPLICITLY. The modal default is 300 seconds, which
  would kill a five-model fold of a 700-residue complex partway through and
  bill for the whole attempt.
- ``retries=0``: a broken image must fail once and immediately rather than be
  retried and billed four times. This is a lesson recorded in
  enrich-bench/LOG.md, learned the expensive way.
- ``max_containers=PILOT_MAX_CONTAINERS``: the main guard against an accidental
  large bill. SPEC.md requires Nina's approval before raising it.
- Designs are processed in CHUNKS so the roughly 60 to 90 seconds of loading
  five sets of AF2 weights is amortised across a batch rather than paid per
  design. Per-design outputs are still written and keyed by ``design_id``, so
  resumability is unaffected.

### Resumability

``fold_chunk`` skips any design whose output already exists on the results
volume. Containers die; a re-run must resume, not restart.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

APP_NAME = "binder-bench"

# Raising this needs Nina's approval. It is the main guard against a runaway
# bill, and SPEC.md lists it as a stop-and-ask.
PILOT_MAX_CONTAINERS = 4

# Designs per container. Amortises weight loading without making a single
# failure expensive.
CHUNK_SIZE = 5

# Explicit. The modal default of 300s would kill these folds mid-run.
FOLD_TIMEOUT_SEC = 3600

# Seconds a container lingers after its last input before shutting down.
SCALEDOWN_WINDOW_SEC = 5

app = modal.App(APP_NAME)

weights = modal.Volume.from_name("bb-weights", create_if_missing=True)
data = modal.Volume.from_name("bb-data", create_if_missing=True)
results = modal.Volume.from_name("bb-results", create_if_missing=True)

WEIGHTS_DIR = "/weights"
DATA_DIR = "/data"
RESULTS_DIR = "/results"

# ---------------------------------------------------------------------------
# IMAGE: NOT YET VERIFIED. DO NOT BUILD THIS WITHOUT READING THE NOTE.
#
# SPEC.md calls containerising ColabFold the largest technical unknown in the
# project and says to start from a working published example rather than
# debugging CUDA and JAX versions from scratch. That is not just a time
# argument: Modal charges for image builds, so a build-fail loop on an
# unverified CUDA/JAX pin spends Nina's money producing nothing.
#
# So this is left deliberately unpinned and unbuilt. Before the first build:
#   1. Find Modal's published protein-folding example or a known-good
#      ColabFold container and start from its exact pins.
#   2. Stage weights into the bb-weights Volume ONCE, or bake them at build
#      time with run_function. Containers must never re-download per
#      invocation.
#   3. Only then set this up and build.
# ---------------------------------------------------------------------------
fold_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget")
    # .uv_pip_install(...)  # PIN EVERY VERSION. See note above.
    .add_local_python_source("binder_bench")
)


def _result_path(design_id: str) -> Path:
    return Path(RESULTS_DIR) / "folds" / f"{design_id}.json"


@app.function(
    image=fold_image,
    gpu="L4",
    volumes={WEIGHTS_DIR: weights, DATA_DIR: data, RESULTS_DIR: results},
    timeout=FOLD_TIMEOUT_SEC,
    scaledown_window=SCALEDOWN_WINDOW_SEC,
    buffer_containers=0,
    retries=0,
    max_containers=PILOT_MAX_CONTAINERS,
)
def fold_chunk(chunk: list[dict]) -> list[dict]:
    """Fold a batch of designs, writing one JSON per design to the volume.

    Each item of ``chunk`` is ``{"design_id": str, "sequence": str}``. The GPU
    type is read from the environment Modal exposes so it lands in provenance,
    because GPU architecture can affect numerics and we want that recorded.
    """
    import numpy as np

    from binder_bench.metrics import aggregate_across_models, interface_metrics
    from binder_bench.target import target_sequence

    target = target_sequence()  # checksum-verified, raises on mismatch
    gpu_type = os.environ.get("MODAL_GPU_TYPE", "unknown")

    out: list[dict] = []
    for item in chunk:
        design_id = item["design_id"]
        dest = _result_path(design_id)

        # Idempotency: never pay twice for the same design.
        if dest.exists():
            out.append({"design_id": design_id, "status": "cached"})
            continue

        binder = item["sequence"]

        # ------------------------------------------------------------------
        # TODO, blocked on the image above: run ColabFold here.
        #
        # Config fixed in DECISIONS.md: 5 models, 1 seed, 3 recycles,
        # templates OFF, relaxation OFF, target MSA plus single-sequence
        # binder, one cached MSA for the 621-mer read from DATA_DIR. Chain
        # order is BINDER FIRST, target second, matching Adaptyv.
        #
        # For each of the 5 models, read its PAE matrix and per-residue pLDDT
        # and call interface_metrics(pae, plddt, len(binder)), which asserts
        # the chain layout and fails loudly on a flip. Then combine with
        # aggregate_across_models, whose primary output is the MEAN across
        # models, not the best of them.
        # ------------------------------------------------------------------
        raise NotImplementedError(
            "ColabFold invocation is blocked on verifying the container image. "
            "See the IMAGE note at the top of this file: building an unverified "
            "CUDA/JAX pin spends money producing nothing."
        )

    results.commit()
    return out


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    limit: int = 10,
    include_controls: bool = True,
    designs_parquet: str = "data/designs.parquet",
):
    """Pilot entrypoint. Defaults are deliberately small.

    The three-card pilot runs this once per card and compares measured dollars
    per design, because rate per hour is not cost: a card three times faster at
    twice the price is cheaper. Spec-sheet throughput favours the bigger cards,
    but AF2's Evoformer is bandwidth and capacity bound and ColabFold's JAX is
    not tuned for recent architectures, so the ratios have to be measured.

    All 378 designs must eventually run on ONE card type. Mixing cards would
    put a hardware confound inside the very metric being measured.
    """
    import pandas as pd

    from binder_bench.costs import assert_within_budget, track_cost

    df = pd.read_parquet(designs_parquet)

    controls = df[df["is_control"]] if include_controls else df.iloc[0:0]
    pool = df[(~df["is_control"]) & df["expressed"]]
    picked = pd.concat([controls, pool.head(limit)])

    items = [
        {"design_id": r.design_id, "sequence": r.sequence}
        for r in picked.itertuples()
    ]
    chunks = [items[i : i + CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]

    # Rough ceiling for the guard: assume no design exceeds the per-call
    # timeout. Replaced by measured numbers after the pilot.
    projected = len(items) * (FOLD_TIMEOUT_SEC / 3600) * 2.0
    assert_within_budget(projected)

    print(
        f"[pilot] gpu={gpu} designs={len(items)} chunks={len(chunks)} "
        f"max_containers={PILOT_MAX_CONTAINERS} scaledown={SCALEDOWN_WINDOW_SEC}s"
    )

    fn = fold_chunk.with_options(
        gpu=gpu,
        max_containers=PILOT_MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SEC,
        timeout=FOLD_TIMEOUT_SEC,
        retries=0,
    )

    records: list[dict] = []
    with track_cost("fold_chunk", gpu, len(items), notes=f"pilot limit={limit}"):
        for batch in fn.map(chunks):
            records.extend(batch)

    print(json.dumps(records, indent=2)[:2000])
