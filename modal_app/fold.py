"""Step 2: uniform re-prediction of every design on Modal.

Verified against the installed modal 1.5.5 API and against ColabFold 1.6.2's
own source, not against documentation. The documentation has been wrong four
times on this project.

### Money discipline, per Nina's standing requirement

Modal bills container lifetime, not just compute, so an idle GPU container
costs real money while doing nothing. Each of these is deliberate:

- ``scaledown_window=5``: a container exits about five seconds after its last
  input rather than loitering for the 60-second default.
- no ``min_containers``, ``buffer_containers=0``: nothing kept warm, nothing
  pre-spawned.
- ``timeout`` set EXPLICITLY. Modal's default is 300 seconds, which would kill
  a five-model fold of a 700-residue complex partway and bill the attempt.
- ``retries=0``: a broken image fails once rather than being billed four times.
  Note that this does NOT cover container STARTUP failures, which crash-loop
  regardless; tests/test_modal_imports.py guards that separately.
- ``max_containers=PILOT_MAX_CONTAINERS``: the main guard against a runaway
  bill. Raising it needs Nina's approval.
- One ``colabfold_batch`` call per CHUNK of designs, not per design. ColabFold
  sorts a directory of inputs by length and reuses compiled GPU kernels across
  the batch, so both the 3.8 GiB weight load and the JAX compilation are
  amortised instead of paid per design.
- Geometry (contacts, buried surface area) is deliberately NOT computed here.
  It is CPU work at about 2.8 seconds per complex, and doing it inside the GPU
  container would rent an A100 to run numpy. It gets its own CPU function over
  the saved structures.

### Resumability

Per-design output is keyed by ``design_id`` on the results Volume and
``fold_chunk`` returns early for any design already done. Containers die; a
re-run must resume, not restart.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from binder_bench.modal_images import (
    CACHE_DIR,
    DATA_DIR,
    RESULTS_DIR,
    colabfold_image,
    cpu_image,
    provenance,
)

APP_NAME = "binder-bench"

#: Raising this needs Nina's approval. SPEC.md lists it as a stop-and-ask.
PILOT_MAX_CONTAINERS = 4

#: Designs per container. Amortises weight loading and kernel compilation
#: without making a single failure expensive.
CHUNK_SIZE = 5

#: Explicit. Modal's default of 300s would kill these folds mid-run. Budgeted
#: generously per chunk: five designs at five models each.
FOLD_TIMEOUT_SEC = 7200

#: Seconds a container lingers after its last input.
SCALEDOWN_WINDOW_SEC = 5

#: The cached single target MSA, staged once by modal_app/stage.py.
TARGET_MSA_PATH = f"{DATA_DIR}/target_msa/EGFR_ectodomain_P00533_25_645.a3m"

app = modal.App(APP_NAME)

weights = modal.Volume.from_name("bb-weights", create_if_missing=True)
data = modal.Volume.from_name("bb-data", create_if_missing=True)
results = modal.Volume.from_name("bb-results", create_if_missing=True)


def _fold_record_path(design_id: str) -> Path:
    """Where a design's metrics land. Keyed by design_id, not by job name.

    This distinction matters. ColabFold names its outputs after a sanitised job
    name, and 106 of our 402 design_ids contain spaces which it rewrites to
    underscores. The record is keyed by the real design_id so it joins back to
    designs.parquet; the job name is recorded inside it for traceability.
    """
    from binder_bench.msa import colabfold_jobname

    return Path(RESULTS_DIR) / "folds" / f"{colabfold_jobname(design_id)}.json"


def _structures_dir(design_id: str) -> Path:
    from binder_bench.msa import colabfold_jobname

    return Path(RESULTS_DIR) / "structures" / colabfold_jobname(design_id)


@app.function(
    image=colabfold_image,
    gpu="L4",
    volumes={CACHE_DIR: weights, DATA_DIR: data, RESULTS_DIR: results},
    timeout=FOLD_TIMEOUT_SEC,
    scaledown_window=SCALEDOWN_WINDOW_SEC,
    buffer_containers=0,
    retries=0,
    max_containers=PILOT_MAX_CONTAINERS,
)
def fold_chunk(chunk: list[dict]) -> list[dict]:
    """Fold a batch of designs, writing one JSON record per design.

    Each item of ``chunk`` is ``{"design_id": str, "sequence": str}``.
    """
    import shutil
    import subprocess
    import time

    from binder_bench.foldconfig import (
        FoldConfig,
        assert_no_ambiguous_flags,
        build_command,
        expected_prediction_count,
    )
    from binder_bench.metrics import aggregate_across_models
    from binder_bench.msa import (
        assert_unique_jobnames,
        colabfold_jobname,
        write_complex_a3m,
    )
    from binder_bench.parse_colabfold import parse_colabfold_output, to_metric_records
    from binder_bench.target import EXPECTED_LENGTH, target_sequence

    cfg = FoldConfig()
    target = target_sequence()  # checksum-verified, raises on mismatch
    gpu_type = os.environ.get("MODAL_GPU_TYPE", "unknown")

    # Refuse a batch where two design_ids would collapse onto one ColabFold job
    # name, because the second would silently overwrite the first's outputs.
    assert_unique_jobnames([it["design_id"] for it in chunk])

    todo = [it for it in chunk if not _fold_record_path(it["design_id"]).exists()]
    skipped = [
        {"design_id": it["design_id"], "status": "cached"}
        for it in chunk
        if _fold_record_path(it["design_id"]).exists()
    ]
    if not todo:
        return skipped

    target_a3m = Path(TARGET_MSA_PATH).read_text()

    work = Path("/tmp/work")
    if work.exists():
        shutil.rmtree(work)
    inputs = work / "inputs"
    outputs = work / "outputs"
    inputs.mkdir(parents=True)
    outputs.mkdir(parents=True)

    # One A3M per design, all in one directory, so ColabFold runs once.
    for item in todo:
        write_complex_a3m(
            inputs,
            item["design_id"],
            item["sequence"],
            target,
            target_a3m,
        )

    cmd = build_command(inputs, outputs, CACHE_DIR, config=cfg)
    assert_no_ambiguous_flags(cmd)

    started = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - started

    if proc.returncode != 0:
        raise RuntimeError(
            f"colabfold_batch failed with code {proc.returncode} after "
            f"{elapsed:.0f}s on {len(todo)} designs\n"
            f"--- command ---\n{' '.join(cmd)}\n"
            f"--- stdout ---\n{proc.stdout[-5000:]}\n"
            f"--- stderr ---\n{proc.stderr[-5000:]}"
        )

    out: list[dict] = list(skipped)
    per_design_seconds = elapsed / len(todo)

    for item in todo:
        design_id = item["design_id"]
        jobname = colabfold_jobname(design_id)
        binder_len = len(item["sequence"])

        job = parse_colabfold_output(
            outputs,
            jobname,
            expected_models=cfg.num_models,
            expected_seeds=cfg.num_seeds,
        )
        if job.n_models != expected_prediction_count(cfg):
            raise RuntimeError(
                f"{design_id}: parsed {job.n_models} predictions, expected "
                f"{expected_prediction_count(cfg)}. Averaging over the wrong "
                f"number of models would corrupt the primary estimator."
            )

        # Asserts chain layout per model and fails loudly on a flip.
        records = to_metric_records(job, binder_len)
        aggregated = aggregate_across_models(records)

        # Keep every predicted structure. Step 4 starts MD from these, and the
        # CPU geometry pass reads them.
        dest = _structures_dir(design_id)
        dest.mkdir(parents=True, exist_ok=True)
        kept = []
        for pred in job.predictions:
            src = Path(pred.structure_path)
            tgt = dest / src.name
            shutil.copy2(src, tgt)
            kept.append(tgt.name)

        record = {
            "design_id": design_id,
            "colabfold_jobname": jobname,
            "status": "folded",
            "binder_length": binder_len,
            "target_length": EXPECTED_LENGTH,
            "complex_length": binder_len + EXPECTED_LENGTH,
            "metrics": aggregated,
            "per_model": records,
            "structures": kept,
            # Provenance. GPU type is recorded because architecture can affect
            # numerics and we want that in the record, not inferred later.
            "gpu_type": gpu_type,
            "seconds_per_design": round(per_design_seconds, 2),
            "chunk_seconds": round(elapsed, 2),
            "chunk_size": len(todo),
            "config": cfg.as_record(),
            **provenance(),
        }

        path = _fold_record_path(design_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=str))
        out.append(
            {
                "design_id": design_id,
                "status": "folded",
                "pae_interface_mean": aggregated.get("pae_interface_mean__mean"),
                "iptm_mean": aggregated.get("iptm__mean"),
                "seconds": round(per_design_seconds, 2),
                "gpu_type": gpu_type,
            }
        )

    results.commit()
    return out


@app.function(
    image=cpu_image,
    volumes={RESULTS_DIR: results},
    timeout=3600,
    scaledown_window=5,
    buffer_containers=0,
    retries=0,
    max_containers=4,
    cpu=2.0,
)
def geometry_chunk(design_ids: list[str]) -> list[dict]:
    """Contacts and buried surface area, on CPU, over the saved structures.

    Separate from the fold on purpose. This is numpy at roughly 2.8 seconds per
    complex, and running it inside the GPU container would mean renting an L40S
    to do arithmetic.
    """
    from binder_bench.geometry import geometry_metrics

    out: list[dict] = []
    for design_id in design_ids:
        record_path = _fold_record_path(design_id)
        if not record_path.exists():
            out.append({"design_id": design_id, "status": "no_fold_record"})
            continue

        record = json.loads(record_path.read_text())
        if "geometry" in record:
            out.append({"design_id": design_id, "status": "cached"})
            continue

        binder_len = record["binder_length"]
        per_structure = []
        for name in record["structures"]:
            path = _structures_dir(design_id) / name
            per_structure.append(
                {"structure": name, **geometry_metrics(path, binder_len)}
            )

        # Averaged across models, consistent with the primary estimator being
        # the mean rather than the best of five.
        keys = ("n_contacts_5A", "bsa", "bsa_per_side")
        summary = {
            f"{k}__mean": sum(s[k] for s in per_structure) / len(per_structure)
            for k in keys
        }
        record["geometry"] = {"summary": summary, "per_structure": per_structure}
        record_path.write_text(json.dumps(record, indent=2, default=str))
        out.append({"design_id": design_id, "status": "computed", **summary})

    results.commit()
    return out


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    limit: int = 10,
    include_controls: bool = True,
    designs_parquet: str = "data/designs.parquet",
    geometry: bool = False,
):
    """Pilot entrypoint. Defaults are deliberately small.

    The three-card pilot runs this once per card and compares measured dollars
    per design. Rate per hour is not cost: a card three times faster at twice
    the price is cheaper. Spec-sheet throughput favours the bigger cards, but
    AF2's Evoformer is bandwidth and capacity bound and ColabFold's JAX is not
    tuned for recent architectures, so the ratio has to be measured.

    All 378 designs must eventually run on ONE card type. Mixing cards would
    put a hardware confound inside the metric being measured.
    """
    import pandas as pd

    from binder_bench.costs import assert_within_budget, spend_to_date, track_cost

    df = pd.read_parquet(designs_parquet)
    controls = df[df["is_control"]] if include_controls else df.iloc[0:0]
    pool = df[(~df["is_control"]) & df["expressed"]]
    picked = pd.concat([controls, pool.head(limit)])

    items = [
        {"design_id": r.design_id, "sequence": r.sequence} for r in picked.itertuples()
    ]
    chunks = [items[i : i + CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]

    print(f"[spend so far] ${spend_to_date():.4f}")
    # Conservative ceiling: assume every chunk runs to its full timeout.
    projected = len(chunks) * (FOLD_TIMEOUT_SEC / 3600) * 2.5
    assert_within_budget(projected)

    print(
        f"[pilot] gpu={gpu} designs={len(items)} chunks={len(chunks)} "
        f"chunk_size={CHUNK_SIZE} max_containers={PILOT_MAX_CONTAINERS} "
        f"scaledown={SCALEDOWN_WINDOW_SEC}s timeout={FOLD_TIMEOUT_SEC}s"
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

    print(json.dumps(records, indent=2)[:4000])

    if geometry:
        ids = [r["design_id"] for r in records if r.get("status") == "folded"]
        with track_cost("geometry_chunk", "cpu", len(ids), notes="pilot geometry"):
            geo = list(geometry_chunk.map([ids[i : i + 10] for i in range(0, len(ids), 10)]))
        print(json.dumps(geo, indent=2)[:2000])
