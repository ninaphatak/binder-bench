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

#: Designs per container, DERIVED FROM MEASUREMENT rather than guessed.
#:
#: Measured over 44 pilot models on an L4: 107.4s per model on average
#: (median 91.5, max 198.5) across the real design-length distribution, so a
#: five-model design costs about 537s. The controls alone suggested 83s per
#: model, which was optimistic because control.Human_EGF at 674 residues sits
#: near the 10th percentile of complex length while the design median is 699
#: and the maximum 871.
#:
#: project_cost.safe_chunk_size(107.4, num_models=5, timeout_sec=7200,
#: salvage_margin_sec=420, weight_stage_sec=60) returns 9, giving about 4833s
#: of folding against a deadline near 6720s. At the earlier value of 12 a chunk
#: would have needed roughly 6450s and would have grazed or crossed the
#: deadline on nearly every chunk, salvaging partial results each time and
#: spending the margin instead of holding it in reserve.
CHUNK_SIZE = 9

#: Explicit. Modal's default of 300s would kill these folds mid-run. Budgeted
#: generously per chunk: five designs at five models each.
FOLD_TIMEOUT_SEC = 7200

#: Seconds held back from the container's budget so that when a chunk runs
#: long we kill ColabFold OURSELVES and still have time to parse, copy and
#: commit whatever designs did finish.
#:
#: The first control run taught this the expensive way. It let Modal's timeout
#: kill the container, which means the results Volume was never committed, so
#: two GPU-hours of work vanished and the Volume was still empty afterwards.
#: A chunk that overruns must degrade to partial results, not to nothing.
SALVAGE_MARGIN_SEC = 420

#: Seconds a container lingers after its last input.
SCALEDOWN_WINDOW_SEC = 5

#: Container-local destination for the AlphaFold parameters. Local disk, not
#: the Volume: see the comment in fold_chunk and src/binder_bench/weights.py.
LOCAL_WEIGHTS_DIR = "/tmp/weights"

#: The cached single target MSA, staged once by modal_app/stage.py.
TARGET_MSA_PATH = f"{DATA_DIR}/target_msa/EGFR_ectodomain_P00533_25_645.a3m"

app = modal.App(APP_NAME)

weights = modal.Volume.from_name("bb-weights", create_if_missing=True)
data = modal.Volume.from_name("bb-data", create_if_missing=True)
results = modal.Volume.from_name("bb-results", create_if_missing=True)


def _run_streamed(
    cmd: list[str], deadline_sec: float, tail_lines: int = 300
) -> tuple[int | None, list[str], float, bool]:
    """Run a command, printing its output line by line as it arrives.

    Returns ``(returncode, last_lines, elapsed_seconds, hit_deadline)``.

    Deliberately NOT ``subprocess.run(capture_output=True)``. That buffers
    everything until the process exits, so when the first control run was
    killed at its timeout, ColabFold's entire progress log was discarded along
    with the container. Two GPU-hours produced no results AND no diagnosis.
    A long-running GPU job must narrate itself while it runs.

    A draining thread does the reading so that the main thread can enforce a
    deadline even if ColabFold stops printing entirely. On deadline we kill the
    child ourselves, which leaves the caller time to salvage finished work
    instead of letting Modal kill the container with the Volume uncommitted.
    """
    import collections
    import subprocess
    import threading
    import time

    # PYTHONUNBUFFERED is load bearing, not boilerplate. CPython block-buffers
    # stdout when it is a pipe rather than a terminal, so a child that prints
    # progress to stdout would deliver nothing for minutes and then a burst,
    # and would lose the buffer entirely if killed. Streaming the output is
    # pointless if the child decides not to emit it.
    # UNIFIED MEMORY OFF, pre-set so ColabFold's import-time
    # "if k not in os.environ" guard leaves our values alone. Without this it
    # forces TF_FORCE_UNIFIED_MEMORY=1 and XLA_PYTHON_CLIENT_MEM_FRACTION=4.0,
    # asking JAX for four times the card's memory; allocations spill to host
    # RAM and the run stalls for hours instead of failing. See
    # foldconfig.DISABLE_UNIFIED_MEMORY.
    from binder_bench.foldconfig import UNIFIED_MEMORY_OFF_ENV

    child_env = dict(os.environ, PYTHONUNBUFFERED="1", **UNIFIED_MEMORY_OFF_ENV)

    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    tail: collections.deque[str] = collections.deque(maxlen=tail_lines)

    def drain() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip()
            tail.append(line)
            print(f"[cf +{time.monotonic() - started:7.1f}s] {line}", flush=True)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()

    hit_deadline = False
    try:
        proc.wait(timeout=max(deadline_sec, 1.0))
    except subprocess.TimeoutExpired:
        hit_deadline = True
        print(
            f"[fold] deadline of {deadline_sec:.0f}s reached; killing "
            f"colabfold_batch to salvage completed designs",
            flush=True,
        )
        proc.kill()
        proc.wait()

    reader.join(timeout=15)
    return proc.returncode, list(tail), time.monotonic() - started, hit_deadline


def _gpu_name() -> str:
    """The card's own name, from the driver.

    Not ``os.environ["MODAL_GPU_TYPE"]``. That variable does not exist: the
    diagnostic run printed MODAL_GPU_TYPE=unknown on a container that
    nvidia-smi correctly identified as an L4. Every fold record would have
    stored "unknown" for the one provenance field SPEC.md asks for because
    architecture can affect numerics.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 - provenance must not kill a fold
        return f"unavailable: {type(exc).__name__}"
    if proc.returncode != 0:
        return f"nvidia-smi rc={proc.returncode}"
    return proc.stdout.strip().splitlines()[0].strip() or "empty"


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
    import math
    import shutil
    import time

    import numpy as np

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
    from binder_bench.control_validation import (
        contact_residue_pairs,
        pae_interface_contacts_mean,
    )
    from binder_bench.target import EXPECTED_LENGTH, target_sequence
    from binder_bench.weights import stage_weights_locally

    # Clock starts HERE, not when colabfold_batch launches. Everything before
    # that point (weight staging, A3M writing, nvidia-smi) spends the same
    # container budget Modal is counting down.
    container_started = time.monotonic()

    cfg = FoldConfig()
    target = target_sequence()  # checksum-verified, raises on mismatch
    gpu_type = _gpu_name()

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

    # WEIGHTS ONTO LOCAL DISK FIRST. Reading parameters directly off the
    # Volume is what stalled L4, A100-40GB and H100 for over 40 minutes each
    # without producing a single prediction. ColabFold's np.load returns a lazy
    # NpzFile and flat_params_to_haiku then touches every one of thousands of
    # arrays, so a network filesystem pays latency thousands of times, twice
    # per model. See src/binder_bench/weights.py for the stack that proved it.
    #
    # Paid once per container and amortised over the whole chunk.
    t_stage = time.monotonic()
    local_data = stage_weights_locally(
        CACHE_DIR, LOCAL_WEIGHTS_DIR, cfg.model_type, cfg.num_models
    )
    stage_seconds = time.monotonic() - t_stage

    cmd = build_command(inputs, outputs, local_data, config=cfg)
    assert_no_ambiguous_flags(cmd)

    print(
        f"[fold] {len(todo)} designs, complex lengths "
        f"{sorted(len(it['sequence']) + EXPECTED_LENGTH for it in todo)}",
        flush=True,
    )
    print(f"[fold] {' '.join(cmd)}", flush=True)

    # Hold back time to parse and commit, measured against the budget that
    # REMAINS rather than the full timeout. Subtracting the setup already spent
    # is the whole point: if the deadline overran the container's real budget,
    # Modal would kill us first and the Volume would never be committed, which
    # is precisely the failure the salvage margin exists to prevent.
    container_budget = float(os.environ.get("BB_FOLD_TIMEOUT_SEC", FOLD_TIMEOUT_SEC))
    setup_spent = time.monotonic() - container_started
    deadline = max(container_budget - setup_spent - SALVAGE_MARGIN_SEC, 60.0)
    print(
        f"[fold] setup took {setup_spent:.0f}s; colabfold deadline "
        f"{deadline:.0f}s of {container_budget:.0f}s budget",
        flush=True,
    )

    rc, tail, elapsed, hit_deadline = _run_streamed(cmd, deadline)

    if rc != 0 and not hit_deadline:
        raise RuntimeError(
            f"colabfold_batch failed with code {rc} after "
            f"{elapsed:.0f}s on {len(todo)} designs\n"
            f"--- command ---\n{' '.join(cmd)}\n"
            f"--- last output ---\n" + "\n".join(tail)
        )

    # Salvage mode. ColabFold writes each query's outputs as it finishes them,
    # so an overrun chunk usually holds complete results for its earlier
    # designs. Those are kept; anything short of a full five-model set is
    # reported as incomplete and left for a re-run rather than averaged over
    # the wrong number of models.
    out: list[dict] = list(skipped)
    folded_ids: list[str] = []

    for item in todo:
        design_id = item["design_id"]
        jobname = colabfold_jobname(design_id)
        binder_len = len(item["sequence"])

        try:
            job = parse_colabfold_output(
                outputs,
                jobname,
                expected_models=cfg.num_models,
                expected_seeds=cfg.num_seeds,
            )
        except Exception as exc:
            if not hit_deadline:
                raise
            print(f"[fold] {design_id}: incomplete, {exc}", flush=True)
            out.append(
                {
                    "design_id": design_id,
                    "status": "incomplete",
                    "reason": f"chunk hit its deadline: {exc}"[:400],
                }
            )
            continue

        if job.n_models != expected_prediction_count(cfg):
            msg = (
                f"{design_id}: parsed {job.n_models} predictions, expected "
                f"{expected_prediction_count(cfg)}. Averaging over the wrong "
                f"number of models would corrupt the primary estimator."
            )
            if not hit_deadline:
                raise RuntimeError(msg)
            print(f"[fold] {msg}", flush=True)
            out.append(
                {"design_id": design_id, "status": "incomplete", "reason": msg[:400]}
            )
            continue

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

        # SECONDARY METRIC S1, pre-registered in ANALYSIS_PLAN.md 1a:
        # pae_interface_contacts_mean, the same PAE restricted to residue pairs
        # actually in contact, which removes the target-size dilution that
        # makes Adaptyv's whole-block average scale with the 621-residue
        # target. Computed here because the PAE matrices are in memory now.
        #
        # The raw matrices are ALSO saved, compressed, because they are the
        # raw model output: any future metric over them then needs no re-fold,
        # and a re-fold is not bit-identical (XLA kernel selection moved ipTM
        # by 0.03 between two runs of the same model and seed).
        s1_per_model: list[float] = []
        pae_to_save: dict[str, np.ndarray] = {}
        for pred in job.predictions:
            if pred.pae is None:
                continue
            tgt = dest / Path(pred.structure_path).name
            b_idx, t_idx = contact_residue_pairs(tgt, binder_len)
            s1_per_model.append(
                pae_interface_contacts_mean(pred.pae, b_idx, t_idx)
            )
            # float16 is ample: PAE is bounded at about 32 Angstrom and three
            # significant digits is far finer than the quantity's meaning.
            pae_to_save[pred.model_id] = pred.pae.astype(np.float16)

        if pae_to_save:
            pae_path = Path(RESULTS_DIR) / "pae" / f"{jobname}.npz"
            pae_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pae_path, **pae_to_save)

        s1_clean = [v for v in s1_per_model if not math.isnan(v)]
        s1_summary = {
            "pae_interface_contacts_mean__mean": (
                sum(s1_clean) / len(s1_clean) if s1_clean else float("nan")
            ),
            "pae_interface_contacts_mean__per_model": s1_per_model,
            "n_models_with_contacts": len(s1_clean),
        }

        record = {
            "design_id": design_id,
            "colabfold_jobname": jobname,
            "status": "folded",
            "binder_length": binder_len,
            "target_length": EXPECTED_LENGTH,
            "complex_length": binder_len + EXPECTED_LENGTH,
            "metrics": aggregated,
            "secondary_metrics": s1_summary,
            "per_model": records,
            "structures": kept,
            # Provenance. GPU type is recorded because architecture can affect
            # numerics and we want that in the record, not inferred later.
            "gpu_type": gpu_type,
            "weight_stage_seconds": round(stage_seconds, 2),
            "chunk_seconds": round(elapsed, 2),
            "chunk_n_todo": len(todo),
            "chunk_hit_deadline": hit_deadline,
            "config": cfg.as_record(),
            **provenance(),
        }

        path = _fold_record_path(design_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, default=str))

        # Commit per design, not once at the end. An uncommitted Volume is a
        # Volume that loses everything if the container is killed, which is
        # exactly how the first control run lost two GPU-hours.
        results.commit()

        folded_ids.append(design_id)
        out.append(
            {
                "design_id": design_id,
                "status": "folded",
                "pae_interface_mean": aggregated.get("pae_interface_mean__mean"),
                "pae_interface_contacts_mean": s1_summary[
                    "pae_interface_contacts_mean__mean"
                ],
                "iptm_mean": aggregated.get("iptm__mean"),
                "gpu_type": gpu_type,
            }
        )

    # Keep ColabFold's own log. It is the ONLY place the per-model recycle
    # count is recorded, and that count varies per design because
    # AF2-multimer's recycle_early_stop_tolerance of 0.5 is active. Step 3 has
    # to be able to test whether recycle count confounds the metric, which
    # needs the counts, and they exist nowhere else: the scores JSON does not
    # carry them. Named by chunk leader so concurrent containers cannot
    # overwrite one another.
    cf_log = outputs / "log.txt"
    if cf_log.exists():
        log_dest = (
            Path(RESULTS_DIR) / "colabfold_logs"
            / f"{colabfold_jobname(todo[0]['design_id'])}.log"
        )
        log_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cf_log, log_dest)
        print(f"[fold] saved ColabFold log to {log_dest}", flush=True)
    else:
        print(f"[fold] WARNING: no ColabFold log at {cf_log}", flush=True)

    # Cost per design is only knowable once we know how many actually finished.
    # Dividing the chunk's seconds by the designs we ASKED for would understate
    # the real cost whenever a chunk overruns, which is the one case where the
    # number matters most.
    if folded_ids:
        per_design = elapsed / len(folded_ids)
        print(
            f"[fold] {len(folded_ids)}/{len(todo)} folded in {elapsed:.0f}s "
            f"= {per_design:.0f}s per design on {gpu_type}",
            flush=True,
        )
        for rec in out:
            if rec.get("status") == "folded":
                rec["seconds_per_design"] = round(per_design, 1)
    else:
        print(
            f"[fold] 0/{len(todo)} folded in {elapsed:.0f}s on {gpu_type}",
            flush=True,
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

    from binder_bench.costs import (
        assert_within_budget,
        spend_to_date,
        track_cost,
        worst_case_usd,
    )

    df = pd.read_parquet(designs_parquet)
    controls = df[df["is_control"]] if include_controls else df.iloc[0:0]
    pool = df[(~df["is_control"]) & df["expressed"]]
    picked = pd.concat([controls, pool.head(limit)])

    items = [
        {"design_id": r.design_id, "sequence": r.sequence} for r in picked.itertuples()
    ]
    chunks = [items[i : i + CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]

    print(f"[spend so far] ${spend_to_date():.4f}")
    # Conservative ceiling at the SELECTED card's real rate, assuming every
    # chunk runs to its full timeout. A flat assumed rate here would have
    # understated an A100 run by threefold.
    # One full timeout PER CHUNK, not per container. max_containers bounds
    # wall-clock time, not the bill: four containers each running three chunks
    # cost the same as twelve running one.
    projected = worst_case_usd(
        gpu, n_containers=len(chunks), timeout_sec=FOLD_TIMEOUT_SEC
    )
    print(f"[budget] worst case on {gpu}: ${projected:.2f}")
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
