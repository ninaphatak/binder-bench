"""Secondary metrics computed AFTER the fold, from saved artifacts only.

Separate from ``fold.py`` on purpose, for three reasons.

It needs no GPU. ipSAE reads a PAE matrix and a set of coordinates and does
numpy arithmetic, so running it in the fold container would rent an L40S to do
what a CPU does. Same argument as the geometry pass.

It must be re-runnable without re-folding. A re-fold is not bit-identical: the
same model and seed moved ipTM from 0.524 to 0.550 across two runs, which is
XLA kernel selection. So every secondary metric has to be computable from
stored output, which is why ``fold_chunk`` saves the raw PAE matrices.

And it must be addable late without touching folded data. ipSAE was
pre-registered (ANALYSIS_PLAN 1a, secondary metric S3) while the full fold was
already running. Computing it here means the 380 folds stay exactly as
produced, and the metric is layered on top.

Inputs, all on the results Volume:
    pae/{jobname}.npz            one array per model id, float16
    structures/{jobname}/*.pdb   one PDB per model
    folds/{jobname}.json         the fold record, updated in place

Nothing here reads a binding outcome. It cannot: ``designs.parquet`` is not
mounted.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

from binder_bench.modal_images import RESULTS_DIR, cpu_image

APP_NAME = "binder-bench-secondary"

#: Designs per container. These are seconds of CPU each, so the batch exists to
#: amortise container startup rather than any expensive setup.
SECONDARY_CHUNK_SIZE = 20

app = modal.App(APP_NAME)
results = modal.Volume.from_name("bb-results", create_if_missing=True)


@app.function(
    image=cpu_image.add_local_file(
        Path(__file__).resolve().parents[1] / "vendor" / "ipsae.py",
        remote_path="/vendor/ipsae.py",
    ),
    volumes={RESULTS_DIR: results},
    timeout=3600,
    scaledown_window=5,
    buffer_containers=0,
    retries=0,
    max_containers=4,
    cpu=2.0,
)
def ipsae_chunk(jobnames: list[str]) -> list[dict]:
    """Compute ipSAE for each model of each design and update its record.

    The vendored reference script is shipped to a fixed path in the container
    rather than resolved from the repo layout, because ``modal_app`` is not
    shipped and only the ``binder_bench`` package is. Its checksum is still
    verified, so a wrong or truncated copy fails loudly.
    """
    import numpy as np

    from binder_bench import ipsae_runner

    # Point the runner at the shipped copy instead of the repo-relative path.
    ipsae_runner.IPSAE_SCRIPT = Path("/vendor/ipsae.py")
    ipsae_runner.assert_reference_script_unmodified()

    out: list[dict] = []
    for jobname in jobnames:
        record_path = Path(RESULTS_DIR) / "folds" / f"{jobname}.json"
        pae_path = Path(RESULTS_DIR) / "pae" / f"{jobname}.npz"
        struct_dir = Path(RESULTS_DIR) / "structures" / jobname

        if not record_path.exists():
            out.append({"jobname": jobname, "status": "no_fold_record"})
            continue
        if not pae_path.exists():
            out.append({"jobname": jobname, "status": "no_pae_saved"})
            continue

        record = json.loads(record_path.read_text())
        if "ipsae" in record.get("secondary_metrics", {}):
            out.append({"jobname": jobname, "status": "cached"})
            continue

        binder_len = record["binder_length"]
        with np.load(pae_path) as npz:
            model_ids = list(npz.files)
            paes = {k: np.asarray(npz[k], dtype=float) for k in model_ids}

        per_model: dict[str, dict] = {}
        failures: list[str] = []
        for model_id in model_ids:
            # Match the structure file for this model id, e.g. model_3_seed_000.
            matches = sorted(struct_dir.glob(f"*{model_id}*.pdb"))
            if len(matches) != 1:
                failures.append(
                    f"{model_id}: {len(matches)} structures matched"
                )
                continue
            try:
                res = ipsae_runner.compute_ipsae(matches[0], paes[model_id])
            except Exception as exc:  # noqa: BLE001 - one model must not kill the design
                failures.append(f"{model_id}: {type(exc).__name__}: {exc}"[:300])
                continue
            per_model[model_id] = res.as_record()

        if not per_model:
            out.append(
                {
                    "jobname": jobname,
                    "status": "failed",
                    "reasons": failures[:5],
                }
            )
            continue

        values = [v["ipsae"] for v in per_model.values()]
        summary = {
            # Mean across models, matching the primary estimator's convention
            # rather than taking the best of five.
            "ipsae__mean": float(sum(values) / len(values)),
            "ipsae__min": float(min(values)),
            "ipsae__max": float(max(values)),
            "ipsae__n_models": len(values),
            "ipsae__per_model": per_model,
        }
        if failures:
            summary["ipsae__failures"] = failures[:5]

        record.setdefault("secondary_metrics", {})
        record["secondary_metrics"]["ipsae"] = summary
        record_path.write_text(json.dumps(record, indent=2, default=str))
        results.commit()

        out.append(
            {
                "jobname": jobname,
                "status": "computed",
                "ipsae__mean": summary["ipsae__mean"],
                "n_models": len(values),
                "n_failed_models": len(failures),
            }
        )

    results.commit()
    return out


@app.local_entrypoint()
def main(limit: int = 0):
    """Run the ipSAE pass over every design that has a saved PAE matrix."""
    from binder_bench.costs import track_cost

    # Which designs are ready: a fold record AND a saved PAE matrix.
    jobnames = sorted(
        p.stem
        for p in _volume_listing("pae")
        if p.suffix == ".npz"
    )
    if limit:
        jobnames = jobnames[:limit]
    if not jobnames:
        print("[ipsae] nothing to do: no saved PAE matrices found", flush=True)
        return

    chunks = [
        jobnames[i : i + SECONDARY_CHUNK_SIZE]
        for i in range(0, len(jobnames), SECONDARY_CHUNK_SIZE)
    ]
    print(
        f"[ipsae] {len(jobnames)} designs in {len(chunks)} chunks, CPU only",
        flush=True,
    )

    records: list[dict] = []
    with track_cost("ipsae_chunk", "cpu", len(jobnames), notes="secondary metric S3"):
        for batch in ipsae_chunk.map(chunks):
            records.extend(batch)

    by_status: dict[str, int] = {}
    for r in records:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print(f"[ipsae] {by_status}", flush=True)

    done = [r for r in records if r["status"] == "computed"]
    if done:
        vals = sorted(r["ipsae__mean"] for r in done)
        mid = vals[len(vals) // 2]
        print(
            f"[ipsae] computed {len(done)}: min {vals[0]:.4f} "
            f"median {mid:.4f} max {vals[-1]:.4f}",
            flush=True,
        )


def _volume_listing(subdir: str) -> list[Path]:
    """Names under a results subdirectory, via the Modal CLI's volume listing.

    Done locally rather than in a container so the entrypoint can plan chunks
    without renting anything.
    """
    import subprocess

    proc = subprocess.run(
        ["modal", "volume", "ls", "bb-results", subdir],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"could not list bb-results/{subdir}: {proc.stderr[-1000:]}"
        )
    return [
        Path(tok)
        for tok in proc.stdout.split()
        if tok.startswith(f"{subdir}/")
    ]
