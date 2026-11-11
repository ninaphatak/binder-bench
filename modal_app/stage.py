"""One-off staging: AlphaFold2 weights, and the single target MSA.

CPU only where possible. Separate app from the fold so a staging failure never
burns GPU time, and so neither job can be scheduled onto the other's container.

Run order:
    uv run modal run modal_app/stage.py::download_weights
    uv run modal run modal_app/stage.py::build_target_msa
    uv run modal run modal_app/stage.py::verify
"""

from __future__ import annotations

import sys
from pathlib import Path

import modal

# Import ONLY from the binder_bench package, never from a sibling file in this
# directory. Modal ships `binder_bench` into the container via
# add_local_python_source; it does not ship this directory, so a sibling import
# resolves locally and then fails inside the container. That failure is a
# container STARTUP error, which crash-loops: Modal keeps relaunching
# containers, and retries=0 does not prevent it because retries govern function
# failures, not startup. tests/test_modal_imports.py enforces this rule.
from binder_bench.modal_images import (  # noqa: E402
    CACHE_DIR,
    DATA_DIR,
    colabfold_image,
    provenance,
)

app = modal.App("binder-bench-stage")

# AF2-multimer v3 weights. The designs are binder/target complexes, so the
# multimer parameters are the ones required; the monomer set would not do.
MODEL_TYPE = "alphafold2_multimer_v3"

weights = modal.Volume.from_name("bb-weights", create_if_missing=True)
data = modal.Volume.from_name("bb-data", create_if_missing=True)

# Weights download is network and disk bound, not GPU bound. No GPU attached.
# An hour is generous; the parameter set is a few gigabytes.
@app.function(
    image=colabfold_image,
    volumes={CACHE_DIR: weights},
    timeout=3600,
    scaledown_window=5,
    buffer_containers=0,
    retries=0,
    max_containers=1,
    cpu=2.0,
)
def download_weights() -> dict:
    """Download the AlphaFold2 parameters into the weights Volume.

    NOT via `python -m colabfold.download`. That is what the ColabFold Docker
    wiki documents, and it exits 0 after ~107 seconds having written nothing to
    our volume, because it targets its own default cache directory inside the
    container, which is discarded when the container exits. There is no flag on
    that entrypoint to redirect it.

    So we call ColabFold's own download function directly with an explicit
    data_dir. Same code path, just told where to put things.
    """
    import inspect

    from colabfold.download import (
        default_data_dir,
        download_alphafold_params,
    )

    dest = Path(CACHE_DIR)
    dest.mkdir(parents=True, exist_ok=True)

    report = {
        "colabfold_default_data_dir": str(default_data_dir),
        "download_signature": str(inspect.signature(download_alphafold_params)),
        "requested_data_dir": str(dest),
        "model_type": MODEL_TYPE,
    }

    download_alphafold_params(MODEL_TYPE, dest)
    weights.commit()

    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())

    if total == 0:
        raise RuntimeError(
            f"Weights download wrote nothing to {dest}. Default data dir is "
            f"{default_data_dir}. Do not proceed to a GPU run: every fold would "
            f"fail or silently re-download."
        )

    report.update(
        {
            "file_count": len(files),
            "files_sample": files[:12],
            "bytes_total": total,
            "gib_total": round(total / 2**30, 2),
            **provenance(),
        }
    )
    return report


@app.function(
    image=colabfold_image,
    volumes={CACHE_DIR: weights, DATA_DIR: data},
    timeout=3600,
    scaledown_window=5,
    buffer_containers=0,
    retries=0,
    max_containers=1,
    cpu=4.0,
)
def build_target_msa() -> dict:
    """Compute the MSA for the 621-residue EGFR target ONCE and cache it.

    The spec is emphatic about this and it is the single most important cost and
    correctness decision in Step 2. The binder is de novo and has no meaningful
    MSA, so standard practice is target MSA plus single-sequence binder. That
    means exactly one MSA is needed, for the target, reused across all 378
    predictions.

    Calling the public MMseqs2 API once per design would be 378 calls from a
    cloud IP, which gets rate limited or blocked, and would introduce
    run-to-run variation in the MSA. That variation is precisely the
    inconsistency Step 2 exists to eliminate.
    """
    import subprocess

    from binder_bench.target import (
        EXPECTED_LENGTH,
        EXPECTED_SHA256,
        target_sequence,
    )

    seq = target_sequence()  # raises on checksum mismatch
    assert len(seq) == EXPECTED_LENGTH

    out_dir = Path(DATA_DIR) / "target_msa"
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "egfr_ectodomain.a3m"

    if marker.exists():
        return {
            "status": "cached",
            "a3m_path": str(marker),
            "a3m_bytes": marker.stat().st_size,
            "target_sha256": EXPECTED_SHA256,
        }

    fasta = out_dir / "egfr_ectodomain.fasta"
    fasta.write_text(f">EGFR_ectodomain_P00533_25_645\n{seq}\n")

    # One call, for one sequence. This is the only time the public MSA server
    # is contacted in the whole project.
    proc = subprocess.run(
        [
            "colabfold_batch",
            str(fasta),
            str(out_dir),
            "--msa-only",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"colabfold_batch --msa-only failed with code {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )

    produced = sorted(p.name for p in out_dir.glob("*.a3m"))
    data.commit()
    return {
        "status": "built",
        "a3m_files": produced,
        "out_dir": str(out_dir),
        "target_sha256": EXPECTED_SHA256,
        "stdout_tail": proc.stdout[-1500:],
    }


@app.function(
    image=colabfold_image,
    volumes={CACHE_DIR: weights, DATA_DIR: data},
    timeout=600,
    scaledown_window=5,
    buffer_containers=0,
    retries=0,
    max_containers=1,
    cpu=1.0,
)
def verify() -> dict:
    """Confirm the staged state without spending GPU time.

    Checks the weights volume is populated, the target MSA exists, and reports
    what python and JAX the official image actually ships, which goes into
    provenance.
    """
    import subprocess

    cache = Path(CACHE_DIR)
    params = sorted(p.name for p in cache.rglob("params_*") if p.is_file())
    total = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())

    msa_dir = Path(DATA_DIR) / "target_msa"
    a3m = sorted(p.name for p in msa_dir.glob("*.a3m")) if msa_dir.exists() else []

    jax_report = subprocess.run(
        [
            sys.executable,
            "-c",
            "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)",
        ],
        capture_output=True,
        text=True,
    )

    return {
        "python": sys.version.split()[0],
        "jax": jax_report.stdout.strip() or jax_report.stderr.strip()[-300:],
        "weights_param_files": params[:10],
        "weights_param_count": len(params),
        "weights_gib": round(total / 2**30, 2),
        "target_a3m": a3m,
        **provenance(),
    }


@app.local_entrypoint()
def main(step: str = "verify"):
    """Run one staging step. Default is the free one."""
    import json

    from binder_bench.costs import spend_to_date, track_cost

    print(f"[spend so far] ${spend_to_date():.4f}")

    fns = {
        "weights": download_weights,
        "msa": build_target_msa,
        "verify": verify,
    }
    if step not in fns:
        raise SystemExit(f"step must be one of {sorted(fns)}")

    # CPU-only staging: gpu_type recorded as "cpu" so the rate lookup returns
    # nothing rather than silently pricing it as a GPU.
    with track_cost(f"stage.{step}", "cpu", 1, notes="staging, cpu only"):
        out = fns[step].remote()

    print(json.dumps(out, indent=2)[:4000])
