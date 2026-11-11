"""Why did a two-design fold burn 7200 seconds and produce nothing?

The failed control run called ``subprocess.run(..., capture_output=True)``, so
ColabFold's own progress log went into a buffer that was discarded when the
timeout killed the container. Two GPU-hours ran blind. This module exists to buy
back the visibility for the smallest possible amount of money.

Two candidate explanations, in order of how badly they would hurt:

1. JAX never saw the GPU. ColabFold prints a warning and carries on using CPU
   rather than failing, and AF2-multimer on an 862-residue complex on CPU would
   not finish in two hours. This is checked FIRST and raises immediately, so the
   container dies in under a minute instead of renting a card for an hour.
2. Unified memory thrashing. The official image sets TF_FORCE_UNIFIED_MEMORY
   and an XLA memory fraction above 1.0, which lets JAX oversubscribe GPU
   memory and page to host RAM over PCIe instead of raising an out-of-memory
   error. The symptom is exactly what we saw: it runs, it never finishes. The
   environment is printed so this is a measurement rather than a guess.

Then it folds ONE model of ONE design with output streamed line by line, which
gives a measured seconds-per-model at the real complex size. Everything about
whether this project is affordable follows from that one number.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

from binder_bench.modal_images import (
    CACHE_DIR,
    DATA_DIR,
    colabfold_image,
)

APP_NAME = "binder-bench-diagnose"

#: Hard ceiling. One model of one design should take minutes. If it takes 40
#: of them, the answer is "not on this card" and we stop paying to confirm it.
DIAGNOSE_TIMEOUT_SEC = 2400

TARGET_MSA_PATH = f"{DATA_DIR}/target_msa/EGFR_ectodomain_P00533_25_645.a3m"

app = modal.App(APP_NAME)

weights = modal.Volume.from_name("bb-weights", create_if_missing=True)
data = modal.Volume.from_name("bb-data", create_if_missing=True)


def _say(msg: str) -> None:
    """Print with a flush. An unflushed line is a line we do not get to read."""
    print(msg, flush=True)


@app.function(
    image=colabfold_image,
    volumes={CACHE_DIR: weights},
    timeout=600,
    scaledown_window=2,
    buffer_containers=0,
    retries=0,
    max_containers=1,
    cpu=1.0,
)
def inspect_image() -> dict:
    """What the pinned image actually contains. CPU only, so nearly free.

    Asked because three GPU cards all stalled in the same place and the two
    remaining explanations both hinge on facts about the image: whether its
    colabfold_batch accepts --compile-mode, and how much CPU a container
    really gets. Guessing either would cost another GPU rental.
    """
    import subprocess

    out: dict = {}

    help_proc = subprocess.run(
        ["colabfold_batch", "--help"], capture_output=True, text=True, timeout=300
    )
    help_text = help_proc.stdout + help_proc.stderr
    out["help_rc"] = help_proc.returncode
    # Only the flags relevant to the stall, to keep the record readable.
    out["flags_present"] = {
        flag: (flag in help_text)
        for flag in (
            "--compile-mode",
            "--use-pallas",
            "--max-msa",
            "--num-recycle",
            "--recycle-early-stop-tolerance",
            "--use-fuse",
            "--pad-msa",
        )
    }
    _say(f"[help] rc={help_proc.returncode}")
    for flag, present in out["flags_present"].items():
        _say(f"    {flag}: {'yes' if present else 'NO'}")

    # The CPU question. os.cpu_count() reports the HOST's cores, which is not
    # what a cgroup-limited container may use, so read the cgroup quota too.
    out["os_cpu_count"] = os.cpu_count()
    try:
        out["sched_affinity"] = len(os.sched_getaffinity(0))
    except AttributeError:
        out["sched_affinity"] = None
    for path in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            out[path] = Path(path).read_text().strip()
        except OSError as exc:
            out[path] = f"unreadable: {type(exc).__name__}"
    _say(f"[cpu] os.cpu_count={out['os_cpu_count']} affinity={out['sched_affinity']}")
    _say(f"[cpu] cgroup cpu.max={out['/sys/fs/cgroup/cpu.max']}")

    for mod in ("colabfold", "jax", "jaxlib", "triton", "alphafold"):
        probe = subprocess.run(
            [
                "python",
                "-c",
                f"import importlib.metadata as m; print(m.version({mod!r}))",
            ],
            capture_output=True,
            text=True,
        )
        out[f"version_{mod}"] = (probe.stdout or probe.stderr).strip()[:120]
        _say(f"[version] {mod}: {out[f'version_{mod}']}")

    return out


@app.function(
    image=colabfold_image,
    gpu="L4",
    volumes={CACHE_DIR: weights, DATA_DIR: data},
    timeout=DIAGNOSE_TIMEOUT_SEC,
    scaledown_window=2,
    buffer_containers=0,
    retries=0,
    max_containers=1,
)
def diagnose(
    design_id: str,
    sequence: str,
    num_models: int = 1,
    compile_mode: str = "",
    max_msa: str = "",
    dump_stacks_every: int = 0,
) -> dict:
    """Report the GPU environment, then time exactly one model.

    Returns a dict rather than printing only, so the measurement survives in
    the local entrypoint's output even if the stream is lost.

    ``compile_mode`` and ``max_msa`` are passed straight through to
    colabfold_batch so a stall can be attributed to one variable at a time.
    ``dump_stacks_every``, when nonzero, runs ColabFold under faulthandler so
    that every N seconds it prints the full Python stack of every thread. That
    turns a stall from "it hung again" into a named function, which is the
    difference between one more GPU rental and several.
    """
    from binder_bench.foldconfig import FoldConfig, build_command
    from binder_bench.msa import colabfold_jobname, write_complex_a3m
    from binder_bench.target import target_sequence

    gpu_type = os.environ.get("MODAL_GPU_TYPE", "unknown")
    _say(f"=== container up, MODAL_GPU_TYPE={gpu_type} ===")

    # ---------------------------------------------------------------- nvidia
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    _say(f"[nvidia-smi] rc={smi.returncode} {smi.stdout.strip()} {smi.stderr.strip()}")

    # ------------------------------------------------------- the environment
    # These are the variables that decide whether JAX pages to host RAM.
    interesting = {
        k: v
        for k, v in sorted(os.environ.items())
        if any(
            tok in k
            for tok in ("XLA", "JAX", "CUDA", "TF_FORCE", "NVIDIA", "LD_LIBRARY")
        )
    }
    _say("[env]")
    for k, v in interesting.items():
        _say(f"    {k}={v}")

    # --------------------------------------------- does JAX see the card at all
    import jax

    _say(f"[jax] version={jax.__version__}")
    devices = jax.devices()
    _say(f"[jax] devices={devices}")
    platform = devices[0].platform if devices else "none"
    _say(f"[jax] default platform={platform}")

    if platform != "gpu":
        raise RuntimeError(
            f"JAX reports platform {platform!r}, devices {devices}. AlphaFold "
            f"would run on CPU and never finish an 862-residue complex. "
            f"Refusing to rent a GPU to run on the CPU. This is almost "
            f"certainly the cause of the 7200s timeout."
        )

    # Confirm a real allocation lands on the device rather than just that a
    # device is listed.
    import jax.numpy as jnp

    t0 = time.monotonic()
    x = jnp.ones((4096, 4096))
    y = float((x @ x).sum())
    _say(
        f"[jax] 4096x4096 matmul ok in {time.monotonic() - t0:.2f}s "
        f"(sum={y:.3e}, device={x.devices()})"
    )

    # ------------------------------------------------------------- the fold
    target = target_sequence()
    work = Path("/tmp/diag")
    inputs, outputs = work / "inputs", work / "outputs"
    inputs.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)

    target_a3m = Path(TARGET_MSA_PATH).read_text()
    write_complex_a3m(inputs, design_id, sequence, target, target_a3m)

    complex_len = len(sequence) + len(target)
    _say(
        f"=== folding {design_id}: binder={len(sequence)} target={len(target)} "
        f"complex={complex_len} models={num_models} ==="
    )

    # Weights onto LOCAL disk first. Reading them straight off the Volume is
    # what stalled three cards for 40 minutes each.
    from binder_bench.weights import stage_weights_locally

    cfg = FoldConfig(num_models=num_models)

    t_stage = time.monotonic()
    local_data = stage_weights_locally(
        CACHE_DIR, "/tmp/weights", cfg.model_type, num_models
    )
    stage_seconds = time.monotonic() - t_stage
    _say(f"[weights] staging took {stage_seconds:.1f}s")

    cmd = build_command(inputs, outputs, local_data, config=cfg)

    # One variable at a time, appended rather than baked into FoldConfig, so
    # the frozen configuration stays frozen while the stall is attributed.
    if compile_mode:
        cmd += ["--compile-mode", compile_mode]
    if max_msa:
        cmd += ["--max-msa", max_msa]

    if dump_stacks_every:
        # Run ColabFold in-process under faulthandler instead of through its
        # console script, so periodic stack dumps show where it is stuck.
        # faulthandler writes to stderr, which is merged into this stream.
        wrapper = (
            "import faulthandler, sys\n"
            f"faulthandler.dump_traceback_later({dump_stacks_every}, "
            "repeat=True, exit=False)\n"
            f"sys.argv = {cmd!r}\n"
            "from colabfold.batch import main\n"
            "main()\n"
        )
        cmd = ["python", "-u", "-c", wrapper]
        _say(f"[stacks] dumping every {dump_stacks_every}s under faulthandler")

    _say(f"[cmd] {' '.join(cmd) if not dump_stacks_every else 'python -u -c <wrapper>'}")

    # STREAMED, not captured. This is the whole point of this module.
    #
    # PYTHONUNBUFFERED is required, not optional. ColabFold's setup_logging
    # installs a TqdmHandler, so its log lines go out through tqdm.write to
    # STDOUT, and CPython block-buffers stdout when it is a pipe rather than a
    # terminal. Without this the child emits nothing until its buffer fills or
    # it exits, which looks exactly like a hang and loses everything if killed.
    # The first diagnostic run was launched without it and ran silent for
    # minutes; ColabFold's own asctime prefixes saved the measurement.
    child_env = dict(os.environ, PYTHONUNBUFFERED="1")

    started = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=child_env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        _say(f"[cf +{time.monotonic() - started:7.1f}s] {line.rstrip()}")
    rc = proc.wait()
    elapsed = time.monotonic() - started

    _say(f"=== colabfold_batch rc={rc} after {elapsed:.0f}s ===")
    produced = sorted(p.name for p in outputs.rglob("*") if p.is_file())
    _say(f"[outputs] {len(produced)} files")
    for name in produced:
        _say(f"    {name}")

    return {
        "design_id": design_id,
        "gpu_type": gpu_type,
        "jax_platform": platform,
        "jax_devices": str(devices),
        "complex_length": complex_len,
        "num_models": num_models,
        "returncode": rc,
        "seconds": round(elapsed, 1),
        "seconds_per_model": round(elapsed / num_models, 1),
        "n_output_files": len(produced),
        "weight_stage_seconds": round(stage_seconds, 1),
        "jobname": colabfold_jobname(design_id),
        "env": interesting,
    }


@app.local_entrypoint()
def inspect():
    """Cheap, CPU-only: report what the pinned image supports."""
    import json

    from binder_bench.costs import track_cost

    with track_cost("inspect_image", "cpu", 1, notes="image capability probe"):
        out = inspect_image.remote()
    print(json.dumps(out, indent=2), flush=True)


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    control: str = "control.Human_EGF",
    num_models: int = 1,
    designs_parquet: str = "data/designs.parquet",
    compile_mode: str = "",
    max_msa: str = "",
    cpu: float = 0.0,
    dump_stacks_every: int = 0,
    timeout: int = DIAGNOSE_TIMEOUT_SEC,
    label: str = "",
):
    """Time one model of one control on one card.

    Defaults to Human_EGF, the 53-residue control, because at 674 residues it
    is the smaller of the two real complexes and therefore the cheapest honest
    measurement of this target's cost.
    """
    import json

    import pandas as pd

    from binder_bench.costs import spend_to_date, track_cost

    df = pd.read_parquet(designs_parquet)
    row = df[df["design_id"] == control]
    if row.empty:
        raise SystemExit(f"no design {control!r} in {designs_parquet}")
    seq = row.iloc[0]["sequence"]

    print(f"[diagnose] spend so far ${spend_to_date():.4f}", flush=True)
    print(f"[diagnose] gpu={gpu} control={control} models={num_models}", flush=True)

    # Tracked like any other GPU rental. A diagnostic that is not in costs.csv
    # is money that left the budget without appearing in the record.
    opts = dict(gpu=gpu, timeout=timeout, retries=0)
    if cpu:
        opts["cpu"] = cpu
    note = (
        f"{control} num_models={num_models} compile_mode={compile_mode or 'default'} "
        f"max_msa={max_msa or 'default'} cpu={cpu or 'default'} {label}"
    )
    print(f"[diagnose] {note}", flush=True)

    with track_cost("diagnose", gpu, 1, notes=note):
        out = diagnose.with_options(**opts).remote(
            control, seq, num_models, compile_mode, max_msa, dump_stacks_every
        )

    print(json.dumps(out, indent=2), flush=True)
    sys.stdout.flush()
