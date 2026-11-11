"""Copy AlphaFold parameters from a network Volume onto local container disk.

### Why this module exists

Three GPU cards (L4, A100-40GB, H100) each stalled for over 40 minutes without
producing a single prediction, all at the same point. A periodic stack dump
showed the process sitting in::

    alphafold/model/utils.py  flat_params_to_haiku
      jax/_src/numpy/array_constructors.py  array
        jax/_src/interpreters/pxla.py  batched_device_put

It was never compiling. It was loading weights.

ColabFold loads parameters with ``np.load(path)``, which returns a LAZY
``NpzFile``: nothing is read until a member is accessed. ``flat_params_to_haiku``
then iterates ``params.items()`` and calls ``jnp.array`` on each one. AF2's
parameter files hold thousands of separate arrays, so that loop performs
thousands of small reads. Pointed at a Modal Volume, which is a network
filesystem, each of those pays network latency, and ColabFold calls the loader
TWICE per model. On local disk the same load takes well under a minute.

So the weights live on the Volume, where they are staged once and cheaply, and
are copied to container-local disk before use: one large sequential read instead
of thousands of small random ones. The copy is paid once per container and
amortised over every design in the chunk.

The success marker file is copied too, and that is not optional. ColabFold's
``download_alphafold_params`` returns early only if
``params/download_complexes_multimer_v3_finished.txt`` exists; without it, it
re-downloads 3.8 GiB into the container on every single run.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

#: ColabFold checks for this exact filename before deciding the parameters are
#: already present. Named per model type in colabfold/download.py.
SUCCESS_MARKERS = {
    "alphafold2_multimer_v3": "download_complexes_multimer_v3_finished.txt",
    "alphafold2_multimer_v2": "download_complexes_multimer_v2_finished.txt",
    "alphafold2_multimer_v1": "download_complexes_multimer_v1_finished.txt",
    "alphafold2_ptm": "download_finished.txt",
}

#: Parameter filename pattern per model type.
PARAM_PATTERNS = {
    "alphafold2_multimer_v3": "params_model_{n}_multimer_v3.npz",
    "alphafold2_multimer_v2": "params_model_{n}_multimer_v2.npz",
    "alphafold2_multimer_v1": "params_model_{n}_multimer_v1.npz",
    "alphafold2_ptm": "params_model_{n}_ptm.npz",
}


def required_files(model_type: str, num_models: int) -> list[str]:
    """Names, relative to the params directory, that must exist locally.

    Only the model files actually used are listed. ColabFold takes models in
    order, so ``num_models=1`` needs model 1 alone, and copying the other four
    would be 1.5 GiB of pointless network traffic.
    """
    if model_type not in PARAM_PATTERNS:
        raise ValueError(
            f"Unknown model_type {model_type!r}. Known: {sorted(PARAM_PATTERNS)}"
        )
    if not 1 <= num_models <= 5:
        raise ValueError(f"num_models must be 1 to 5, got {num_models}")

    pattern = PARAM_PATTERNS[model_type]
    names = [pattern.format(n=n) for n in range(1, num_models + 1)]
    names.append(SUCCESS_MARKERS[model_type])
    return names


def stage_weights_locally(
    volume_dir: Path | str,
    local_dir: Path | str,
    model_type: str = "alphafold2_multimer_v3",
    num_models: int = 5,
    *,
    verbose: bool = True,
) -> Path:
    """Copy the parameters needed onto local disk and return the data dir.

    The returned path is what should be passed to ``colabfold_batch --data``.
    Both directories carry a ``params`` subdirectory, because AlphaFold always
    joins "params" onto the data dir internally.

    Files already present locally with a matching size are not recopied, so a
    warm container pays nothing. A size mismatch IS recopied rather than
    trusted: a half-copied parameter file would load as garbage or raise deep
    inside haiku, and neither failure would name its own cause.
    """
    volume_params = Path(volume_dir) / "params"
    local = Path(local_dir)
    local_params = local / "params"
    local_params.mkdir(parents=True, exist_ok=True)

    names = required_files(model_type, num_models)
    copied_bytes = 0
    started = time.monotonic()

    for name in names:
        src = volume_params / name
        dst = local_params / name
        if not src.exists():
            raise FileNotFoundError(
                f"{src} is missing from the weights Volume. Staging did not "
                f"complete, or the model type is wrong. Refusing to let "
                f"ColabFold silently re-download 3.8 GiB instead."
            )
        size = src.stat().st_size
        if dst.exists() and dst.stat().st_size == size:
            if verbose:
                print(f"[weights] {name} already local ({size / 2**20:.0f} MiB)", flush=True)
            continue

        t0 = time.monotonic()
        # copyfile, not copy2: metadata is irrelevant here and copystat on a
        # Volume-backed source has failed for others.
        shutil.copyfile(src, dst)
        elapsed = time.monotonic() - t0
        copied_bytes += size
        if verbose:
            rate = (size / 2**20) / elapsed if elapsed > 0 else float("inf")
            print(
                f"[weights] copied {name} {size / 2**20:.0f} MiB in "
                f"{elapsed:.1f}s ({rate:.0f} MiB/s)",
                flush=True,
            )

        if dst.stat().st_size != size:
            raise OSError(
                f"{name} copied to {dst.stat().st_size} bytes but source is "
                f"{size}. Refusing to fold from a truncated parameter file."
            )

    if verbose:
        print(
            f"[weights] local data dir {local} ready in "
            f"{time.monotonic() - started:.1f}s, "
            f"{copied_bytes / 2**20:.0f} MiB copied",
            flush=True,
        )
    return local
