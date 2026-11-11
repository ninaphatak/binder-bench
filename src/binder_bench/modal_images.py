"""Every Modal image definition, in one place. Pinned by content digest.

The spec hoped for a published Modal ColabFold example to start from. There
isn't one. What exists instead is better: the ColabFold authors publish their
own built containers, so we never choose JAX or CUDA versions ourselves and
never debug them.

Registry: ghcr.io/sokrypton/colabfold
Tags available as of 2026-11-10: 1.5.3-cuda11.8.0, 1.5.3-cuda12.2.2,
1.5.5-cuda11.8.0, 1.5.5-cuda12.2.2, 1.6.0-cuda12, 1.6.1-cuda12, 1.6.2-cuda12,
1.6.2-cuda13, main-cuda12, main-cuda13.

We take 1.6.2-cuda12, pinned by digest rather than by tag so the image cannot
change under us. CUDA 12 rather than 13 because 13 exists for Blackwell cards
we are not using, and the newer stack is where JAX's "no supported devices
found for platform CUDA" failures appear. Ada and Ampere, which is what the L4,
L40S and A100 are, are well covered by CUDA 12.
"""

from __future__ import annotations

import modal

COLABFOLD_VERSION = "1.6.2"
COLABFOLD_CUDA = "cuda12"
COLABFOLD_TAG = f"{COLABFOLD_VERSION}-{COLABFOLD_CUDA}"
COLABFOLD_DIGEST = (
    "sha256:22f1243a8feca427b6942889d4dc3b549a3b75b23df9cf85c5f512a85e4f7f77"
)
COLABFOLD_REF = f"ghcr.io/sokrypton/colabfold@{COLABFOLD_DIGEST}"

# WEIGHTS MOUNT PATH. Not /cache, even though that is what the ColabFold Docker
# wiki uses. The official image already ships files at /cache, and Modal refuses
# with "cannot mount volume on non-empty path" when a Volume target is not
# empty. So the weights Volume goes at /weights and ColabFold is pointed at it
# explicitly. Weights are NOT baked into the image: someone else reported a
# 33 GB image from doing that, mostly duplicated CUDA plus parameters.
CACHE_DIR = "/weights"
DATA_DIR = "/data"
RESULTS_DIR = "/results"

colabfold_image = modal.Image.from_registry(
    COLABFOLD_REF,
    add_python=None,  # the official image ships its own python
).add_local_python_source("binder_bench")

# Small CPU-only image for staging and for anything that does not need a GPU.
# Separate on purpose: "one function, one job", and a CPU job must never be
# scheduled onto a GPU container we are paying for.
cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("numpy==2.2.6", "pandas==2.3.3", "pyarrow==21.0.0")
    .add_local_python_source("binder_bench")
)


def provenance() -> dict[str, str]:
    """Recorded with every run. The image digest is the real version record."""
    return {
        "colabfold_version": COLABFOLD_VERSION,
        "colabfold_cuda": COLABFOLD_CUDA,
        "colabfold_image_digest": COLABFOLD_DIGEST,
        "colabfold_image_ref": COLABFOLD_REF,
        "weights_license": "AlphaFold2 / ColabFold params, CC BY 4.0, openly licensed",
    }
