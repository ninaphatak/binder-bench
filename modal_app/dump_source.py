"""One-off: dump ColabFold 1.6.2 source locally for offline analysis. CPU only."""
from __future__ import annotations

import json
from pathlib import Path

import modal

from binder_bench.modal_images import colabfold_image

app = modal.App("binder-bench-dump")


@app.function(image=colabfold_image, timeout=600, scaledown_window=5,
              buffer_containers=0, retries=0, max_containers=1, cpu=1.0)
def dump() -> dict:
    import colabfold
    pkg = Path(colabfold.__file__).parent
    want = [
        "batch.py",
        "input.py",
        "colabfold.py",
        "utils.py",
        "download.py",
        "alphafold/models.py",
    ]
    out = {}
    for n in want:
        f = pkg / n
        if f.exists():
            out[n.replace("/", "__")] = f.read_text()
    return out


@app.local_entrypoint()
def main(dest: str = "/tmp/colabfold_src"):
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    files = dump.remote()
    for name, text in files.items():
        (out / name).write_text(text)
        print(f"{name}: {len(text.splitlines())} lines")
    print("wrote to", out)
