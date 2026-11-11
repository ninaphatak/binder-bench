"""One-off CPU introspection of ColabFold 1.6.2 inside the pinned image.

Writes its findings to the local filesystem via stdout so the exact,
version-correct interfaces can be analysed offline instead of guessed at from
the project's GitHub main branch, which may have drifted from 1.6.2.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

from binder_bench.modal_images import CACHE_DIR, colabfold_image

app = modal.App("binder-bench-introspect")
weights = modal.Volume.from_name("bb-weights", create_if_missing=True)


@app.function(
    image=colabfold_image,
    volumes={CACHE_DIR: weights},
    timeout=900,
    scaledown_window=5,
    buffer_containers=0,
    retries=0,
    max_containers=1,
    cpu=1.0,
)
def introspect() -> dict:
    out: dict = {}

    h = subprocess.run(["colabfold_batch", "--help"], capture_output=True, text=True)
    out["help_rc"] = h.returncode
    out["help"] = h.stdout or h.stderr

    import colabfold
    pkg = Path(colabfold.__file__).parent
    out["package_dir"] = str(pkg)
    out["modules"] = sorted(p.name for p in pkg.glob("*.py"))

    batch = (pkg / "batch.py").read_text()
    out["batch_lines"] = len(batch.splitlines())

    # Filename patterns: anything that looks like an output file being named.
    import re
    pats = sorted(set(re.findall(r'f?"[^"\n]*\{[^"\n]*\}[^"\n]*\.(?:json|pdb|png|a3m|pickle|cif)"', batch)))
    out["filename_patterns"] = pats

    # The functions that write results, by name.
    out["defs"] = sorted(set(re.findall(r"^def (\w+)", batch, re.M)))

    # a3m / chain handling: lines mentioning cardinality headers or chain joins.
    out["a3m_lines"] = [
        ln.strip() for ln in batch.splitlines()
        if ("#" in ln and "card" in ln.lower())
        or "cardinality" in ln.lower()
        or 'join' in ln and 'seq' in ln.lower()
    ][:40]

    return out


@app.local_entrypoint()
def main(dest: str = "/tmp/colabfold_introspect.json"):
    d = introspect.remote()
    Path(dest).write_text(json.dumps(d, indent=2))
    print(f"wrote {dest}")
    print("help_rc:", d["help_rc"], "| batch.py lines:", d["batch_lines"])
    print("modules:", d["modules"])
