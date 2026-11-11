"""Throwaway probe: learn the ColabFold container's interfaces. CPU only."""
from __future__ import annotations
import subprocess, sys, json
from pathlib import Path
import modal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from images import CACHE_DIR, colabfold_image  # noqa: E402

app = modal.App("binder-bench-probe")
weights = modal.Volume.from_name("bb-weights", create_if_missing=True)


def _run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": " ".join(cmd), "rc": p.returncode,
            "out": p.stdout[-2500:], "err": p.stderr[-1200:]}


@app.function(image=colabfold_image, volumes={CACHE_DIR: weights},
              timeout=900, scaledown_window=5, buffer_containers=0,
              retries=0, max_containers=1, cpu=1.0)
def probe() -> dict:
    out = {"python": sys.version.split()[0], "executable": sys.executable}
    out["ls_cache"] = sorted(p.name for p in Path("/cache").glob("*"))[:20] \
        if Path("/cache").exists() else "missing"
    out["ls_weights_mount"] = sorted(p.name for p in Path(CACHE_DIR).glob("*"))[:20]
    out["download_help"] = _run([sys.executable, "-m", "colabfold.download", "--help"])
    out["batch_help_tail"] = _run(["colabfold_batch", "--help"])
    out["download_src"] = _run([sys.executable, "-c",
        "import colabfold.download as d, inspect; "
        "print('FILE', d.__file__); "
        "print([n for n in dir(d) if not n.startswith('_')]); "
        "print(inspect.getsource(d))"])
    out["jax"] = _run([sys.executable, "-c",
        "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"])
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(probe.remote(), indent=2)[:9000])
