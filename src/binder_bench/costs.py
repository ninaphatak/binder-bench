"""Cost and provenance accounting, mirroring enrich-bench's costs.csv schema.

Why this schema and not something simpler: the derived number we actually want
later is dollars per design evaluated, and that is unrecoverable from Modal's
own records because they do not know n_inputs. A git sha answers "exactly what
code produced this number, and can I check it out and rerun it", which is
meaningless if the tree was dirty, hence git_dirty.

Rates are read from a recorded snapshot, never from memory. See
RATE_CARD_SOURCE. Re-read the pricing page and update the snapshot rather than
assuming a rate has held.
"""

from __future__ import annotations

import csv
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

RATE_CARD_SOURCE = "modal.com/pricing, read 2026-11-09"

# USD per GPU-second.
GPU_RATES_USD_PER_SEC = {
    "H100": 0.001097,
    "A100-80GB": 0.000694,
    "A100-40GB": 0.000583,
    "L40S": 0.000542,
    "A10": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}
CPU_RATE_USD_PER_CORE_SEC = 0.0000131
MEM_RATE_USD_PER_GIB_SEC = 0.00000222

FIELDS = [
    "timestamp",
    "function_name",
    "gpu_type",
    "gpu_seconds",
    "cpu_seconds",
    "dollars_est",
    "n_inputs",
    "git_sha",
    "git_dirty",
    "produces_results",
    "notes",
]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_provenance() -> tuple[str, bool]:
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return sha, dirty


def estimate_dollars(gpu_type: str, gpu_seconds: float) -> float | None:
    rate = GPU_RATES_USD_PER_SEC.get(gpu_type)
    if rate is None:
        return None
    return rate * gpu_seconds


def log_measured_cost(
    function_name: str,
    gpu_type: str,
    gpu_seconds: float,
    n_inputs: int,
    *,
    produces_results: bool = False,
    notes: str = "",
    path: Path = Path("costs.csv"),
) -> float:
    """Write a ledger row from MEASURED container-seconds.

    ``track_cost`` times the local client, which equals GPU-seconds only when
    exactly one container runs. The first full fold ran four in parallel and
    track_cost recorded $10.21 against a true $38.43, understating by 3.77x.
    A budget guard fed that number would let a parallel run consume four times
    its apparent allowance.

    So any function that fans out must sum the container-seconds its workers
    actually report and log them here instead.
    """
    dollars = estimate_dollars(gpu_type, gpu_seconds)
    sha, dirty = git_provenance()
    row = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "function_name": function_name,
        "gpu_type": gpu_type,
        "gpu_seconds": f"{gpu_seconds:.3f}",
        "cpu_seconds": "",
        "dollars_est": f"{dollars:.6f}" if dollars is not None else "",
        "n_inputs": n_inputs,
        "git_sha": sha,
        "git_dirty": dirty,
        "produces_results": produces_results,
        "notes": notes,
    }
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(
        f"[cost] MEASURED {function_name} gpu={gpu_type} "
        f"gpu_seconds={gpu_seconds:.0f} n={n_inputs} "
        f"est=${row['dollars_est'] or '?'}"
    )
    return dollars or 0.0


@contextmanager
def track_cost(
    function_name: str,
    gpu_type: str,
    n_inputs: int,
    *,
    produces_results: bool = False,
    notes: str = "",
    path: Path = Path("costs.csv"),
):
    """Wrap a .remote() or .map() call in the LOCAL entrypoint, not remotely.

    WARNING: this records LOCAL WALL TIME, which equals GPU-seconds only for a
    single container. For a fanned-out .map() it UNDERSTATES cost by roughly
    the number of parallel containers; the first full fold understated by
    3.77x. Use log_measured_cost with summed container-seconds for those, and
    treat this row as wall-clock context only.

    It has to be local: code inside a Modal function runs in the container, and
    a file written there is discarded when the container exits. The alternative
    is writing under a Volume and committing, which is what the fold's
    per-design records do, but the cost ledger stays local so a dead container
    cannot lose it.
    """
    started = time.monotonic()
    try:
        yield
    finally:
        wall = time.monotonic() - started
        sha, dirty = git_provenance()
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "function_name": function_name,
            "gpu_type": gpu_type,
            "gpu_seconds": f"{wall:.3f}",
            "cpu_seconds": f"{wall:.3f}",
            "dollars_est": (
                f"{estimate_dollars(gpu_type, wall):.6f}"
                if estimate_dollars(gpu_type, wall) is not None
                else ""
            ),
            "n_inputs": n_inputs,
            "git_sha": sha,
            "git_dirty": dirty,
            "produces_results": produces_results,
            "notes": notes,
        }
        new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(
            f"[cost] {function_name} gpu={gpu_type} wall={wall:.1f}s "
            f"n={n_inputs} est=${row['dollars_est'] or '?'}"
        )


def spend_to_date(path: Path = Path("costs.csv")) -> float:
    """Total estimated spend so far. Printed before any run that costs money."""
    if not path.exists():
        return 0.0
    total = 0.0
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("dollars_est"):
                total += float(row["dollars_est"])
    return total


BUDGET_USD = float(os.environ.get("BINDER_BENCH_BUDGET_USD", "260"))


def worst_case_usd(gpu_type: str, n_containers: int, timeout_sec: float) -> float:
    """What a run costs if EVERY container runs to its full timeout.

    The ceiling, not the estimate. Used before launching, because the honest
    question is not "what should this cost" but "what is the most this can
    cost if it goes as badly as last time". The first control run went exactly
    that way: both containers ran the full 7200s and produced nothing.

    Raises on an unknown card rather than assuming a price, since silently
    assuming a cheap rate for an expensive card is the one error here that
    costs money.
    """
    rate = GPU_RATES_USD_PER_SEC.get(gpu_type)
    if rate is None:
        raise ValueError(
            f"No rate known for GPU {gpu_type!r}. Known: "
            f"{sorted(GPU_RATES_USD_PER_SEC)}. Refusing to project a cost from "
            f"a guessed price."
        )
    return rate * timeout_sec * n_containers


def assert_within_budget(projected_usd: float, path: Path = Path("costs.csv")) -> None:
    """Refuse to launch a run that would take the project past its budget.

    A guard, not a substitute for the stop-and-ask gates in SPEC.md.
    """
    spent = spend_to_date(path)
    if spent + projected_usd > BUDGET_USD:
        raise RuntimeError(
            f"Refusing to launch. Spent ${spent:.2f}, this run projects "
            f"${projected_usd:.2f}, budget is ${BUDGET_USD:.2f}. "
            f"Raise BINDER_BENCH_BUDGET_USD only after talking to Nina."
        )
    print(
        f"[budget] spent ${spent:.2f} + projected ${projected_usd:.2f} "
        f"of ${BUDGET_USD:.2f}"
    )
