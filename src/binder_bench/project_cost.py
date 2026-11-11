"""Project the cost of the full fold from a measured pilot.

This exists because the question "can we afford this" was answered by guesswork
until the first GPU run, and the guess was wrong. Everything here takes a
MEASURED seconds-per-model and projects from it. Nothing here estimates
AlphaFold's speed from first principles, because that estimate is what failed.

Two facts about this target shape the arithmetic.

The target dominates the complex. The EGFR ectodomain is 621 residues and the
binder median is 78, so complexes run 634 to 871 residues with a median of 699.
Designs therefore cost roughly the same as each other, and a measurement on one
control transfers to the set far better than it would for a set of varied
lengths.

Rate per hour is not cost. A card that costs three times as much per hour and
runs three times faster costs the same per design. So the comparison that
matters is dollars per design measured on each card, which is what
``per_design_usd`` reports and what the three-card pilot is for.
"""

from __future__ import annotations

from dataclasses import dataclass

from .costs import GPU_RATES_USD_PER_SEC

#: Expressed designs in the tested population, from designs.parquet.
N_EXPRESSED = 378

#: Median and mean complex length over those 378, measured not assumed.
MEDIAN_COMPLEX_LENGTH = 699
MEAN_COMPLEX_LENGTH = 724


@dataclass(frozen=True)
class Projection:
    """What one configuration on one card would cost for the whole set."""

    gpu: str
    n_designs: int
    num_models: int
    seconds_per_model: float
    compile_seconds_per_chunk: float
    chunk_size: int

    @property
    def n_chunks(self) -> int:
        return -(-self.n_designs // self.chunk_size)  # ceiling division

    @property
    def gpu_seconds(self) -> float:
        """Total billed GPU seconds.

        Compilation is paid once per CHUNK, not once per design: ColabFold
        sorts a directory of inputs by length and reuses its compiled kernels
        across the batch. That is the entire reason chunking exists, and at
        these complex sizes it is a large share of the bill.
        """
        folding = self.n_designs * self.num_models * self.seconds_per_model
        compiling = self.n_chunks * self.compile_seconds_per_chunk
        return folding + compiling

    @property
    def total_usd(self) -> float:
        rate = GPU_RATES_USD_PER_SEC.get(self.gpu)
        if rate is None:
            raise ValueError(
                f"No rate known for {self.gpu!r}. Known: "
                f"{sorted(GPU_RATES_USD_PER_SEC)}."
            )
        return rate * self.gpu_seconds

    @property
    def per_design_usd(self) -> float:
        return self.total_usd / self.n_designs

    @property
    def gpu_hours(self) -> float:
        return self.gpu_seconds / 3600.0

    def wall_clock_hours(self, max_containers: int) -> float:
        """Wall time if chunks are spread over ``max_containers`` containers.

        Separate from cost on purpose. Parallelism buys time, never money: the
        bill is total container-seconds either way.
        """
        if max_containers < 1:
            raise ValueError("max_containers must be at least 1")
        chunks_per_container = -(-self.n_chunks // max_containers)
        seconds_per_chunk = self.gpu_seconds / self.n_chunks
        return chunks_per_container * seconds_per_chunk / 3600.0

    def as_row(self) -> dict:
        return {
            "gpu": self.gpu,
            "n_designs": self.n_designs,
            "num_models": self.num_models,
            "seconds_per_model": round(self.seconds_per_model, 1),
            "gpu_hours": round(self.gpu_hours, 1),
            "per_design_usd": round(self.per_design_usd, 3),
            "total_usd": round(self.total_usd, 2),
        }


def scale_to_length(
    seconds: float,
    measured_length: int,
    target_length: int,
    exponent: float = 2.5,
) -> float:
    """Scale a runtime from one complex length to another.

    AlphaFold's Evoformer mixes terms that grow as the square of the length
    with triangle-update terms that grow as its cube, so the effective
    exponent sits between 2 and 3 and drifts with length. 2.5 is a midpoint,
    and it is an ASSUMPTION, flagged as one.

    It is used only for small corrections, such as the 674-residue control to
    the 699-residue median, a 1.04x length ratio and about 1.1x in time. Over
    that span the choice of exponent barely matters: at 2.0 the factor is 1.07
    and at 3.0 it is 1.11. Do not use it to extrapolate across a large
    truncation, where the error compounds and a direct measurement is cheap.
    """
    if measured_length <= 0 or target_length <= 0:
        raise ValueError("lengths must be positive")
    ratio = target_length / measured_length
    if not 0.5 <= ratio <= 2.0:
        raise ValueError(
            f"Refusing to scale across a {ratio:.2f}x length ratio. The "
            f"exponent is a midpoint guess and the error compounds. Measure "
            f"the new length directly instead."
        )
    return seconds * ratio**exponent


def compare_cards(
    seconds_per_model_by_gpu: dict[str, float],
    *,
    n_designs: int = N_EXPRESSED,
    num_models: int = 5,
    compile_seconds_per_chunk: float = 0.0,
    chunk_size: int = 5,
) -> list[Projection]:
    """One projection per card, cheapest per design first."""
    out = [
        Projection(
            gpu=gpu,
            n_designs=n_designs,
            num_models=num_models,
            seconds_per_model=secs,
            compile_seconds_per_chunk=compile_seconds_per_chunk,
            chunk_size=chunk_size,
        )
        for gpu, secs in seconds_per_model_by_gpu.items()
    ]
    return sorted(out, key=lambda p: p.per_design_usd)


def safe_chunk_size(
    seconds_per_model: float,
    *,
    num_models: int = 5,
    timeout_sec: float,
    salvage_margin_sec: float,
    weight_stage_sec: float = 120.0,
    safety: float = 0.75,
) -> int:
    """Largest chunk that should finish inside its own deadline.

    A chunk sized by guesswork rather than by measurement either wastes the
    amortisation that chunking exists for, or overruns its deadline on every
    single chunk and salvages partial results every time. The first fold run
    used CHUNK_SIZE=5 against a 7200s timeout without knowing the per-model
    cost, which at a realistic rate could not have fitted.

    ``safety`` leaves headroom because designs differ in length by up to 237
    residues, so the slowest design in a chunk costs meaningfully more than
    the median one, and it is the slowest that decides whether the chunk fits.

    Returns at least 1: a chunk of one design that overruns is a problem to
    solve by raising the timeout, not by chunking less.
    """
    if seconds_per_model <= 0:
        raise ValueError("seconds_per_model must be positive")
    if timeout_sec <= salvage_margin_sec:
        raise ValueError(
            f"timeout {timeout_sec}s does not exceed salvage margin "
            f"{salvage_margin_sec}s, so no time remains to fold anything"
        )

    usable = (timeout_sec - salvage_margin_sec) * safety - weight_stage_sec
    per_design = seconds_per_model * num_models
    if usable <= per_design:
        return 1
    return max(1, int(usable // per_design))
