"""Interface metrics from a ColabFold prediction. Pure, no Modal imports.

The pae_interaction definition here is deliberately Adaptyv's, not ours, so our
numbers are directly comparable to the ones they published. From their
compute_af2_metrics.py:

    pae_interaction = (pae[:binder_len, binder_len:].mean()
                       + pae[binder_len:, :binder_len].mean()) / 2

Chain order is binder first, target second. Their code states the assumption in
a comment; we assert it per design instead of trusting it.
"""

from __future__ import annotations

import numpy as np

from .target import assert_chain_layout


def interface_metrics(
    pae: np.ndarray,
    plddt: np.ndarray,
    expected_binder_length: int,
) -> dict[str, float]:
    """Compute the per-prediction interface metrics.

    ``pae`` is the square predicted-aligned-error matrix over the whole
    complex; ``plddt`` is per-residue. Both come straight from ColabFold output.
    """
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE matrix must be square, got shape {pae.shape}")
    if plddt.shape[0] != pae.shape[0]:
        raise ValueError(
            f"pLDDT has {plddt.shape[0]} residues but PAE has {pae.shape[0]}. "
            f"These must describe the same complex."
        )

    # Fails loudly on a chain-order flip rather than computing a plausible
    # looking number over the wrong block.
    b = assert_chain_layout(pae.shape[0], expected_binder_length)

    upper = pae[:b, b:]
    lower = pae[b:, :b]

    return {
        "pae_interface_mean": float((upper.mean() + lower.mean()) / 2.0),
        "pae_interface_min": float(min(upper.min(), lower.min())),
        "plddt_binder_mean": float(plddt[:b].mean()),
        "binder_length": int(b),
        "complex_length": int(pae.shape[0]),
    }


def aggregate_across_models(per_model: list[dict[str, float]]) -> dict[str, float]:
    """Combine the five AF2 models into the pre-specified primary estimator.

    The primary metric is the MEAN across models, not the best of them.
    Adaptyv read their numbers off rank_001, the highest-scoring of five, which
    is an optimistic estimator: a design need only fool one network out of five
    to post a good score. For a project about adversarial designs, a
    max-selection step inside the primary metric works against the experiment.

    The best-of-five value is kept as a secondary for comparability with the
    competition's published numbers, and the spread across models is recorded as
    a free ensemble-disagreement baseline. That spread is a weaker signal than
    EvoBind2's AF2-versus-AlphaFold-multimer check, because these five share an
    architecture and training data, so they disagree less and for less
    interesting reasons. It costs nothing once the five are paid for.
    """
    if not per_model:
        raise ValueError("No per-model metrics to aggregate")

    out: dict[str, float] = {"n_models": len(per_model)}
    keys = [
        "pae_interface_mean",
        "pae_interface_min",
        "plddt_binder_mean",
        "iptm",
        "ptm",
    ]
    for key in keys:
        vals = [m[key] for m in per_model if key in m]
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        out[f"{key}__mean"] = float(arr.mean())  # PRIMARY
        out[f"{key}__sd"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[f"{key}__min"] = float(arr.min())
        out[f"{key}__max"] = float(arr.max())

    # Secondary, for comparability only: the value from the model ColabFold
    # would rank first. Lower pae_interaction is better, higher ipTM is better,
    # and ColabFold ranks multimers by ipTM, so rank_001 is the max-ipTM model.
    if any("iptm" in m for m in per_model):
        best = max(per_model, key=lambda m: m.get("iptm", float("-inf")))
        out["pae_interface_mean__rank001"] = float(best["pae_interface_mean"])
        out["iptm__rank001"] = float(best.get("iptm", float("nan")))

    return out
