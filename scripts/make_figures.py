#!/usr/bin/env python
"""Render the headline figure, in a light and a dark variant.

Run it with the figures dependency group, which is kept out of the main
dependency list so the Modal image definition (and therefore its cached hash)
never changes because of a plotting library:

    uv run --group figures python scripts/make_figures.py

Everything plotted comes from results/tables/per_design_metrics.csv and
data/designs.parquet, the same two files the statistics read, so the figure
cannot drift from the tables. The ROC curve helper is tested against auc() to
make sure the picture and the headline number agree.

Design notes, since they are not arbitrary:

- Two line panels rather than bars. The left one shows four curves sitting on
  the no-skill diagonal, which is the result. The right one shows the WHOLE
  cumulative-precision curve rather than the three pre-registered k values, so
  a reader can see where the ipSAE effect lives and where it decays instead of
  taking three points on trust.
- Series are ordered so the primary metric and the one exception take the two
  highest-contrast colour slots. The palette passed a CVD and contrast
  validator in both modes before anything was drawn.
- Identity is never carried by colour alone: every series has its own dash
  pattern, the legend shows it, and the right panel is directly labelled. That
  keeps the figure readable for colour-vision deficiency, in print, and in
  grayscale.
- Dark mode is a separate render against the dark surface, not an inverted
  image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from binder_bench.stats import auc, cumulative_precision, roc_curve

#: (column, label, lower_is_better, dash pattern). Order is the colour-slot
#: order: the pre-registered primary first, then the one metric that shows an
#: effect, so those two get the two most legible hues.
SERIES: tuple[tuple[str, str, str, bool, object], ...] = (
    # column, legend label, short in-plot label, lower_is_better, dash
    ("pae_interface_mean", "interface PAE (primary)", "interface PAE", True, "solid"),
    ("ipsae", "ipSAE", "ipSAE", False, (0, (5, 1.6))),
    ("pae_interface_contacts_mean", "interface PAE, contacts only",
     "contacts only", True, (0, (1.4, 1.6))),
    ("iptm", "ipTM", "ipTM", False, (0, (6, 1.6, 1.4, 1.6))),
)

#: Validated categorical slots 1, 2, 3, 4. Light and dark are separately
#: stepped for their own surface, not derived from one another.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_soft": "#52514e",
        "grid": "#e4e3df",
        "reference": "#8a8884",
        "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink_soft": "#c3c2b7",
        "grid": "#333330",
        "reference": "#7d7b76",
        "series": ("#3987e5", "#d95926", "#199e70", "#c98500"),
    },
}

K_PREREGISTERED = (10, 20, 50)


def load(metrics_csv: Path, designs_parquet: Path):
    """The same join the statistics use: expressed, non-control, in order."""
    df = pd.read_parquet(designs_parquet)
    pool = df[(~df["is_control"]) & df["expressed"]].copy()
    table = pd.read_csv(metrics_csv, float_precision="round_trip")
    table = table.set_index("design_id").loc[list(pool["design_id"])]
    binds = pool["binds"].to_numpy().astype(bool)
    return table, binds


def render(table, binds, theme_name: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import matplotlib.patheffects as pe

    t = THEMES[theme_name]
    n = len(binds)
    base_rate = binds.mean()

    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(11.2, 4.5), dpi=200, facecolor=t["surface"]
    )

    for ax in (axl, axr):
        ax.set_facecolor(t["surface"])
        ax.grid(True, color=t["grid"], linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["grid"])
        ax.tick_params(colors=t["ink_soft"], labelsize=9, length=3)

    # ---- left: ROC, the null result ---------------------------------------
    axl.plot(
        [0, 1], [0, 1], color=t["reference"], linewidth=1.3,
        linestyle=(0, (3, 3)), zorder=1,
    )
    axl.annotate(
        "no skill", xy=(0.63, 0.63), xytext=(0.73, 0.41),
        color=t["ink_soft"], fontsize=8.5, zorder=4,
        arrowprops=dict(
            arrowstyle="-", color=t["reference"], linewidth=0.8,
            shrinkA=3, shrinkB=2,
        ),
    )

    handles, labels = [], []
    for (col, label, _short, lower, dash), colour in zip(SERIES, t["series"]):
        scores = table[col].to_numpy(dtype=float)
        fpr, tpr = roc_curve(scores, binds, lower_is_better=lower)
        a = auc(scores, binds, lower_is_better=lower)
        (line,) = axl.plot(
            fpr, tpr, color=colour, linewidth=2.0, linestyle=dash,
            solid_capstyle="round", zorder=3,
        )
        handles.append(line)
        labels.append(f"{label}   AUC {a:.3f}")

    axl.set_xlim(0, 1)
    axl.set_ylim(0, 1)
    axl.set_aspect("equal")
    axl.set_xlabel("false positive rate", color=t["ink_soft"], fontsize=9.5)
    axl.set_ylabel("true positive rate", color=t["ink_soft"], fontsize=9.5)
    axl.set_title(
        "All four metrics sit on the no-skill line",
        color=t["ink"], fontsize=11.5, fontweight="bold", pad=10, loc="left",
    )

    # ---- right: cumulative bind rate, the one exception -------------------
    ks = np.arange(1, n + 1)
    axr.axhline(
        base_rate, color=t["reference"], linewidth=1.3, linestyle=(0, (3, 3)),
        zorder=1,
    )
    axr.annotate(
        f"base rate {base_rate:.1%}", xy=(300, base_rate),
        xytext=(132, base_rate + 0.042), color=t["ink_soft"], fontsize=8.5,
        zorder=5,
        path_effects=[pe.withStroke(linewidth=3.2, foreground=t["surface"])],
    )

    anchors = []
    for (col, _label, short, lower, dash), colour in zip(SERIES, t["series"]):
        scores = table[col].to_numpy(dtype=float)
        curve = cumulative_precision(scores, binds, lower_is_better=lower)
        axr.plot(
            ks, curve, color=colour, linewidth=2.0, linestyle=dash,
            solid_capstyle="round", zorder=3,
        )
        anchors.append([curve[9], short, col])

    # Direct labels go at the left edge, where the curves are furthest apart;
    # the right edge is where they all converge on the base rate. Two of them
    # start at the same value, so nudge upward to a minimum gap rather than
    # letting the text overprint.
    anchors.sort(key=lambda a: a[0])
    min_gap = 0.052
    for i in range(1, len(anchors)):
        if anchors[i][0] - anchors[i - 1][0] < min_gap:
            anchors[i][0] = anchors[i - 1][0] + min_gap
    halo = [pe.withStroke(linewidth=3.2, foreground=t["surface"])]
    for y, short, col in anchors:
        axr.annotate(
            short, xy=(11.2, y), xytext=(11.2, y),
            color=t["ink"], fontsize=9, va="center", zorder=5,
            fontweight="bold" if col == "ipsae" else "normal",
            path_effects=halo,
        )

    for k in K_PREREGISTERED:
        axr.plot([k], [-0.004], marker="^", markersize=4.5,
                 color=t["ink_soft"], clip_on=False, zorder=4)

    axr.set_xscale("log")
    axr.set_xlim(10, n)
    axr.set_ylim(0, 0.92)
    axr.set_xticks([10, 20, 50, 100, 200, 378])
    axr.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v)}")
    )
    axr.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )
    axr.set_xlabel(
        "top k by score  (▲ = the three k fixed in advance)",
        color=t["ink_soft"], fontsize=9.5,
    )
    axr.set_ylabel("share of the top k that bind", color=t["ink_soft"], fontsize=9.5)
    axr.set_title(
        "But the very top of an ipSAE ranking is enriched",
        color=t["ink"], fontsize=11.5, fontweight="bold", pad=10, loc="left",
    )

    leg = fig.legend(
        handles, labels, loc="lower center", ncol=2, frameon=False,
        fontsize=9, bbox_to_anchor=(0.5, -0.045), handlelength=3.0,
        columnspacing=2.4,
    )
    for text in leg.get_texts():
        text.set_color(t["ink_soft"])

    fig.suptitle(
        f"AlphaFold interface confidence vs measured binding, {n} de novo EGFR "
        f"designs ({int(binds.sum())} bind)",
        color=t["ink"], fontsize=10, y=1.0, x=0.009, ha="left",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.945))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"[figures] wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/tables/per_design_metrics.csv")
    ap.add_argument("--designs", default="data/designs.parquet")
    ap.add_argument("--out", default="results/figures")
    args = ap.parse_args()

    table, binds = load(Path(args.metrics), Path(args.designs))
    print(f"[figures] {len(binds)} designs, {int(binds.sum())} binders")
    for theme in ("light", "dark"):
        render(table, binds, theme, Path(args.out) / f"headline_{theme}.png")


if __name__ == "__main__":
    main()
