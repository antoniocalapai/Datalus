#!/usr/bin/env python3
"""
HomeCagePaper · Biomechanical / inference-quality checks — per-check details.

Loads cached events from tables/validation/biomech_events.csv (produced by
validation.py) and the detection table from curated/master.h5, and writes one
dedicated multi-panel figure per check into figures/biomech/.

Each figure is a 2x2 grid:
  (top-left)      rate per session (% of detections or events / min),
  (top-right)     severity distribution of the triggering metric,
  (bottom-left)   breakdown by camera / animal / session duration,
  (bottom-right)  schematic example of the violation pattern.

The rule and its rationale sit in a compact banner underneath.

Run:
    python3 HomeCagePaper/biomech_details.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _style import apply_style, W_DOUBLE, PAL_WONG, ANIMALS  # noqa: E402

CURATED = HERE / "curated" / "master.h5"
EVENTS  = HERE / "tables"  / "validation" / "biomech_events.csv"
SUMMARY = HERE / "tables"  / "validation" / "biomech_summary.csv"
OUT_DIR = HERE / "figures" / "biomech"
OUT_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

# ─── Rule definitions ────────────────────────────────────────────────────────
RULES = {
    "id_swap": {
        "title":    "ID change (2D)",
        "subtitle": "bbox persists across frames, animal label flips",
        "rule":     "IoU(bbox_t, bbox_{t+1}) ≥ 0.40   and   name_t ≠ name_{t+1}   (Δt ≤ 2 frames)",
        "why":      ("The same body occupies essentially the same pixels from one frame to the next, "
                     "but the tracker reassigns it to the other monkey. High IoU rules out a real "
                     "re-detection — the swap is an identity error in the tracker, not a physical event."),
        "metric_label": "IoU of consecutive bboxes",
    },
    "overlap": {
        "title":    "Overlap (3D)",
        "subtitle": "two animals triangulated to the same place",
        "rule":     "‖centroid_Elm − centroid_Jok‖ < 15 cm   in the same frame",
        "why":      ("Elm and Jok resolve to overlapping 3D centroids — both IDs land on the same body. "
                     "This can be a real event (grooming, huddling) or a tracker error fitting two IDs "
                     "to one animal; the metric alone cannot disambiguate. The 15 cm cutoff is tight "
                     "enough that true body centroids rarely coincide that closely."),
        "metric_label": "inter-animal distance (cm)",
    },
    "teleport": {
        "title":    "Teleport (3D)",
        "subtitle": "centroid jumps farther than any real motion allows",
        "rule":     "‖centroid_{t+Δ} − centroid_t‖ > 80 cm   with   Δ ≤ 5 frames (≤ 1 s)",
        "why":      ("A macaque cannot translate its centre of mass more than ~80 cm within a single "
                     "second inside this cage (~4 m/s). A jump above that threshold is a triangulation "
                     "failure or an ID swap across frames, not a real movement."),
        "metric_label": "inter-frame centroid jump (cm)",
    },
    "size_jump": {
        "title":    "Growth spurt (3D)",
        "subtitle": "skeleton size changes impossibly fast",
        "rule":     "|size_{t+Δ} − size_t| / size_t > 50 %   with   size ≥ 5 cm and Δ ≤ 5 frames",
        "why":      ("The body diagonal of the 8 kept keypoints is a rough skeleton size. Between "
                     "consecutive frames (0.2 s) it cannot change by more than ~50 %. A spike usually "
                     "means a keypoint jumped to the wrong animal or the triangulation picked up a "
                     "spurious ray."),
        "metric_label": "frame-to-frame size change (%)",
    },
}

CHECK_COLOR = {
    "id_swap":     PAL_WONG["warn"],
    "overlap":     PAL_WONG["neutral"],
    "teleport":    PAL_WONG["accent"],
    "size_jump":   PAL_WONG["Jok"],
}


# ─── Data loading ────────────────────────────────────────────────────────────
def load_events():
    if not EVENTS.exists():
        raise FileNotFoundError(
            f"{EVENTS} not found. Run validation.py first to generate it.")
    ev = pd.read_csv(EVENTS)
    ev["session"] = ev["session"].astype(str)
    return ev


def load_summary():
    s = pd.read_csv(SUMMARY)
    s["session"] = s["session"].astype(str)
    return s


def session_detection_totals():
    det = pd.read_hdf(CURATED, "detections")
    det["session"] = det["session"].astype(str)
    return det.groupby("session").size().rename("n_det")


def session_order(summary_df):
    return sorted(summary_df["session"].unique())


# ─── Plot helpers ────────────────────────────────────────────────────────────
def _save(fig, stem):
    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    print(f"  {stem}.png")


def _open_figure():
    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.75))
    gs = fig.add_gridspec(
        2, 2,
        left=0.07, right=0.98, top=0.92, bottom=0.22,
        wspace=0.30, hspace=0.55,
    )
    return fig, gs


def _add_rule_banner(fig, spec, color):
    ax = fig.add_axes([0.04, 0.02, 0.94, 0.15])
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                           facecolor=color, alpha=0.07,
                           edgecolor=color, linewidth=0.5))
    ax.text(0.015, 0.82, "Rule",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7, fontweight="bold")
    ax.text(0.07, 0.82, spec["rule"],
            transform=ax.transAxes, ha="left", va="center",
            fontsize=7, family="monospace")
    ax.text(0.015, 0.48, "Why",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7, fontweight="bold")
    ax.text(0.07, 0.48, spec["why"],
            transform=ax.transAxes, ha="left", va="top",
            fontsize=6.5, linespacing=1.35, wrap=True)


def _per_session_bar(ax, rates, color, ylabel):
    sessions = list(rates.index)
    x = np.arange(len(sessions))
    ax.bar(x, rates.values, 0.72, color=color,
           edgecolor="black", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=90, fontsize=5.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("session")
    sns.despine(ax=ax)


def _severity_hist(ax, values, color, xlabel, bins=40, log_x=False,
                   xlim=None, threshold=None, threshold_label=None):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if log_x:
        values = values[values > 0]
        ax.set_xscale("log")
        edges = np.logspace(np.log10(max(values.min(), 1e-3)),
                            np.log10(values.max()), bins)
    else:
        edges = bins
    ax.hist(values, bins=edges, color=color, edgecolor="black", linewidth=0.3)
    if threshold is not None:
        ax.axvline(threshold, color="black", lw=0.6, ls="--",
                   label=threshold_label)
        if threshold_label:
            ax.legend(loc="upper right", fontsize=6, handlelength=1.2)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("events")
    sns.despine(ax=ax)


def _style_example(ax, title="Example"):
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.4); s.set_color("0.7")
    ax.set_title(title)


def _draw_skeleton(ax, cx, cy, scale=1.0, color="black", lw=0.8, alpha=1.0):
    """Minimal stick figure: head + torso + 2 arms + 2 legs around (cx, cy)."""
    s = scale
    # torso
    ax.plot([cx, cx], [cy + 1.2*s, cy - 1.0*s], color=color, lw=lw, alpha=alpha)
    # head
    ax.add_patch(Circle((cx, cy + 1.6*s), 0.4*s, facecolor=color,
                        edgecolor="black", lw=0.4, alpha=alpha))
    # arms
    ax.plot([cx, cx - 0.9*s], [cy + 0.8*s, cy - 0.2*s],
            color=color, lw=lw, alpha=alpha)
    ax.plot([cx, cx + 0.9*s], [cy + 0.8*s, cy - 0.2*s],
            color=color, lw=lw, alpha=alpha)
    # legs
    ax.plot([cx, cx - 0.5*s], [cy - 1.0*s, cy - 2.2*s],
            color=color, lw=lw, alpha=alpha)
    ax.plot([cx, cx + 0.5*s], [cy - 1.0*s, cy - 2.2*s],
            color=color, lw=lw, alpha=alpha)


# ─── Schematic examples ──────────────────────────────────────────────────────
def example_id_swap(ax):
    """Two frames: same bbox contents, label flips Elm → Jok."""
    _style_example(ax, "Example")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    for i, (x0, label, col) in enumerate([(0.6, "Elm", PAL_WONG["Elm"]),
                                          (5.2, "Jok", PAL_WONG["Jok"])]):
        ax.add_patch(Rectangle((x0, 1.0), 3.6, 4.0,
                               fill=False, edgecolor=col, lw=1.2))
        _draw_skeleton(ax, x0 + 1.8, 3.0, scale=0.75, color="0.25", lw=1.0)
        ax.text(x0 + 0.1, 5.15, label, color=col, fontsize=7, fontweight="bold")
        ax.text(x0 + 1.8, 0.55, f"frame t{'+1' if i else ''}",
                ha="center", fontsize=6, color="0.3")

    ax.annotate("", xy=(5.1, 3.0), xytext=(4.3, 3.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.text(4.7, 3.4, "IoU ≈ 1.0", ha="center", fontsize=6,
            style="italic", color="0.25")
    ax.text(4.7, 2.6, "label flips", ha="center", fontsize=6,
            fontweight="bold", color=PAL_WONG["warn"])


def example_overlap(ax):
    """Top-down cage view: two skeletons inside a 15 cm ball."""
    _style_example(ax, "Example (top-down)")
    ax.set_xlim(-5, 5); ax.set_ylim(-4, 4)

    # 15 cm collapse ball (in schematic units — label does the work)
    ax.add_patch(Circle((0, 0), 2.2, fill=False,
                        edgecolor=PAL_WONG["warn"], ls="--", lw=0.9))
    ax.text(0, 2.5, "< 15 cm",
            ha="center", fontsize=6, color=PAL_WONG["warn"], fontweight="bold")

    # two stick figures overlapping
    _draw_skeleton(ax, -0.6, 0.2, scale=0.55, color=PAL_WONG["Elm"], lw=1.0)
    _draw_skeleton(ax,  0.6, -0.1, scale=0.55, color=PAL_WONG["Jok"], lw=1.0,
                   alpha=0.85)
    ax.plot(-0.6, 0.2, "o", color=PAL_WONG["Elm"], ms=3,
            markeredgecolor="black", mew=0.3)
    ax.plot(0.6, -0.1, "o", color=PAL_WONG["Jok"], ms=3,
            markeredgecolor="black", mew=0.3)
    ax.plot([-0.6, 0.6], [0.2, -0.1], color="0.3", lw=0.5, ls=":")

    ax.text(-4.6, 3.4, "centroid Elm", color=PAL_WONG["Elm"], fontsize=6)
    ax.text(-4.6, 2.9, "centroid Jok", color=PAL_WONG["Jok"], fontsize=6)


def example_teleport(ax):
    """Side view: skeleton at A, dashed arrow to skeleton at B, > 80 cm."""
    _style_example(ax, "Example (side view)")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    _draw_skeleton(ax, 1.6, 3.0, scale=0.75, color=PAL_WONG["Elm"], lw=1.0)
    _draw_skeleton(ax, 8.4, 3.0, scale=0.75, color=PAL_WONG["Elm"], lw=1.0,
                   alpha=0.95)
    ax.text(1.6, 0.7, "frame t", ha="center", fontsize=6, color="0.3")
    ax.text(8.4, 0.7, "frame t+Δ", ha="center", fontsize=6, color="0.3")

    arr = FancyArrowPatch((2.5, 3.3), (7.5, 3.3),
                          arrowstyle="->", color=PAL_WONG["accent"],
                          lw=1.2, linestyle="--", mutation_scale=10)
    ax.add_patch(arr)
    ax.text(5.0, 3.9, "> 80 cm in ≤ 1 s",
            ha="center", fontsize=6.5,
            color=PAL_WONG["accent"], fontweight="bold")


def example_size_jump(ax):
    """Two adjacent frames: same animal, skeleton grows ≥ 50 % in one step."""
    _style_example(ax, "Example")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    _draw_skeleton(ax, 2.4, 2.6, scale=0.55, color=PAL_WONG["Jok"], lw=1.0)
    _draw_skeleton(ax, 7.6, 2.8, scale=1.05, color=PAL_WONG["Jok"], lw=1.1)
    ax.text(2.4, 0.7, "frame t  (size X)",
            ha="center", fontsize=6, color="0.3")
    ax.text(7.6, 0.7, "frame t+Δ  (size ≈ 1.5 X)",
            ha="center", fontsize=6, color="0.3")

    ax.annotate("", xy=(6.0, 3.0), xytext=(4.0, 3.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=0.8))
    ax.text(5.0, 3.7, "Δ size > 50 %",
            ha="center", fontsize=6.5,
            color=PAL_WONG["warn"], fontweight="bold")


EXAMPLES = {
    "id_swap":     example_id_swap,
    "overlap":     example_overlap,
    "teleport":    example_teleport,
    "size_jump":   example_size_jump,
}


# ─── Figures: one per check ──────────────────────────────────────────────────
def figure_id_swap(events, totals, sessions):
    kind = "id_swap"; spec = RULES[kind]; color = CHECK_COLOR[kind]
    ev = events[events["kind"] == kind].copy()

    n_per_sess = ev.groupby("session").size().reindex(sessions, fill_value=0)
    pct = (n_per_sess / totals.reindex(sessions)) * 100

    fig, gs = _open_figure()
    ax_sess = fig.add_subplot(gs[0, 0])
    ax_iou  = fig.add_subplot(gs[0, 1])
    ax_cam  = fig.add_subplot(gs[1, 0])
    ax_ex   = fig.add_subplot(gs[1, 1])

    _per_session_bar(ax_sess, pct, color, ylabel="% of detections")
    ax_sess.set_title("Rate per session")

    _severity_hist(ax_iou, ev["iou"], color,
                   xlabel=spec["metric_label"],
                   bins=30, xlim=(0.3, 1.02),
                   threshold=0.4, threshold_label="IoU = 0.40")
    ax_iou.set_title("Severity distribution")

    cam_counts = (ev.groupby("cam").size()
                    .reindex([102, 108, 113, 117], fill_value=0))
    x = np.arange(len(cam_counts))
    ax_cam.bar(x, cam_counts.values, 0.6, color=color,
               edgecolor="black", lw=0.3)
    ax_cam.set_xticks(x)
    ax_cam.set_xticklabels([str(int(c)) for c in cam_counts.index])
    ax_cam.set_xlabel("camera")
    ax_cam.set_ylabel("events (all sessions)")
    ax_cam.set_title("Per-camera breakdown")
    sns.despine(ax=ax_cam)

    EXAMPLES[kind](ax_ex)

    fig.suptitle(f"{spec['title']} — {spec['subtitle']}",
                 fontsize=9, fontweight="bold", y=0.98)
    _add_rule_banner(fig, spec, color)
    _save(fig, kind)


def figure_overlap(events, totals, sessions, summary):
    kind = "overlap"; spec = RULES[kind]; color = CHECK_COLOR[kind]
    ev = events[events["kind"] == kind].copy()

    rates = (summary[summary["kind"] == kind]
             .set_index("session")["rate_per_min"]
             .reindex(sessions, fill_value=0))
    minutes = (summary[summary["kind"] == kind]
               .set_index("session")["minutes"]
               .reindex(sessions, fill_value=0))
    counts  = (summary[summary["kind"] == kind]
               .set_index("session")["n"]
               .reindex(sessions, fill_value=0))

    fig, gs = _open_figure()
    ax_sess = fig.add_subplot(gs[0, 0])
    ax_d    = fig.add_subplot(gs[0, 1])
    ax_dur  = fig.add_subplot(gs[1, 0])
    ax_ex   = fig.add_subplot(gs[1, 1])

    _per_session_bar(ax_sess, rates, color, ylabel="events / min")
    ax_sess.set_title("Rate per session")

    if not ev.empty:
        _severity_hist(ax_d, ev["dist_cm"], color,
                       xlabel=spec["metric_label"],
                       bins=25, xlim=(0, 16),
                       threshold=15, threshold_label="15 cm cutoff")
    else:
        ax_d.text(0.5, 0.5, "no events", ha="center", va="center",
                  transform=ax_d.transAxes)
        ax_d.set_xticks([]); ax_d.set_yticks([])
    ax_d.set_title("Severity distribution")

    ax_dur.scatter(minutes.values, counts.values, s=28,
                   color=color, edgecolor="black", lw=0.4)
    for s, m, n in zip(sessions, minutes.values, counts.values):
        if n > 0:
            ax_dur.annotate(s, (m, n), xytext=(2, 2),
                            textcoords="offset points", fontsize=5)
    ax_dur.set_xlabel("session length (min)")
    ax_dur.set_ylabel("events")
    ax_dur.set_title("Events vs. session duration")
    sns.despine(ax=ax_dur)

    EXAMPLES[kind](ax_ex)

    fig.suptitle(f"{spec['title']} — {spec['subtitle']}",
                 fontsize=9, fontweight="bold", y=0.98)
    _add_rule_banner(fig, spec, color)
    _save(fig, kind)


def figure_teleport(events, totals, sessions, summary):
    kind = "teleport"; spec = RULES[kind]; color = CHECK_COLOR[kind]
    ev = events[events["kind"] == kind].copy()

    rates = (summary[summary["kind"] == kind]
             .set_index("session")["rate_per_min"]
             .reindex(sessions, fill_value=0))

    fig, gs = _open_figure()
    ax_sess = fig.add_subplot(gs[0, 0])
    ax_d    = fig.add_subplot(gs[0, 1])
    ax_an   = fig.add_subplot(gs[1, 0])
    ax_ex   = fig.add_subplot(gs[1, 1])

    _per_session_bar(ax_sess, rates, color, ylabel="events / min")
    ax_sess.set_title("Rate per session")

    _severity_hist(ax_d, ev["dist_cm"], color,
                   xlabel=spec["metric_label"],
                   bins=50, log_x=True,
                   threshold=80, threshold_label="80 cm cutoff")
    ax_d.set_title("Severity distribution (log-x)")

    per_animal = (ev.groupby("animal").size()
                    .reindex(ANIMALS, fill_value=0))
    x = np.arange(len(per_animal))
    ax_an.bar(x, per_animal.values, 0.6,
              color=[PAL_WONG[a] for a in ANIMALS],
              edgecolor="black", lw=0.3)
    ax_an.set_xticks(x); ax_an.set_xticklabels(ANIMALS)
    ax_an.set_xlabel("animal")
    ax_an.set_ylabel("events (all sessions)")
    ax_an.set_title("Per-animal breakdown")
    sns.despine(ax=ax_an)

    EXAMPLES[kind](ax_ex)

    fig.suptitle(f"{spec['title']} — {spec['subtitle']}",
                 fontsize=9, fontweight="bold", y=0.98)
    _add_rule_banner(fig, spec, color)
    _save(fig, kind)


def figure_size_jump(events, totals, sessions, summary):
    kind = "size_jump"; spec = RULES[kind]; color = CHECK_COLOR[kind]
    ev = events[events["kind"] == kind].copy()

    rates = (summary[summary["kind"] == kind]
             .set_index("session")["rate_per_min"]
             .reindex(sessions, fill_value=0))

    fig, gs = _open_figure()
    ax_sess = fig.add_subplot(gs[0, 0])
    ax_p    = fig.add_subplot(gs[0, 1])
    ax_an   = fig.add_subplot(gs[1, 0])
    ax_ex   = fig.add_subplot(gs[1, 1])

    _per_session_bar(ax_sess, rates, color, ylabel="events / min")
    ax_sess.set_title("Rate per session")

    p = ev["pct_change"].dropna().values
    p_clip = p[p < np.nanpercentile(p, 95)]
    _severity_hist(ax_p, p_clip, color,
                   xlabel=spec["metric_label"],
                   bins=40, xlim=(0, 300),
                   threshold=50, threshold_label="50 % cutoff")
    ax_p.set_title(f"Severity distribution (≤95th pct, max={p.max():.0f} %)")

    per_animal = (ev.groupby("animal").size()
                    .reindex(ANIMALS, fill_value=0))
    x = np.arange(len(per_animal))
    ax_an.bar(x, per_animal.values, 0.6,
              color=[PAL_WONG[a] for a in ANIMALS],
              edgecolor="black", lw=0.3)
    ax_an.set_xticks(x); ax_an.set_xticklabels(ANIMALS)
    ax_an.set_xlabel("animal")
    ax_an.set_ylabel("events (all sessions)")
    ax_an.set_title("Per-animal breakdown")
    sns.despine(ax=ax_an)

    EXAMPLES[kind](ax_ex)

    fig.suptitle(f"{spec['title']} — {spec['subtitle']}",
                 fontsize=9, fontweight="bold", y=0.98)
    _add_rule_banner(fig, spec, color)
    _save(fig, kind)


# ─── Overview figure: stacked bars per session, % of session detections ──────
def _fmt_count(n):
    if n >= 1_000_000: return f"{n/1e6:.1f} M"
    if n >= 1_000:     return f"{n/1e3:.0f} k"
    return str(int(n))


def figure_overview(summary, totals, sessions):
    kinds = ["id_swap", "overlap", "teleport", "size_jump"]
    det_totals = totals.reindex(sessions)
    n_by_sk = (summary.pivot(index="session", columns="kind", values="n")
                      .reindex(index=sessions, columns=kinds, fill_value=0))
    pct_by_sk = n_by_sk.div(det_totals.values, axis=0) * 100

    fig, ax = plt.subplots(figsize=(W_DOUBLE, W_DOUBLE * 0.45))
    x = np.arange(len(sessions))

    bottoms = np.zeros(len(sessions))
    for k in kinds:
        vals = pct_by_sk[k].values
        ax.bar(x, vals, 0.7,
               bottom=bottoms,
               color=CHECK_COLOR[k], edgecolor="black", lw=0.3,
               label=RULES[k]["title"])
        bottoms = bottoms + vals

    totals_pct = pct_by_sk.sum(axis=1).values
    for xi, pct, ntot in zip(x, totals_pct, det_totals.values):
        ax.text(xi, pct + 0.4,
                f"{pct:.1f}%\nn={_fmt_count(ntot)}",
                ha="center", va="bottom",
                fontsize=5.5, linespacing=1.05, color="0.15")

    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha="right")
    ax.set_ylabel("% of session detections")
    ax.set_title("All inference-quality checks — stacked % per session "
                 "(n = detections used as denominator)")
    ax.set_ylim(0, max(totals_pct) * 1.18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
              fontsize=6.5, frameon=False, handlelength=1.2)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, "overview")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    print(f"Writing biomech detail figures to {OUT_DIR}")
    events  = load_events()
    summary = load_summary()
    totals  = session_detection_totals()
    sessions = session_order(summary)
    print(f"  sessions: {len(sessions)}  · events: {len(events):,}")

    figure_id_swap(events, totals, sessions)
    figure_overlap(events, totals, sessions, summary)
    figure_teleport(events, totals, sessions, summary)
    figure_size_jump(events, totals, sessions, summary)
    figure_overview(summary, totals, sessions)

    print(f"\nDone. {len(list(OUT_DIR.glob('*.png')))} PNGs in {OUT_DIR}.")


if __name__ == "__main__":
    main()
