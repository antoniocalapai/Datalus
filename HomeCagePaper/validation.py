#!/usr/bin/env python3
"""
HomeCagePaper · Section 1 — Validation

Loads curated/master.csv, generates publication-ready PNGs in
figures/validation/.

Functional, top-to-bottom in publication order. Re-runs overwrite.

Run:
    python3 HomeCagePaper/curate_data.py     # once
    python3 HomeCagePaper/validation.py
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sps

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _style import (                                        # noqa: E402
    apply_style, STRIP_KW,
    W_SINGLE, W_DOUBLE, PAL_WONG,
    ANIMALS, KP_NAMES, KEPT_KP_IDX, COCO_PAIRS_BODY,
    ROOM_DIAG_CM, MM_TO_CM,
)

CURATED  = HERE / "curated" / "master.h5"
OUT_DIR  = HERE / "_figures" / "validation"
TBL_DIR  = HERE / "tables"  / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

apply_style()
UNIT = "cm"


# ─── 1. Load and filter ──────────────────────────────────────────────────────
def load_master():
    return pd.read_hdf(CURATED, "keypoints")


def load_detections():
    """Per-camera RT detections from full sessions."""
    return pd.read_hdf(CURATED, "detections")


# ─── 2. Metric helpers ───────────────────────────────────────────────────────
def remove_outliers(df, value_col, group_cols, k=1.5):
    if df.empty:
        return df
    keep = pd.Series(False, index=df.index)
    for _, g in df.groupby(group_cols):
        q1 = g[value_col].quantile(0.25)
        q3 = g[value_col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        keep.loc[g.index] = (g[value_col] >= lo) & (g[value_col] <= hi)
    return df.loc[keep]


def errors_2d(master):
    """Pixel error per matched (cam, frame, animal, kp_idx) where both
    GT and RT have a 2D detection."""
    gt = master[master["model"] == "GT"][
        ["session", "cam", "frame", "animal", "kp_idx", "x_2d", "y_2d"]]
    rt = master[master["model"] == "RT"][
        ["session", "cam", "frame", "animal", "kp_idx", "x_2d", "y_2d"]]
    m = gt.merge(rt, on=["session", "cam", "frame", "animal", "kp_idx"],
                 suffixes=("_gt", "_rt"))
    m = m.dropna(subset=["x_2d_gt", "y_2d_gt", "x_2d_rt", "y_2d_rt"])
    m["err_px"] = np.hypot(m["x_2d_gt"] - m["x_2d_rt"],
                           m["y_2d_gt"] - m["y_2d_rt"])
    return m


def errors_3d(master):
    """3D error per (frame, animal, kp_idx). 3D is denormalised across cams,
    so dedupe to one row per (session, frame, animal, kp_idx)."""
    cols = ["session", "model", "frame", "animal", "kp_idx",
            "x_3d", "y_3d", "z_3d", "animal_size_cm"]
    de = master[cols].dropna(subset=["x_3d"]).drop_duplicates(
        subset=["session", "model", "frame", "animal", "kp_idx"])
    gt = de[de["model"] == "GT"].drop(columns="model")
    rt = de[de["model"] == "RT"].drop(columns="model")
    m = gt.merge(rt, on=["session", "frame", "animal", "kp_idx"],
                 suffixes=("_gt", "_rt"))
    dx = (m["x_3d_rt"] - m["x_3d_gt"]) * MM_TO_CM
    dy = (m["y_3d_rt"] - m["y_3d_gt"]) * MM_TO_CM
    dz = (m["z_3d_rt"] - m["z_3d_gt"]) * MM_TO_CM
    m["dx"], m["dy"], m["dz"] = dx, dy, dz
    m["err_cm"] = np.sqrt(dx**2 + dy**2 + dz**2)
    asize = m["animal_size_cm_gt"]
    m["animal_size_cm"] = asize
    m["err_pct_room"]   = m["err_cm"] / ROOM_DIAG_CM * 100
    m["err_pct_animal"] = np.where(asize.notna() & (asize > 0),
                                   m["err_cm"] / asize * 100, np.nan)
    return m


def detection_universe(master):
    """Per (session, cam, frame, animal): is the animal detected (any kp) by
    GT and/or RT? Used for confusion matrix and recall."""
    rows = []
    for (session, model, cam, frame, animal), g in master.groupby(
            ["session", "model", "cam", "frame", "animal"]):
        present = g["conf_2d"].notna().any()
        rows.append({"session": session, "model": model, "cam": cam,
                     "frame": int(frame), "animal": animal,
                     "present": bool(present)})
    return pd.DataFrame(rows)


def detection_confusion(det):
    """Per-camera TP/FN/FP/TN counts."""
    out = {}
    pivot = det.pivot_table(index=["session", "cam", "frame", "animal"],
                            columns="model", values="present",
                            aggfunc="any", fill_value=False).reset_index()
    for cam, g in pivot.groupby("cam"):
        tp = int((g["GT"] & g["RT"]).sum())
        fn = int((g["GT"] & ~g["RT"]).sum())
        fp = int((~g["GT"] & g["RT"]).sum())
        tn = int((~g["GT"] & ~g["RT"]).sum())
        rec = tp / max(1, tp + fn)
        out[int(cam)] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                         "recall": rec}
    return out


def id_confusion(det):
    """Counts of (GT animal → RT prediction in {Elm, Jok, none})."""
    rt_labels = ANIMALS + ["none"]
    total = {(g, r): 0 for g in ANIMALS for r in rt_labels}
    for (sess, cam, frame), g in det.groupby(["session", "cam", "frame"]):
        gt_set = set(g.loc[(g["model"] == "GT") & g["present"], "animal"])
        rt_set = set(g.loc[(g["model"] == "RT") & g["present"], "animal"])
        for ganimal in gt_set:
            if ganimal in rt_set:
                total[(ganimal, ganimal)] += 1
            else:
                others = [a for a in ANIMALS if a != ganimal and a in rt_set]
                if others:
                    total[(ganimal, others[0])] += 1
                else:
                    total[(ganimal, "none")] += 1
    return total


def bone_length_errors(df_3d):
    """Per-frame bone length differences (RT − GT) in cm, pooled per animal."""
    rows = []
    for (sess, animal, frame), g in df_3d.groupby(["session", "animal", "frame"]):
        gt = {int(r["kp_idx"]): np.array([r["x_3d_gt"], r["y_3d_gt"], r["z_3d_gt"]])
              for _, r in g.iterrows()}
        rt = {int(r["kp_idx"]): np.array([r["x_3d_rt"], r["y_3d_rt"], r["z_3d_rt"]])
              for _, r in g.iterrows()}
        for a, b in COCO_PAIRS_BODY:
            if a in gt and b in gt and a in rt and b in rt:
                d = (np.linalg.norm(rt[a] - rt[b]) -
                     np.linalg.norm(gt[a] - gt[b])) * MM_TO_CM
                rows.append({"animal": animal, "session": sess, "err": d})
    return pd.DataFrame(rows)


# ─── 3. Save helper ──────────────────────────────────────────────────────────
def _save(fig, stem):
    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=600)
    print(f"  {stem}.png")
    plt.close(fig)


# ─── 4. Panel drawers (mid-density: titles + legends, no in-plot annotations) ─
def draw_recall(ax, det):
    cm = detection_confusion(det)
    cams = sorted(cm)
    rec = [cm[c]["recall"] * 100 for c in cams]
    overall = sum(cm[c]["tp"] for c in cams) / max(1, sum(cm[c]["tp"] + cm[c]["fn"] for c in cams)) * 100
    ax.bar(range(len(cams)), rec, color=PAL_WONG["Elm"],
           edgecolor="black", linewidth=0.4, width=0.65)
    ax.axhline(overall, color=PAL_WONG["warn"], lw=0.8, ls="--",
               label=f"overall {overall:.0f}%")
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([f"{c}" for c in cams])
    ax.set_xlabel("camera")
    ax.set_ylabel("recall (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Detection recall")
    ax.legend(loc="lower right", fontsize=6)
    sns.despine(ax=ax)


def draw_id_confusion(ax, det):
    total = id_confusion(det)
    rt_labels = ANIMALS + ["none"]
    matrix = np.array([[total[(g, r)] for r in rt_labels] for g in ANIMALS],
                      dtype=float)
    row_norm = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f"{int(matrix[i,j])}\n({row_norm[i,j]*100:.0f}%)"
                       for j in range(matrix.shape[1])]
                      for i in range(matrix.shape[0])])
    sns.heatmap(row_norm, ax=ax, annot=annot, fmt="",
                cmap="rocket_r", cbar=True, linewidths=0.6, linecolor="white",
                xticklabels=rt_labels, yticklabels=ANIMALS,
                annot_kws={"fontsize": 6},
                cbar_kws={"label": "row fraction", "shrink": 0.7, "pad": 0.02})
    ax.set_xlabel("RT prediction")
    ax.set_ylabel("GT label")
    ax.set_title("ID confusion")


def draw_id_breakdown(ax, det):
    total = id_confusion(det)
    rt_labels = ANIMALS + ["none"]
    n_per = {g: sum(total[(g, r)] for r in rt_labels) for g in ANIMALS}
    correct = [total[(g, g)] / max(1, n_per[g]) * 100 for g in ANIMALS]
    swap    = [total[(g, [a for a in ANIMALS if a != g][0])] / max(1, n_per[g]) * 100
               for g in ANIMALS]
    none    = [total[(g, "none")] / max(1, n_per[g]) * 100 for g in ANIMALS]
    x = np.arange(len(ANIMALS))
    ax.bar(x, correct, 0.6, color=PAL_WONG["accent"],
           edgecolor="black", lw=0.4, label="correct")
    ax.bar(x, swap, 0.6, bottom=correct, color=PAL_WONG["warn"],
           edgecolor="black", lw=0.4, label="ID swap")
    ax.bar(x, none, 0.6, bottom=np.array(correct)+np.array(swap),
           color=PAL_WONG["neutral"], edgecolor="black", lw=0.4, label="missed")
    ax.set_xticks(x); ax.set_xticklabels(ANIMALS)
    ax.set_ylabel("% of GT detections")
    ax.set_ylim(0, 100)
    ax.set_title("Per-animal classification")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.0)
    sns.despine(ax=ax)


def draw_2d_per_camera(ax, df_2d):
    df = remove_outliers(df_2d, "err_px", ["animal", "cam"])
    cams = sorted(df["cam"].unique())
    sns.violinplot(data=df, x="cam", y="err_px", hue="animal",
                   palette=PAL_WONG, split=True, inner="quartile",
                   cut=0, density_norm="width", linewidth=0.6,
                   order=cams, ax=ax)
    sns.stripplot(data=df, x="cam", y="err_px", hue="animal",
                  palette=PAL_WONG, dodge=True, order=cams,
                  ax=ax, legend=False, **STRIP_KW)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([f"{c}" for c in cams])
    ax.set_xlabel("camera")
    ax.set_ylabel("2D error (px)")
    ax.set_title("2D error per camera")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2],
              loc="upper right", fontsize=6, handlelength=1.0)
    sns.despine(ax=ax)


def draw_3d_per_animal(ax, df_3d):
    df = remove_outliers(df_3d, "err_cm", ["animal"])
    sns.violinplot(data=df, x="animal", y="err_cm", hue="animal",
                   palette=PAL_WONG, inner="quartile", cut=0,
                   density_norm="width", linewidth=0.6,
                   order=ANIMALS, legend=False, ax=ax)
    sns.stripplot(data=df, x="animal", y="err_cm", hue="animal",
                  palette=PAL_WONG, order=ANIMALS, ax=ax,
                  legend=False, **STRIP_KW)
    ax.set_xlabel("")
    ax.set_ylabel(f"3D error ({UNIT})")
    ax.set_title("3D error per animal")
    sns.despine(ax=ax)


def draw_per_axis(ax, df_3d):
    long = pd.melt(df_3d, id_vars=["animal"], value_vars=["dx", "dy", "dz"],
                   var_name="axis", value_name="err")
    long["axis"] = long["axis"].map({"dx": "X", "dy": "Y", "dz": "Z"})
    long = remove_outliers(long, "err", ["animal", "axis"])
    sns.violinplot(data=long, x="axis", y="err", hue="animal",
                   palette=PAL_WONG, split=True, inner="quartile",
                   cut=0, density_norm="width", linewidth=0.6, ax=ax)
    sns.stripplot(data=long, x="axis", y="err", hue="animal",
                  palette=PAL_WONG, dodge=True, ax=ax,
                  legend=False, **STRIP_KW)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("")
    ax.set_ylabel(f"RT − GT ({UNIT})")
    ax.set_title("Per-axis signed error")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2],
              loc="upper right", fontsize=6, handlelength=1.0)
    sns.despine(ax=ax)


def draw_bones(ax, df_3d):
    bones = bone_length_errors(df_3d)
    if bones.empty:
        return
    bones = remove_outliers(bones, "err", ["animal"])
    sns.violinplot(data=bones, x="animal", y="err", hue="animal",
                   palette=PAL_WONG, inner="quartile", cut=0,
                   density_norm="width", linewidth=0.6,
                   order=ANIMALS, legend=False, ax=ax)
    sns.stripplot(data=bones, x="animal", y="err", hue="animal",
                  palette=PAL_WONG, order=ANIMALS, ax=ax,
                  legend=False, **STRIP_KW)
    ax.axhline(0, color="black", lw=0.4)
    ax.set_xlabel("")
    ax.set_ylabel(f"Δ bone length ({UNIT})")
    ax.set_title("Bone-length error")
    sns.despine(ax=ax)


def draw_room_ecdf(ax, df_3d):
    df = remove_outliers(df_3d, "err_pct_room", ["animal"])
    sns.ecdfplot(data=df, x="err_pct_room", hue="animal",
                 palette=PAL_WONG, ax=ax, lw=1.2)
    ax.set_xlabel("% room diagonal")
    ax.set_ylabel("cumulative fraction")
    ax.set_xlim(0, max(2, df["err_pct_room"].quantile(0.99)))
    ax.set_title(f"Error / room diagonal ({ROOM_DIAG_CM:.0f} cm)")
    sns.despine(ax=ax)


def cam_recall_any(det_universe):
    """From the validation universe, fraction of GT (frame, animal) events
    detected by ≥ 1 camera. Returns dict animal → recall%."""
    out = {}
    pivot = det_universe.pivot_table(
        index=["session", "frame", "animal"],
        columns="model", values="present",
        aggfunc="any", fill_value=False).reset_index()
    for animal, g in pivot.groupby("animal"):
        gt = g["GT"]
        rt = g["RT"]
        denom = int(gt.sum())
        num = int((gt & rt).sum())
        out[animal] = num / max(1, denom) * 100
    return out


def coincidence_distribution(detections):
    """For each (session, frame, animal) in the full sessions, how many
    cameras detected it. Returns DataFrame[session, animal, n_cams, count]."""
    g = (detections.groupby(["session", "frame", "animal"])["cam"]
                   .nunique().reset_index(name="n_cams"))
    counts = (g.groupby(["session", "animal", "n_cams"])
                .size().reset_index(name="count"))
    return counts


def data_loss_summary(detections, val_recall_any):
    """Per (session, animal) summary:
       - n_total          frames where ≥1 cam detected (after RT detector)
       - n_triangulable   frames with ≥2 cams (kept in /centroids)
       - lost_to_tri_pct  fraction lost because <2 cams agreed
       - val_recall_pct   detector recall from validation (≥1 cam vs GT)
       - effective_pct    val_recall × (1 − lost_to_tri_pct)
    """
    rows = []
    g = (detections.groupby(["session", "frame", "animal"])["cam"]
                   .nunique().reset_index(name="n_cams"))
    for (session, animal), sub in g.groupby(["session", "animal"]):
        n_total = len(sub)
        n_tri   = int((sub["n_cams"] >= 2).sum())
        lost_pct = (1 - n_tri / max(1, n_total)) * 100
        det_rec = val_recall_any.get(animal, np.nan)
        eff = (det_rec / 100) * (1 - lost_pct / 100) * 100 if not np.isnan(det_rec) else np.nan
        rows.append({
            "session":         session,
            "animal":          animal,
            "n_total":         n_total,
            "n_triangulable":  n_tri,
            "lost_to_tri_pct": lost_pct,
            "val_recall_pct":  det_rec,
            "effective_pct":   eff,
        })
    return pd.DataFrame(rows)


# ─── 4b. Biomechanical / inference-quality checks ───────────────────────────
#
# Each check is a small function that walks the curated tables and returns a
# DataFrame of flagged events with these mandatory columns:
#
#   session, kind, frame
#
# Plus any extra columns useful for that specific check.
#
# All flagged events are concatenated into a single DataFrame by
# `run_biomech_checks()`. Add new checks by writing another `check_*` function
# and registering it in `BIOMECH_CHECKS`.

def _bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(1.0, a1 + a2 - inter)


def check_id_swap(iou_thresh=0.4, max_gap=2):
    """Per-camera ID swap: a bbox in frame F overlaps a bbox in frame F+gap
    (IoU >= iou_thresh) but the monkey_name attached to it has changed.
    Operates on /detections (full-session 2D bboxes), so works even when
    only one animal is detected at a time."""
    det = pd.read_hdf(CURATED, "detections")
    det = det.dropna(subset=["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])
    rows = []
    for (sess, cam), grp in det.groupby(["session", "cam"]):
        per_frame = {}
        for _, r in grp.iterrows():
            per_frame.setdefault(int(r["frame"]), []).append(
                (r["animal"], (float(r["bbox_x1"]), float(r["bbox_y1"]),
                               float(r["bbox_x2"]), float(r["bbox_y2"]))))
        frames = sorted(per_frame)
        for i in range(len(frames) - 1):
            fa, fb = frames[i], frames[i + 1]
            if fb - fa > max_gap:
                continue
            for name_a, box_a in per_frame[fa]:
                best = (0.0, None, None)
                for name_b, box_b in per_frame[fb]:
                    iou = _bbox_iou(box_a, box_b)
                    if iou > best[0]:
                        best = (iou, name_b, box_b)
                if best[0] >= iou_thresh and best[1] is not None and best[1] != name_a:
                    rows.append({
                        "session":       sess,
                        "kind":          "id_swap",
                        "frame":         int(fa),
                        "frame_b":       int(fb),
                        "cam":           int(cam),
                        "animal_before": name_a,
                        "animal_after":  best[1],
                        "iou":           float(best[0]),
                    })
    return pd.DataFrame(rows)


def check_overlap(overlap_dist_cm=15.0):
    """Both animals' centroids inside the same `overlap_dist_cm` ball in a
    single frame — the tracker is fitting two IDs to one body."""
    ct = pd.read_hdf(CURATED, "centroids")
    rows = []
    for sess, sub in ct.groupby("session"):
        per_frame = {}
        for _, r in sub.iterrows():
            per_frame.setdefault(int(r["frame"]), {})[r["animal"]] = (
                float(r["x_cm"]), float(r["y_cm"]), float(r["z_cm"]))
        for f, d in per_frame.items():
            if "Elm" in d and "Jok" in d:
                ea = np.array(d["Elm"]); ja = np.array(d["Jok"])
                dist = float(np.linalg.norm(ea - ja))
                if dist < overlap_dist_cm:
                    rows.append({"session": sess, "kind": "overlap",
                                 "frame": int(f), "dist_cm": dist})
    return pd.DataFrame(rows)


def check_centroid_teleport(max_dist_cm=80.0, max_gap=5):
    """One animal's 3D centroid jumps more than `max_dist_cm` between
    consecutive frames — physically implausible motion (~ 4 m/s for a single
    0.2 s step)."""
    ct = pd.read_hdf(CURATED, "centroids")
    ct = ct.sort_values(["session", "animal", "frame"])
    rows = []
    for (sess, animal), grp in ct.groupby(["session", "animal"]):
        prev = None
        for _, r in grp.iterrows():
            cur = (int(r["frame"]),
                   float(r["x_cm"]), float(r["y_cm"]), float(r["z_cm"]))
            if prev is not None:
                gap = cur[0] - prev[0]
                if 0 < gap <= max_gap:
                    d = float(np.linalg.norm(np.array(cur[1:]) - np.array(prev[1:])))
                    if d > max_dist_cm:
                        rows.append({"session": sess, "kind": "teleport",
                                     "frame": prev[0], "frame_b": cur[0],
                                     "animal": animal, "dist_cm": d})
            prev = cur
    return pd.DataFrame(rows)


def check_size_jump(max_change_pct=50.0, max_gap=5):
    """The triangulated body 'size' (bbox diagonal of the kept keypoints)
    suddenly changes by more than `max_change_pct` % between consecutive
    frames — usually means one of the keypoints flipped to the wrong body."""
    ct = pd.read_hdf(CURATED, "centroids")
    ct = ct.sort_values(["session", "animal", "frame"])
    rows = []
    for (sess, animal), grp in ct.groupby(["session", "animal"]):
        prev = None
        for _, r in grp.iterrows():
            cur = (int(r["frame"]), float(r["size_cm"]))
            if prev is not None and 0 < cur[0] - prev[0] <= max_gap:
                if prev[1] > 5 and cur[1] > 5:
                    pct = abs(cur[1] - prev[1]) / prev[1] * 100
                    if pct > max_change_pct:
                        rows.append({"session": sess, "kind": "size_jump",
                                     "frame": prev[0], "frame_b": cur[0],
                                     "animal": animal, "pct_change": pct})
            prev = cur
    return pd.DataFrame(rows)


BIOMECH_CHECKS = (check_id_swap, check_overlap,
                  check_centroid_teleport, check_size_jump)


def run_biomech_checks():
    """Run every registered check and return one combined DataFrame."""
    parts = []
    for fn in BIOMECH_CHECKS:
        df = fn()
        print(f"  {fn.__name__:<28}{len(df):>6} events")
        if not df.empty:
            parts.append(df)
    if not parts:
        return pd.DataFrame(columns=["session", "kind", "frame"])
    return pd.concat(parts, ignore_index=True, sort=False)


def biomech_summary(events):
    """Per (session, kind) count + per-minute rate. Denominator is the total
    minutes of any-animal-detected time per session."""
    det = pd.read_hdf(CURATED, "detections")
    rows = []
    for sess in sorted(det["session"].unique()):
        # number of distinct (cam, frame) where at least one animal was detected
        n_frames = int(det[det["session"] == sess]
                       .groupby("frame")["animal"].nunique().shape[0])
        minutes = n_frames / 5.0 / 60.0
        for kind in ("id_swap", "overlap", "teleport", "size_jump"):
            n = int(((events["session"] == sess) &
                     (events["kind"] == kind)).sum()) if not events.empty else 0
            rows.append({"session": sess, "kind": kind,
                         "n": n, "minutes": minutes,
                         "rate_per_min": n / minutes if minutes > 0 else np.nan})
    return pd.DataFrame(rows)


SUPERPOWER_NAMES = {
    "id_swap":     "ID change",
    "teleport":    "teleport",
    "size_jump":   "growth spurt",
    # "overlap" deliberately dropped — too rare to plot
}
SUPERPOWER_COLORS = {
    "ID change":    PAL_WONG["warn"],
    "teleport":     PAL_WONG["accent"],
    "growth spurt": PAL_WONG["Jok"],
}
SUPERPOWER_DESC = {
    "ID change":    "2D bbox persists (IoU≥0.4)\nbut Elm/Jok label swaps",
    "teleport":     "centroid jumps\n>80 cm in ≤1 s",
    "growth spurt": "skeleton bbox diagonal\nchanges >50 % between frames",
}
SUPERPOWER_ORDER = ["ID change", "teleport", "growth spurt"]


def draw_inference_quality(ax, summary_df):
    """Side-by-side bars per session, % of detections per superpower."""
    if summary_df.empty:
        return
    det = pd.read_hdf(CURATED, "detections")
    totals = det.groupby("session").size().rename("n_det")

    df = summary_df.copy()
    df["power"] = df["kind"].map(SUPERPOWER_NAMES)
    df = df.dropna(subset=["power"])
    df = df.merge(totals, left_on="session", right_index=True)
    df["pct"] = df["n"] / df["n_det"] * 100

    pivot = df.pivot(index="session", columns="power",
                     values="pct").fillna(0)
    pivot = pivot.reindex(columns=SUPERPOWER_ORDER, fill_value=0)
    sessions = sorted(pivot.index)
    pivot = pivot.reindex(sessions)

    n_pow = len(SUPERPOWER_ORDER)
    n_sess = len(sessions)
    group_w = 0.78
    bar_w = group_w / n_pow
    x = np.arange(n_sess)
    for i, k in enumerate(SUPERPOWER_ORDER):
        offset = (i - (n_pow - 1) / 2) * bar_w
        ax.bar(x + offset, pivot[k].values, bar_w,
               color=SUPERPOWER_COLORS[k], edgecolor="black", lw=0.3, label=k)
    ax.set_xticks(x)
    ax.set_xticklabels(sessions, rotation=45, ha="right")
    ax.set_ylabel("% of detections")
    ax.set_title("Inference superpowers")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=7, frameon=False, handlelength=1.2)
    desc_lines = []
    for k in SUPERPOWER_ORDER:
        desc_lines.append(f"$\\bf{{{k}}}$  " + SUPERPOWER_DESC[k].replace("\n", "\n   "))
    desc_text = "\n\n".join(desc_lines)
    ax.text(1.02, 0.55, desc_text, transform=ax.transAxes,
            fontsize=6, va="top", ha="left",
            family="monospace", color="#222")
    sns.despine(ax=ax)


def draw_effective_recall(ax, loss_df, val_recall_any):
    """Stacked bar: detector recall × triangulation success → effective recall."""
    if loss_df.empty:
        return
    rows = []
    for animal in ANIMALS:
        sub = loss_df[loss_df["animal"] == animal]
        if sub.empty: continue
        det_rec = val_recall_any.get(animal, np.nan)
        n_total = sub["n_total"].sum()
        n_tri   = sub["n_triangulable"].sum()
        tri_succ = n_tri / max(1, n_total) * 100
        eff = (det_rec / 100) * (tri_succ / 100) * 100 if not np.isnan(det_rec) else np.nan
        rows.append((animal, det_rec, tri_succ, eff))

    x = np.arange(len(rows)); width = 0.6
    detector_loss         = np.array([100 - r[1] for r in rows])
    tri_loss_of_remaining = np.array([(100 - r[2]) * r[1] / 100 for r in rows])
    effective             = np.array([r[3] for r in rows])

    ax.bar(x, effective,             width, color=PAL_WONG["accent"],
           edgecolor="black", lw=0.4, label="kept (≥2 cams)")
    ax.bar(x, tri_loss_of_remaining, width, bottom=effective,
           color=PAL_WONG["neutral"], edgecolor="black", lw=0.4, label="1 cam only")
    ax.bar(x, detector_loss,         width, bottom=effective + tri_loss_of_remaining,
           color=PAL_WONG["warn"],    edgecolor="black", lw=0.4, label="detector miss")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows])
    ax.set_ylabel("% of GT events")
    ax.set_ylim(0, 100)
    ax.set_title("Effective recall")
    ax.legend(loc="upper right", fontsize=6, handlelength=1.0)
    sns.despine(ax=ax)


def draw_animal_norm(ax, df_3d):
    sub = df_3d.dropna(subset=["err_pct_animal"]).copy()
    sub = remove_outliers(sub, "err_pct_animal", ["animal"])
    sns.violinplot(data=sub, x="animal", y="err_pct_animal", hue="animal",
                   palette=PAL_WONG, inner="quartile", cut=0,
                   density_norm="width", linewidth=0.6,
                   order=ANIMALS, legend=False, ax=ax)
    sns.stripplot(data=sub, x="animal", y="err_pct_animal", hue="animal",
                  palette=PAL_WONG, order=ANIMALS, ax=ax,
                  legend=False, **STRIP_KW)
    ax.set_xlabel("")
    ax.set_ylabel("% animal size")
    ax.set_title("Error / animal size")
    sns.despine(ax=ax)


# ─── 5. Statistics ───────────────────────────────────────────────────────────
def write_stats(df_2d, df_3d):
    rows = []

    def mw(label, a, b):
        a = np.asarray(a); b = np.asarray(b)
        if a.size < 5 or b.size < 5:
            rows.append({"test": label, "n_Elm": int(a.size), "n_Jok": int(b.size),
                         "median_Elm": np.nan, "median_Jok": np.nan,
                         "U": np.nan, "p": np.nan}); return
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"test": label, "n_Elm": int(a.size), "n_Jok": int(b.size),
                     "median_Elm": float(np.median(a)),
                     "median_Jok": float(np.median(b)),
                     "U": float(u), "p": float(p)})

    mw("2D error (px)",
       df_2d.loc[df_2d["animal"] == "Elm", "err_px"],
       df_2d.loc[df_2d["animal"] == "Jok", "err_px"])
    mw("3D error (cm)",
       df_3d.loc[df_3d["animal"] == "Elm", "err_cm"],
       df_3d.loc[df_3d["animal"] == "Jok", "err_cm"])

    out = pd.DataFrame(rows)
    out.to_csv(TBL_DIR / "stats.csv", index=False)
    print("\nMann-Whitney U (two-sided):")
    print(out.to_string(index=False))


# ─── 6. Figure builders (chronological) ──────────────────────────────────────
def build_figure1(det):
    print("\n── figure1 — Detection performance ──")
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, W_DOUBLE * 0.34))
    draw_recall(axes[0], det)
    draw_id_confusion(axes[1], det)
    draw_id_breakdown(axes[2], det)
    fig.subplots_adjust(wspace=0.40)
    _save(fig, "figure1")
    for letter, drawer in (("a", draw_recall),
                           ("b", draw_id_confusion),
                           ("c", draw_id_breakdown)):
        fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
        drawer(ax, det)
        _save(fig, f"figure1_{letter}")


def build_figure2(df_2d, df_3d):
    print("\n── figure2 — Keypoint accuracy ──")
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.36),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})
    draw_2d_per_camera(axes[0], df_2d)
    draw_3d_per_animal(axes[1], df_3d)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "figure2")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_2d_per_camera(ax, df_2d); _save(fig, "figure2_a")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_3d_per_animal(ax, df_3d); _save(fig, "figure2_b")


def build_figure3(df_3d):
    print("\n── figure3 — 3D error structure ──")
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.36),
                             gridspec_kw={"width_ratios": [1.2, 1.0]})
    draw_per_axis(axes[0], df_3d)
    draw_bones(axes[1],    df_3d)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "figure3")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_per_axis(ax, df_3d); _save(fig, "figure3_a")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_bones(ax, df_3d);    _save(fig, "figure3_b")


def build_figure4(df_3d):
    print("\n── figure4 — Error normalisation ──")
    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, W_DOUBLE * 0.36))
    draw_room_ecdf(axes[0],   df_3d)
    draw_animal_norm(axes[1], df_3d)
    fig.subplots_adjust(wspace=0.32)
    _save(fig, "figure4")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_room_ecdf(ax, df_3d);   _save(fig, "figure4_a")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_animal_norm(ax, df_3d); _save(fig, "figure4_b")


def build_figure5(loss_df, val_recall_any):
    print("\n── figure5 — Effective recall ──")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.85))
    draw_effective_recall(ax, loss_df, val_recall_any)
    _save(fig, "figure5")


def build_figure6(summary_df):
    print("\n── figure6 — Superpowers (inference issues) ──")
    fig, ax = plt.subplots(figsize=(W_DOUBLE * 0.62, W_SINGLE * 0.95))
    draw_inference_quality(ax, summary_df)
    _save(fig, "figure6")


# ─── 7. main ─────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {CURATED.name}")
    master = load_master()
    df_2d  = errors_2d(master)
    df_3d  = errors_3d(master)
    det    = detection_universe(master)
    print(f"  master: {len(master)} rows  ·  2D pairs: {len(df_2d)}  ·  3D pairs: {len(df_3d)}")
    print(f"\nWriting figures to {OUT_DIR}")

    detections = load_detections()
    val_rec_any = cam_recall_any(det)
    loss_df    = data_loss_summary(detections, val_rec_any)
    print(f"  detections: {len(detections):>7}  ·  loss summary rows: {len(loss_df)}")

    build_figure1(det)
    build_figure2(df_2d, df_3d)
    build_figure3(df_3d)
    build_figure4(df_3d)
    build_figure5(loss_df, val_rec_any)

    print("\nRunning biomechanical / inference-quality checks:")
    events = run_biomech_checks()
    summary = biomech_summary(events)
    print("\nPer-session events / min:")
    print(summary.pivot(index="session", columns="kind",
                        values="rate_per_min").fillna(0).round(2)
          .to_string())
    events.to_csv(TBL_DIR / "biomech_events.csv", index=False)
    summary.to_csv(TBL_DIR / "biomech_summary.csv", index=False)
    build_figure6(summary)

    write_stats(df_2d, df_3d)
    loss_df.to_csv(TBL_DIR / "data_loss.csv", index=False)
    print(f"\n  data_loss.csv written ({len(loss_df)} rows)")

    print(f"\nDone. {len(list(OUT_DIR.glob('figure*.png')))} PNGs.")


if __name__ == "__main__":
    main()
