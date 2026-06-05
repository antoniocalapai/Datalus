#!/usr/bin/env python3
"""
HomeCagePaper · qc_inspector.py

Lightweight PyQt6 interface to browse biomechanical QC violations produced by
biomechanical_qc.py. Select a row, see the triangulated skeleton at that
frame with the offending bone/keypoint highlighted.

Run:
    python3 HomeCagePaper/qc_inspector.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# Reuse the pipeline from curate_data + biomechanical_qc
from curate_data import (
    SESSIONS_ROOT, CALIBRATIONS, CURATED,
    ANIMALS, KP_NAMES, KP_CONF_PRED,
    parse_full_session, projection_matrices, triangulate_session,
)
from biomechanical_qc import BONES, KP, BODY_KPS

OUT_DIR    = CURATED / "plausibility"
CALIB_PATH = CALIBRATIONS / "human_pose.json"

ANIMAL_COLOR = {
    "Elm": (0.27, 0.67, 1.00, 1.0),   # blue
    "Jok": (1.00, 0.47, 0.27, 1.0),   # orange
}
HIGHLIGHT = (1.0, 0.15, 0.15, 1.0)    # red for the offending element
DIMMED    = (0.75, 0.75, 0.75, 0.6)   # other animal / context

# Skeleton we'll draw (same as BONES)
SKELETON = [(a, b) for _, a, b in BONES]


# ─── Triangulation cache ────────────────────────────────────────────────────
class SessionCache:
    """Lazy-load + cache per-session triangulation."""
    def __init__(self, calib_path):
        calib = json.loads(Path(calib_path).read_text())
        self.Ps = projection_matrices(calib)
        self._cache = {}   # session -> {(frame, animal): {kp_idx: (x,y,z)|None}}

    def get(self, session):
        if session not in self._cache:
            sess_dir = SESSIONS_ROOT / session
            per_cam = parse_full_session(sess_dir, KP_CONF_PRED)
            self._cache[session] = triangulate_session(per_cam, self.Ps) if per_cam else {}
        return self._cache[session]


# ─── Main window ────────────────────────────────────────────────────────────
class Inspector(QtWidgets.QMainWindow):
    def __init__(self, v3, v2):
        super().__init__()
        self.setWindowTitle("HomeCage QC Inspector")
        self.resize(1500, 900)

        self.v3 = v3          # 3D violations DataFrame
        self.v2 = v2          # 2D violations DataFrame
        self.cache = SessionCache(CALIB_PATH)

        self._build_ui()
        self._populate_filters()
        self._apply_filter()

    # ── UI construction ────────────────────────────────────────────────────
    def _build_ui(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        root = QtWidgets.QHBoxLayout(w)

        # ── Left panel ─────────────────────────────────────────────────────
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 2)

        # filters
        filt = QtWidgets.QFormLayout()
        self.cb_layer   = QtWidgets.QComboBox()
        self.cb_session = QtWidgets.QComboBox()
        self.cb_check   = QtWidgets.QComboBox()
        self.cb_animal  = QtWidgets.QComboBox()
        self.sb_min_sev = QtWidgets.QDoubleSpinBox()
        self.sb_min_sev.setRange(0.0, 1e6); self.sb_min_sev.setDecimals(2)
        self.sb_min_sev.setValue(0.0)
        filt.addRow("Layer",      self.cb_layer)
        filt.addRow("Session",    self.cb_session)
        filt.addRow("Check",      self.cb_check)
        filt.addRow("Animal",     self.cb_animal)
        filt.addRow("Min severity", self.sb_min_sev)
        left.addLayout(filt)

        self.lbl_count = QtWidgets.QLabel("0 rows")
        left.addWidget(self.lbl_count)

        # table
        self.tbl = QtWidgets.QTableWidget()
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.tbl.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tbl.setSortingEnabled(True)
        left.addWidget(self.tbl, 1)

        # prev / next
        nav = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("◀ Prev")
        self.btn_next = QtWidgets.QPushButton("Next ▶")
        nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next)
        left.addLayout(nav)

        # ── Right panel: 3D view + details ─────────────────────────────────
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 3)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(pg.mkColor(20, 20, 25))
        self.view.opts['distance'] = 4500
        right.addWidget(self.view, 1)

        # axes + grid for context
        g = gl.GLGridItem(); g.setSize(4000, 4000); g.setSpacing(500, 500)
        self.view.addItem(g)

        self.txt_detail = QtWidgets.QTextEdit()
        self.txt_detail.setReadOnly(True)
        self.txt_detail.setFixedHeight(160)
        self.txt_detail.setFont(QtGui.QFont("Menlo", 11))
        right.addWidget(self.txt_detail)

        # wire signals
        for cb in (self.cb_layer, self.cb_session, self.cb_check, self.cb_animal):
            cb.currentIndexChanged.connect(self._apply_filter)
        self.sb_min_sev.valueChanged.connect(self._apply_filter)
        self.tbl.itemSelectionChanged.connect(self._on_select)
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        self.btn_next.clicked.connect(lambda: self._step(+1))

        # 3D scene items (managed)
        self._scene_items = []

    def _populate_filters(self):
        layers = []
        if not self.v3.empty: layers.append("3D")
        if not self.v2.empty: layers.append("2D")
        if not layers: layers = ["(none)"]
        self.cb_layer.addItems(["(all)"] + layers)

        # Build union of sessions / checks / animals across both tables
        def uniq(col):
            s = set()
            for d in (self.v3, self.v2):
                if not d.empty and col in d.columns:
                    s.update(d[col].dropna().astype(str).unique())
            return ["(all)"] + sorted(s)

        self.cb_session.addItems(uniq("session"))
        self.cb_check.addItems(uniq("check"))
        self.cb_animal.addItems(uniq("animal"))

    # ── Filtering ──────────────────────────────────────────────────────────
    def _current_df(self):
        """Return the filtered, severity-sorted DataFrame."""
        layer = self.cb_layer.currentText()
        frames = []
        if layer in ("(all)", "3D") and not self.v3.empty:
            d = self.v3.copy(); d["layer"] = "3D"; frames.append(d)
        if layer in ("(all)", "2D") and not self.v2.empty:
            d = self.v2.copy(); d["layer"] = "2D"; frames.append(d)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)

        for col, cb in (("session", self.cb_session),
                        ("check",   self.cb_check),
                        ("animal",  self.cb_animal)):
            v = cb.currentText()
            if v != "(all)":
                df = df[df[col].astype(str) == v]

        df = df[df["severity"].fillna(0) >= self.sb_min_sev.value()]
        df = df.sort_values("severity", ascending=False).reset_index(drop=True)
        return df

    def _apply_filter(self):
        df = self._current_df()
        self.lbl_count.setText(f"{len(df):,} rows")
        self._fill_table(df)
        self._current = df
        if len(df) > 0:
            self.tbl.selectRow(0)
        else:
            self._clear_scene()
            self.txt_detail.clear()

    def _fill_table(self, df):
        cols = ["layer", "session", "frame", "animal", "check", "detail",
                "value", "severity", "cam"]
        # 'cam' only exists in 2D; fill missing with ""
        self.tbl.setSortingEnabled(False)
        self.tbl.clear()
        self.tbl.setColumnCount(len(cols))
        self.tbl.setHorizontalHeaderLabels(cols)
        self.tbl.setRowCount(len(df))
        for r, row in df.iterrows():
            for c, key in enumerate(cols):
                v = row.get(key, "")
                if pd.isna(v): v = ""
                if isinstance(v, float):
                    txt = f"{v:.2f}"
                else:
                    txt = str(v)
                item = QtWidgets.QTableWidgetItem(txt)
                if key in ("frame", "value", "severity", "cam"):
                    item.setData(QtCore.Qt.ItemDataRole.EditRole,
                                 float(v) if v != "" else 0.0)
                self.tbl.setItem(r, c, item)
        self.tbl.resizeColumnsToContents()
        self.tbl.setSortingEnabled(True)

    # ── Selection / rendering ──────────────────────────────────────────────
    def _on_select(self):
        rows = self.tbl.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        if r >= len(self._current):
            return
        # map visible (possibly sorted) row back to our df by reading displayed cells
        session = self.tbl.item(r, 1).text()
        frame   = int(float(self.tbl.item(r, 2).text()))
        animal  = self.tbl.item(r, 3).text()
        check   = self.tbl.item(r, 4).text()
        detail  = self.tbl.item(r, 5).text()
        layer   = self.tbl.item(r, 0).text()
        value   = self.tbl.item(r, 6).text()
        sev     = self.tbl.item(r, 7).text()
        cam     = self.tbl.item(r, 8).text()

        self._render(session, int(frame), animal, check, detail)
        self.txt_detail.setPlainText(
            f"layer   : {layer}\n"
            f"session : {session}\n"
            f"frame   : {frame}\n"
            f"animal  : {animal}\n"
            f"check   : {check}\n"
            f"detail  : {detail}\n"
            f"value   : {value}\n"
            f"severity: {sev}\n"
            f"cam     : {cam}\n"
        )

    def _step(self, d):
        r = self.tbl.currentRow() + d
        if 0 <= r < self.tbl.rowCount():
            self.tbl.selectRow(r)

    # ── 3D rendering ───────────────────────────────────────────────────────
    def _clear_scene(self):
        for it in self._scene_items:
            self.view.removeItem(it)
        self._scene_items = []

    def _add(self, item):
        self.view.addItem(item); self._scene_items.append(item)

    def _render(self, session, frame, animal, check, detail):
        self._clear_scene()
        kps3d = self.cache.get(session)

        # Figure out which kp / bone to highlight
        hl_kp = None
        hl_bone = None
        if check == "bone_length" or check == "bone_symmetry":
            # detail is the bone name (or pair); map to one bone
            bone_name = detail
            # symmetry detail is the pair (torso/upper_arm/upper_leg); pick both sides
            hl_bone = bone_name
        elif check in ("keypoint_velocity", "out_of_room"):
            hl_kp = detail   # kp_name

        # Draw both animals (focus animal bright, the other dim)
        for who in ANIMALS:
            kp_dict = kps3d.get((frame, who))
            if not kp_dict:
                continue
            is_focus = (who == animal)
            base = ANIMAL_COLOR.get(who, (1, 1, 1, 1)) if is_focus else DIMMED
            self._draw_skeleton(kp_dict, base, focus=is_focus,
                                hl_kp=hl_kp if is_focus else None,
                                hl_bone=hl_bone if is_focus else None)

        # Auto-centre the view on the focus animal (or fallback to origin)
        focus = kps3d.get((frame, animal))
        if focus:
            pts = [np.asarray(p) for p in focus.values() if p is not None]
            if pts:
                c = np.mean(pts, axis=0)
                self.view.opts['center'] = pg.Vector(float(c[0]),
                                                    float(c[1]),
                                                    float(c[2]))
                self.view.update()

    def _draw_skeleton(self, kp_dict, base_color, focus=True,
                       hl_kp=None, hl_bone=None):
        # Points
        pts, colors, sizes = [], [], []
        for k in BODY_KPS:
            p = kp_dict.get(k)
            if p is None:
                continue
            pts.append(p)
            if focus and hl_kp is not None and KP_NAMES[k] == hl_kp:
                colors.append(HIGHLIGHT); sizes.append(18.0)
            else:
                colors.append(base_color); sizes.append(10.0)
        if pts:
            scat = gl.GLScatterPlotItem(
                pos=np.asarray(pts, dtype=float),
                color=np.asarray(colors, dtype=float),
                size=np.asarray(sizes, dtype=float),
                pxMode=True,
            )
            self._add(scat)

        # Bones
        sym_both = None
        if hl_bone in ("torso", "upper_leg"):
            sym_both = {"torso":    ("torso_left",  "torso_right"),
                        "upper_leg":("upper_leg_L", "upper_leg_R")}[hl_bone]

        for name, a, b in BONES:
            pa = kp_dict.get(a); pb = kp_dict.get(b)
            if pa is None or pb is None:
                continue
            is_hl = focus and (
                (hl_bone is not None and name == hl_bone)
                or (sym_both is not None and name in sym_both)
            )
            color = HIGHLIGHT if is_hl else base_color
            line = gl.GLLinePlotItem(
                pos=np.asarray([pa, pb], dtype=float),
                color=color,
                width=4.0 if is_hl else 2.0,
                antialias=True,
            )
            self._add(line)


# ─── Entry point ────────────────────────────────────────────────────────────
def load_violations():
    v3_path = OUT_DIR / "violations_3d.parquet"
    v2_path = OUT_DIR / "violations_2d.parquet"
    v3 = pd.read_parquet(v3_path) if v3_path.exists() else pd.DataFrame()
    v2 = pd.read_parquet(v2_path) if v2_path.exists() else pd.DataFrame()
    return v3, v2


def main():
    v3, v2 = load_violations()
    if v3.empty and v2.empty:
        print(f"No violations found in {OUT_DIR}.")
        print("Run: python3 HomeCagePaper/biomechanical_qc.py  first.")
        return
    app = QtWidgets.QApplication(sys.argv)
    w = Inspector(v3, v2)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
