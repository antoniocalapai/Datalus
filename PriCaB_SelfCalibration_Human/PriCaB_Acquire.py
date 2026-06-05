#!/usr/bin/env python3
"""
PriCaB_Acquire.py  —  Synchronized multi-camera video acquisition
==================================================================

Records one MP4 per Hikrobot GigE camera, all named and laid out so
they drop straight into PriCaB_HumanCalib.py.

QUICK START
-----------
    python3 PriCaB_Acquire.py                       # session name = timestamp
    python3 PriCaB_Acquire.py MySession             # session name = MySession
    python3 PriCaB_Acquire.py MySession --fps 10    # override frame rate

Output
------
    Measurements/<session_name>/
        <session_name>_<cam_id>_<datetime>.mp4   (one per camera)

Then run the calibration directly on the output folder:
    python3 PriCaB_HumanCalib.py Measurements/<session_name>

Recording control
-----------------
    Press  Enter  to stop recording (clean shutdown, finalises all files).
    Press  Ctrl+C for emergency stop (files may be incomplete).

Trigger modes (--trigger)
--------------------------
    freerun   (default) — cameras run independently at the same FPS.
              Good enough for most uses; sync error ≈ 1 / (2 × fps).
    software  — main thread fires a MV_CC_SetCommandValue("TriggerSoftware")
              to every camera once per frame period.  Better sync on a
              single-machine setup, slight CPU overhead.

Camera IDs
----------
    Derived from the "UserDefinedName" field programmed into each camera
    in the Hikrobot MVS client (e.g. "102", "108", …).  If the field is
    empty the last digits of the serial number are used instead.

Dependencies
------------
    - libMvCameraControl.so / .dylib  (Hikrobot MVS SDK, installed system-wide)
    - opencv-python
    - numpy
"""

import argparse
import ctypes
import os
import queue
import signal
import sys
import threading
import time
from ctypes import *
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Locate MvImport ────────────────────────────────────────────────────────────
_BASE     = Path(__file__).parent
_MV_DIR   = _BASE.parent / "_docs" / "ABT_Software-main" / "utils" / "MvImport"
sys.path.insert(0, str(_MV_DIR))

# MvCameraControl_class.py hardcodes "libMvCameraControl.so" (Linux).
# Patch the load path for macOS before importing.
import platform as _platform
if _platform.system() == "Darwin":
    import ctypes as _ctypes
    _lib_candidates = [
        "/opt/MVS/lib/64/libMvCameraControl.dylib",
        "/usr/local/lib/libMvCameraControl.dylib",
        "libMvCameraControl.dylib",
    ]
    _loaded = False
    for _lib in _lib_candidates:
        try:
            _ctypes.cdll.LoadLibrary(_lib)
            _loaded = True
            break
        except OSError:
            pass
    if not _loaded:
        sys.exit(
            "[ERROR] libMvCameraControl not found on macOS.\n"
            "        Install the Hikrobot MVS SDK for macOS and retry.\n"
            f"        Tried: {_lib_candidates}"
        )

try:
    from MvCameraControl_class import MvCamera  # type: ignore
    from CameraParams_header import (             # type: ignore
        MV_CC_DEVICE_INFO_LIST, MV_CC_DEVICE_INFO,
        MV_GIGE_DEVICE, MV_FRAME_OUT, MVCC_INTVALUE, MVCC_FLOATVALUE,
        PixelType_Gvsp_YUV422_YUYV_Packed,
    )
    from CameraParams_const import MV_ACCESS_Exclusive  # type: ignore
except (ImportError, OSError) as e:
    sys.exit(
        f"[ERROR] Cannot load Hikrobot SDK.\n"
        f"        Install libMvCameraControl (MVS SDK) and ensure\n"
        f"        ABT_Software-main/utils/MvImport is present.\n"
        f"        Detail: {e}"
    )

# ── Output root ────────────────────────────────────────────────────────────────
MEASUREMENTS_DIR = _BASE / "Measurements"
MEASUREMENTS_DIR.mkdir(exist_ok=True)

# ── Defaults ───────────────────────────────────────────────────────────────────
DEFAULT_FPS       = 5       # frames per second
DEFAULT_BINNING   = 1       # 1 = no binning, 2 = 2×2
QUEUE_MAXSIZE     = 64      # per-camera frame buffer (frames)
CODEC             = "mp4v"  # OpenCV fourcc  (use 'avc1' for H.264 if available)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _bytes_to_str(arr) -> str:
    """Convert a ctypes c_ubyte array to a Python string, strip nulls."""
    return bytes(arr).rstrip(b"\x00").decode("utf-8", errors="replace").strip()


def _enumerate_cameras():
    """
    Return a list of (cam_id_str, MV_CC_DEVICE_INFO pointer) for all
    reachable GigE cameras.  cam_id comes from UserDefinedName; falls back
    to the last 3 digits of the serial number.
    """
    dev_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE, dev_list)
    if ret != 0:
        raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:08x}")

    n = dev_list.nDeviceNum
    if n == 0:
        return []

    cameras = []
    for i in range(n):
        info_ptr = dev_list.pDeviceInfo[i]
        info     = cast(info_ptr, POINTER(MV_CC_DEVICE_INFO)).contents

        # Try UserDefinedName first, then serial number tail
        gige = info.SpecialInfo.stGigEInfo
        user_name = _bytes_to_str(gige.chUserDefinedName)
        serial     = _bytes_to_str(gige.chSerialNumber)

        if user_name:
            cam_id = user_name
        else:
            cam_id = serial[-3:] if len(serial) >= 3 else serial or str(i)

        cameras.append((cam_id, info_ptr))

    cameras.sort(key=lambda x: x[0])
    return cameras


def _yuv422_to_bgr(frame_out):
    """Convert a YUV422 (YUYV) MV_FRAME_OUT to a BGR numpy array."""
    w = frame_out.stFrameInfo.nWidth
    h = frame_out.stFrameInfo.nHeight
    n = frame_out.stFrameInfo.nFrameLen
    raw = cast(frame_out.pBufAddr, POINTER(c_ubyte * n)).contents
    yuyv = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 2)
    return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)


# ══════════════════════════════════════════════════════════════════════════════
# Camera worker
# ══════════════════════════════════════════════════════════════════════════════

class CamWorker:
    """
    One instance per camera.  Runs two threads:
      _grab_thread  — calls MV_CC_GetImageBuffer in a tight loop
      _write_thread — reads from self.q and writes to VideoWriter
    """

    def __init__(self, cam_id, info_ptr, out_path, fps, binning, trigger_mode):
        self.cam_id      = cam_id
        self.info_ptr    = info_ptr
        self.out_path    = out_path
        self.fps         = fps
        self.binning     = binning
        self.trigger_mode = trigger_mode   # "freerun" | "software"

        self.cam         = MvCamera()
        self.q           = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self.stop_event  = threading.Event()
        self.writer      = None            # cv2.VideoWriter, set after first frame
        self._w = self._h = 0
        self.frame_count = 0
        self.payload_size = 0
        self.data_buf    = None
        self._grab_thr   = None
        self._write_thr  = None
        self.error       = None            # set if something goes wrong

    # ── Setup ──────────────────────────────────────────────────────────────────

    def open(self):
        info = cast(self.info_ptr, POINTER(MV_CC_DEVICE_INFO)).contents
        ret = self.cam.MV_CC_CreateHandle(info)
        if ret != 0:
            raise RuntimeError(f"cam {self.cam_id}: CreateHandle failed 0x{ret:08x}")

        ret = self.cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            self.cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"cam {self.cam_id}: OpenDevice failed 0x{ret:08x}")

        # Optimal packet size for GigE
        nps = self.cam.MV_CC_GetOptimalPacketSize()
        if nps > 0:
            self.cam.MV_CC_SetIntValue("GevSCPSPacketSize", nps)

        # Pixel format: YUV422 YUYV
        self.cam.MV_CC_SetEnumValue("PixelFormat", 0x02100032)

        # Binning
        self.cam.MV_CC_SetEnumValue("BinningHorizontal", self.binning)
        self.cam.MV_CC_SetEnumValue("BinningVertical",   self.binning)

        # Auto-exposure, auto-gain
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 2)                  # Continuous
        self.cam.MV_CC_SetIntValue("AutoExposureTimeUpperLimit", 100000)
        self.cam.MV_CC_SetIntValue("AutoExposureTimeLowerLimit", 100)
        self.cam.MV_CC_SetEnumValue("GainAuto", 2)                      # Continuous

        # Frame rate
        self.cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", c_bool(True))
        self.cam.MV_CC_SetFloatValue("AcquisitionFrameRate", float(self.fps))

        # Trigger
        if self.trigger_mode == "software":
            self.cam.MV_CC_SetEnumValueByString("TriggerMode",   "On")
            self.cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")
        else:
            self.cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")

        # Payload size
        stParam = MVCC_INTVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
        ret = self.cam.MV_CC_GetIntValue("PayloadSize", stParam)
        if ret != 0:
            raise RuntimeError(f"cam {self.cam_id}: GetIntValue(PayloadSize) failed 0x{ret:08x}")
        self.payload_size = stParam.nCurValue
        self.data_buf = (c_ubyte * self.payload_size)()

    def start(self):
        ret = self.cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"cam {self.cam_id}: StartGrabbing failed 0x{ret:08x}")
        self._grab_thr  = threading.Thread(target=self._grab_loop,  daemon=True, name=f"grab-{self.cam_id}")
        self._write_thr = threading.Thread(target=self._write_loop, daemon=True, name=f"write-{self.cam_id}")
        self._grab_thr.start()
        self._write_thr.start()

    # ── Fire software trigger (called from main thread) ────────────────────────

    def trigger(self):
        if self.trigger_mode == "software":
            self.cam.MV_CC_SetCommandValue("TriggerSoftware")

    # ── Grab thread ────────────────────────────────────────────────────────────

    def _grab_loop(self):
        stOut = MV_FRAME_OUT()
        memset(byref(stOut), 0, sizeof(stOut))
        timeout_ms = max(2000, int(3000 / self.fps))

        while not self.stop_event.is_set():
            ret = self.cam.MV_CC_GetImageBuffer(stOut, timeout_ms)
            if ret == 0 and stOut.pBufAddr is not None:
                if stOut.stFrameInfo.enPixelType == PixelType_Gvsp_YUV422_YUYV_Packed:
                    try:
                        bgr = _yuv422_to_bgr(stOut)
                        # Drop frame rather than block if writer is slow
                        if not self.q.full():
                            self.q.put(bgr)
                    except Exception as ex:
                        print(f"  [WARN] cam {self.cam_id}: frame convert error: {ex}")
                self.cam.MV_CC_FreeImageBuffer(stOut)
            else:
                if not self.stop_event.is_set():
                    time.sleep(0.005)

        self.q.put(None)   # sentinel

    # ── Write thread ───────────────────────────────────────────────────────────

    def _write_loop(self):
        fourcc = cv2.VideoWriter_fourcc(*CODEC)
        while True:
            frame = self.q.get()
            if frame is None:
                break
            if self.writer is None:
                self._h, self._w = frame.shape[:2]
                self.writer = cv2.VideoWriter(
                    str(self.out_path), fourcc, float(self.fps),
                    (self._w, self._h))
                if not self.writer.isOpened():
                    print(f"  [ERROR] cam {self.cam_id}: VideoWriter failed to open {self.out_path}")
                    self.writer = None
                    continue
            self.writer.write(frame)
            self.frame_count += 1

        if self.writer:
            self.writer.release()
            self.writer = None

    # ── Shutdown ───────────────────────────────────────────────────────────────

    def stop(self):
        self.stop_event.set()
        self.cam.MV_CC_StopGrabbing()
        if self._grab_thr:
            self._grab_thr.join(timeout=5)
        if self._write_thr:
            self._write_thr.join(timeout=10)

    def close(self):
        self.cam.MV_CC_CloseDevice()
        self.cam.MV_CC_DestroyHandle()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Record synchronized video from all Hikrobot GigE cameras.")
    ap.add_argument("session", nargs="?", default=None,
                    help="Session name (default: YYYYMMDD_HHMMSS)")
    ap.add_argument("--fps",      type=float, default=DEFAULT_FPS,
                    help=f"Target frame rate (default: {DEFAULT_FPS})")
    ap.add_argument("--binning",  type=int,   default=DEFAULT_BINNING, choices=[1, 2, 4],
                    help=f"Pixel binning (default: {DEFAULT_BINNING})")
    ap.add_argument("--trigger",  default="freerun", choices=["freerun", "software"],
                    help="Sync mode: freerun (default) or software trigger")
    ap.add_argument("--duration", type=float, default=None,
                    help="Auto-stop after N seconds (default: manual stop)")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = args.session if args.session else ts

    out_dir = MEASUREMENTS_DIR / session_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover cameras ───────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("PriCaB Acquire")
    print('═'*60)
    print(f"  Session  : {session_name}")
    print(f"  Output   : {out_dir}")
    print(f"  FPS      : {args.fps}  Binning: {args.binning}×  Trigger: {args.trigger}")
    print()
    print("  Enumerating cameras…")

    try:
        cameras = _enumerate_cameras()
    except RuntimeError as e:
        sys.exit(f"[ERROR] {e}")

    if not cameras:
        sys.exit("[ERROR] No Hikrobot GigE cameras found.  Check network and MVS SDK.")

    print(f"  Found {len(cameras)} camera(s): {[c for c,_ in cameras]}")
    print()

    # ── Build output paths ─────────────────────────────────────────────────────
    workers = []
    for cam_id, info_ptr in cameras:
        fname   = f"{session_name}_{cam_id}_{ts}.mp4"
        out_path = out_dir / fname
        w = CamWorker(cam_id, info_ptr, out_path,
                      fps=args.fps, binning=args.binning,
                      trigger_mode=args.trigger)
        workers.append(w)

    # ── Open all cameras ───────────────────────────────────────────────────────
    opened = []
    for w in workers:
        try:
            w.open()
            opened.append(w)
            print(f"  cam {w.cam_id}: opened  →  {w.out_path.name}")
        except RuntimeError as e:
            print(f"  cam {w.cam_id}: [FAILED] {e}")

    if not opened:
        sys.exit("[ERROR] No cameras could be opened.")

    # ── Start grabbing ─────────────────────────────────────────────────────────
    print()
    print("  Starting capture…  Press Enter to stop.\n")
    for w in opened:
        w.start()

    t_start = time.time()
    stop_flag = threading.Event()

    def _on_signal(sig, frame):
        print("\n  [!] Interrupted — stopping…")
        stop_flag.set()

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # Software-trigger loop (freerun: just sleeps at the right rate)
    frame_interval = 1.0 / args.fps

    def _input_waiter():
        try:
            input()
        except Exception:
            pass
        stop_flag.set()

    input_thr = threading.Thread(target=_input_waiter, daemon=True)
    input_thr.start()

    last_tick = time.perf_counter()
    while not stop_flag.is_set():
        if args.trigger == "software":
            for w in opened:
                w.trigger()

        elapsed = time.time() - t_start
        if args.duration and elapsed >= args.duration:
            break

        # Status line (overwrite in-place)
        counts = "  ".join(f"cam{w.cam_id}: {w.frame_count:>5} fr" for w in opened)
        print(f"\r  {elapsed:6.1f}s  |  {counts}   ", end="", flush=True)

        # Wait until next frame slot
        now  = time.perf_counter()
        wait = frame_interval - (now - last_tick)
        if wait > 0:
            time.sleep(wait)
        last_tick = time.perf_counter()

    print()
    print("\n  Stopping cameras…")

    # ── Shutdown ───────────────────────────────────────────────────────────────
    for w in opened:
        w.stop()
        w.close()

    # ── Summary ────────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print()
    print(f"{'═'*60}")
    print(f"  Recording complete  ({total_time:.1f} s)")
    print(f"{'─'*60}")
    any_ok = False
    for w in opened:
        size_mb = w.out_path.stat().st_size / 1e6 if w.out_path.exists() else 0
        if w.frame_count > 0:
            any_ok = True
            print(f"  cam {w.cam_id}: {w.frame_count:>6} frames  {size_mb:.1f} MB  →  {w.out_path.name}")
        else:
            print(f"  cam {w.cam_id}: [WARN] 0 frames written — file may be missing/empty")
    print(f"{'═'*60}")

    if any_ok:
        print(f"\n  Run calibration:")
        print(f"    python3 PriCaB_HumanCalib.py Measurements/{session_name}\n")


if __name__ == "__main__":
    main()
