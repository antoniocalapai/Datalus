#!/usr/bin/env python3
# Datalous — simple, linear batch runner for ABT headless 2D prediction

import os, glob, re, subprocess, sys, getpass, tempfile, shlex

# ====== SIMPLE SETTINGS ======
PROCESS_ALL = 0  # 1 = reprocess everything; 0 = skip if any date token from src appears in any filename in OUT_ROOT

# SMB mount
SMB_SERVER = "storage2.dpz.lokal"
SMB_SHARE = "DPZ"
MOUNTPOINT = "/mnt/storage2_cnl"
SMB_VERS = "3.1.1"  # try "3.0" or "2.1" if your server requires it
SMB_SEC = "ntlmssp"

# Paths on the share
VIDEO_DIR = "KognitiveNeurowissenschaften/PriCaB/HomeCage/Videos/_automated/"
OUT_ROOT = "KognitiveNeurowissenschaften/PriCaB/HomeCage/Videos/_processed"

# ABT headless repo (module invocation)
ABT_VENV = "/home/cams/Desktop/ABT/ABT_venv/bin/python"
print("ABT venv Python:", ABT_VENV)  # quick sanity print
if not os.path.isfile(ABT_VENV):
    print(f"ERROR: ABT venv python not found: {ABT_VENV}", file=sys.stderr); sys.exit(1)

ABT_ROOT = "/home/cams/Desktop/ABT/ABT_Software-headless"
BOXPOSE = os.path.join(ABT_ROOT, "models/box_pose_detector/monkey_box_pose.pt")

# Prediction args
NUM_MONKEYS = "2"
BBOX_CONF = "0.2"
IOU_CONF = "0.75"
KPT_CONF = "0.4"

# Date tokens to detect processed counterparts
DATE_TOKEN_RE = re.compile(r"20\d{6}(?:\d{6})?")  # matches YYYYMMDD or YYYYMMDDHHMMSS starting with '20'
# =============================

# --- Ask DPZ creds (password hidden) ---
print("=== DPZ SMB login ===")
user_in = input(f"Username [{os.environ.get('USER', 'acalapai')}]: ").strip() or os.environ.get('USER', 'acalapai')
domain_in = input("Domain [DPZ]: ").strip() or "DPZ"
pwd_in = getpass.getpass("DPZ password (hidden): ")

# --- Prepare mountpoint ---
os.makedirs(MOUNTPOINT, exist_ok=True)

# --- Write secure temp creds file (deleted later) ---
fd, cred_path = tempfile.mkstemp(prefix="dpz-creds-", dir="/tmp")
os.close(fd)
with open(cred_path, "w") as f:
    f.write(f"username={user_in}\npassword={pwd_in}\ndomain={domain_in}\n")
os.chmod(cred_path, 0o600)

# --- Mount (may prompt for your LOCAL sudo password) ---
mount_src = f"//{SMB_SERVER}/{SMB_SHARE}"
mount_opts = f"vers={SMB_VERS},sec={SMB_SEC},credentials={cred_path},iocharset=utf8,uid={os.getuid()},gid={os.getgid()}"
cmd_mount = ["sudo", "mount", "-t", "cifs", mount_src, MOUNTPOINT, "-o", mount_opts]
print(f"\nMounting {mount_src} -> {MOUNTPOINT}")
if subprocess.run(cmd_mount).returncode != 0:
    try:
        os.remove(cred_path)
    except:
        pass
    print("Mount failed. Adjust SMB_VERS/SMB_SEC or check credentials.", file=sys.stderr)
    sys.exit(1)

# --- Delete the temp creds file ASAP (mount stays active) ---
try:
    os.remove(cred_path)
except:
    pass

# --- Compose absolute paths on the mounted share ---
video_dir_abs = os.path.join(MOUNTPOINT, VIDEO_DIR)
out_root_abs = os.path.join(MOUNTPOINT, OUT_ROOT)
os.makedirs(out_root_abs, exist_ok=True)

print("\n=== Paths ===")
print("Input Videos :", video_dir_abs)
print("Output Folder :", out_root_abs)
print("ABT virtual Environment: ", ABT_VENV)
print("ABT-Software :", ABT_ROOT)
print()

if not os.path.exists(BOXPOSE):
    print(f"ERROR: BoxPose weight not found at {BOXPOSE}", file=sys.stderr)
    sys.exit(1)

# --- Collect only .mp4 recursively ---
sources = sorted(glob.glob(os.path.join(video_dir_abs, "**", "*.mp4"), recursive=True))
print(f"Found {len(sources)} .mp4 videos.")

# --- Preload output names once for quick membership checks ---
out_names = [os.path.basename(x) for x in glob.glob(os.path.join(out_root_abs, "*"))]

# --- Process loop with date-token skip logic ---
total = len(sources)
for i, src in enumerate(sources, 1):
    base = os.path.basename(src)
    stem = os.path.splitext(base)[0]

    tokens = set(DATE_TOKEN_RE.findall(stem))
    if not tokens:
        tokens = {stem}  # fallback (rare)

    already = False
    if not PROCESS_ALL:
        for name in out_names:
            # if ANY token from the source name appears in ANY filename in OUT_ROOT, we treat it as processed
            if any(tok in name for tok in tokens):
                already = True
                break

    if already and not PROCESS_ALL:
        print(f"[{i}/{total}] skip  {base}  (token match in output)")
        continue

    print(f"[{i}/{total}] run   {base}  (tokens: {', '.join(sorted(tokens))})")

    # Build the module command (NO identification)
    VENV_PY = os.path.expanduser("~/Desktop/ABT/ABT_venv/bin/python")

    cmd = [
        ABT_VENV, "-m", "Modules_2D.box_pose_identification_2d",
        "-b", BOXPOSE,
        "-v", src,
        "-m", NUM_MONKEYS,
        "-o", out_root_abs,  # write directly to _processed
        "-sv", "-st",
        "-bc", BBOX_CONF,
        "-ic", IOU_CONF,
        "-kc", KPT_CONF,
    ]

    # Ensure module imports resolve (repo root + Modules_2D + its utils)
    env = os.environ.copy()
    extra_pp = f"{ABT_ROOT}:{ABT_ROOT}/Modules_2D:{ABT_ROOT}/Modules_2D/utils"
    env["PYTHONPATH"] = extra_pp + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

    print("   $", " ".join(shlex.quote(x) for x in cmd))
    r = subprocess.run(cmd, cwd=ABT_ROOT, env=env)
    if r.returncode != 0:
        print(f"   ! command failed (exit {r.returncode}) — continuing", file=sys.stderr)

print("\nAll done.")
print("(Note: the mount remains active; no credentials are stored on disk.)")
