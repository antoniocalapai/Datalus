#!/usr/bin/env python3
from pathlib import Path
import re
from collections import Counter
from datetime import datetime, timedelta
import calendar

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# CONFIG
# ============================================================
INFILE = Path("ABT_github_activity.rtf")
TOP_N_BRANCHES = 12

sns.set_theme(style="whitegrid", context="talk")

# ============================================================
# RTF -> lines
# ============================================================
rtf = INFILE.read_text(encoding="utf-8", errors="ignore")
rtf = rtf.replace("\\par", "\n").replace("\\line", "\n")

txt = re.sub(r"{\\\*\\[^}]*}", " ", rtf)   # drop hidden groups
txt = re.sub(r"[{}]", " ", txt)           # drop braces

def hex_to_char(m):
    try:
        return bytes.fromhex(m.group(1)).decode("latin-1")
    except Exception:
        return " "

txt = re.sub(r"\\'([0-9a-fA-F]{2})", hex_to_char, txt)
txt = re.sub(r"\\[a-zA-Z]+\d* ?", " ", txt)  # remove control words

txt = re.sub(r"[ \t]+", " ", txt)
txt = re.sub(r"\n\s+", "\n", txt)

lines = [l.strip() for l in txt.splitlines() if l.strip()]
lines = [l for l in lines if not re.fullmatch(r"[;.\w-]+;.*", l)]  # strip header-ish garbage

# ============================================================
# Helpers + regex
# ============================================================
BUL = ""  # bullet char in your export

def strip_trailing_backslash(s: str) -> str:
    return s[:-1].rstrip() if s.endswith("\\") else s

def parse_when(s: str):
    """
    Returns datetime for:
      - yesterday
      - N days ago
      - on 14 Nov
      - 15 Aug 2024
    """
    s = s.strip().lower()
    now = datetime.now()

    if s == "yesterday":
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    m = re.match(r"(\d+)\s+days?\s+ago", s)
    if m:
        days = int(m.group(1))
        return (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)

    m = re.match(r"on\s+(\d{1,2})\s+([a-z]{3})", s)
    if m:
        day = int(m.group(1))
        mon = m.group(2).title()
        dt = datetime.strptime(f"{day} {mon} {now.year}", "%d %b %Y")
        if dt.month > now.month:
            dt = dt.replace(year=now.year - 1)
        return dt

    try:
        return datetime.strptime(s.title(), "%d %b %Y")
    except Exception:
        return None

# 3-line push block:
push_head_re = re.compile(
    r"^(?P<author>[A-Za-z0-9_.-]+)\s*pushed\s*(?P<n>\d+)\s*commits?\s*to\s*(?P<branch>[A-Za-z0-9_.\-/]+)\s*"
    + re.escape(BUL) +
    r"\s*(?P<sha1>[0-9a-f]{6,40})\s*$",
    re.IGNORECASE
)

push_tail_re = re.compile(
    r"^(?P<sha2>[0-9a-f]{6,40})\s*"
    + re.escape(BUL) +
    r"\s*(?P<when>yesterday|\d+\s+days?\s+ago|on\s+\d{1,2}\s+[A-Za-z]{3}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\\?$",
    re.IGNORECASE
)

# 2-line create/delete block:
create_del_re = re.compile(
    r"^(?P<author>[A-Za-z0-9_.-]+)\s*(?P<action>created|deleted)\s*(?P<branch>[A-Za-z0-9_.\-/]+)\s*"
    + re.escape(BUL) +
    r"\s*(?P<when>yesterday|\d+\s+days?\s+ago|on\s+\d{1,2}\s+[A-Za-z]{3}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\\?$",
    re.IGNORECASE
)

# ============================================================
# Parse blocks -> counters + event list
# ============================================================
total_commits = 0
commits_by_author = Counter()
commits_by_branch = Counter()
commits_by_month = Counter()
created = Counter()
deleted = Counter()

# store push events so we can infer "module added" from messages
# each: {dt, author, branch, n, msg}
push_events = []

i = 0
matched = 0

while i < len(lines):
    l1 = lines[i]

    if len(l1) <= 2 and not l1.endswith("\\"):
        i += 1
        continue

    # CASE A: pushed commit block (3 lines)
    if i + 2 < len(lines):
        msg = strip_trailing_backslash(l1)
        l2 = strip_trailing_backslash(lines[i + 1])
        l3 = strip_trailing_backslash(lines[i + 2])

        m2 = push_head_re.match(l2)
        m3 = push_tail_re.match(l3)

        if msg and m2 and m3:
            author = m2.group("author")
            n = int(m2.group("n"))
            branch = m2.group("branch")
            dt = parse_when(m3.group("when"))

            total_commits += n
            commits_by_author[author] += n
            commits_by_branch[branch] += n
            if dt is not None:
                commits_by_month[dt.strftime("%Y-%m")] += n

            push_events.append(
                {"dt": dt, "author": author, "branch": branch, "n": n, "msg": msg}
            )

            matched += 1
            i += 3
            continue

    # CASE B: created/deleted block (2 lines)
    if i + 1 < len(lines):
        msg = strip_trailing_backslash(l1)
        l2 = strip_trailing_backslash(lines[i + 1])

        m = create_del_re.match(l2)
        if msg and m:
            action = m.group("action").lower()
            branch = m.group("branch")
            if action == "created":
                created[branch] += 1
            else:
                deleted[branch] += 1
            matched += 1
            i += 2
            continue

    i += 1

print(f"Lines: {len(lines)} | Matched events: {matched}")

# ============================================================
# Infer "major modules added" from commit messages
# ============================================================
# Edit/extend these rules to match what *you* consider major modules.
# The first match (earliest date) will be reported.
MODULE_RULES = [
    ("Initialization / first import", [r"\binitialization\b", r"\bcreate\s+readme\b"]),
    ("PTP sync pipeline",            [r"\bptp\b", r"\bsynchronized frames\b", r"\bsoftware triggering\b"]),
    ("Frame manager",                [r"\bframe_manager\b", r"\bframe manager\b", r"\bmultiple cams\b"]),
    ("Jetson frame manager",         [r"\bframe_manager_jetson\b", r"\bjetson\b"]),
    ("2D live processing",           [r"\b2d_live\b", r"\b2d\b.*\blive\b", r"\b2d modules\b"]),
    ("3D reconstruction/visualization", [r"\b3d\b", r"\b3d visualization\b", r"\b3d reconstruction\b", r"\b3d transformation\b"]),
    ("Identification model",         [r"\bidentification\b", r"\bget ready\b.*\bidentification\b"]),
    ("Labeling workflow",            [r"\blabel(ing)?\b", r"\bsplit data\b.*\btraining\b", r"\badd label\b"]),
    ("Automation / scheduling",      [r"\bautomation\b", r"\bschedul(ing|e)\b", r"\bevery day\b"]),
    ("Human pose version",           [r"\bhuman_version\b", r"\bhuman pose\b"]),
    ("Acquisition + processing module", [r"\bacqui(sition|sation)\b", r"\bacquisition and processing\b"]),
    ("Docker packaging",             [r"\bdocker\b", r"\bdocker version\b", r"\bdockerfile\b"]),
]

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

# Only events with a real date are reliable for "when"; keep relative if you want, but it’s noisy.
dated_events = [e for e in push_events if e["dt"] is not None]
dated_events.sort(key=lambda e: e["dt"])

module_first = {}
module_first_event = {}

for module_name, patterns in MODULE_RULES:
    cre = re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)
    for e in dated_events:
        if cre.search(e["msg"]):
            module_first[module_name] = e["dt"]
            module_first_event[module_name] = e
            break

print("\n=== Estimated module introduction (from commit messages) ===")
if not module_first:
    print("No module-introduction patterns matched. Adjust MODULE_RULES.")
else:
    for module_name, dt in sorted(module_first.items(), key=lambda kv: kv[1]):
        e = module_first_event[module_name]
        print(f"- {dt.strftime('%Y-%m-%d')}: {module_name}  |  '{e['msg']}'  (branch: {e['branch']}, author: {e['author']})")

# ============================================================
# Build last-12-month series (ensures complete x-axis)
# ============================================================
now = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
last12 = []
for k in range(11, -1, -1):
    y = now.year
    m = now.month - k
    while m <= 0:
        m += 12
        y -= 1
    last12.append(f"{y:04d}-{m:02d}")

month_vals = [commits_by_month.get(ym, 0) for ym in last12]
month_letters = [calendar.month_abbr[int(ym.split("-")[1])][0] for ym in last12]

# ============================================================
# Plot: 16:9 with 2 panels
# ============================================================
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.2, 1.0], hspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
sns.lineplot(x=list(range(len(month_vals))), y=month_vals, marker="o", ax=ax1)
ax1.set_title("Commits per month (last 12 months)")
ax1.set_xlabel("")
ax1.set_ylabel("Commits")
ax1.set_xticks(list(range(len(month_letters))))
ax1.set_xticklabels(month_letters)
ax1.grid(axis="y", alpha=0.25)

ax2 = fig.add_subplot(gs[1, 0])
branch_items = commits_by_branch.most_common(TOP_N_BRANCHES)
branch_names = [b for b, _ in branch_items]
branch_counts = [c for _, c in branch_items]
sns.barplot(y=branch_names, x=branch_counts, ax=ax2)
ax2.set_title(f"Commits per branch (top {TOP_N_BRANCHES})")
ax2.set_xlabel("Commits")
ax2.set_ylabel("")
ax2.grid(axis="x", alpha=0.25)

plt.tight_layout()
plt.show()

# ============================================================
# Metrics output
# ============================================================
print("\n=== Metrics ===")
print(f"Total commits (from log): {total_commits}")
print(f"Commits last 12 months (from reconstructed dates): {sum(month_vals)}")
print(f"Unique branches (with commits): {len(commits_by_branch)}")
print(f"Unique authors (with commits): {len(commits_by_author)}")

if commits_by_author:
    a, v = commits_by_author.most_common(1)[0]
    print(f"Top author: {a} ({v} commits)")
if commits_by_branch:
    b, v = commits_by_branch.most_common(1)[0]
    print(f"Top branch: {b} ({v} commits)")