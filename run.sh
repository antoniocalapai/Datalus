#!/usr/bin/env bash
# Datalus calibration runner — logs all output to run.log
# Usage:  ./run.sh DatulusCalib_Step3_Stereo.py
#         ./run.sh DatulusCalib_Full.py --from 4

LOG="$SCRIPT_DIR/run.log"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: ./run.sh <script.py> [args...]"
    exit 1
fi

echo "======================================" | tee -a "$LOG"
echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"      | tee -a "$LOG"
echo "======================================" | tee -a "$LOG"

python "$SCRIPT_DIR/$@" 2>&1 | tee -a "$LOG"
