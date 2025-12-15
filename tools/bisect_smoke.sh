#!/usr/bin/env bash
set -e

# Config
LOG_FILE="/workspace/aimo/LM/lmdeploy/build/bisect_run.log"
BUILD_SCRIPT="/workspace/aimo/LM/lmdeploy/tools/build_for_bisect.sh"

echo "==================================================" >> "$LOG_FILE"
echo "Starting bisect step at $(date)" >> "$LOG_FILE"
echo "Commit: $(git rev-parse HEAD)" >> "$LOG_FILE"

# 1. Build
echo "Running build..." >> "$LOG_FILE"
bash "$BUILD_SCRIPT" >> "$LOG_FILE" 2>&1
BUILD_STATUS=$?

if [ $BUILD_STATUS -ne 0 ]; then
    echo "Build FAILED" >> "$LOG_FILE"
    exit 1 # Bad
fi
echo "Build PASSED" >> "$LOG_FILE"

# 2. Import Check
echo "Running import check..." >> "$LOG_FILE"
python - <<'PY' >> "$LOG_FILE" 2>&1
import sys
import os

# Ensure we import from the local source, not installed package if any
sys.path.insert(0, "/workspace/aimo/LM/lmdeploy")

try:
    from lmdeploy.lib import _turbomind as tm
    print(f"Import SUCCESS. Build ID: {tm.build_id()}")
except Exception as e:
    print(f"Import FAILED: {e}")
    sys.exit(1)
PY
IMPORT_STATUS=$?

if [ $IMPORT_STATUS -ne 0 ]; then
    echo "Import FAILED" >> "$LOG_FILE"
    exit 1 # Bad
fi

echo "Step SUCCESS (Good commit)" >> "$LOG_FILE"
exit 0 # Good
