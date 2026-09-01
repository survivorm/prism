#!/usr/bin/env bash
# Push the user-approved v2 kernels: exp16r (dose-curve) + exp17 (dual-cadence).
# Uses the REAL kaggle CLI (kagglesdk, /usr/bin/kaggle) — NOT the phantom `kaggle_cli` module.
# exp19 is built but NOT pushed (awaiting user approval).
# Run when the Kaggle weekly GPU quota (30h) resets.
set -u
cd "$(cd "$(dirname "$0")/.." && pwd)"
KAGGLE="${KAGGLE:-kaggle}"

push_one() {
  local m="$1"
  [ -f "kaggle_kernel/$m/kernel-metadata.json" ] || { echo "SKIP $m (no metadata)"; return 1; }
  local title
  title=$(grep -oP '"title":\s*"\K[^"]+' "kaggle_kernel/$m/kernel-metadata.json")
  echo "PUSH $m  title=\"$title\""
  (cd "kaggle_kernel/$m" && "$KAGGLE" kernels push -p . 2>&1 | grep -vE "outdated|consider upgrading")
}

echo "=== weekly quota check + push 16r + 17 ==="
push_one "kernel-exp16r"
push_one "kernel-exp17"
