#!/usr/bin/env bash
# Package kaggle_kernel/ for `kaggle kernels push`.
set -euo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"

DEST="kaggle_kernel"
rm -rf "$DEST/prism" "$DEST/exp13_*.py" "$DEST/exp14_*.py" "$DEST/exp15_*.py"

# Package
cp -r prism "$DEST/"

# Experiments
cp experiments/exp13_domain_esg.py "$DEST/"
cp experiments/exp14_snr_pr.py "$DEST/"
cp experiments/exp15_gdn_anchors.py "$DEST/"

echo "kaggle_kernel/ ready:"
find "$DEST" -maxdepth 2 -type f | sort
echo ""
echo "Push:  kaggle kernels push -p $DEST/"
