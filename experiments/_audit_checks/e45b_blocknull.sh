#!/usr/bin/env bash
# Обёртка для run_one.sh: cwd нужен experiments/ (импорт exp45b_analysis), root — артефакт.
set -eu
cd "$(dirname "$0")/.."
exec ../.venv/bin/python _audit_checks/e45b_blocknull.py
