#!/usr/bin/env bash
# Хвост раунда E-45: дожидается основного прогона и доделывает то, что в нём не помещалось.
#   1) E45_SEEDNULL=8 --resume — 8 draw'ов того же семейства при дрейфе 0.12: это полоса
#      «разброс одного draw'а», без неё знак Δκ и величина gap16 у rand_* нечитаемы
#      (строк seednull_* в текущем JSON нет). Строки kind=sparse, кадровой дырой не задеты.
#   2) явная доводка исправленных dense-строк из /tmp/e45_finish.json: iid-проход пишет главный
#      JSON из своего .partial, то есть из состаренных строк, и перезаписал бы починку кадра,
#      если бы успел раньше внутрепроцессного merge'а доборки.
#   3) сводный вердикт exp45_verdicts.py по смёрженному результату.
# Повторы flat-правок со сдвигом seed в хвост НЕ включены: seed зашит в код (manual_seed(0)),
# а кэш flat-носителя общий на все seed'ы, так что сначала надо разнести кэш по seed'ам.
# Шаг 1 обязан проверять по метке в логе, а не по наличию процесса: pgrep ловит этот же скрипт.
set -uo pipefail
cd "$(dirname "$0")"
LOG=../results/exp45_damage_vs_structure.log
MAIN=../results/exp45_damage_vs_structure.json
FIN=../results/exp45_seednull.log
ROWS=/tmp/e45_finish.json
KEEP=/home/survivor/work/prism/results/exp45_finish_rows.json

while ! grep -q "сохранено: $MAIN" "$LOG" 2>/dev/null; do
  if ! pgrep -f "exp45_damage_vs_structure.py" > /dev/null; then
    echo "[tail] $(date +%H:%M:%S) основного прогона нет, метки завершения нет — выходить, см. $LOG"
    exit 1
  fi
  sleep 120
done
echo "[tail] $(date +%H:%M:%S) основной прогон завершён"

if grep -q "seednull_s0" "$FIN" 2>/dev/null; then
  echo "[tail] iid-семейство уже посчитано — пропускаю"
else
  echo "[tail] $(date +%H:%M:%S) E45_SEEDNULL=8 --resume (8 draw'ей при дрейфе 0.12)"
  E45_SEEDNULL=8 OMP_NUM_THREADS=4 ../.venv/bin/python exp45_damage_vs_structure.py --resume \
    >> "$FIN" 2>&1 || echo "[tail] iid-проход упал, см. $FIN"
fi

if [ -f "$ROWS" ]; then
  cp -f "$ROWS" "$KEEP"
  echo "[tail] $(date +%H:%M:%S) доводка исправленных dense-строк (главный JSON + .partial)"
  ../.venv/bin/python - "$MAIN" "$MAIN.partial" "$ROWS" <<'PY' || echo "[tail] merge упал"
import json
import sys

main_p, part_p, rows_p = sys.argv[1:4]
fix = json.load(open(rows_p))
for p in (main_p, part_p):
    try:
        d = json.load(open(p))
    except Exception as e:
        print(f"  {p}: нет файла ({e.__class__.__name__}) — пропускаю")
        continue
    by = {(r["name"], r["layer"]): r for r in fix["frame_layers"]}
    names = {r["name"] for r in fix["edits"]}
    kept = [r for r in d["edits"] if r["name"] not in names]
    kept += [r for r in fix["edits"] if r["name"] in names]
    d["edits"] = kept
    if "frame_layers" in d:
        for r in d["frame_layers"]:
            r.update(by.get((r["name"], r["layer"]), {}))
    done = {tuple(x) for x in d.get("done", [])} | {tuple(x) for x in fix["done"]}
    d["done"] = sorted(done)
    dl = d.get("done_layers", {})
    dl.update(fix["done_layers"])
    d["done_layers"] = dl
    dm = d.get("doc_metrics", {})
    dm.update(fix.get("doc_metrics", {}))
    d["doc_metrics"] = dm
    with open(p, "w") as f:
        json.dump(d, f)
    print(f"  {p}: правок {len(d['edits'])}, done {len(d['done'])}")
PY
else
  echo "[tail] $ROWS нет — доборка не дописалась, dense-строки остались состаренными"
fi

echo "[tail] $(date +%H:%M:%S) сводный вердикт"
OMP_NUM_THREADS=4 ../.venv/bin/python exp45_verdicts.py > ../results/exp45_verdicts.log 2>&1 \
  || echo "[tail] exp45_verdicts.py упал, см. ../results/exp45_verdicts.log"
echo "[tail] $(date +%H:%M:%S) готово"
