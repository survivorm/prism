"""Минимальная проверка закрепления пула (PLAN «Запрос 11», воспроизводимость данных).

Запуск: .venv/bin/python experiments/_test_calib_pin.py
Без сети и без весов работают проверки 1–3; проверка 4 требует локального токенизатора
и закоммиченных пинов — иначе пропускается.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prism.calibration import (  # noqa: E402
    _pin_path,
    load_calibration_data,
    pool_fingerprint,
)

MODEL = "Qwen/Qwen3.5-0.8B"

# 1. отпечаток: стабилен, чувствителен к порядку, не зависит от типа строки
t = ["альфа", "бета", "гамма"]
assert pool_fingerprint(t) == pool_fingerprint(list(t))
assert pool_fingerprint(t) != pool_fingerprint(t[::-1]), "порядок пула обязан влиять"
assert pool_fingerprint(["a", "b"]) != pool_fingerprint(["ab"]), "границы строк значимы"

# 2. имя пина кодирует весь протокол (другой фильтр = другой пин, а не тихая подмена)
p1 = _pin_path("wiki", MODEL, 8, 64, 64, False)
p2 = _pin_path("wiki", MODEL, 8, 1, 64, False)
p3 = _pin_path("wiki", MODEL, 8, 64, 64, True)
p4 = _pin_path("code", MODEL, 8, 64, 64, False)
assert len({p1.name, p2.name, p3.name, p4.name}) == 4, (p1.name, p2.name, p3.name, p4.name)
assert p1.parent.name == "calib_pins"

# 3. формат пина, который пишем сам себе: читабельный JSON с протоколом и отпечатком
pin = _pin_path("wiki", MODEL, 8, 64, 64, False)
if pin.exists():
    meta = json.loads(pin.read_text(encoding="utf-8"))
    assert set(meta) >= {"protocol", "source", "fp", "texts"}, sorted(meta)
    assert len(meta["texts"]) == meta["protocol"]["n_samples"]
    assert meta["fp"] == pool_fingerprint(meta["texts"]), "отпечаток не сходится с текстами"
    print(f"3. пин {pin.name}: строк={len(meta['texts'])} fp={meta['fp']} ок")

    # 4. детерминизм: два прогона через пин дают бит-идентичные input_ids
    try:
        a = load_calibration_data(MODEL, max_length=64, min_length=64, n_samples=8, domain="wiki")
        b = load_calibration_data(MODEL, max_length=64, min_length=64, n_samples=8, domain="wiki")
    except Exception as e:  # нет весов/сети — не роняем проверку механизма
        print(f"4. пропущено ({type(e).__name__}: {str(e)[:80]})")
        raise SystemExit(0)
    ha = hashlib.sha256(str([x["input_ids"].tolist() for x in a]).encode()).hexdigest()
    hb = hashlib.sha256(str([x["input_ids"].tolist() for x in b]).encode()).hexdigest()
    assert ha == hb and len(a) == 8, (ha, hb, len(a))
    print(f"4. два чтения пина → идентичные input_ids ({ha[:16]}) ок")
else:
    print(f"3–4. пропущено: нет пина {pin.name}")

# 5. flattening-Kaggle-ядро: модуля нет на диске, `__file__` не определён. Именно на этом
# упал exp3 после введения пинов (NameError в _pin_path) — ядро собиралось и падало на загрузке
# данных, поэтому проверка входит в самопроверку, а не остаётся на совести ревью.
src = (Path(__file__).resolve().parent.parent / "prism" / "calibration.py").read_text()
g: dict = {"__name__": "calib_flattened"}
old_root = os.environ.pop("PRISM_CALIB_PIN_DIR", None)
try:
    exec(compile(src, "calibration.py(ядро)", "exec"), g)
    assert g["_pin_path"]("wiki", MODEL, 8, 64, 64, False) is None, "без __file__ пин-пути быть не должно"
    g["os"].environ["PRISM_CALIB_PIN_DIR"] = "/tmp/pins"
    q = g["_pin_path"]("wiki", MODEL, 8, 64, 64, False)
    assert q is not None and q.name.startswith("wiki__"), q
finally:
    if old_root is not None:
        os.environ["PRISM_CALIB_PIN_DIR"] = old_root
    g["os"].environ.pop("PRISM_CALIB_PIN_DIR", None)
print("5. без __file__ (flattening-ядро) пин-путь = None, с PRISM_CALIB_PIN_DIR = путь — ок")

print("OK")
