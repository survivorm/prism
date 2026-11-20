"""Атомарная запись JSON-артефактов: без NaN/Infinity и без оборванных файлов.

Зачем (класс 3 аудита): `results/AT2e/at2e_smoke_r1.json` попал в git оборванным — запись
прервалась, а `json.dump` не жалуется на нефинитные float, потому что пишет литералы `NaN`/
`Infinity`, которые не являются JSON (строгий парсер их не примет, а читатель-питон примет и
получит мусор). Здесь: `allow_nan=False` (падаем громко) + запись в tmp + `os.replace`
(читатель видит либо старый файл, либо целый новый).
"""
from __future__ import annotations

import json
import os


def dump_json_atomic(obj, path, indent=1):
    """Записать obj в path атомарно; ValueError на NaN/Infinity, целевой файл не тронут."""
    tmp = f"{path}.tmp.{os.getpid()}"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=indent, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path
