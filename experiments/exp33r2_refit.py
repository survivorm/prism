#!/usr/bin/env python3
"""E-33r.2 refit A/E: p0 = НОРМИРОВАННАЯ медиана (Σp=1) + assert.

Причина (ревью 2026-10-11): ненормированная медиана по строкам симплекса
имеет Σ ≈ 0.006, и P(c·p) = c·diag(p) − c²·ppᵀ ≈ c·diag(p) — крошечный
диагональный скалинг, не gauge-оператор. Старые dgp/E_fixed_p мерили
«насколько малое диагональное возмущение сдвигает подпространство», а не
«возвращает ли фиксация P структуру».

Версия 2026-10-12: адаптация под API exp33r_gauge_anatomy.py (load_layer(l)
-> list doc dict, p_apply, SENT; agg_positions/FLAT_REL/load_doc не существуют
— заменены локальными); фикс mislabeling E_fixed_p["32"] (запись "32" = копия
r=16 — ложный ключ больше не пишется, при dump-замене dict он исчезает);
идемпотентная MD-правка (line-based, безопасен повторный прогон);
refit_note -> refit_notes (список, история сохраняется); done ключуется по
doc['key'] (dom/si), не по индексу; dgp_{dt}_r16 = alias dgp_{dt} (нет
дублирующего SVD-цикла). published-значения старого кода НЕ воспроизводятся
(его точная p0-формула не идентифицируется — см. note в JSON); старый
published дубльруется в refit_notes.published_old.

Диагностики C/D/F/B не зависят от p0 — не трогаем. Пересчитываем только
dgp_* (A/E), обновляем layers[23].diag.E_fixed_p / E_fixed_p_r.
Вход: results/exp33r_gauge_anatomy.json (полный прогон E-33r.2).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import torch

torch.set_num_threads(1)   # не забирать все ядра

import exp33r_gauge_anatomy as E33R

L23 = 23
R_MAIN = 16
DELTA_GRID = [1, 4, 8, 16]
FLAT_REL = 1e-6            # σ_r/σ_1 < FLAT_REL -> вырожденное подпространство, NaN
JSON = "results/exp33r_gauge_anatomy.json"
MD = "docs/Exp33r2.md"
CKPT = "results/exp33r2_refit.ckpt.json"


def p0_for_doc(Pdoc, sent):
    """p0: медиана по не-сентинельным позициям, нормировка Σ=1 (E-34: P=diag(p)−ppᵀ).
    Тип = тип Pdoc (f32 в main) — не повышать до f64: p_apply/Jdoc f32."""
    sent_t = torch.from_numpy(np.ascontiguousarray(sent))
    alive = ~sent_t
    p0 = Pdoc[alive].median(dim=0).values if alive.any() else Pdoc.mean(dim=0)
    p0 = p0 / p0.sum().clamp_min(1e-30)
    assert abs(float(p0.sum()) - 1.0) < 1e-5, f"p0 not normalized: {float(p0.sum())}"
    return p0


def dgp_for_doc(Jdoc, Pdoc, sent):
    """A/E с нормированным p0: dgp_{dt} (r=16) + dgp_{dt}_r{rt} для rt∈{1,2,8,16}.
    Возвращает dict {name: [T-dt] (None=NaN)}."""
    T = Jdoc.shape[0]
    sent_t = torch.from_numpy(np.ascontiguousarray(sent))
    p0 = p0_for_doc(Pdoc, sent).to(Jdoc.dtype)
    _, sfp, Vhfp = torch.linalg.svd(
        E33R.p_apply(Jdoc, p0.unsqueeze(0).expand(T, -1)), full_matrices=False)
    out = {}
    for dt in DELTA_GRID:
        mask = (~(sent_t[:-dt] | sent_t[dt:])).numpy()
        for r in (1, 2, 8, R_MAIN):
            M = Vhfp[:-dt, :r] @ Vhfp[dt:, :r].transpose(-1, -2)
            sv = torch.linalg.svdvals(M).clamp(max=1.0 - 1e-12)
            flat = (sfp[:-dt, r - 1] / sfp[:-dt, 0].clamp_min(1e-30) < FLAT_REL).numpy()
            a = torch.acos(sv).max(-1).values.numpy()
            vals = [None if (not m or f) else float(x) for m, f, x in zip(mask, flat, a)]
            out[f"dgp_{dt}_r{r}"] = vals
            if r == R_MAIN:
                out[f"dgp_{dt}"] = vals   # alias, не дублируем
    return out


def agg(doct):
    """per-doc nanmedian -> med/iqr [p25,p75] по докам, n_docs = число finite (схема 08:30)."""
    meds = []
    for arr in doct:
        a = np.asarray(arr, dtype=float)   # None -> nan
        if np.isfinite(a).any():
            m = float(np.nanmedian(a))     # NaN = sentinel/вырождение — пропускаем
            if np.isfinite(m):
                meds.append(m)
    if not meds:
        return None
    meds = np.asarray(meds)
    return {'med': float(np.median(meds)),
            'iqr': [float(np.percentile(meds, 25)), float(np.percentile(meds, 75))],
            'n_docs': len(meds)}


def set_row(md, prefix, cell):
    """0 совпадений -> warning + нет-оп (MD уже отрефитчен); >1 -> SystemExit (амбигуитет)."""
    lines = md.split('\n')
    hits = [i for i, ln in enumerate(lines) if ln.startswith(prefix)]
    if len(hits) == 0:
        print(f'  MD: префикс {prefix!r} не найден — пропускаю (уже обновлён?)')
        return md
    if len(hits) > 1:
        raise SystemExit(f'set_row: префикс {prefix!r} совпал с {len(hits)} строками (нужно ровно 1)')
    i = hits[0]
    parts = lines[i].split('|')
    parts[-2] = f' {cell} '
    lines[i] = '|'.join(parts)
    return '\n'.join(lines)


def selfcheck():
    torch.manual_seed(0)
    P = (torch.rand(8, 4) + 0.01).double()   # [S, K], f64: точное сравнение
    sent = np.array([False, True, False, False, True, False, False, False])
    p0 = p0_for_doc(P, sent)
    assert p0.dtype == torch.float64
    assert abs(float(p0.sum()) - 1.0) < 1e-12, f'p0 sum {float(p0.sum())}'
    m = P[~torch.from_numpy(sent)].median(0).values
    assert torch.allclose(p0, m / m.sum(), atol=1e-12)
    # тип следует за Pdoc: f32 в -> f32 (main-путь: Jdoc f32)
    p32 = p0_for_doc(P.float(), sent)
    assert p32.dtype == torch.float32 and abs(float(p32.sum()) - 1.0) < 1e-5
    # set_row: ровно 1 совпадение; дубль -> SystemExit
    md = 'a\n| 1 | 1.461 | x |\nb\n'
    md2 = set_row(md, '| 1 | 1.461', 'y')
    assert '| 1 | 1.461 | y |' in md2.split('\n')
    try:
        set_row(md + '| 1 | 1.461 | z |\n', '| 1 | 1.461', 'q')
        raise AssertionError('set_row: двойное совпадение не поймано')
    except SystemExit:
        pass
    print('refit selfcheck OK')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    t0 = time.time()
    res = json.load(open(JSON))
    docs = E33R.load_layer(L23)
    n_docs = res["layers"][str(L23)]["n_docs"]
    assert len(docs) == n_docs, f"len(docs)={len(docs)} vs n_docs={n_docs}"
    order = [d['key'] for d in docs]

    done = {}
    if os.path.exists(CKPT):
        done = json.load(open(CKPT))
        print(f"resume: {len(done)}/{len(order)}")

    for doc in docs:
        k = doc['key']
        if k in done:
            continue
        Jdoc, Pdoc = doc['JR'].float(), doc['P128'].float()
        sent = (doc['RAW'].float().sum(-1) < E33R.SENT).numpy()
        done[k] = dgp_for_doc(Jdoc, Pdoc, sent)
        del Jdoc, Pdoc, sent
        json.dump(done, open(CKPT, "w"))
        print(f"  {k}  t={time.time()-t0:.0f}s", flush=True)

    # --- update layers[23].diag ---
    diag = res["layers"][str(L23)]["diag"]
    e16 = {str(dt): agg([done[k][f"dgp_{dt}"] for k in order]) for dt in DELTA_GRID}
    # NOTE: старый published-значения воспроизвести НЕВОЗМОЖНО (код 08:30 потерян,
    # точная p0-формула не идентифицируется: P128-med 0.20014 / RAW-med 0.19598 /
    # argmax-TЗ 0.17635 / без сентинел 0.1554-0.1570 vs published 0.202142).
    # Published вычислены с ненормированным p0 (Σ≈0.006, P≈diag) — это и есть баг;
    # refit их заменяет, det-check на совпадение намеренно НЕ ставится.
    old_e16 = {str(dt): diag["E_fixed_p"]["16"][str(dt)]["med"] for dt in DELTA_GRID}
    diag["E_fixed_p"] = {"16": e16}   # целая замена -> ложный ключ "32" исчезает
    diag["E_fixed_p_r"] = {str(rt): {str(dt): agg([done[k][f"dgp_{dt}_r{rt}"] for k in order])
                                     for dt in DELTA_GRID} for rt in (1, 2, 8, 16)}

    meta = res["meta"]
    notes = list(meta.get("refit_notes", []))
    if "refit_note" in meta:
        notes.append(meta.pop("refit_note"))   # мигрируем историю, не затираем
    if not any(n.get("published_old") == old_e16 for n in notes):   # идемпотентность
        notes.append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "E-33r.2 A/E (dgp, E_fixed_p, E_fixed_p_r) пересчитаны с "
                "нормированным p0 (Σp=1): p0 = медиана НЕ-сентинельных строк P128, "
                "нормировка. Старые значения (published_old в этом note) "
                "использовали ненормированную медиану (Σ≈0.006): P ≈ diag — не "
                "gauge-оператор. Точная p0-формула старого кода не идентифицируется "
                "(код потерян; варианты давали 0.155-0.204 vs published 0.20214). "
                "C/D/F/B не зависят от p0 — не тронуты. Зафиксировано doc↔code "
                "расхождение: Exp33r2.md «p_0 = аргмакс p_max», код — медиана; "
                "публикуемая версия = медиана + нормировка. Ложный ключ "
                "E_fixed_p['32'] удалён.",
        "published_old": old_e16,
        "refit_timing_sec": round(time.time() - t0, 1),
    })
    meta["refit_notes"] = notes

    with open(JSON, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, allow_nan=False)
    os.remove(CKPT)

    # --- MD: idempotent update A/E таблицы + фикс даты 07-07 -> 10-11 ---
    md = open(MD).read()
    new1 = e16["1"]["med"]
    new16 = e16["16"]["med"]
    md = md.replace("2026-07-07", "2026-10-11")
    md = set_row(md, "| 1 | 1.461", f"**{new1:.3f}** (refit)")
    md = set_row(md, "| 16 | 1.567", f"{new16:.3f} (refit)")
    open(MD, "w").write(md)
    print(f"done in {time.time()-t0:.0f}s; JSON + MD updated "
          f"(dgp med dt=1: {new1:.6f}, dt=16: {new16:.6f})")


if __name__ == "__main__":
    main()
