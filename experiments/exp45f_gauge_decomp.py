#!/usr/bin/env python
"""E-45f: сколько калибровки сидит в каждой правке таблицы E-45 и что она делает с `κ`.

Зачем (пререгистрация — `docs/E45_damage.md` §6, критерии F1-F3 зафиксированы до чисел).
E-45c показал: свобода считывания `W → W + 1·cᵗ` не меняет поведение НИКАК (KL 2e-06 — это
fp32-под), но двигает `κ(J)` на 0.67-23.3 % в зависимости от направления `c`. Сдвиги `κ` в
таблице E-45 — 0.01-0.68 % при полосе разброса одного draw'а ±0.52 %. Значит если в правках
есть калибровочная компонента хоть какого-то порядка, колонка `κ` таблицы неатрибутируема и
«плотные и разреженные правки не двигают лист» — не утверждение. Здесь измеряется, сколько её
там, и печатает колонку `κ(gauge-free)`.

Раскладка (по строкам `U128`, только они входят в J слоя 23):

    c := (1/K) Σ_k ΔW_k,     ΔW_gauge := 1_K cᵗ,     ΔW_gf := ΔW − ΔW_gauge.

Ключевое упрощение: `J` в `E45.make_j_fn` ЛИНЕЕН по `W_U` при фиксированном `h`, поэтому
`ΔJ = J(ΔW)` — базовый якобиан на вычитание не нужен, две правки на чанк вместо трёх.
Линейность проверяется численно (`checks.linearity_J`), потому что на ней держится вся
экономика прогона.

Плотные правки (`gptq4`, `rtn8`, `rand_full_*`) таблица не сохраняет — они восстанавливаются
здесь теми же детерминированными вызовами (`_quantize_gptq` на том же Гессиане из 4096 кэшевых
позиций, `_quantize_rtn`, тот же `default_rng(11)`), и каждая сверяется с таблицей по дрейфу и
по `κ`: не сошлось — восстановлена не та правка, скрипт падает громко.

Критерии (дословно из §6):
  F1: `s ≥ 0.2` хотя бы у одной правки ⇒ всем κ-сравнениям таблицы нужна колонка κ(gf);
  F2: `|Δκ_gauge| ≥ 0.5·|Δκ_full|` ⇒ вывод об «уплощении листа» по этой правке снимается;
  F3: у плотных правок `s` заметно больше, чем у разреженных — предсказание, обязательное к
      опровержению.
Самопроверки: `s = 0` у базы; `s = 1` и ранг-1 у чистой калибровки из E-45c.

Запуск:  cd experiments && ../.venv/bin/python exp45f_gauge_decomp.py     (~20 минут CPU)
Память: как в E-45c — всё, что смотрит в J, идёт чанками по CHUNK позиций.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import exp45_damage_vs_structure as E45                   # noqa: E402
import exp45c_gauge_null as E45C                          # noqa: E402  (j_chunk, kappa_chunk)
from _e38_cache import jcache_path                        # noqa: E402
from exp39b_ptq_perturbation import _quantize_gptq, _quantize_rtn   # noqa: E402

S, R, D, K_TOK = E45.S, E45.R, 1024, E45.K_TOK
SPEC_DOCS = E45.SPEC_DOCS                 # те же 8 документов, что у колонки κ таблицы
SP_POS = np.concatenate([np.arange(i * S, (i + 1) * S) for i in SPEC_DOCS])
SPARSE = os.path.join(_HERE, "..", "results", "exp45_edits_sparse.npz")
TABLE = os.path.join(_HERE, "..", "results", "exp45_damage_vs_structure.json")
OUT = os.path.join(_HERE, "..", "results", "exp45f_gauge_decomp.json")
CHUNK = 512                               # [CHUNK,K,D] fp64 = 0.54 ГБ
TOL_LINEAR = 1e-9                         # J линеен по W_U (fp64, остаток — округление)
TOL_RANK1 = 1e-9                          # остаток чистой калибровки против ранг-1 предсказания
TOL_KAPPA_TABLE = 5e-3                    # восстановленный κ против колонки таблицы
TOL_DRIFT_TABLE = 2e-2                    # …и дрейф (он же проверка «та же правка»)
GAUGE_ALPHA = 0.15                        # масштаб чистой калибровки в самопроверке s = 1
S_F1 = 0.2
F2_RATIO = 0.5
SMOKE = int(os.environ.get("E45F_SMOKE", "0"))    # >0: только разреженные правки, строк SMOKE
                                                  # (плотные восстанавливаются дорого: GPTQ [V,D])
log = E45.log


def dJ_pos(Hsub, g, dW):
    """‖ΔJ_t‖_F по позициям `Hsub` для приращения строк `dW` [K,D]; ΔJ = J(dW) по линейности."""
    out = np.empty(len(Hsub))
    for c0 in range(0, len(Hsub), CHUNK):
        c1 = min(c0 + CHUNK, len(Hsub))
        Jc = E45C.j_chunk(Hsub, g, dW, c0, c1)
        out[c0:c1] = np.linalg.norm(Jc.reshape(c1 - c0, -1), axis=1)
        del Jc
    return out


def main():
    t0 = time.time()
    torch.set_num_threads(len(os.sched_getaffinity(0)))
    log("loading jcache L23 ...")
    ck = torch.load(jcache_path(E45.L), map_location="cpu", weights_only=False)
    U128 = [int(t) for t in (ck["U128"].numpy() if hasattr(ck["U128"], "numpy") else ck["U128"])]
    DOMS, NS = list(ck["domains"]), int(ck["n_samples"])
    Hst = np.stack([ck["H"][(d, s, t)].double().numpy()
                    for d in DOMS for s in range(NS) for t in range(S)])
    del ck
    Hs = Hst[SP_POS]                      # позиции геометрии = те же 8 документов, что в таблице
    log(f"позиции: корпус {Hst.shape[0]}, геометрия {Hs.shape[0]} ({len(SPEC_DOCS)} документов)")

    log("loading model (fp32) ...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        E45.MODEL_DIR, dtype=torch.float32, attn_implementation="eager",
        low_cpu_mem_usage=True).eval()
    torch.set_grad_enabled(False)
    W = model.lm_head.weight.detach().float().clone()
    g_np = (1.0 + model.model.norm.weight.detach().float()).numpy().astype(np.float64)
    sigW = float(W.std())
    WU0 = W[U128].numpy().astype(np.float64)

    table = {e["name"]: e for e in json.load(open(TABLE))["edits"]}
    sparse = np.load(SPARSE)

    # ---------------- восстанавливаем плотные правки таблицы ----------------
    # Гессиан GPTQ — по ВСЕМ 4096 кэшевым позициям, как в таблице (не по позициям геометрии),
    # иначе восстановленная правка отличается от той, чей κ лежит в таблице.
    # нормы базового J: J линеен по строкам, поэтому J(W_U0) и есть базовый якобиан (а J(0) = 0,
    # на это и наступил первый прогон: n_base вышло нулём, и s чистой калибровки улетело в 1e178)
    n_base = np.maximum(dJ_pos(Hs, g_np, WU0), 1e-300)
    edits = [("base", np.zeros((K_TOK, D)))]
    lin_F = float("nan")
    if not SMOKE:
        log("восстанавливаю gptq4 (Гессиан = 4096 позиций кэша) ...")
        Ha = torch.from_numpy(Hst).float()
        W_gq = _quantize_gptq(W.double(), Ha.double(), E45.GPTQ_BITS, E45.GPTQ_GROUP,
                              E45.GPTQ_DAMP, torch.device("cpu")).float()
        edits.append(("gptq4", W_gq[U128].numpy().astype(np.float64) - WU0))
        del W_gq, Ha
        log("восстанавливаю rtn8 ...")
        W_rtn = _quantize_rtn(W.double(), E45.RTN_BITS, E45.RTN_GROUP).float()
        edits.append(("rtn8", W_rtn[U128].numpy().astype(np.float64) - WU0))
        del W_rtn
        # rand_full: в J входят только строки U128, значит его приращение в J — это σ_full·G_U.
        # σ — из той же формулы линейности, что в таблице (пересчитан здесь, не списан из лога).
        G_all = np.random.default_rng(11).standard_normal((W.shape[0], D)) * sigW
        GU = G_all[U128].astype(np.float64)
        del G_all
        lin_F = float(np.median(dJ_pos(Hs, g_np, GU) / n_base))
        log(f"drift_full(σ=1) = {lin_F:.4f} (в логе таблицы было 0.9230)")
        for dt in E45.DRIFTS:
            edits.append((f"rand_full_d{dt}", GU * (dt / lin_F)))
    for dt in E45.DRIFTS:
        edits.append((f"rand_K_d{dt}", sparse[f"rand_K_d{dt}"].astype(np.float64) - WU0))
    for k in range(8):
        edits.append((f"seednull_s{k}", sparse[f"seednull_s{k}"].astype(np.float64) - WU0))
    for nm in ("flat_l0.01", "flat_l0.1", "flat_noleash"):
        edits.append((nm, sparse[nm].astype(np.float64) - WU0))
    if SMOKE:
        # в smoke требуются только те строки таблицы, что реально в списке (плотные не
        # восстанавливаются, остальные разреженные не входят)
        edits = [e for e in edits if e[0] == "base"] + [e for e in edits if e[0] != "base"][:SMOKE]
        log(f"SMOKE={SMOKE}: правки {[n for n, _ in edits]}")
    names = [n for n, _ in edits]
    assert len(set(names)) == len(names), "дубли правок"
    missing = [] if SMOKE else sorted(set(table) - set(names))
    assert not missing, f"нет раскладки для правок таблицы: {missing}"
    log(f"раскладка готова: {len(edits)} правок")

    # ---------------- самопроверки алгебры ----------------
    checks = {}
    dW_test = np.random.default_rng(3).standard_normal((K_TOK, D))
    J_lin = (E45C.j_chunk(Hs, g_np, WU0 + dW_test, 0, 256)
             - E45C.j_chunk(Hs, g_np, WU0, 0, 256))
    J_dir = E45C.j_chunk(Hs, g_np, dW_test, 0, 256)
    lin_rel = float(np.linalg.norm(J_lin - J_dir) / np.linalg.norm(J_lin))
    checks["linearity_J"] = lin_rel
    log(f"линейность J по W_U: ‖J(W+ΔW) − J(W) − J(ΔW)‖/‖ΔJ‖ = {lin_rel:.2e} "
        f"(допуск {TOL_LINEAR:.0e})")
    assert lin_rel < TOL_LINEAR, f"J не линеен по W_U ({lin_rel:.2e}) — экономика скрипта неверна"

    c_pure = torch.from_numpy((Hs.mean(0) * g_np).astype(np.float32))
    cn = (c_pure / c_pure.norm()).numpy().astype(np.float64) * (GAUGE_ALPHA * sigW * np.sqrt(D))
    dW_pure = np.tile(cn, (K_TOK, 1))
    # не-тривиальная часть: раскладка (c = среднее по строкам) обязана ПУСКАТЬ чистую калибровку
    # в gauge-половину целиком, то есть её gf-остаток в J ровно нуль
    checks["pure_gauge_gf_zero"] = float(
        np.linalg.norm(dJ_pos(Hs, g_np, dW_pure - np.tile(dW_pure.mean(0), (K_TOK, 1))))
        / np.linalg.norm(dJ_pos(Hs, g_np, dW_pure)))
    Jp = (E45C.j_chunk(Hs, g_np, WU0 + dW_pure, 0, 256)
          - E45C.j_chunk(Hs, g_np, WU0, 0, 256))
    pred = E45C.gauge_dJ_pred(Hs, g_np, cn, 0, 256)
    checks["pure_rank1_resid"] = float(np.linalg.norm(Jp - pred) / np.linalg.norm(Jp))
    log(f"чистая калибровка: gf-остаток в J = {checks['pure_gauge_gf_zero']:.2e}, ранг-1 остаток = "
        f"{checks['pure_rank1_resid']:.2e}, её дрейф = "
        f"{float(np.median(dJ_pos(Hs, g_np, dW_pure) / n_base)):.4f}; "
        f"нулевая правка: max‖ΔJ‖ = {float(dJ_pos(Hs[:64], g_np, np.zeros((K_TOK, D))).max()):.1e}")
    assert checks["pure_gauge_gf_zero"] < 1e-12, "раскладка не узнаёт чистую калибровку"
    assert checks["pure_rank1_resid"] < TOL_RANK1, "чистая калибровка не ранг-1"
    assert float(dJ_pos(Hs[:64], g_np, np.zeros((K_TOK, D))).max()) == 0.0, "база дала ΔJ ≠ 0"

    # ---------------- основная таблица ----------------
    kap_base = E45C.kappa_chunk(Hs, g_np, WU0, False)
    log(f"база: κ(r={R}) на {len(SPEC_DOCS)} документах = {kap_base:.8f}")
    rows_out = []
    for name, dW in edits:
        t1 = time.time()
        c = dW.mean(0)
        dW_g = np.tile(c, (K_TOK, 1))
        dW_gf = dW - dW_g
        n_tot = dJ_pos(Hs, g_np, dW)
        n_g = dJ_pos(Hs, g_np, dW_g)
        n_gf = dJ_pos(Hs, g_np, dW_gf)
        # половины ортогональны ТОЧНО: у ΔW_gf строки в сумме дают ноль, значит и Σ_k ΔJ_k = 0,
        # а gauge-половина это 1_K ⊗ v. Отсюда Пифагор по Frobenius — проверка, что `s` есть доля,
        # а не отношение двух непредставимых одно другому норм.
        checks[f"pythag_{name}"] = abs((n_g ** 2).sum() + (n_gf ** 2).sum() - (n_tot ** 2).sum()) \
            / max((n_tot ** 2).sum(), 1e-300)
        if name != "base":
            assert checks[f"pythag_{name}"] < 1e-12, \
                f"{name}: половины не ортогональны ({checks[f'pythag_{name}']:.2e})"
        row = dict(name=name, c_norm=float(np.linalg.norm(c)),
                   c_over_sigWsqD=float(np.linalg.norm(c)) / (sigW * np.sqrt(D)),
                   drift_med=float(np.median(n_tot / n_base)),
                   s_norm=float(np.sqrt((n_g ** 2).sum()) / max(np.sqrt((n_tot ** 2).sum()),
                                                                1e-300)),
                   s_median=float(np.median(n_g / np.maximum(n_tot, 1e-300))))
        if name == "base":
            row.update(kappa_full=kap_base, kappa_gauge=kap_base, kappa_gf=kap_base,
                       dkap_full=0.0, dkap_gauge=0.0, dkap_gf=0.0, dkap_gauge_over_full=0.0,
                       kappa_table=float(table["base"]["kappa_A"]), wall_sec=0.0)
            rows_out.append(row)
            log(f"{name:>16s}: s = 0 по построению, κ = {kap_base:.6f}")
            continue
        kap_full = E45C.kappa_chunk(Hs, g_np, WU0 + dW, False)
        kap_g = E45C.kappa_chunk(Hs, g_np, WU0 + dW_g, False)
        kap_gf = E45C.kappa_chunk(Hs, g_np, WU0 + dW_gf, False)
        tbl = table[name]
        checks[f"kappa_vs_table_{name}"] = abs(kap_full / tbl["kappa_A"] - 1)
        checks[f"drift_vs_table_{name}"] = abs(row["drift_med"] / tbl["drift_med"] - 1)
        bad = [k for k, v in checks.items()
               if k.startswith(f"kappa_vs_table_{name}") and v > TOL_KAPPA_TABLE]
        bad += [k for k, v in checks.items()
                if k.startswith(f"drift_vs_table_{name}") and v > TOL_DRIFT_TABLE]
        assert not bad, (f"{name}: восстановлена НЕ та правка "
                         f"(κ rel {checks[f'kappa_vs_table_{name}']:.2e}, "
                         f"drift rel {checks[f'drift_vs_table_{name}']:.2e})")
        row.update(kappa_full=kap_full, kappa_gauge=kap_g, kappa_gf=kap_gf,
                   kappa_table=float(tbl["kappa_A"]),
                   dkap_full=kap_full - kap_base, dkap_gauge=kap_g - kap_base,
                   dkap_gf=kap_gf - kap_base,
                   dkap_gauge_over_full=abs(kap_g - kap_base)
                   / max(abs(kap_full - kap_base), 1e-30),
                   wall_sec=round(time.time() - t1, 1))
        rows_out.append(row)
        log(f"{name:>16s}: ‖c‖={row['c_over_sigWsqD']:.4f}·σ_W√D  s={row['s_norm']:.4f} "
            f"(мед {row['s_median']:.4f})  κ {kap_base:.6f}→{kap_full:.6f} "
            f"(табл {row['kappa_table']:.6f})  gauge-only {kap_g:.6f}  gf {kap_gf:.6f}  "
            f"[{row['wall_sec']:.0f} с]")
        del kap_full, kap_g, kap_gf, n_tot, n_g, n_gf

    # ---------------- вердикты ----------------
    by = {r["name"]: r for r in rows_out if r["name"] != "base"}
    dense_n = [n for n in by if n.startswith(("gptq", "rtn", "rand_full"))]
    sparse_n = [n for n in by if n.startswith(("rand_K", "seednull", "flat"))]
    f1 = sorted(n for n in by if by[n]["s_norm"] >= S_F1)
    f2 = sorted(n for n in by if abs(by[n]["dkap_gauge"]) >= F2_RATIO * abs(by[n]["dkap_full"]))
    s_dense = [by[n]["s_norm"] for n in dense_n]
    s_sparse = [by[n]["s_norm"] for n in sparse_n]
    verdict = dict(
        F1_needs_gf_column=f1,
        F2_withdraw_flattening=f2,
        F3_dense_gt_sparse=(bool(min(s_dense) > max(s_sparse)) if s_dense and s_sparse else None),
        s_dense={n: round(by[n]["s_norm"], 4) for n in sorted(dense_n)},
        s_sparse={n: round(by[n]["s_norm"], 4) for n in sorted(sparse_n)},
        kappa_spread_gauge_only_pct=100 * max(abs(r["dkap_gauge"]) / kap_base
                                              for r in rows_out),
        checks=checks)
    log("\n== вердикт ==")
    log(f"  F1 (s ≥ {S_F1} ⇒ таблице обязательна колонка κ(gauge-free)): "
        f"{f1 if f1 else 'нет таких правок'}")
    log(f"  F2 (gauge-only ≥ половины сдвига κ ⇒ уплощение по правке не утверждаем): "
        f"{f2 if f2 else 'нет таких правок'}")
    log(f"  F3 (у плотных s больше всех разреженных): {verdict['F3_dense_gt_sparse']}; "
        f"dense {verdict['s_dense']} vs sparse {verdict['s_sparse']}")
    log(f"  разброс κ от ОДНОЙ калибровочной половины: "
        f"{verdict['kappa_spread_gauge_only_pct']:.2f} % от κ базы "
        f"(для сравнения: полоса draw'ей таблицы ±0.52 %)")
    json.dump(dict(experiment="E-45f", L=E45.L, r=R, spec_docs=list(SPEC_DOCS),
                   n_spec_pos=int(len(SP_POS)), kappa_base=kap_base, sigma_W=sigW,
                   rows=rows_out, verdict=verdict, wall_sec=round(time.time() - t0, 1)),
              open(OUT, "w"), ensure_ascii=False, indent=1)
    log(f"→ {OUT}, wall {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
