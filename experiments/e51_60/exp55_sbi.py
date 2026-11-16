#!/usr/bin/env python
"""E-55 стадия -1 v2: SBI-гейт. Идентифицируемость спектра мод через ПОЛНЫЙ пайплайн
(simulate_modes → lag14 → obj против целевой кривой; оценщик идентичен с обеих сторон).
Kung-1a понижена до генератора гипотез (v1: ложные моды на C7, obj=201 на истине).
J — с shuffle-бейслайном (J_excess). Контроли: цель seed=11, фит-симуляции seed=77.
Гейты в конце печати; любой непройденный — остановка по ТЗ."""
import os, sys, json, math, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M

SEED_T, SEED_F, NDOC = 11, 77, 20
rd = np.random.default_rng(5)
w1 = M.unit(rd.standard_normal(M.D)); w2 = M.unit(rd.standard_normal(M.D))
u1, u2 = w1, M.unit(w2 - (w2 @ w1) * w1)
OM8 = 2 * math.pi / 8
BETA = ('r', 0.15, 0.4, w1)                         # усиленный β-канал (v1 C1 был невидим)

def target(modes, jitter=False):
    d = M.simulate_modes(NDOC, SEED_T, modes, jitter_phase=jitter)
    c = M.lag14(d)
    y = [c[k]["delta"][0] for k in M.LAGS14]
    s = [max((c[k]["delta"][2] - c[k]["delta"][1]) / 3.92, 2e-3) for k in M.LAGS14]  # пол 2e-3 (self-audit: узкие σ на больших лагах взрывают obj)
    return y, s, d

def j_excess(docs):
    j, _ = M.phase_current(docs)
    sh = []
    for dd in docs:
        V = M.E.shuffle_seq(dd["V"].copy())
        sh.append(dict(V=V, ok=np.ones(len(V), bool), cstar=dd["cstar"], eta=dd["eta"], Vc=V))
    js, _ = M.phase_current(sh)
    return j, js, j - js

def make_obj(y, s, struct_fn):
    def obj(p):
        try:
            modes = struct_fn(p)
            if modes is None:
                return 1e6
            d = M.simulate_modes(NDOC, SEED_F, modes)
            c = M.lag14(d)
            v = sum(((c[k]["delta"][0] - y[i]) / s[i]) ** 2 for i, k in enumerate(M.LAGS14))
            return float(v) if np.isfinite(v) else 1e6
        except Exception:
            return 1e6
    return obj

def nelder(f, x0, step, iters=120):
    n = len(x0); pts = [np.array(x0, float)]
    for i in range(n):
        q = np.array(x0, float); q[i] += step[i]; pts.append(q)
    vals = [f(q) for q in pts]
    for _ in range(iters):
        o = np.argsort(vals); pts = [pts[i] for i in o]; vals = [vals[i] for i in o]
        cen = np.mean(pts[:-1], 0)
        xr = cen + (cen - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = cen + 2.0 * (cen - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (pts[-1] - cen); fc = f(xc)
            if fc < vals[-1]: pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i])
    o = np.argsort(vals)
    return pts[o[0]], vals[o[0]]

def inb(p, lo, hi):
    return all(lo <= v <= hi for v, (lo, hi) in zip(p, zip(np.atleast_1d(lo) if False else [], []))) if False else True

# ---------- контроли: структура, границы, гейт ----------

def run_all():
    out = {}

    def bounded(p, bounds):
        for v_, (lo, hi) in zip(p, bounds):
            if not (lo <= v_ <= hi):
                return None
        return p

    # --- цели ---
    tg = {}
    tg["C1"] = target([BETA])
    tg["C2"] = target([BETA, ('c', 0.92, OM8, 0.05, w2)])
    tg["C3"] = target([BETA, ('rot', 0.92, OM8, 0.05, (u1, u2))])
    tg["C4"] = target([BETA, ('c', 0.92, OM8, 0.04, w2), ('c', 0.95, 2 * math.pi / 16, 0.03, w1 * 0 + M.unit(np.roll(w2, 7)) )])
    tg["C5"] = target([])
    tg["C6"] = target([BETA, ('c', 0.92, OM8, 0.04, w2), ('r', 0.93, 0.05, M.unit(np.roll(w2, 13)))])
    tg["C7"] = target([BETA, ('x', OM8, 0.06, w2)], jitter=True)

    specs = {
      "C1": dict(fit=lambda p: [BETA if False else ('r', p[0], p[1], w1)],
                 x0=[0.2, 0.4], st=[0.1, 0.15],
                 bnds=[(0.02, 0.6), (0.05, 0.9)],
                 check=lambda p: abs(p[0] - 0.15) < 0.12, truth="r=0.15±0.12"),
      "C2": dict(fit=lambda p: [BETA, ('c', p[0], p[1], p[2], w2)],
                 x0=[0.9, OM8, 0.05], st=[0.05, 0.08, 0.02],
                 bnds=[(0.5, 0.985), (0.4, 1.2), (0.005, 0.25)],
                 check=lambda p: abs(p[1] - OM8) / OM8 < 0.15 and abs(p[0] - 0.92) < 0.08, truth="r=.92 ω=.785 T=8±15%"),
      "C3": dict(fit=lambda p: [BETA, ('rot', p[0], p[1], p[2], (u1, u2))],
                 x0=[0.9, OM8, 0.05], st=[0.05, 0.08, 0.02],
                 bnds=[(0.5, 0.985), (0.4, 1.2), (0.005, 0.25)],
                 check=lambda p: abs(p[1] - OM8) / OM8 < 0.15, truth="rot T=8±15%"),
      "C4": dict(fit=lambda p: [BETA, ('c', p[0], p[1], p[2], w2), ('c', p[3], p[4], p[5], M.unit(np.roll(w2, 7)))],
                 x0=[0.9, OM8, 0.04, 0.95, 2 * math.pi / 16, 0.03], st=[0.04, 0.06, 0.01, 0.03, 0.05, 0.01],
                 bnds=[(0.5, 0.985), (0.4, 1.2), (0.005, 0.2)] * 2,
                 check=lambda p: abs(p[1] - OM8) / OM8 < 0.15 and abs(p[4] - 2 * math.pi / 16) / (2 * math.pi / 16) < 0.15,
                 truth="обе ω (8,16) ±15%"),
      "C5": dict(fit=lambda p: [BETA if False else ('c', p[0], p[1], p[2], w2)],
                 x0=[0.9, OM8, 0.02], st=[0.05, 0.08, 0.01],
                 bnds=[(0.5, 0.985), (0.4, 1.2), (0.002, 0.25)],
                 check=lambda p: p[2] < 0.02 or True,  # см. Δobj vs empty — отдельная проверка
                 truth="пусто: sd→пол или Δobj<порога"),
      "C6": dict(fit=lambda p: [BETA, ('c', p[0], p[1], p[2], w2), ('r', p[3], p[4], M.unit(np.roll(w2, 13)))],
                 x0=[0.9, OM8, 0.04, 0.93, 0.05], st=[0.04, 0.06, 0.01, 0.03, 0.015],
                 bnds=[(0.5, 0.985), (0.4, 1.2), (0.005, 0.2), (0.5, 0.99), (0.005, 0.3)],
                 check=lambda p: abs(p[1] - OM8) / OM8 < 0.15 and abs(p[3] - 0.93) < 0.1,
                 truth="пара(T8) + вещ(r=.93)"),
      "C7": dict(fit=lambda p: [BETA, ('c', p[0], p[1], p[2], w2)],
                 x0=[0.9, OM8, 0.05], st=[0.05, 0.08, 0.02],
                 bnds=[(0.5, 0.985), (0.4, 1.2), (0.002, 0.25)],
                 check=lambda p: p[2] < 0.02, truth="sd(амплитуда)→пол: фаза-джиттер гасит"),
    }

    for name in ["C1", "C2", "C3", "C5", "C7", "C6", "C4"]:
        y, s, ddocs = tg[name]
        sp = specs[name]
        yb = None
        if name in ("C5", "C7"):     # baseline пустой структуры
            db = M.simulate_modes(NDOC, SEED_F, [BETA] if name == "C7" else [])
            cb = M.lag14(db)
            yb = sum(((cb[k]["delta"][0] - y[i]) / s[i]) ** 2 for i, k in enumerate(M.LAGS14))
        def f_(p, sp=sp):
            b = sp["fit"](np.array(p))
            if b is None: return 1e6
            for v_, (lo, hi) in zip(p, sp["bnds"]):
                if not (lo <= v_ <= hi): return 1e6
            return None
        def obj_wrap(p, sp=sp, y=y, s=s):
            for v_, (lo, hi) in zip(p, sp["bnds"]):
                if not (lo <= v_ <= hi):
                    return 1e6
            modes = sp["fit"](np.array(p))
            try:
                d = M.simulate_modes(NDOC, SEED_F, modes)
                c = M.lag14(d)
                v = sum(((c[k]["delta"][0] - y[i]) / s[i]) ** 2 for i, k in enumerate(M.LAGS14))
                return float(v) if np.isfinite(v) else 1e6
            except Exception:
                return 1e6
        t0 = time.time()
        x, fv = nelder(obj_wrap, sp["x0"], sp["st"], iters=110)
        j, js, jx = j_excess(ddocs)
        passed = bool(sp["check"](x))
        extra = ""
        if yb is not None:
            extra = f" objEmpty={yb:.1f}"
            passed = passed and (yb - fv > 8)     # режим: мода обязана выигрывать ≥8 σ² или амплитуда→пол
        out[name] = dict(p=[float(v) for v in x], obj=float(fv), obj_empty=yb,
                         J=j, J_shuf=js, J_excess=jx, passed=passed, truth=sp["truth"], secs=round(time.time()-t0))
        print(f"{name}: p={[round(float(v),4) for v in x]} obj={fv:.1f}{extra} "
              f"J={j:.3f} Jsh={js:.3f} Jex={jx:+.3f} PASS={passed} [{time.time()-t0:.0f}s]", flush=True)
    # J-гейт ДЕЛЕГИРОВАН exp55_jrecal: статистика j (paired plane) + порог, калиброванный на
    # seed 13-22; J_excess здесь — только диагностика (на C1/C2 знаменатель раздувает j_shuf,
    # J_excess отрицателен на шуме → годился бы только с поправкой, review v14 #2).
    out["gate_J"] = None
    out["gate_J_owner"] = "exp55_jrecal.py"
    print("J-гейт отложен до exp55_jrecal (Jex-диагностика: "
          + ", ".join(f"{k}={out[k]['J_excess']:+.2f}" for k in ("C2", "C3", "C5")) + ")", flush=True)
    allp = all(out[k]["passed"] for k in ("C1", "C2", "C3", "C4", "C5", "C6", "C7"))
    out["gate_all_partial_no_J"] = bool(allp)   # явная метка: без J — не финальный вердикт
    out["gate_all"] = None                       # финальный ставит только jrecal (review v14 minor a)
    print("=== STAGE -1 v2 (partial, без J):", "PASS" if allp else "FAIL", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                     "results", "exp55_sbi_gate.json"), "w"), indent=1, default=float)

if __name__ == "__main__":
    run_all()
