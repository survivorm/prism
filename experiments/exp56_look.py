#!/usr/bin/env python
"""E-56 «посмотри прежде чем фитить» (ревью E56_pre): 4 дешёвых диагностических шага.
1) Δ(k) с σ-барами (pooled, из stage0.json) + расширенные лаги до 64 (3) — заодно
   проверяет, реальный ли dip на 24 / подъём на 32 или край окна (здесь нет: T=384, пар много).
2) Per-document кривые Δ_doc(k) + доменная классификация (dip-only / hump<=6 / 7-9 / >=10).
   Ответ на «pooled медиана — обманка или нет».
4) Траектория u_t в top-3 PCA по 3 документам на домен (окружность/эллипс/спираль/изломы).
+ честный null: j_paired на реальных 64 доках против ЕГО ШЕ же shuffled-бейслайна и против
  синтетики C1/C3/C5 на ТОМ ЖЕ бюджете 64 доков (прежний thr калиброван на 200 — несравнимо).
Оценка delta — ровно оценщик lag14 (per-doc median_t cos(v_i+k, v_i)); ничего не переопределяем."""
import os, sys, json
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M
import exp53b_servo as E
import exp55_sbi as S

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "e56")
os.makedirs(OUT, exist_ok=True)
res = {"predregistered_rules": {
    "dip": "raw med_t cos(v_{t+1},v_t) < -0.05 (СЫРАЯ медиана, не delta vs shuffle — честная",
    "": "формулировка после review v15; порог тот же, что в коде с самого начала)",
    "hump_pos": "argmax_k med_t cos по k в 3..16",
    "types": "no_dip | hump<=6 | hump7_9 | hump>=10"}}

docs = [E.derive(d) for d in E.load_real()]
doms = sorted({d["dom"] for d in docs})
COL = {d: c for d, c in zip(doms, ("tab:blue", "tab:orange", "tab:green", "tab:red"))}
print("docs:", len(docs), flush=True)

# ---------- шаг 1: pooled Δ(k)±σ (stage0) + шаг 3: расширение лагов ----------
st0 = json.load(open(os.path.join(HERE, "..", "results", "exp55_stage0.json")))
k14 = sorted(int(k) for k in st0["curve"])
d14 = [st0["curve"][str(k)]["delta"][0] for k in k14]
s14 = [st0["sigma"][str(k)] for k in k14]
LAGS_EXT = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64)
cur_ext = M.lag14(docs, lags=LAGS_EXT)
ke = list(LAGS_EXT)
de = [cur_ext[k]["delta"][0] for k in ke]
se = [max((cur_ext[k]["delta"][2] - cur_ext[k]["delta"][1]) / 3.92, 2e-3) for k in ke]
res["ext_curve"] = {str(k): dict(delta=cur_ext[k]["delta"], med=cur_ext[k]["med"],
                                 null=cur_ext[k]["null"]) for k in ke}

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].errorbar(k14, d14, yerr=s14, fmt="o-", capsize=3)
ax[0].axhline(0, color="k", lw=.6); ax[0].set_title("pooled Δ(k) ± σ (стаг0, 14 лагов)")
ax[0].set_xlabel("lag k"); ax[0].grid(alpha=.3)
ax[1].errorbar(ke, de, yerr=se, fmt="s-", capsize=3)
ax[1].axhline(0, color="k", lw=.6); ax[1].set_title("расширенные лаги до 64 (тот же оценщик)")
ax[1].set_xlabel("lag k"); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/f1_delta_sigma.png", dpi=110); plt.close(fig)

# ---------- шаг 2: per-doc кривые + классификация ----------
def perdoc(doc, lags):
    Vn = doc["V"][doc["ok"]]
    Vn = Vn / np.maximum(np.linalg.norm(Vn, axis=1, keepdims=True), 1e-12)
    if len(Vn) < 2:
        return None
    return {k: (float(np.median([float(Vn[i + k] @ Vn[i]) for i in range(len(Vn) - k)]))
                if len(Vn) > k else np.nan) for k in lags}

KS = tuple(sorted(set(k14) | set(range(3, 17))))
pdoc = {}
for d in docs:
    r = perdoc(d, KS)
    if r is None:
        continue
    pdoc[d["name"]] = dict(dom=d["dom"], d1=r[1], hump_pos=int(max(range(3, 17), key=lambda k: r[k])),
                           hump_amp=r[max(range(3, 17), key=lambda k: r[k])], curve=r)
    r["_type"] = ("no_dip" if r[1] >= -0.05 else
                  "hump<=6" if pdoc[d["name"]]["hump_pos"] <= 6 else
                  "hump7_9" if pdoc[d["name"]]["hump_pos"] <= 9 else "hump>=10")
cnt = {}
for v in pdoc.values():
    cnt.setdefault(v["dom"], {}).setdefault(v["curve"]["_type"], 0)
    cnt[v["dom"]][v["curve"]["_type"]] += 1
res["perdoc_types"] = cnt
res["perdoc_hump_pos_hist"] = {str(k): sum(1 for v in pdoc.values() if v["hump_pos"] == k)
                               for k in range(3, 17)}

fig, ax = plt.subplots(figsize=(11, 5))
for nm, v in pdoc.items():
    xs = [k for k in KS if k <= 20]
    ax.plot(xs, [v["curve"][k] for k in xs], color=COL[v["dom"]], alpha=.35, lw=.9)
ax.axhline(0, color="k", lw=.6); ax.set_xlabel("lag k")
ax.set_title(f"Δ_doc(k), {len(pdoc)} доков (цвет — домен); типы: {cnt}")
ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{OUT}/f2_perdoc_curves.png", dpi=110); plt.close(fig)

# ---------- шаг 4: траектории u_t в top-3 PCA ----------
picks = [d for dom in doms for d in docs if d["dom"] == dom][:3]
fig = plt.figure(figsize=(13, 8.5))
trajs = {}
for i, d in enumerate(picks):
    U = d["U"] if "U" in d else None
    if U is None:
        U = E.derive.__globals__["np"].array([])
    X = U - U.mean(0)
    _, _, Vt = np.linalg.svd(X[:200], full_matrices=False)   # топ-3 по первым 200 шагам
    Y = X @ Vt[:3].T
    trajs[d["name"]] = Y[:, :3].astype(np.float32).tolist()
    for a, b, lbl in ((0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")):
        axx = fig.add_subplot(3, 3, i * 3 + (lbl == "xy") + 2 * (lbl == "xz") + 3 * (lbl == "yz"))
        sc = axx.scatter(Y[:, a], Y[:, b], c=np.arange(len(Y)), cmap="viridis", s=7)
        axx.set_title(f"{d['name']} {lbl}"); axx.grid(alpha=.3); axx.set_aspect("equal")
fig.colorbar(sc, ax=fig.axes, fraction=.02, label="t")
fig.tight_layout(); fig.savefig(f"{OUT}/f3_pca_trajectories.png", dpi=100); plt.close(fig)
json.dump(trajs, open(f"{OUT}/f3_traj_coords.json", "w"))

# ---------- честный null: j_paired на бюджете real (64), не 200 ----------
jg, j, js, jx = M.j_paired(docs)
res["j_real64"] = dict(j_glob=jg, j=j, j_shuf=js, J_excess=jx)
sim64 = {}
for nm, md, kw in (("C1", [S.BETA], {}),
                   ("C5", [], {}),
                   ("C3", [S.BETA, ('rot', 0.92, S.OM8, 0.08, (S.u1, S.u2))], {}),
                   ("C2", [S.BETA, ('c', 0.92, S.OM8, 0.08, S.w2)], {})):
    dd = M.simulate_modes(len(docs), S.SEED_T, md, **kw)
    a, b, c, e = M.j_paired(dd)
    sim64[nm] = dict(j_glob=a, j=b, j_shuf=c, J_excess=e)
res["j_synth64"] = sim64
print("j real64:", res["j_real64"], flush=True)
print("j synth64:", sim64, flush=True)
print("типы:", cnt, flush=True)

json.dump(res, open(os.path.join(OUT, "e56_look.json"), "w"), indent=1, default=float)
print("saved ->", OUT, flush=True)
