#!/usr/bin/env python
"""E-59b (docs/E59b.md, пред-рег до прогона): лаги 1..24; полоса ω∈[0.1,2.0], r≤0.92;
per-doc веса = bootstrap-σ медиан продуктов по позициям; CI(ω̂) из профиля χ² (Δ≤1).
Миры A/B/C с гейтом identifiability; эксплораторика ω̂×hump_pos и по доменам."""
import os, sys, json, math
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp55_modes as M
import exp53b_servo as E

HERE = os.path.dirname(os.path.abspath(__file__))
K = np.arange(1, 25)
n = len(K)
WG = np.round(np.arange(0.10, 2.001, 0.02), 3)
RG = np.round(np.arange(0.02, 0.921, 0.02), 3)
rng = np.random.default_rng(11)
docs = [E.derive(d) for d in E.load_real()]

def unit(doc):
    V = doc["V"][doc["ok"]]
    return V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)

def curve_and_sigma(Vn, B=200):
    y, sg = [], []
    for k in K:
        P = np.einsum("ij,ij->i", Vn[:-k], Vn[k:])
        y.append(float(np.median(P)))
        bm = np.median(P[rng.integers(0, len(P), (B, len(P)))], axis=1)
        sg.append(max(float(bm.std()), 2e-3))
    return np.array(y), np.array(sg)

def wls(y, w, cols):
    G = np.stack(cols, 1)
    A = G * w[:, None] ** .5
    AB = np.linalg.lstsq(A, y * w ** .5, rcond=None)[0]
    return float(np.sum(w * (y - G @ AB) ** 2))

out = []
for d in docs:
    Vn = unit(d)
    if len(Vn) <= 24:
        continue
    y, sd = curve_and_sigma(Vn)
    w = 1.0 / sd ** 2
    tot = float(np.sum(w * y ** 2))
    # EXP
    be = None
    for a in RG:
        for b in RG[RG > a + 0.02]:
            sse = wls(y, w, [a ** K, b ** K, np.ones(n)])
            if be is None or sse < be[0]:
                be = (sse, a, b)
    # OSC + профиль χ² по ω
    prof = np.empty(len(WG))
    bi_ = None
    for i, om in enumerate(WG):
        c, s = np.cos(om * K), np.sin(om * K)
        best = min((wls(y, w, [(r ** K) * c, (r ** K) * s, np.ones(n)]), r) for r in RG)
        prof[i] = best[0]
        if bi_ is None or best[0] < bi_[0]:
            bi_ = (best[0], om, best[1])
    bic_e = n * math.log(be[0] / n) + 5 * math.log(n)
    bic_o = n * math.log(bi_[0] / n) + 5 * math.log(n)
    dch = prof - bi_[0]
    p = int(np.argmin(dch))
    l_ = p
    while l_ > 0 and dch[l_ - 1] <= 1:
        l_ -= 1
    r_ = p
    while r_ < len(WG) - 1 and dch[r_ + 1] <= 1:
        r_ += 1
    lo_, hi_ = float(WG[l_]), float(WG[r_])
    clip = bool(l_ == 0 or r_ == len(WG) - 1)   # CI упёрся в границу полосы
    # secondary = второй ЛОКАЛЬНЫЙ минимум профиля (не просто «далеко хорошо» — review v17)
    loc = np.r_[False, (dch[1:-1] < dch[:-2]) & (dch[1:-1] <= dch[2:]), False]
    far = np.abs(WG - bi_[1]) > 0.3
    secondary = bool((loc & far & (dch < 4.0)).any())
    hr = {k: float(np.median(np.einsum("ij,ij->i", Vn[:-k], Vn[k:]))) for k in range(3, 17)}
    hump = int(max(hr, key=hr.get))
    out.append(dict(name=d["name"], dom=d["dom"], R2_exp=1 - be[0] / tot, R2_osc=1 - bi_[0] / tot,
                    r=[be[1], be[2]], omega=bi_[1], r_osc=bi_[2], osc_wins=bool(bic_o < bic_e),
                    hump_pos=hump, r_clip=bool(bi_[2] >= 0.919),
                    dBIC=bic_e - bic_o, ci=[lo_, hi_], ci_w=hi_ - lo_, ci_clip=clip,
                    ident_ci_only=bool(hi_ - lo_ < 0.6 and not clip),
                    identifiable=bool(hi_ - lo_ < 0.6 and not clip and bi_[2] < 0.919),
                    secondary_min=secondary))
    if len(out) % 16 == 0:
        print(f"...{len(out)} доков, текущий ω̂={bi_[1]:.2f} R2o={1 - bi_[0] / tot:.2f}", flush=True)

win = [o for o in out if o["osc_wins"]]
W = np.array([o["omega"] for o in win])
W_id = np.array([o["omega"] for o in win if o["identifiable"]])      # strict: без r_clip
W_ci = np.array([o["omega"] for o in win if o["ident_ci_only"]])     # пред-реговый ident
med_osc = float(np.median([o["R2_osc"] for o in out]))
ident_frac = float(np.mean([o["identifiable"] for o in win])) if win else 0.0
ident_frac_ci = float(np.mean([o["ident_ci_only"] for o in win])) if win else 0.0

def gmm_bic(x, k):
    if len(x) < 4 * k:
        return np.inf, 0.0
    mu = np.linspace(x.min(), x.max(), k + 2)[1:-1] if k > 1 else np.array([x.mean()])
    sd = np.full(k, x.std() / max(k, 1) + 1e-6); pk = np.full(k, 1.0 / k); ll = -np.inf
    for _ in range(150):
        L = -np.log(2 * np.pi * sd ** 2) / 2 - ((x[:, None] - mu) / sd) ** 2 / 2 + np.log(pk)
        m = L.max(1, keepdims=True)
        r = np.exp(L - m); r /= r.sum(1, keepdims=True)
        pk = r.mean(0); mu = (r * x[:, None]).sum(0) / r.sum(0)
        sd = np.sqrt((r * (x[:, None] - mu) ** 2).sum(0) / r.sum(0)) + 1e-6
        ll = float(np.sum(m[:, 0] + np.log(np.exp(L - m).sum(1))))
    return -2 * ll + (2 * k + k - 1) * math.log(len(x)), float(pk.min())

bi = {}
bimodal = False
if len(W) >= 12:
    b1, _ = gmm_bic(W, 1)
    b2, wmin = gmm_bic(W, 2)
    bimodal = (b1 - b2) >= 6.0 and wmin >= 0.25
    bi = dict(bic1=float(b1), bic2=float(b2), min_w=wmin)

# эксплораторика: spearman(ω̂, hump_pos) без scipy — hump_pos уже в out (review v17: без O(n²))
A_ = np.array([o["omega"] for o in out]); B_ = np.array([o["hump_pos"] for o in out])
ra = np.argsort(np.argsort(A_)); rb = np.argsort(np.argsort(B_))
rho = float(np.corrcoef(ra, rb)[0, 1])
dommed = {dm: float(np.median([o["omega"] for o in out if o["dom"] == dm])) for dm in sorted({o["dom"] for o in out})}

if med_osc < 0.5:
    world = "A"
elif med_osc >= 0.7 and ident_frac < 0.5:
    world = "НЕИДЕНТИФИЦИРУЕМО (strict ident < 0.5; B/C не различимы)"
elif med_osc >= 0.7 and len(W) >= 12:
    world = "B: два типа документов" if bimodal else "C: континуум частот"
else:
    world = "неопределённо"

res = dict(med_R2_exp=float(np.median([o["R2_exp"] for o in out])), med_R2_osc=med_osc,
           win_frac=len(win) / len(out),
           omega_all=dict(n=len(W), med=float(np.median(W)), p25=float(np.percentile(W, 25)),
                          p75=float(np.percentile(W, 75))),
           omega_ident=dict(n=int(len(W_id)), med=float(np.median(W_id)),
                            p25=float(np.percentile(W_id, 25)), p75=float(np.percentile(W_id, 75))),
           omega_ci_only=dict(n=int(len(W_ci)), med=float(np.median(W_ci))),
           ident_frac=ident_frac, ident_frac_ci_only=ident_frac_ci,
           ident_note="strict identifiable (исключает r_clip>=0.919) введена post-hoc по review v18; "
                      "world считается по строгой — ужесточение конзервативно",
           secondary_min_frac=float(np.mean([o["secondary_min"] for o in win])) if win else None,
           gmm=bi, bimodal=bool(bimodal), spearman_omega_hump=rho,
           rho_note="не значимо при n=64; знак противоположен ожиданию ревьюера; damp-swing и "
                    "края окна 3..16 смешаны — exploratory, не интерпретировать",
           dom_med_omega=dommed, world=world, docs=out)
print(f"R² osc med={med_osc:.3f} | winners {len(win)}/{len(out)} | ω̂ all med={np.median(W):.2f} "
      f"| ω̂ ident med={np.median(W_id):.2f} (n={len(W_id)}) | ident strict={ident_frac:.2f} "
      f"ci-only={ident_frac_ci:.2f} | gmm={bi}", flush=True)
print(f"explore: spearman(ω̂,hump)={rho:+.2f} | домены ω̂: " + ", ".join(f"{k}:{v:.2f}" for k, v in dommed.items()), flush=True)
print("МИР:", world, flush=True)

def sane(o):
    if isinstance(o, dict):  return {k: sane(v) for k, v in o.items()}
    if isinstance(o, list):  return [sane(v) for v in o]
    if isinstance(o, (float, np.floating)) and not math.isfinite(float(o)):  return None
    return o
json.dump(sane(res), open(os.path.join(HERE, "..", "results", "exp59b.json"), "w"), indent=1, allow_nan=False)
