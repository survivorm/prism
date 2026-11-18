#!/usr/bin/env python3
"""AT2d dose recompute — disk-only, БЕЗ форвардов (ревью r5, F3).

Канон проекта (paper §3.1/§2.5, модель Qwen3_5RMSNorm.forward:
output = (x/rms(x)) * (1 + weight), weight init=0):
  δn(арм) = (1+w) ⊙ [(I − ûûᵀ) (P_V (h/rms(h)))],  û = h/‖h‖
  dose = mean_t ‖δn_t‖²

Код r5.1 считал (баг F3):
  δn_old = (I − ûûᵀ) [ (P_V (h/rms)) ⊙ w ]   # w, не 1+w; gain ПЕРВЫЙ, радиальный ПОТОМ

Скрипт:
(1) OLD-формула — ДОЛЖНА дословно совпасть с dose_tab/alpha_tab артефакта
    (санитарная проверка верности пересчёта);
(2) NEW-формула (канон) — отчёт D_new, ratio = D_new/D_old, alpha_new,
    подразумеваемое рассогласование dose:
    mismatch(арм, док) = alpha_old² * D_new(арм) / D_new(randG)  (1.0 = матч).
Решение по перепуску: max |mismatch − 1| < 10% → «matched by construction»
состоит количественно; больше → перепуск r6.
"""
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import AT2d as A  # noqa: E402  (импорт без main: константы + функции)

LAYS = (6, 12, 18, 22, 19)
KS = (4, 8)
ART = os.path.join(A.ROOT, "results", "AT2d", "AT2d.json")

t0 = time.time()
art = json.load(open(ART))
arms_art = art["arms"]
print(f"artifact: {ART} (md5 сверен вне скрипта)")

def build_basis(L):
    """Дословно AT2d.main(): J/gbar-базисы + per-doc pca/randG + haar."""
    ck = torch.load(os.path.join(A.CACHE_DIR, f"exp27b_jcache_L{L}.pt"),
                    map_location="cpu", weights_only=False)
    Hf = A.f2a_inv(ck["H"], (A.D,)).numpy().astype(np.float32)
    JRf = A.f2a_inv(ck["JR"], (128, A.D))
    Pf = A.f2a_inv(ck["P128"], (128,))
    Jagg = JRf.mean(dim=(0, 1)).double()
    Gagg = Jagg.T @ Jagg
    Gf = A.gbar_from(JRf, Pf, range(A.S_LEN))
    VJ = {}
    for k in KS:
        VJ[f"top{k}"] = A.eigh_basis(Gagg, k, "desc")
        VJ[f"weak{k}"] = A.eigh_basis(Gagg, k, "asc")
        VJ[f"gbar{k}"] = A.eigh_basis(Gf, k, "asc")
    del Gagg, Gf
    Vpca, Vrg = {}, {}
    for k in KS:
        arr = np.zeros((A.N_DOCS, A.D, k), np.float32)
        for d in range(A.N_DOCS):
            Hc = torch.from_numpy(np.ascontiguousarray(
                Hf[d] - Hf[d].mean(0, keepdims=True))).double()
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            arr[d] = Vh[:k].cpu().numpy().T.astype(np.float32)
        Vpca[k] = arr
        arr = np.zeros((A.N_DOCS, A.D, k), np.float32)
        for d in range(A.N_DOCS):
            rng = np.random.default_rng(200_000 + L * 10 + k + 1000 * d)
            arr[d] = np.linalg.qr(rng.standard_normal((A.D, k)))[0]
        Vrg[k] = arr
    rng_base = np.random.default_rng(A.SEED)
    G32 = rng_base.standard_normal((A.D, 32))
    Vhaar = {k: np.linalg.qr(G32[:, :k])[0] for k in KS}
    return Hf, VJ, Vpca, Vrg, Vhaar

def doses(Hdd, V, w, win0, win1, new):
    """Per-doc (mean по окну) dose. new=False: формула r5.1 (bug);
    new=True: канон (radial first, g = 1 + w). Возврат (dose_mean, dose1_mean),
    dose1 — при g ≡ 1 (fingerprint: ≈ E*1024)."""
    Hw = torch.from_numpy(np.ascontiguousarray(Hdd[win0:win1])).double()
    Vd = torch.as_tensor(np.ascontiguousarray(V), dtype=torch.float64)
    vh = Hw @ Vd
    proj = vh @ Vd.T
    hn = (Hw ** 2).mean(1, keepdim=True).sqrt()
    u = Hw / (Hw.norm(dim=1, keepdim=True) + 1e-30)
    a = proj / hn
    P_a = a - (a * u).sum(1, keepdim=True) * u      # radial first (канон)
    if new:
        r = P_a * (1.0 + w[None, :])                 # gain ПОТОМ, g = 1 + w
    else:
        r0 = a * w[None, :]                          # gain ПЕРВЫЙ, g = w (bug)
        r = r0 - (r0 * u).sum(1, keepdim=True) * u
    r1 = P_a                                          # g ≡ 1
    return float((r ** 2).sum(1).mean()), float((r1 ** 2).sum(1).mean())

# веса (input_layernorm СЛЕДУЮЩЕГО слоя / final norm для L23) — как в AT2d
cfg_j = json.load(open(os.path.join(A.MODEL_DIR, "config.json")))
ltypes = cfg_j["text_config"]["layer_types"] if "text_config" in cfg_j else cfg_j["layer_types"]
from safetensors import safe_open
W = {}
with safe_open(A.MODEL_SFT, framework="pt") as f:
    for L in LAYS:
        key = (f"model.language_model.layers.{L + 1}.input_layernorm.weight"
               if L + 1 < len(ltypes)
               else "model.language_model.norm.weight")
        W[L] = f.get_tensor(key).double().numpy()
print(f"weights loaded: L={list(W)}; w-rms: "
      + " ".join(f"L{L}={np.sqrt((W[L]**2).mean()):.4f}" for L in LAYS))

out = {"per_cell": {}, "fingerprint": {}}
for L in LAYS:
    Hf, VJ, Vpca, Vrg, Vhaar = build_basis(L)
    for k in KS:
        cell = f"L{L}k{k}"
        per_arm = {}
        for d in range(A.N_DOCS):
            wL = W[L]
            for name in ("top", "weak", "gbar", "pca", "haar", "randG"):
                V = VJ[f"{name}{k}"] if name in ("top", "weak", "gbar") else (
                    Vpca[k][d] if name == "pca" else
                    (Vhaar[k] if name == "haar" else Vrg[k][d]))
                d_old, d1 = doses(Hf[d], V, wL, 0, A.AB_W, new=False)
                d_new, _ = doses(Hf[d], V, wL, 0, A.AB_W, new=True)
                per_arm.setdefault(name, []).append(
                    (d_old, d_new, d1))
        # сверка с артефактом (OLD должен совпасть; randG без "D" в per_doc)
        worst = 0.0
        n_sane = 0
        for name in ("top", "weak", "gbar", "pca", "haar", "randG"):
            arts = arms_art[cell]["P_future"][name]["per_doc"]
            for d in range(A.N_DOCS):
                if "D" not in arts[d]:
                    continue
                aD = arts[d]["D"]
                oD = per_arm[name][d][0]
                rel = abs(oD - aD) / max(aD, 1e-30)
                worst = max(worst, rel)
                n_sane += 1
        # агрегаты
        agg = {}
        rg_old = np.array([per_arm["randG"][d][0] for d in range(A.N_DOCS)])
        rg_new = np.array([per_arm["randG"][d][1] for d in range(A.N_DOCS)])
        for name in ("top", "weak", "gbar", "pca", "haar", "randG"):
            o = np.array([per_arm[name][d][0] for d in range(A.N_DOCS)])
            nw = np.array([per_arm[name][d][1] for d in range(A.N_DOCS)])
            a_old = np.sqrt(rg_old / o)
            a_new = np.sqrt(rg_new / nw)
            mismatch = (a_old ** 2) * nw / rg_new   # 1.0 = точный матч
            agg[name] = {
                "D_old_mean": float(o.mean()), "D_new_mean": float(nw.mean()),
                "ratio_new_old_mean": float((nw / o).mean()),
                "ratio_new_old_range": [float((nw / o).min()), float((nw / o).max())],
                "alpha_old_mean": float(a_old.mean()),
                "alpha_new_mean": float(a_new.mean()),
                "mismatch_mean": float(mismatch.mean()),
                "mismatch_range": [float(mismatch.min()), float(mismatch.max())],
            }
        # fingerprint g≡1 (randG): ≈ E*1024? (E — считается из кэша, в
        # артефакте randG пер-док без E_raw)
        d1_rg = np.array([per_arm["randG"][d][2] for d in range(A.N_DOCS)])
        E_rg = np.array([])
        for d in range(A.N_DOCS):
            Hw = torch.from_numpy(np.ascontiguousarray(Hf[d][:A.AB_W])).double()
            Vd = torch.as_tensor(np.ascontiguousarray(Vrg[k][d]), dtype=torch.float64)
            vh = Hw @ Vd
            E_rg = np.append(E_rg, float(((vh ** 2).sum(1) / (Hw ** 2).sum(1)).mean()))
        out["fingerprint"].setdefault(str(L), {})[f"randG{k}"] = {
            "D_g1_mean": float(d1_rg.mean()), "E1024_mean": float((E_rg * 1024).mean())}
        out["per_cell"][cell] = {"old_vs_artifact_max_rel_diff": float(worst),
                                 "n_sanity_checked": int(n_sane),
                                 "per_arm": agg}
    del Hf, VJ, Vpca, Vrg, Vhaar
    print(f"L{L}: done ({time.time() - t0:.0f}s)")

json.dump(out, open("/tmp/at2d_dose_check.json", "w"), indent=1, ensure_ascii=False)
print(f"\n=== SANITY: OLD формула vs артефакт (max rel diff per cell) ===")
for cell, v in out["per_cell"].items():
    print(f"  {cell}: max_rel_diff = {v['old_vs_artifact_max_rel_diff']:.3e}")
print("\n=== FINGERPRINT g≡1 (D_g1 vs E*1024; reviewer: 4.096 vs ~4.1) ===")
for L, cells in out["fingerprint"].items():
    for kk, v in cells.items():
        print(f"  L{L} {kk}: D_g1 = {v['D_g1_mean']:.4f}, E*1024 = {v['E1024_mean']:.4f}")
print("\n=== PER ARM: D_old → D_new, ratio, α, mismatch (1.0 = матч) ===")
for cell, v in out["per_cell"].items():
    print(f"  {cell}:")
    for name, a in v["per_arm"].items():
        print(f"    {name:6s} D_old={a['D_old_mean']:.5f} D_new={a['D_new_mean']:.5f} "
              f"ratio={a['ratio_new_old_mean']:.3f} "
              f"α_old={a['alpha_old_mean']:.4f} α_new={a['alpha_new_mean']:.4f} "
              f"mismatch={a['mismatch_mean']:.4f} "
              f"range=[{a['mismatch_range'][0]:.3f},{a['mismatch_range'][1]:.3f}]")
print(f"\nDONE ({time.time() - t0:.0f}s) -> /tmp/at2d_dose_check.json")
