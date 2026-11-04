"""Аудит E-39b: magnitude-matched null для κ — случайный ΔW той же величины, что GPTQ-дрейф.
Всё аналитически на кэше (analytic_J воспроизводит JR с rel 1.3e-07, проверено выше)."""
import sys, math, json, numpy as np, torch
sys.path.insert(0, ".")
import exp28_leaf_structure as E28
from _e38_cache import jcache_path
from transformers import AutoModelForCausalLM

L, r, D_ = 23, 16, 1024
ck = torch.load(jcache_path(L), map_location="cpu", weights_only=False)
m = AutoModelForCausalLM.from_pretrained("/data/Qwen/Qwen3.5-0.8B", dtype=torch.float32,
                                         attn_implementation="eager", low_cpu_mem_usage=True).eval()
W = m.lm_head.weight.detach().float().numpy()
g = (1.0 + m.model.norm.weight.detach().float()).numpy().astype(np.float64)
U = ck["U128"].numpy().astype(np.int64) if hasattr(ck["U128"], "numpy") else np.asarray(ck["U128"])
WU = W[U].astype(np.float64)                      # [K, D]
DOMS, NS, S = ck["domains"], ck["n_samples"], 128
Hs = [ck["H"][(d, s, t)].double().numpy() for d in DOMS for s in range(NS) for t in range(S)]
Hst = np.stack(Hs)                                # [4096, D]
del ck
print("H stack", Hst.shape, flush=True)

def J_of(WU_):
    """J_t = √D/‖h‖ (M − (M u)uᵀ), M = W_U ⊙ g — векторизованно по всем 4096 позициям."""
    M = WU_ * g[None, :]
    n = np.linalg.norm(Hst, axis=1, keepdims=True)
    u = Hst / n
    Mu = u @ M.T                                    # [N, K]
    return (math.sqrt(D_) / n)[:, 0][:, None, None] * (M[None, :, :] - Mu[:, :, None] * u[:, None, :])

def drift_med(J0, J1):
    d = np.linalg.norm((J1 - J0).reshape(len(J0), -1), axis=1) / np.linalg.norm(J0.reshape(len(J0), -1), axis=1)
    return float(np.median(d))

def kappa(J):
    """Медиана κ(r=16) по всем документам (стек [128,K,D] на документ), как в E-39b/E-28."""
    ks = []
    for i in range(0, len(J), S):
        Jd = torch.from_numpy(J[i:i + S].astype(np.float32))
        _, _, k2, _ = E28.document_profile(Jd, [1, r, 32], r_main=r)
        ks.append(k2[~np.isnan(k2)])
    return float(np.median(np.concatenate(ks)))

J0 = J_of(WU)
k0 = kappa(J0)
print("базовый κ (аналитически): %.6f  (E-39b: 0.01623332)" % k0, flush=True)
res = {"kappa_base_analytic": k0, "kappa_base_e39b": 0.01623332313133695,
       "kappa_gptq4_e39b": 0.016042994668936907, "ratio_gptq_e39b": 0.9882754466931893,
       "drift_target_e39b": 0.12, "nulls": []}
g_std = WU.std()
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    G = rng.standard_normal(WU.shape) * g_std
    lo, hi = 0.01, 2.0
    for _ in range(6):                       # бисекция по σ под медианный drift = 0.12
        s_ = (lo + hi) / 2
        if drift_med(J0, J_of(WU + s_ * G)) < 0.12: lo = s_
        else: hi = s_
    s_ = (lo + hi) / 2
    J1 = J_of(WU + s_ * G)
    dr, k1 = drift_med(J0, J1), kappa(J1)
    res["nulls"].append({"seed": seed, "sigma_rel_to_Wstd": s_, "drift_median": dr,
                         "kappa": k1, "ratio": k1 / k0})
    print("null seed %d: drift %.4f  κ %.6f  ratio %.4f" % (seed, dr, k1, k1 / k0), flush=True)
json.dump(res, open("/tmp/null39b.json", "w"), indent=1)
print("GPTQ ratio 0.9883 | null ratios:", [round(n["ratio"], 4) for n in res["nulls"]])
