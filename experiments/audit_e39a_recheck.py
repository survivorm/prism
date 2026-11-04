#!/usr/bin/env python3
"""Аудит E-39a (независимая пересборка гейт-чисел): r_obs, r_eff(body,0.05),
режим пула, ρ_noise(a/b/c), угол между V_a и V_b. Только H из кэша + веса M_K.
Сверка с results/exp39a_u_obs_rank.json."""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
from _e38_cache import load_layer

SEED, R, R_POOL, BODY_LO, BODY_HI, QR_RANK_TOL = 0, 16, 64, 16, 126, 1e-6
TAUS = (0.01, 0.05, 0.1, 0.5)

def r_eff(sig):
    s1 = sig[0].item()
    return {str(t): int((sig > t * s1).sum().item()) for t in TAUS}

def qr_guarded(P):
    s = torch.linalg.svdvals(P)
    return torch.linalg.qr(P)[0], bool(s[-1] > QR_RANK_TOL * s[0])

ck = load_layer(23, want=("H",))
H, U128 = ck["H"], ck["U128"]
u = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-12)
U = u.reshape(-1, 1024).double()
_, sU, Vh = torch.linalg.svd(U, full_matrices=False)
r_obs = int(len(sU) - int((sU <= 1e-6 * sU[0].item()).sum()))
body = np.zeros(128, dtype=bool); body[BODY_LO:BODY_HI+1] = True
Ub = U.reshape(32, 128, 1024)[:, body].reshape(-1, 1024)
_, sb, _ = torch.linalg.svd(Ub, full_matrices=False)
print("r_obs =", r_obs, "(ожидание 1024)   r_eff(body,0.05) =", r_eff(sb)["0.05"], "(ожидание 178)")
VU = Vh.T
null_thr = 1e-6 * sU[0].item()
null_idx = (sU <= null_thr).nonzero().flatten()
mode = "true_null" if len(null_idx) >= R else f"bottom_{R_POOL}"
P_pool = VU[:, null_idx] if mode == "true_null" else VU[:, -R_POOL:]
print("null_mode =", mode, "| пул ортонормирован:",
      bool((P_pool.T @ P_pool - torch.eye(R_POOL, dtype=torch.float64)).abs().max() < 1e-10))

# стратегии — тот же rng-порядок, что в скрипте
rng = np.random.default_rng(SEED)
Aa = torch.from_numpy(rng.normal(size=(R_POOL, R))).double()
V_a, ok_a = qr_guarded(P_pool @ torch.linalg.qr(Aa)[0])

from transformers import AutoModelForCausalLM
from prism.interventions import _resolve_lm_head_weight
from prism.jacobian import _lm_module
from prism.jstruct import build_M_K
os.environ.setdefault("HF_HUB_OFFLINE", "1")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B", dtype=torch.float32,
                                             attn_implementation="eager", local_files_only=True)
W_lm = _resolve_lm_head_weight(model); g = _lm_module(model).norm.weight + 1
M_K = build_M_K(W_lm, list(U128), g).double().detach()
del model
Uk, sk, Vk = torch.linalg.svd(M_K, full_matrices=False)
V_top = Vk[:R, :].T
Pn = P_pool @ P_pool.T
V_b, ok_b = qr_guarded(Pn @ V_top)
V_c, ok_c = qr_guarded(torch.cat([V_a[:, :8], V_b[:, :8]], dim=1))
u64 = u.double().reshape(-1, 1024)
for name, V in (("a_random", V_a), ("b_principal", V_b), ("c_balanced", V_c)):
    print(f"rho_noise[{name}] = {(u64 @ V).pow(2).sum(-1).mean().item():.6f}  qr_ok={ok_a and ok_b and ok_c}")
cos = torch.linalg.svdvals(V_a.T @ V_b)
print("max principal angle a↔b =", float(torch.acos(cos.clamp(-1,1)).max()*180/np.pi), "° (ожидание 89.1)")
print("artefact: r_eff=178, ρ = 0.000193 / 0.000257 / 0.000211, угол 89.1°, distinct=True")
