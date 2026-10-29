"""E-39c_pre — KL pre-gate: behavioral budget for M_K-surgery (pre-stage of
E-39c, docs/E39.md §3).  REVISED per docs/E39_fixes.md (E-39b review).

ID HISTORY (fixes §0): the first implementation of this script was named
`exp39b_kl_surgery.py` — an ID collision: E-39b per TЗ §2 is the PTQ/GPTQ
gate (a different hypothesis). Renamed to E-39c_pre: this is the KL-budget
pre-stage of E-39c (surgery), not E-39b. E-39b (real, GPTQ) runs on Kaggle
(kernel-exp39b) with this metrics harness.

BUGFIX (fixes §1, the critical one): the old code applied
`W_lm[U128] += (M_kp − M_K)` — a delta in M_K-space added to W_lm-space.
Since M_K = W_lm·diag(g), the achieved M_K was M_K_old·(1−g) + M_kp·g,
i.e. the surgery was row-scaled by γ≠1 and NEITHER κ' nor KL described the
intended surgery. Fix: inverse normalization per TЗ §3.3 step 6:
    W_lm'[token_set] = M_kp · diag(γ)⁻¹.

Also fixed per review:
  §2   — κ' explicitly marked DRY-RUN (base-model û_t); post-edit κ not
         claimed; gate_pass is KL-only.
  §3   — row_diag in one space (W_lm-space: dW = dM/g vs W_lm rows).
  §4.1 — κ' vs E-39a dry-run consistency check (warn + flag, not assert).
  §4.2 — Jp per-doc float32 (no 4 GB materialization).
  §4.4 — V_new: explicit true-null/bottom-R branch mirroring E-39a (not a
         hardcoded bottom-R).
  §4.5 — explicit row assignment (no in-place += on the shared tensor).
  §5.1 — KL over attention_mask tokens (padding excluded), reported
         with/without t=0.
  §5.3 — C4 docs via ds.skip(4096).take(8): the 27b cache used the first
         per-domain C4 docs; skip ≫ 8 ⇒ disjoint (no hash, stated as
         assumption in limitations).
  §5.4 — KL_PASS = 0.05 = half of the E-16 KL leash (0.1): conservative.
  §5.5 — tied-weights recorded as a real limitation (no "moves
         consistently" claim).

Protocol (CPU, deterministic seed=0):
  1. Base forward on 8 fresh C4 docs (skip 4096), S=128 → logprobs_base.
  2. M_K = build_M_K(W_lm, U128, g); surgery: top-16 right subspace of M_K
     rotated onto the bottom-16 right-SV subspace of U=span{û_t} (E-39a
     construction, deterministic QR); M_kp = U_top Σ_top V_newᵀ + M_rest
     (Frobenius preserved — asserted).
  3. W_lm'[U128] = M_kp / g  (inverse normalization, the bugfix).
  4. Edited forward on the same docs → KL(P_edit ‖ P_base) per token.
  5. κ' dry-run on the 27b cache (J' = M_kp(I−ûûᵀ)) — preview only.

Gating: PASS iff mean KL (real tokens, t≥1) < 0.05 nats/token.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _e38_cache import D, load_layer  # noqa: E402
from exp39a_u_obs_rank import _kappa_dryrun  # noqa: E402
from prism.interventions import _resolve_lm_head_weight  # noqa: E402
from prism.jacobian import _lm_module  # noqa: E402
from prism.jstruct import build_M_K  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = "Qwen/Qwen3.5-0.8B"
OUT = os.path.join(REPO, "results", "exp39c_pre_kl_surgery.json")
SEED = 0
R = 16
N_DOCS = 8
S_LEN = 128
C4_SKIP = 4096        # 27b cache used the first per-domain C4 docs; 4096 ≫ 8
# E-16 KL leash = 0.1 nats; half-leash = conservative behavioral budget.
KL_PASS = 0.05
KAPPA_39A_REF = 0.0042   # E-39a dry-run κ' (same V_new construction)
QR_RANK_TOL = 1e-6


def _c4_docs(n_docs, s_len, tokenizer, skip=C4_SKIP):
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.skip(skip).take(n_docs)
    texts = [s["text"] for s in ds]
    if len(texts) < n_docs:
        raise RuntimeError(f"C4 streaming yielded {len(texts)}/{n_docs} docs")
    enc = tokenizer(texts, truncation=True, max_length=s_len,
                    padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def _logprobs(model, ids):
    out = model(input_ids=ids).logits
    return torch.log_softmax(out.float(), dim=-1)


def _obs_subspace(sU, VU, R_r, report):
    """True null space if ≥ R dims, else bottom-R right SVs (E-39a branch).

    Fixes §4.4: explicit branch + assertions, not a hardcoded bottom-R.
    """
    null_thr = 1e-6 * sU[0].item()
    null_idx = (sU <= null_thr).nonzero().flatten()
    if len(null_idx) >= R_r:
        report["mode"] = "true_null_space"
        return VU[:, null_idx[:R_r]], report
    P = VU[:, -R_r:]
    report["mode"] = f"bottom_{R_r}_right_svs_of_U (no true null space)"
    assert P.shape[1] == R_r, f"VU width {P.shape[1]} < R={R_r}"
    return P, report


@torch.no_grad()
def main(results_dir=None):
    # no_grad: inference-only script; W_lm is a requires_grad parameter —
    # without this, row_diag tensors carry grad (np.percentile crashes,
    # float() warns).
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = results_dir if results_dir else os.path.join(REPO, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp39c_pre_kl_surgery.json")

    ck = load_layer(23, want=("H",))
    H, U128 = ck["H"], ck["U128"]
    u = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    U = u.reshape(-1, D).double()
    print("U SVD ...")
    _, sU, VU = torch.linalg.svd(U, full_matrices=False)
    sub_rep = {}
    P_sub, sub_rep = _obs_subspace(sU, VU, R, sub_rep)
    sv = torch.linalg.svdvals(P_sub)
    assert sv[-1] > QR_RANK_TOL * sv[0], "V_new pool rank-deficient"
    V_new = torch.linalg.qr(P_sub)[0]
    assert (V_new.T @ V_new - torch.eye(R, dtype=V_new.dtype)).abs().max() < 1e-8

    print("loading model fp32 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager")
    model.eval()

    # --- 1. base forward (BEFORE any edit) ---
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids, attn = _c4_docs(N_DOCS, S_LEN, tok)
    print(f"C4 docs: {ids.shape} (skip {C4_SKIP})")
    print("base forward ...")
    lp_base = _logprobs(model, ids)

    # --- 2. surgery (M_K-space) ---
    W_lm = _resolve_lm_head_weight(model)
    assert W_lm.shape[1] == D, W_lm.shape
    vocab = W_lm.shape[0]
    g = _lm_module(model).norm.weight + 1                    # [D]
    assert g.min() > 1e-6, f"γ has zeros: min={g.min()}"
    g_dev = g.to(W_lm.device, dtype=W_lm.dtype)
    M_K = build_M_K(W_lm, list(U128), g).double().detach()   # [K, D]
    mnorm_old = float(torch.linalg.norm(M_K))
    Uk, sk, Vk = torch.linalg.svd(M_K, full_matrices=False)
    U_top, s_top = Uk[:, :R], sk[:R]
    M_rest = Uk[:, R:] @ torch.diag(sk[R:]) @ Vk[R:, :]
    M_kp = U_top @ torch.diag(s_top) @ V_new.T + M_rest
    mnorm_new = float(torch.linalg.norm(M_kp))
    # Frobenius preserved by the orthogonal top-16 rotation (smoke test)
    assert abs(mnorm_new - mnorm_old) < 1e-6, (
        f"M_kp Frobenius drift: {mnorm_old:.6f} → {mnorm_new:.6f}")

    # --- 3. apply: INVERSE NORMALIZATION (fixes §1 bugfix) ---
    # TЗ §3.3 step 6: W_lm'[token_set] = M_K' · diag(γ)⁻¹
    W_new_rows = (M_kp / g.double()[None, :]).to(W_lm.dtype).to(W_lm.device)
    # row diagnostics in ONE space (W_lm-space): fixes §3
    dW = W_new_rows.double().cpu() - W_lm[list(U128)].double().cpu()
    row_norm = W_lm[list(U128)].double().cpu().norm(dim=-1).clamp_min(1e-9)
    dratio = dW.norm(dim=-1) / row_norm
    row_diag = {
        "space": "W_lm (dW = M_kp/g − W_lm_old; ratios in W_lm-space)",
        "mean_dW_over_row": float(dratio.mean()),
        "max_dW_over_row": float(dratio.max()),
        "p90_dW_over_row": float(np.percentile(dratio, 90)),
    }
    # explicit assignment, not in-place += on the shared tensor (fixes §4.5)
    W_lm.data[list(U128)] = W_new_rows
    print(f"M_K norm: {mnorm_old:.4f} → {mnorm_new:.4f} (Frobenius preserved)")
    print(f"row edit: mean dW/row = {row_diag['mean_dW_over_row']:.4f}")

    # --- 4. edited forward + KL (attention-masked, fixes §5.1/§5.2) ---
    print("edited forward ...")
    lp_edit = _logprobs(model, ids)
    p_edit = lp_edit.exp()
    kl = (p_edit * (lp_edit - lp_base)).sum(dim=-1)          # [N, S]
    real = attn.bool()                                        # excludes padding
    t1 = real.clone(); t1[:, 0] = False                       # excludes t=0
    kl_mean_real = float(kl[real].mean())
    kl_mean_t1 = float(kl[t1].mean())
    kl_mean_all = float(kl.mean())
    per_doc = [float(kl[i, real[i]].mean()) for i in range(N_DOCS)]
    print(f"mean KL (real tokens) = {kl_mean_real:.5f} nats/token "
          f"(gate: {KL_PASS}); t≥1: {kl_mean_t1:.5f}; all-incl-pad: {kl_mean_all:.5f}")

    # --- 5. κ' dry-run (preview only, fixes §2) ---
    u64 = u.double()
    kap_out = _kappa_dryrun(M_kp, u64, H.shape[0], S_LEN, R)
    kap_new, kap_ci = kap_out["kappa"], kap_out["kappa_ci"]
    if kap_new is None:
        warnings.warn("κ' dry-run: no valid κ (n_valid=%d) — preview unavailable"
                      % kap_out["n_valid"])
        consistent = False
    else:
        consistent = abs(kap_new - KAPPA_39A_REF) < 0.003
        if not consistent:
            warnings.warn(f"κ' mismatch: {kap_new:.5f} vs E-39a dry-run "
                          f"≈ {KAPPA_39A_REF}")
    print(f"κ' dry-run (27b cache, base û_t): {kap_new} "
          f"(39a ≈ {KAPPA_39A_REF}, consistent={consistent})")

    gate_pass = bool(kl_mean_real < KL_PASS)
    res = {
        "experiment": "E-39c_pre: KL pre-gate for M_K-surgery",
        "id_note": "Renamed from exp39b_kl_surgery (ID collision with the "
                   "real E-39b = PTQ/GPTQ gate, TЗ §2; see E39_fixes §0).",
        "tz": "docs/E39.md §3 (pre-stage), fixes: docs/E39_fixes.md",
        "params": {"seed": SEED, "model": MODEL, "n_docs": N_DOCS,
                   "s_len": S_LEN, "r": R, "kl_pass": KL_PASS,
                   "kl_pass_rationale": "half of the E-16 KL leash (0.1)",
                   "c4_skip": C4_SKIP, "vocab": vocab},
        "v_new_subspace": sub_rep,
        "sigma_bottom16_U": [float(x) for x in sU[-R:]],
        "m_k_norm": {"before": mnorm_old, "after": mnorm_new,
                     "frobenius_preserved": abs(mnorm_new - mnorm_old) < 1e-6},
        "row_diag": row_diag,
        "kappa_dryrun": {
            "median": kap_new, "ci": list(kap_ci) if kap_ci is not None else None,
            "n_valid": kap_out["n_valid"],
            "kappa_median_all_triples": kap_out["kappa_median_all_triples"],
            "status": "DRY-RUN PREVIEW (base-model û_t; the edited model's "
                      "û_t differ due to tied weights — NOT a measurement)",
            "ref_39a": KAPPA_39A_REF, "consistent": consistent,
        },
        "kl_nats_per_token": {
            "mean_real_tokens": kl_mean_real,
            "mean_t_ge_1": kl_mean_t1,
            "mean_incl_padding": kl_mean_all,
            "per_doc_real": per_doc,
            "max": float(kl.max()),
            "p99_real": float(np.percentile(kl[real], 99)),
        },
        "gate": {"metric": "mean KL over real tokens, t≥0, padding excluded",
                 "threshold": KL_PASS, "value": kl_mean_real,
                 "pass": gate_pass,
                 "basis": "KL only (κ' is a dry-run preview, not a gate input)"},
        "limitations": [
            "Tied weights: editing W_lm = editing embed_tokens — the forward "
            "changes beyond the readout (TЗ §3.5 acknowledges this; not "
            "claimed away).",
            "C4 disjointness: ds.skip(4096) vs the 27b cache's first docs — "
            "disjoint by offset, NOT hash-verified.",
            "KL_PASS=0.05 is a protocol choice (half of E-16's 0.1 leash), "
            "not derived from data.",
            "κ' is a dry-run preview on base-model û_t (see status); the real "
            "E-39c measurement happens on the edited model.",
        ],
        "runtime_s": round(time.time() - t0, 1),
        "deterministic": True,
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nE-39c_pre done in {time.time() - t0:.1f}s → {out_path}")
    print(f"gate: {'PASS' if gate_pass else 'FAIL'} "
          f"(mean KL {kl_mean_real:.5f} < {KL_PASS})")


if __name__ == "__main__":
    main()
