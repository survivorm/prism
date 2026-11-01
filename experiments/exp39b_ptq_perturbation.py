"""E-39b — κ(L23) и M_K-структура под GPTQ-4bit: что ломает квантование (docs/E39.md §2).

PTQ (GPTQ/AWQ) квантует lm_head.weight → M_K → J_t, при этом û_t заморожен
(активации кэшированы в L23): неконтролируемая модификация M_K.
Эксперимент измеряет, насколько PTQ ломает лист.

Уровни (spec):
  * "fp16"  — неоквантованный baseline: загруженный fp16 W_lm (→ fp64).
              Воспроизводит кэшированное J в L23 до ~1e-7 (self-check).
  * "int8"  — RTN (per-group min-max, round-to-nearest), без Hessian.
  * "gptq4" — GPTQ 4-bit, group_size=128, desc_act=True, damp=0.01
              (port IST-DASLab/gptq: H = (2/n)·Σxxᵀ, cholesky-inverse,
              per-group scale/zero, in-block error compensation).

Метрики (per layer L ∈ {6,12,18,19,22,23}, per level):
  * κ(L)      = ‖H_012 − I‖_F, top-16 right-singular bases (E-38d, r=16).
  * β/σ₁(M_K) — consistency (E-38e, baseline ≈ 0.9 ± 0.15).
  * cos(u_g, 1/√K) — gauge-axis alignment (E-37).
  * J_rel_drift = ‖J_q − J‖_F / ‖J‖_F (per (doc,t) median/p95/max).
  * PPL на C4 holdout (lm_head.replace не используется — tied weights;
    logits = norm(h_last) @ W_qᵀ в fp32).

GATE на κ(L23) (spec §2.4):
  < 0.02  → PTQ не ломает лист, flatness не приоритет для PTQ.
  0.02–0.10 → умеренно, E-39c как post-hoc recovery.
  > 0.10  → катастрофически, E-39c срочно + E-39e flatness-регуляризатор.

C4 split (disjoint from E-16/E-27b calibration):
  * GPTQ Hessian:  C4 train skip 8192, 128 docs × 128 tok = 16384 vectors.
  * PPL holdout:   C4 val take 32 docs (fallback train skip 24576).

J_t recompute (kappa/beta/u_g/drift) uses the jcache H fields (32 docs ×
128 pos, the 27b split) per spec — the same split for base and quantized,
so base-vs-quantized is internally consistent.
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

# NB: no sys.path insertions — locally the script dir is already sys.path[0];
# on Kaggle the notebook cell has no __file__ (the old lines crashed the cell).

from _e38_cache import D, load_layer  # noqa: E402

# --- inlined prism helpers (canonical: prism/{jacobian,interventions,jstruct}.py) ---
# build_kernel.py inlines only a fixed prism module list and strips prism
# imports from the exp cell; prism.jstruct/_interventions are not in the list
# (relative cross-imports). These three are pure tensor ops, copied verbatim.
def _lm_module(model):
    """Text model with .layers/.norm (prism/jacobian.py:_lm_module)."""
    mm = model.model
    if hasattr(mm, "layers"):
        return mm
    return mm.language_model


def _resolve_lm_head_weight(model) -> torch.Tensor:
    """model.lm_head.weight, get_output_embeddings() fallback."""
    lm = getattr(model, "lm_head", None)
    if lm is not None and getattr(lm, "weight", None) is not None:
        return lm.weight
    out = model.get_output_embeddings()
    if out is None or getattr(out, "weight", None) is None:
        raise AttributeError("model has no lm_head.weight / output embeddings")
    return out.weight


def build_M_K(W_lm, token_set, g):
    """M_K = W_lm[token_set] · diag(g) → [K, D] (prism/jstruct.py verbatim)."""
    if len(token_set) == 0:
        raise ValueError("build_M_K: token_set is empty")
    if W_lm.dim() != 2:
        raise ValueError(f"build_M_K: W_lm must be 2-D, got {W_lm.dim()}-D")
    if g.dim() != 1 or W_lm.shape[1] != g.shape[0]:
        raise ValueError("build_M_K: W_lm.shape[1] != g dim")
    V = W_lm.shape[0]
    for i, idx in enumerate(token_set):
        if idx < 0 or idx >= V:
            raise ValueError(f"build_M_K: token_set[{i}] = {idx} out of [0, {V})")
    if not torch.isfinite(g).all():
        raise ValueError("build_M_K: non-finite g")
    ts = torch.tensor(token_set, device=W_lm.device, dtype=torch.long)
    return W_lm[ts] * g.to(W_lm.dtype).unsqueeze(0)

MODEL = "Qwen/Qwen3.5-0.8B"
SEED = 0
R = 16
# jcache layers (superset of spec's 5 — L19 added for completeness)
L_CACHED = [6, 12, 18, 19, 22, 23]
# spec's 5 layers for the gate report
L_GATE = [6, 12, 18, 22, 23]
# C4 splits
C4_CAL_SKIP = 8192   # GPTQ Hessian: 128 docs × 128 tok
C4_PPL_SKIP = 24576  # PPL holdout (val split preferred, this is fallback)
N_CAL = 128
S_CAL = 128
N_PPL = 32
S_PPL = 128
# GPTQ params (spec: group_size=128, desc_act=True, damp=0.01)
GPTQ_BITS = 4
GPTQ_GROUP = 128
GPTQ_DAMP = 0.01
# RTN (int8) params
RTN_BITS = 8
RTN_GROUP = 128
# Self-check tolerance (local run: rel err max 7.2e-7)
SELFCHECK_TOL = 1e-5
# Gate thresholds (spec §2.4)
KAPPA_PASS = 0.02
KAPPA_MODERATE = 0.10


def _c4_docs(n_docs, s_len, tokenizer, skip=0, split="train"):
    """Load n_docs C4 docs of length s_len, skipping `skip` docs."""
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
    texts = []
    n_skip = 0
    for doc in ds:
        if n_skip < skip:
            n_skip += 1
            continue
        t = doc["text"]
        if t:
            texts.append(t[:s_len])
        if len(texts) >= n_docs:
            break
    if len(texts) < n_docs:
        raise RuntimeError(f"C4 streaming yielded {len(texts)}/{n_docs} docs")
    enc = tokenizer(texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=s_len)
    return enc["input_ids"], enc["attention_mask"]


def _resolve_device(dev):
    """Resolve a device string (from runner) to a torch.device."""
    if dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


def _procrustes_batched(V_a, V_b):
    """Vectorized Procrustes: V_a, V_b [B, D, r] orthonormal columns.
    M = V_b^T V_a, SVD M = P Σ Qt, R = Qt^T P^T.
    Returns (R [B, r, r] fp64, residual_sq [B]).
    """
    M = torch.bmm(V_b.transpose(1, 2), V_a).double()
    P, s, Qt = torch.linalg.svd(M, full_matrices=False)
    R = torch.bmm(Qt.transpose(1, 2), P.transpose(1, 2))
    res = (2.0 * V_a.shape[2] - 2.0 * s.sum(dim=-1)).clamp(min=0.0)
    return R, res


def _kappa_from_J(J, r, device, chunk_docs=4):
    """κ(L) = ‖H_012 − I‖_F over triples (t, t+1, t+2), r=16.

    J: [n_docs, S, K, D] (fp32/64). V_i = top-r right-singular basis of J_i.
    H_012 = R_01 @ R_12 @ R_20 in fp64. Returns kappa_median (float).
    SVD in fp32 (T4 fp64 SVD too slow); batched per doc-chunk.
    """
    n, S, K, Dd = J.shape
    n_triples = S - 2
    Jf = J.float().to(device)
    # Batched SVD per doc-chunk: V_i [D, r] top-r right-singular basis.
    V_all = torch.empty(n * S, Dd, r, device=device, dtype=torch.float32)
    for c0 in range(0, n, chunk_docs):
        c1 = min(c0 + chunk_docs, n)
        B = (c1 - c0) * S
        Jb = Jf[c0:c1].reshape(B, K, Dd)  # [B, K, D]
        _, s, Vh = torch.linalg.svd(Jb, full_matrices=False)  # Vh [B, K, D]: right-SVs as rows
        V_all[c0 * S:c1 * S] = Vh[:, :r, :].transpose(1, 2)  # [B, D, r] right-SV columns
        del Jb, s, Vh
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Per-triple: V0, V1, V2 at (t, t+1, t+2). Doc-major flat index.
    i_doc = torch.arange(n, device=device).repeat_interleave(n_triples)
    t_pos = torch.arange(n_triples, device=device).repeat(n)
    f0 = (i_doc * S + t_pos)
    f1 = f0 + 1
    f2 = f0 + 2
    kappa_list = []
    # Process in flat chunks to bound memory.
    n_flat = n * n_triples
    chunk = max(1, min(n_flat, (1500 * 10**6) // (3 * Dd * r * 4 + 48 * r * r)))
    for b0 in range(0, n_flat, chunk):
        b1 = min(b0 + chunk, n_flat)
        V0 = V_all[f0[b0:b1]]  # [B, D, r]
        V1 = V_all[f1[b0:b1]]
        V2 = V_all[f2[b0:b1]]
        R01, _ = _procrustes_batched(V0, V1)
        R12, _ = _procrustes_batched(V1, V2)
        R20, _ = _procrustes_batched(V2, V0)
        H = torch.einsum("bij,bjk,bkl->bil", R01, R12, R20)  # [B, r, r] fp64
        I = torch.eye(r, device=device, dtype=torch.float64)
        kappa = torch.linalg.norm(H - I, ord="fro", dim=[1, 2])  # [B]
        kappa_list.append(kappa.cpu().numpy())
        del V0, V1, V2, R01, R12, R20, H
        if device.type == "cuda":
            torch.cuda.empty_cache()
    kappa_arr = np.concatenate(kappa_list)
    return float(np.median(kappa_arr))


def _beta_sigma1(M_K, H):
    """β/σ₁(M_K) consistency metric (E-38e).

    w = M_K^T (1/√K); per position: α_t = û_t^T w, β_t = ‖w − α_t û_t‖.
    û_t from cached H. Returns (median β/σ₁(M_K), n_valid).
    """
    K, Dd = M_K.shape
    w = M_K.T @ (torch.ones(K, device=M_K.device, dtype=M_K.dtype) / np.sqrt(K))
    s1 = torch.linalg.svdvals(M_K.float())[0].item()
    beta_list = []
    hnorm = H.norm(dim=2)
    U_hat = H / hnorm.clamp(min=1e-12).unsqueeze(2)  # [n, S, D]
    alpha = torch.einsum("ntd,cd->nt", U_hat, w)  # [n, S]
    beta = (w.unsqueeze(0).unsqueeze(0) - alpha.unsqueeze(2) * U_hat).norm(dim=2)
    beta_flat = (beta / s1).flatten().cpu().numpy()
    beta_flat = beta_flat[np.isfinite(beta_flat)]
    if len(beta_flat) == 0:
        return float("nan"), 0
    return float(np.median(beta_flat)), len(beta_flat)


def _cos_ug_1k(J):
    """cos(u_g, 1/√K) where u_g = JJ^T 1 / ‖JJ^T 1‖ (K-dim token-space).

    J: [n, S, K, D]. Returns (median cos, n_valid).
    """
    K = J.shape[2]
    ones = torch.ones(K, device=J.device, dtype=J.dtype) / np.sqrt(K)
    JG = torch.einsum("ntkd,ntld->ntkl", J, J)  # JJ^T [n, S, K, K]
    v = torch.einsum("ntkl,kl->ntk", JG, ones)  # [n, S, K]
    nv = v.norm(dim=2)  # [n, S]
    with torch.no_grad():
        u_g = v / nv.clamp(min=1e-12).unsqueeze(2)
        cos = (u_g * ones.unsqueeze(0).unsqueeze(0)).sum(dim=2)  # [n, S]
    cos_flat = cos.flatten().cpu().numpy()
    cos_flat = cos_flat[np.isfinite(cos_flat)]
    if len(cos_flat) == 0:
        return float("nan"), 0
    return float(np.median(cos_flat)), len(cos_flat)


def _j_rel_drift(J_q, J_base):
    """J_rel_drift = ‖J_q − J_base‖_F / ‖J_base‖_F, per (doc, t).

    J_q, J_base: [n, S, K, D]. Returns (median, p95, max).
    """
    n, S, K, Dd = J_q.shape
    num = torch.linalg.norm(J_q - J_base, ord="fro", dim=[2, 3])  # [n, S]
    den = torch.linalg.norm(J_base, ord="fro", dim=[2, 3])  # [n, S]
    with torch.no_grad():
        ratio = (num / den.clamp(min=1e-12)).flatten()
    ratio_np = ratio.cpu().numpy()
    ratio_np = ratio_np[np.isfinite(ratio_np)]
    if len(ratio_np) == 0:
        return float("nan"), float("nan"), float("nan")
    return (float(np.median(ratio_np)),
            float(np.percentile(ratio_np, 95)),
            float(np.max(ratio_np)))


def _recompute_J(M_K, H, chunk_docs=8):
    """Recompute J from M_K and cached H (readout-linearized).

    M_K: [K, D] (fp32). H: [n, S, D] (fp32) cached hidden states.
    J[i, t, k, d] = c[t] · (M_K[k,d] − (M_K @ û_t)[k] · û_t[d])
    with c[t] = √D/‖h_t‖, û_t = h_t/‖h_t‖.
    Chunked over docs to bound peak memory. Returns [n, S, K, D] fp32.
    """
    n, S, Dd = H.shape
    K = M_K.shape[0]
    sqD = torch.tensor(float(Dd), device=H.device) ** 0.5
    Jout = torch.empty(n, S, K, Dd, device=H.device, dtype=torch.float32)
    for c0 in range(0, n, chunk_docs):
        c1 = min(c0 + chunk_docs, n)
        h = H[c0:c1]  # [nc, S, D]
        hnorm = h.norm(dim=2)  # [nc, S]
        c = sqD / hnorm.clamp(min=1e-12)  # [nc, S]
        U_hat = h / hnorm.clamp(min=1e-12).unsqueeze(2)  # [nc, S, D]
        MK_Ut = M_K @ U_hat.transpose(1, 2)  # [nc, K, S]
        cross = MK_Ut.permute(0, 2, 1).unsqueeze(3) * U_hat.unsqueeze(2)  # [nc, S, K, D]
        Jc = c.unsqueeze(2).unsqueeze(3) * (M_K.unsqueeze(0).unsqueeze(0) - cross)
        Jout[c0:c1] = Jc  # [nc, S, K, D]
        del h, hnorm, c, U_hat, MK_Ut, cross, Jc
        if H.device.type == "cuda":
            torch.cuda.empty_cache()
    return Jout


def _quantize_rtn(W, bits, group):
    """RTN (round-to-nearest) quantization, per-group min-max affine.
    W: [V, D] fp32/64. Returns dequantized W_q [V, D] (same dtype as W).
    """
    V, Dd = W.shape
    Wq = W.clone()
    for j0 in range(0, Dd, group):
        j1 = min(j0 + group, Dd)
        Wg = W[:, j0:j1]
        xmax = Wg.max(dim=1, keepdim=True).values
        xmin = Wg.min(dim=1, keepdim=True).values
        maxq = (1 << bits) - 1
        scale = (xmax - xmin) / maxq
        zero = -xmin / scale.clamp(min=1e-12)
        q = torch.clamp(torch.round(Wg / scale.clamp(min=1e-12) + zero),
                        0, maxq)
        Wq[:, j0:j1] = scale * (q - zero)
    return Wq


def _quantize_gptq(W, X, bits, group, damp_frac, device):
    """GPTQ (IST-DASLab/gptq port). W: [V, D] fp32. X: [n, D] fp32.

    H = (2/n)·Σxxᵀ, damp = damp_frac·mean(diag H), desc_act, per-group
    scale/zero, in-block + cross-block error compensation.
    Hinv = upper-triangular Cholesky factor of H⁻¹ (matches reference).
    Returns dequantized W_q [V, D] fp32 (unpermuted).
    """
    V, Dd = W.shape
    W = W.clone().float()
    n = X.shape[0]
    H = (2.0 / n) * (X.float().T @ X.float())  # [D, D] fp32 (X: [n, D])
    dead = H.diagonal() == 0
    if dead.any():
        H[dead, dead] = 1.0
        W[:, dead] = 0.0
    damp = damp_frac * torch.mean(H.diagonal())
    diag = torch.arange(Dd, device=device)
    H[diag, diag] += damp
    perm = torch.argsort(H.diagonal(), descending=True)
    invperm = torch.argsort(perm)
    Wp = W[:, perm].clone()
    Hp = H[perm][:, perm]
    maxq = (1 << bits) - 1
    # Hinv = chol(chol_inv(chol(H)), upper=True) — upper, R·Rᵀ = H⁻¹
    L = torch.linalg.cholesky(Hp)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(Wp)
    group_scales = []  # per-group [V] scales (permuted order)
    group_zeros = []
    for i1 in range(0, Dd, group):
        i2 = min(i1 + group, Dd)
        W1 = Wp[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        # Group scale/zero from current (error-compensated) values.
        xmax = W1.max(dim=1, keepdim=True).values
        xmin = W1.min(dim=1, keepdim=True).values
        scale = ((xmax - xmin) / maxq).squeeze(1)  # [V]
        zero = (-xmin / scale.clamp(min=1e-12).unsqueeze(1)).squeeze(1)  # [V]
        group_scales.append(scale.clone())
        group_zeros.append(zero.clone())
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i].clamp(min=1e-12)
            q = torch.clamp(torch.round(w / scale.clamp(min=1e-12) + zero),
                            0, maxq)
            Q1[:, i] = q
            err1 = (w - scale * (q - zero)) / d
            W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:]
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        if i2 < Dd:
            Wp[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    # Dequantize in permuted space, then unpermute.
    Wq_p = torch.zeros_like(Wp)
    for gi in range(len(group_scales)):
        j0 = gi * group
        j1 = min(j0 + group, Dd)
        Wq_p[:, j0:j1] = group_scales[gi].unsqueeze(1) * (Q[:, j0:j1] - group_zeros[gi].unsqueeze(1))
    Wq = torch.zeros(V, Dd, device=device, dtype=torch.float32)
    Wq[:, perm] = Wq_p  # Wq[:, perm[j]] = Wq_p[:, j] (unpermute)
    return Wq


def _ppl_from_logits(logits, ids, attn):
    """PPL from precomputed logits [N, S, V] and ids [N, S] with attn mask.
    Returns mean per-token NLL (real tokens only).
    """
    lp = torch.log_softmax(logits.float(), dim=-1)  # [N, S, V]
    # Shift: predict token t+1 from logits at t.
    N, S, V = lp.shape
    target = ids[:, 1:]  # [N, S-1]
    lp_shift = lp[:, :-1, :]  # [N, S-1, V]
    real = attn[:, 1:].bool()  # [N, S-1]
    # Gather logprob of target.
    lp_target = lp_shift.gather(2, target.unsqueeze(2)).squeeze(2)  # [N, S-1]
    if real.sum() == 0:
        return float("nan")
    return float(lp_target[real].mean().abs().item())


@torch.no_grad()
def main(device=None, results_dir=None, small_model=None):
    t0 = time.time()
    dev = _resolve_device(device)
    out_dir = results_dir or os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp39b_ptq_perturbation.json")
    MODEL_P = small_model or MODEL
    print(f"E-39b PTQ perturbation: device={dev}, model={MODEL_P}")

    # ---------------- model ----------------
    print("loading model fp16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_P, torch_dtype=torch.float16, attn_implementation="eager")
    model.to(dev)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_P)
    W_lm = _resolve_lm_head_weight(model)  # [V, D] fp16 (GPU)
    W_lm64 = W_lm.double().detach()  # fp16 baseline → fp64, no quantization
    vocab = W_lm.shape[0]
    g = _lm_module(model).norm.weight.float() + 1  # [D] (1+w)
    print(f"W_lm: {tuple(W_lm.shape)}, g: min={g.min().item():.3f}")

    # ---------------- self-check: J reconstruction at L23 ----------------
    print("self-check: J reconstruction at L23 ...")
    data = load_layer(L_GATE[-1])  # L23
    n_docs, S = data["J"].shape[:2]
    U128 = data["U128"]
    # H fields: [n, S, D] (hidden states at L23) — moved to GPU to match M_K.
    H = data["H"].float().to(dev)  # [n, S, D]
    # J_t = M_K · (I − û_t û_tᵀ) · c_t, per (doc, t).
    M_K_base = build_M_K(W_lm64, U128, g).float()  # [K, D]
    J_rec = _recompute_J(M_K_base, H)  # [n, S, K, D]
    J_cache = data["J"].float().to(dev)  # [n, S, K, D]
    del data
    rel = (J_rec - J_cache).norm(dim=[2, 3]) / J_cache.norm(dim=[2, 3]).clamp(min=1e-12)
    rel_arr = rel.flatten().cpu().numpy()
    rel_arr = rel_arr[np.isfinite(rel_arr)]
    print(f"  self-check: rel err median={np.median(rel_arr):.2e} "
          f"p95={np.percentile(rel_arr, 95):.2e} max={rel_arr.max():.2e}")
    assert rel_arr.max() < SELFCHECK_TOL, (
        f"J reconstruction self-check failed: max rel err {rel_arr.max():.2e} "
        f"> {SELFCHECK_TOL} — W_lm/g/c_t chain broken")

    # ---------------- GPTQ: capture Hessian inputs ----------------
    print("capturing GPTQ Hessian inputs (C4 train skip "
          f"{C4_CAL_SKIP}) ...")
    ids_cal, attn_cal = _c4_docs(N_CAL, S_CAL, tok, skip=C4_CAL_SKIP)
    ids_cal = ids_cal.to(dev)
    attn_cal = attn_cal.to(dev)
    captured = []

    def _hook(module, inp, out):
        # inp: (x,) where x [B, S, D] fp16 (lm_head input)
        captured.append(inp[0].float().cpu())

    hook = model.lm_head.register_forward_pre_hook(_hook)
    with torch.no_grad():
        model(input_ids=ids_cal, attention_mask=attn_cal)
    hook.remove()
    X = torch.cat(captured, dim=1).reshape(-1, D)  # [n*S, D]
    print(f"  Hessian inputs: {tuple(X.shape)}")
    X_dev = X.to(dev)

    # ---------------- quantize all 3 levels ----------------
    levels = {}
    # fp16 baseline (no quantization)
    levels["fp16"] = W_lm64
    # int8 RTN
    print("quantizing int8 (RTN) ...")
    levels["int8"] = _quantize_rtn(W_lm64, RTN_BITS, RTN_GROUP)
    # gptq 4-bit
    print("quantizing GPTQ 4-bit ...")
    t_gptq = time.time()
    levels["gptq4"] = _quantize_gptq(W_lm64, X_dev, GPTQ_BITS, GPTQ_GROUP,
                                     GPTQ_DAMP, dev).double()
    print(f"  GPTQ done in {time.time() - t_gptq:.1f}s", flush=True)
    del X_dev, X
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------- PPL ----------------
    print("computing PPL (C4 holdout) ...")
    try:
        ids_ppl, attn_ppl = _c4_docs(N_PPL, S_PPL, tok, skip=0, split="val")
    except Exception as e:
        print(f"  C4 val unavailable ({e}), falling back to train skip "
              f"{C4_PPL_SKIP}")
        ids_ppl, attn_ppl = _c4_docs(N_PPL, S_PPL, tok, skip=C4_PPL_SKIP)
    ids_ppl = ids_ppl.to(dev)
    attn_ppl = attn_ppl.to(dev)
    # Base forward: capture last_hidden_state.
    with torch.no_grad():
        out = model.model(input_ids=ids_ppl, attention_mask=attn_ppl)
        h_last = out.last_hidden_state  # [N, S, D] fp16 (pre-norm)
    # lm_head input = final RMSNorm(h_last). Apply once, reuse for all levels.
    with torch.no_grad():
        h_normed = _lm_module(model).norm(h_last.float())  # [N, S, D] fp32
    del h_last
    ppl = {}
    for name, Wq in {"base": W_lm64, "int8": levels["int8"],
                     "gptq4": levels["gptq4"]}.items():
        with torch.no_grad():
            logits_q = h_normed @ Wq.float().T  # [N, S, V]
        ppl[name] = _ppl_from_logits(logits_q, ids_ppl, attn_ppl)
        print(f"  PPL {name} = {ppl[name]:.4f}")
    del h_normed
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------- per-layer metrics ----------------
    print("computing per-layer metrics ...")
    metrics = {}
    for L in L_CACHED:
        data = load_layer(L)
        n_docs_L, S_L = data["J"].shape[:2]
        U128_L = data["U128"]
        H_L = data["H"].float().to(dev)  # [n, S, D] (GPU — matches M_K)
        del data
        metrics[L] = {}
        # Base J: same readout-linearized formula with fp16 W_lm — identical
        # construction for base and quantized (exact at L23; convention for
        # L<23, isolates M_K-only quantization effect). At L23 reproduces
        # the cached J to ~1e-7 (self-check above).
        J_base = _recompute_J(build_M_K(W_lm64, U128_L, g).float(), H_L)
        for name, Wq in levels.items():
            # M_K for this Wq (g is the same — RMSNorm not quantized).
            M_K_q = build_M_K(Wq, U128_L, g).float()
            # J_q = M_K_q · (I − û_t û_tᵀ) · c_t.
            Jq = _recompute_J(M_K_q, H_L)  # [n, S, K, D]
            # κ
            kap = _kappa_from_J(Jq, R, dev)
            # β/σ₁ (uses M_K_q and cached U_hat — approx: use Jq's M_K)
            beta_med, beta_n = _beta_sigma1(M_K_q, H_L)
            # cos(u_g, 1/√K)
            cos_med, cos_n = _cos_ug_1k(Jq)
            # J_rel_drift (base = cached J; for fp16 this is ~0 by self-check)
            drift_med, drift_p95, drift_max = _j_rel_drift(Jq, J_base)
            metrics[L][name] = {
                "kappa": kap,
                "beta_sigma1": beta_med,
                "beta_n_valid": beta_n,
                "cos_ug_1k": cos_med,
                "cos_n_valid": cos_n,
                "j_rel_drift_median": drift_med,
                "j_rel_drift_p95": drift_p95,
                "j_rel_drift_max": drift_max,
            }
            print(f"  L{L} {name}: kappa={kap:.5f} "
                  f"beta/s1={beta_med:.4f} cos={cos_med:.4f} "
                  f"drift={drift_med:.5f}")
            del Jq, M_K_q
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del J_base

    # ---------------- gate ----------------
    kap23_base = metrics[23]["fp16"]["kappa"]
    kap23_gptq = metrics[23]["gptq4"]["kappa"]
    kap23_int8 = metrics[23]["int8"]["kappa"]
    # Gate on κ(L23) GPTQ vs base.
    if kap23_gptq < KAPPA_PASS:
        gate = "pass"
    elif kap23_gptq < KAPPA_MODERATE:
        gate = "moderate"
    else:
        gate = "catastrophic"
    # Key check: κ ×10 with PPL only +1–2% → κ not tied to PPL.
    ppl_base_v = ppl["base"]
    ppl_gptq_v = ppl["gptq4"]
    ppl_delta_pct = (ppl_gptq_v - ppl_base_v) / ppl_base_v * 100 if ppl_base_v > 0 else float("nan")
    kap_ratio = kap23_gptq / kap23_base if kap23_base > 0 else float("nan")
    print(f"\nGATE: κ(L23) base={kap23_base:.5f} gptq4={kap23_gptq:.5f} "
          f"int8={kap23_int8:.5f}")
    print(f"  gate={gate} (κ_gptq < {KAPPA_PASS}: pass; "
          f"< {KAPPA_MODERATE}: moderate; else catastrophic)")
    print(f"  PPL base={ppl_base_v:.4f} gptq4={ppl_gptq_v:.4f} "
          f"(Δ={ppl_delta_pct:+.2f}%)")
    print(f"  κ ratio (gptq4/base) = {kap_ratio:.3f}")

    # ---------------- output ----------------
    result = {
        "experiment": "E-39b PTQ perturbation",
        "model": MODEL_P,
        "device": str(dev),
        "r": R,
        "layers": L_CACHED,
        "gate_layers": L_GATE,
        "levels": ["fp16", "int8", "gptq4"],
        "gptq_bits": GPTQ_BITS,
        "gptq_group": GPTQ_GROUP,
        "gptq_damp": GPTQ_DAMP,
        "c4_cal_skip": C4_CAL_SKIP,
        "c4_ppl_skip": C4_PPL_SKIP,
        "n_cal": N_CAL,
        "s_cal": S_CAL,
        "n_ppl": N_PPL,
        "s_ppl": S_PPL,
        "selfcheck": {
            "rel_err_median": float(np.median(rel_arr)),
            "rel_err_p95": float(np.percentile(rel_arr, 95)),
            "rel_err_max": float(rel_arr.max()),
            "tol": SELFCHECK_TOL,
            "passed": bool(rel_arr.max() < SELFCHECK_TOL),
        },
        "ppl": ppl,
        "ppl_delta_pct_gptq4_vs_base": float(ppl_delta_pct),
        "kappa_l23": {
            "base": kap23_base,
            "int8": kap23_int8,
            "gptq4": kap23_gptq,
            "ratio_gptq4_vs_base": float(kap_ratio),
        },
        "gate": {
            "threshold_pass": KAPPA_PASS,
            "threshold_moderate": KAPPA_MODERATE,
            "verdict": gate,
        },
        "metrics": metrics,
        "wall_sec": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\noutput: {out_path}")
    print(f"wall: {time.time() - t0:.0f}s")
    return result


if __name__ == "__main__":
    main()
