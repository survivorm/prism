"""E-39b — κ(L23) и M_K-структура под GPTQ-4bit: что ломает квантование (docs/E39.md §2).

PTQ (GPTQ/AWQ) квантует lm_head.weight → M_K → J_t, при этом û_t заморожен
(активации кэшированы в L23): неконтролируемая модификация M_K.
Эксперимент измеряет, насколько PTQ ломает лист.

Уровни (spec):
  * "fp16"  — неоквантованный baseline: загруженный fp16 W_lm (→ fp64).
              Воспроизводит кэшированное J в L23 до ~1e-7 (self-check).
  * "int8_rtn" — RTN (per-group min-max, round-to-nearest), без Hessian.
    (P1.7: RTN, не GPTQ-8bit — переименовано int8 → int8_rtn.)
  * "gptq4" — GPTQ 4-bit, group_size=128, desc_act=True, damp=0.01
              (port IST-DASLab/gptq: H = (2/n)·Σxxᵀ, cholesky-inverse,
              per-group scale/zero, in-block error compensation).

Метрики (per layer L ∈ {6,12,18,19,22,23}, per level):
  * κ(L)      = ‖H_012 − I‖_F, top-16 right-singular bases (E-38d, r=16).
  * β/σ₁(M_K) — consistency (E-38e, baseline ≈ 0.9 ± 0.15).
  * cos(u_g, 1/√K) — gauge-axis alignment, PROXY JJᵀ1 (E-37/E-33r; не сравним
              с exact u_g из E-38f).
  * J_rel_drift = ‖J_q − J‖_F / ‖J‖_F (per (doc,t) median/p95/max).
  * PPL на C4 holdout (lm_head.replace не используется — tied weights;
    PPL readout-only: квантуется только W_lm, h от базовой модели — P0.5).

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

P1.3: sentinel-фильтра нет — κ/β/cos считаются на всех тройках/позициях
для консистентности с E-39a (baseline 0.0162 тоже без фильтра).

Как читать JSON (round 3, разбор `docs/E39b_fixes.md`) — числа корректны, но:
  1. `readout_only_ptq` → verdict относится ТОЛЬКО к readout-каналу. Полный
     tied-weight PTQ (квантуется `embed_tokens` → сдвиг h по 24 слоям → û_t)
     не измерен; масштаб upstream-канала по E-39c_pre: embed_only KL 5.23 vs
     head_only 1.36.
  2. `kappa` при L ≠ 23 — синтетика: J readout-линеаризован (нет промежуточных
     слоёв и D_l), поэтому κ(L6) ≈ 0.005 НЕ сравнимо с настоящей leaf-кривизной
     E-28 (κ(L6) = 3.96, κ(L22) = 1.56 — на 3 порядка). Сравнивать с E-28/E-39a
     можно только κ(L23).
  3. `cos_ug_1k_proxy` — proxy (JJᵀ1), НЕ точный u_g = argmax|cos(u,1)| из E-38f
     (там 0.862 на L23, здесь 0.917): разные объекты, не сравнимы.
  4. `int8_rtn` — не baseline, а control: 8-bit RTN здесь практически lossless
     (PPL −0.003%, drift 0.58%); сравнение по существу — fp16 vs gptq4.
  5. Робастность κ к PTQ частично свойство самой метрики: κ = функционал от
     углов подпространства (Procrustes-углы топ-r базиса), инвариантный к
     амплитуде и гладким малым возмущениям M_K. Поэтому «12% ошибки веса →
     0% изменения κ» — не парадокс, а следование определению.
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
# N2: base self-checks — known numbers from E-38d/E-38e (L23, r=16).
KAPPA_BASE_REF = 0.0162   # E-38d / E-39a kappa_median (L23, r=16)
KAPPA_BASE_TOL = 0.005
BETA_BASE_REF = 0.882     # E-38e beta/sigma1 baseline (~0.9 ± 0.15)
BETA_BASE_TOL = 0.05


def _c4_docs(n_docs, s_len, tokenizer, skip=0, split="train"):
    """Load n_docs C4 docs, tokenize FULL text to s_len tokens (no char-slice).

    Truncation to s_len happens in the tokenizer (by tokens), not by cutting
    the raw string — so the calibration/holdout sets are ~s_len real tokens,
    not ~s_len characters (P0.1).
    """
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
    texts = []
    n_skip = 0
    for doc in ds:
        if n_skip < skip:
            n_skip += 1
            continue
        t = doc["text"]
        if t:
            texts.append(t)
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
    alpha = U_hat @ w  # [n, S] (w: [D])
    beta = (w.unsqueeze(0).unsqueeze(0) - alpha.unsqueeze(2) * U_hat).norm(dim=2)
    beta_flat = (beta / s1).flatten().cpu().numpy()
    beta_flat = beta_flat[np.isfinite(beta_flat)]
    if len(beta_flat) == 0:
        return float("nan"), 0
    return float(np.median(beta_flat)), len(beta_flat)


def _cos_ug_1k(J):
    """cos(u_g, 1/√K) where u_g = JJ^T 1 / ‖JJ^T 1‖ (K-dim token-space).

    PROXY, не точный u_g: E-38f берёт u_g = argmax |cos(u, 1/√K)| по левым СВ J,
    здесь это JJᵀ1 (один проход, без SVD). В E-33r align(u_top, u_g) = 0.85–0.99,
    то есть числа НЕ сравнимы с E-38f (L23: 0.917 здесь против 0.862 там).

    J: [n, S, K, D]. Returns (median cos, n_valid).
    """
    K = J.shape[2]
    inv_sqrt_K = 1.0 / np.sqrt(K)
    s = J.sum(dim=2)  # [n, S, D]: s[d] = sum over K tokens
    v = torch.einsum("ntkd,ntd->ntk", J, s)  # [n, S, K] = JJ^T 1
    nv = v.norm(dim=2)  # [n, S]
    with torch.no_grad():
        u_g = v / nv.clamp(min=1e-12).unsqueeze(2)
        cos = (u_g * inv_sqrt_K).sum(dim=2)  # [n, S] = u_g · (1/√K)
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
        out = Jout[c0:c1]  # [nc, S, K, D] view — no extra [nc,S,K,D] temporary
        torch.sub(M_K.unsqueeze(0).unsqueeze(0), cross, out=out)
        out.mul_(c.unsqueeze(2).unsqueeze(3))
        del h, hnorm, c, U_hat, MK_Ut, cross, out
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
    W = W.to(device).clone().float()
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


@torch.no_grad()
def main(device=None, results_dir=None, small_model=None):
    t0 = time.time()
    dev = _resolve_device(device)
    # P1.4: deterministic — seed RNGs (GPTQ itself is deterministic given
    # inputs; this seeds any stochastic path). use_deterministic_algorithms
    # is best-effort (some T4 cuBLAS ops lack a deterministic path).
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    try:
        torch.use_deterministic_algorithms(True)  # P1.4: best-effort, CPU too
    except Exception as e:
        print(f"  (deterministic_algorithms not fully settable: {e})")
    out_dir = results_dir or os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp39b_ptq_perturbation.json")
    MODEL_P = small_model or MODEL
    print(f"E-39b PTQ perturbation: device={dev}, model={MODEL_P}")

    # ---------------- model ----------------
    print("loading model fp16 ...")
    # N3: fp16 only on GPU (CPU fp16 forward is 5-10x slower); W_lm.double()
    # keeps fp64 metrics regardless of model dtype.
    model_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_P, torch_dtype=model_dtype, attn_implementation="eager")
    model.to(dev)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_P)
    W_lm = _resolve_lm_head_weight(model)  # [V, D] fp16 (GPU)
    # T4 memory: the full [V,D] weight copies live on CPU — the per-layer loop
    # only needs the 128 token-set rows (M_K), PPL moves one fp32 copy at a time.
    vocab = W_lm.shape[0]
    W_lm64 = W_lm.double().detach().cpu()  # fp16 baseline → fp64, no quantization
    W_lm32 = W_lm.float().detach().cpu()  # N4: one fp32 copy for PPL matmuls (~1 GB)
    del W_lm
    g = _lm_module(model).norm.weight.float().cpu() + 1  # [D] (1+w)
    print(f"W_lm: {tuple(W_lm64.shape)}, g: min={g.min().item():.3f}")

    # ---------------- self-check: J reconstruction at L23 ----------------
    print("self-check: J reconstruction at L23 ...")
    data = load_layer(L_GATE[-1])  # L23
    n_docs, S = data["J"].shape[:2]
    U128 = data["U128"]
    # H fields: [n, S, D] (hidden states at L23) — moved to GPU to match M_K.
    H = data["H"].float().to(dev)  # [n, S, D]
    # J_t = M_K · (I − û_t û_tᵀ) · c_t, per (doc, t).
    M_K_base = build_M_K(W_lm64, U128, g).float().to(dev)  # [K, D]
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
    # P1.5: release self-check GPU tensors (~4 GB) before the per-layer loop.
    del J_rec, J_cache, H, M_K_base, rel
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------- GPTQ: capture Hessian inputs ----------------
    print("capturing GPTQ Hessian inputs (C4 train skip "
          f"{C4_CAL_SKIP}) ...")
    ids_cal, attn_cal = _c4_docs(N_CAL, S_CAL, tok, skip=C4_CAL_SKIP)
    ids_cal = ids_cal.to(dev)
    attn_cal = attn_cal.to(dev)
    # lm_head input == model.model(...).last_hidden_state (post final RMSNorm).
    # Capture it directly (chunked to bound memory) instead of a full-model
    # forward: no 8 GB logits tensor (fixes the v5 OOM) and no double norm.
    captured = []
    cal_chunk = 16
    for c0 in range(0, N_CAL, cal_chunk):
        c1 = min(c0 + cal_chunk, N_CAL)
        with torch.no_grad():
            out = model.model(input_ids=ids_cal[c0:c1],
                              attention_mask=attn_cal[c0:c1])
        captured.append(out.last_hidden_state.float().cpu())
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    X_all = torch.cat(captured, dim=0)  # [n, S, D]
    del captured
    # Mask out padding (P0.2): keep only real tokens in the Hessian.
    mask = attn_cal.cpu().bool()  # [n, S]
    X = X_all[mask]  # [n_real, D]
    del X_all
    print(f"  Hessian inputs: {tuple(X.shape)} "
          f"(real tokens, masked) of {attn_cal.sum().item()} expected")
    X_dev = X.to(dev)

    # ---------------- quantize all 3 levels ----------------
    levels = {}
    # fp16 baseline (no quantization)
    levels["fp16"] = W_lm64
    # int8 RTN (P1.7: renamed int8 → int8_rtn, it is RTN not GPTQ-8bit)
    print("quantizing int8 (RTN) ...")
    levels["int8_rtn"] = _quantize_rtn(W_lm64, RTN_BITS, RTN_GROUP)
    # gptq 4-bit
    print("quantizing GPTQ 4-bit ...")
    t_gptq = time.time()
    levels["gptq4"] = _quantize_gptq(W_lm64, X_dev, GPTQ_BITS, GPTQ_GROUP,
                                     GPTQ_DAMP, dev).cpu().double()
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
    attn_ppl_dev = attn_ppl.to(dev)  # N1: NLL mask on the same device
    # P0.4 check: last_hidden_state (post final RMSNorm) must equal the lm_head
    # input, i.e. model.lm_head(last_hidden_state) == model(...).logits. Verified
    # on a 4-doc batch (cheap); a mismatch would mean double-norm / wrong input.
    with torch.no_grad():
        lref = model.model(input_ids=ids_ppl[:4], attention_mask=attn_ppl_dev[:4]).last_hidden_state
        lref2 = model(input_ids=ids_ppl[:4], attention_mask=attn_ppl_dev[:4]).logits
        ref_err = (lref.float() @ W_lm32.T.to(dev) - lref2.float()).abs().max().item()
    print(f"  P0.4 check: |lm_head(lh) - model.logits| max = {ref_err:.3e} "
          f"(should be ~0 for a few docs; fp16 round-off allowed)")
    # P0.4 (fix): raise instead of warn — a mismatch would silently corrupt PPL.
    if ref_err > 2.0:
        raise RuntimeError(
            f"P0.4 check FAILED (ref_err={ref_err:.3e} > 2.0): "
            "last_hidden_state is not the lm_head input (double-norm?) — "
            "PPL would be INVALID; aborting.")
    # Capture last_hidden_state chunked (lm_head input, post final RMSNorm).
    # No double norm (P0.4), no full-batch logits tensor (OOM fix).
    ppl = {}
    V_vocab = W_lm64.shape[0]
    ppl_chunk = 8
    # N4: base uses the single precast fp32 copy; other levels cast once.
    for name, Wq in {"base": W_lm32, "int8_rtn": levels["int8_rtn"],
                     "gptq4": levels["gptq4"]}.items():
        Wq32 = (Wq if Wq.dtype == torch.float32 else Wq.float()).to(dev)
        nll_sum = 0.0
        n_real = 0
        for c0 in range(0, N_PPL, ppl_chunk):
            c1 = min(c0 + ppl_chunk, N_PPL)
            with torch.no_grad():
                h = model.model(input_ids=ids_ppl[c0:c1],
                                attention_mask=attn_ppl_dev[c0:c1]).last_hidden_state.float()
                logits_c = h @ Wq32.T  # [chunk, S, V] (N4: no per-chunk cast)
                del h
                lp = torch.log_softmax(logits_c, dim=-1)
                del logits_c
                target = ids_ppl[c0:c1, 1:]
                # N1: mask on the same device as the CUDA logprobs.
                real = attn_ppl_dev[c0:c1, 1:].bool()
                lp_t = lp[:, :-1, :].gather(2, target.unsqueeze(2)).squeeze(2)
                nll_sum += float((-lp_t[real]).sum().item())
                n_real += int(real.sum().item())
                del lp, lp_t, target, real
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        ppl[name] = float(np.exp(nll_sum / max(n_real, 1)))
        print(f"  PPL {name} = {ppl[name]:.4f} (n_real={n_real})")
        del Wq32
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
        J_base = _recompute_J(build_M_K(W_lm64, U128_L, g).float().to(dev), H_L)
        for name, Wq in levels.items():
            # M_K for this Wq (g is the same — RMSNorm not quantized).
            M_K_q = build_M_K(Wq, U128_L, g).float().to(dev)
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
                # P1.1/P1.8: L<23 J is readout-linearized (no D_i); L19 is
                # extra (not in the gate) — readers must not mistake
                # kappa(L6) for a real L6 leaf kappa.
                "synthetic_readout_j": L != 23,
                "extra_layer": L not in L_GATE,
                "kappa": kap,
                "beta_sigma1": beta_med,
                "beta_n_valid": beta_n,
                "cos_ug_1k_proxy": cos_med,
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

    # N2: base self-checks against known E-38d/E-38e numbers — a drift here
    # would flip the gate verdict, so fail loudly instead of reporting it.
    kap_b = metrics[23]["fp16"]["kappa"]
    beta_b = metrics[23]["fp16"]["beta_sigma1"]
    assert abs(kap_b - KAPPA_BASE_REF) < KAPPA_BASE_TOL, (
        f"κ(L23) fp16 = {kap_b:.4f}, expected ~{KAPPA_BASE_REF} "
        f"(E-38d/E-39a) — SVD/driver drift, gate unreliable")
    assert abs(beta_b - BETA_BASE_REF) < BETA_BASE_TOL, (
        f"β/σ₁(L23) fp16 = {beta_b:.3f}, expected ~{BETA_BASE_REF} (E-38e)")
    print(f"  N2 base self-checks OK: κ={kap_b:.5f} "
          f"(ref {KAPPA_BASE_REF}), β/σ₁={beta_b:.4f} (ref {BETA_BASE_REF})")

    # ---------------- gate ----------------
    kap23_base = metrics[23]["fp16"]["kappa"]
    kap23_gptq = metrics[23]["gptq4"]["kappa"]
    kap23_int8 = metrics[23]["int8_rtn"]["kappa"]
    kap_ratio = kap23_gptq / kap23_base if kap23_base > 0 else float("nan")
    # Gate on κ(L23) GPTQ vs base. P1.6: pass also requires the relative
    # bound ≤ +25% (ratio ≤ 1.25), not just the absolute 0.02 threshold.
    if kap23_gptq < KAPPA_PASS and kap_ratio <= 1.25:
        gate = "pass"
    elif kap23_gptq < KAPPA_MODERATE:
        gate = "moderate"
    else:
        gate = "catastrophic"
    # Key check: κ ×10 with PPL only +1–2% → κ not tied to PPL.
    ppl_base_v = ppl["base"]
    ppl_gptq_v = ppl["gptq4"]
    ppl_delta_pct = (ppl_gptq_v - ppl_base_v) / ppl_base_v * 100 if ppl_base_v > 0 else float("nan")
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
        "levels": ["fp16", "int8_rtn", "gptq4"],
        "readout_only_ptq": True,  # P0.5: W_lm only quantized; h from base model
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
        # P0.4 was a print-only check; without the value in JSON there is no way
        # to verify it ran. It raises on ref_err > 2.0, so reaching the output
        # already implies passed=True — but the number is the evidence.
        "p04_ref_err": float(ref_err),
        "p04_passed": bool(ref_err <= 2.0),
        # Round-3 разбор: κ для L≠23 считается на readout-линеаризованном J.
        "warning_kappa_synthetic": (
            "kappa at L != 23 is computed on the readout-linearized J "
            "(c_t·M_K·(I−ûûᵀ), no intermediate layers, no D_l): it is NOT the leaf "
            "curvature of E-28 (E-28: kappa(L6)=3.96, kappa(L22)=1.56 vs 0.0046/0.0098 "
            "here — 3 orders apart). Only kappa(L23) is a real leaf kappa; compare "
            "only L23 with E-28/E-39a."
        ),
        "kappa_semantics_note": (
            "kappa is a Procrustes-angle functional of the top-r subspace of J: "
            "invariant to amplitude and to smooth small perturbations of M_K. The "
            "robustness of the leaf to readout-PTQ is therefore partly a property of "
            "the metric, not only of the model."
        ),
        "levels_note": {
            "fp16": "unquantized baseline (also the level the J-reconstruction self-check runs on)",
            "int8_rtn": (
                "control, not baseline: 8-bit RTN is effectively lossless here "
                "(PPL −0.003%, drift 0.58%); the comparison of record is fp16 vs gptq4"
            ),
            "gptq4": "GPTQ 4-bit group 128 desc_act — the level the gate is read on",
        },
        "ppl": ppl,
        "ppl_delta_pct_gptq4_vs_base": float(ppl_delta_pct),
        "kappa_l23": {
            "base": kap23_base,
            "int8_rtn": kap23_int8,
            "gptq4": kap23_gptq,
            "ratio_gptq4_vs_base": float(kap_ratio),
        },
        "gate": {
            "threshold_pass": KAPPA_PASS,
            "threshold_moderate": KAPPA_MODERATE,
            "verdict": gate,
            # The verdict must not be read as "PTQ is safe": only the lm_head
            # channel was quantized, with h/û_t frozen at the base model.
            "readout_channel_only": True,
            "scope_note": (
                "verdict covers quantization of lm_head.weight only (activations and "
                "embed_tokens frozen at the base model). Full tied-weight PTQ also "
                "quantizes embed_tokens → h shifts through 24 layers → û_t moves → κ "
                "may move; not measured here. Upstream channel dominates per E-39c_pre "
                "(embed_only KL 5.23 vs head_only 1.36)."
            ),
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
