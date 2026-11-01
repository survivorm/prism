"""
E-39d: L23 readout training with flatness regularizer (Reading A).

docs/E39.md §4.

Question: can TRAINING find a W_lm' that flattens the L23 leaf (κ → 0)
BETTER than the algebraic rotation (E-39c), while preserving behavior
(KL leash)?

E-39c (pure algebra) was REFUTED on KL: the full bottom-16 rotation costs
6.31 nats (126× the 0.05 budget). E-39d asks: can a learned W_lm' achieve
similar flatness with lower KL?

Design (Reading A, docs/E39.md §4):
  - Trainable: W_lm[token_set] (128×1024 = 131K params). g (RMSNorm) FROZEN.
    Rationale: g is global (scales all 248K tokens) → training it would make
    the KL leash require a full-vocab matmul per step and is high-risk for
    global PPL. W_rows ⊙ g_base (g_base ≈ 1, all entries O(1)) spans the
    same leaf space, so W_rows alone can flatten the 16-dim subspace, and
    the 128-column KL swap is EXACTLY correct.
  - û_t FROZEN (cached H from exp27b L23 cache). No model forward in the loss.
  - Leaf: M_K = W_rows ⊙ g_base  (E-39a/c factorization, g_base frozen).
  - Loss: L = λ · L_flat + λ_KL · KL_leash
      L_flat  = E_t ‖P_t − P̄‖_F²   (projector dispersion, differentiable via
               eigh(J Jᵀ) — no grad through SVD). Sentinel positions (E-34)
               excluded.
      KL_leash = E_t KL(p_edit(t) || p_base(t))  (readout-only, cached H,
               full 248K vocab, via logsumexp trick → only [P,128] per step).
               ROUND 8 CRITICAL #1: ACTUAL logits (E-38e factorization)
               logit = √D · M_K · û, M_K = W_lm·diag(1+w) — the old H·W_lmᵀ
               were pseudo-logits (off by the per-position/per-token factor
               ‖h‖/(√D·g)); verified against the real model forward to 9.5e-6.
               The final verdict KL is measured via a REAL model forward, so
               it is directly comparable to E-39c_pre's real-forward KL.
  - No bridge (self-bridge degenerate); KL leash preserves behavior.
  - λ sweep {0.01,0.1,1,10}, 100 Adam steps, lr=1e-3, 3 seeds, P=512 pos/step.
  - Metrics every 20 steps: κ' (EXACT E-39a _kappa_dryrun, so κ_base=0.0162
    is comparable) + PPL (C4 held-out, skip 4096, 8 DISTINCT docs).
  - Verdict (E39d_fixes.md #3): SUCCESS requires κ'<0.005 AND PPL ≤+2% AND
    KL<0.05 (full-dataset, not last minibatch — #6).

Round 8 (E39d_fixes.md): fixed the CRITICAL flatness_loss bug (#1a/#1b —
bottom-r + row-flipped eigenvectors instead of top-r right SVs, which made
the surrogate optimize a different subspace than the κ' metric), PPL on 8
identical docs (#2), verdict missing kl_ok (#3), non-restoring weight swap
(#4), weights_only (#5), full-dataset KL in final (#6), best-run selection
(#7), normalizer on random subset (#8), sqrt clamp (#9), doc-length check
(#10), dtype kwarg (#11), and (round 8, E39d_fixes.md round 8) ACTUAL
logits in the KL leash + sentinel (CRITICAL #1), u64_docs on the compute
device (CRITICAL #2 — T4 cuda×cpu crash), NaN check BEFORE opt.step
(MEDIUM #3), sqrt-division guard in flatness (MEDIUM #4), H-norm clamp
(LOW #7). (Round-8 #5/#8 — conclusion/budget_note/sweep[1] text — do not
exist in E-39d; they are E-39c_pre patterns.)

Output: results/exp39d_l23_training.json

CPU estimate: ~0.4 s/step (flatness 256×1024×1024) × 1200 steps ≈ 8 min,
+ PPL (full 0.8B fwd, 120×) ≈ 2 min → ~10-15 min.  T4: ~1-2 min.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- make exp39a importable for the EXACT κ' dry-run ----
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from exp39a_u_obs_rank import _kappa_dryrun   # exact κ' (38d Procrustes cycle)

REPO = os.path.abspath(os.path.join(_HERE, ".."))

# ---- constants ----
MODEL = "Qwen/Qwen3.5-0.8B"
R = 16                     # top-16 right SVs (flatness, E-28/38d/39a)
KAPPA_TARGET = 0.005       # flat-leaf threshold
KL_PASS = 0.05             # KL budget (half of E-16's 0.1 leash)
LAMBDA_KL = 0.1            # KL leash weight (spec §4.4)
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0]
N_STEPS = 100
LR = 1e-3
SEEDS = [0, 1, 2]
P_STEP = 512               # positions per gradient step (stochastic; micro-batched)
MB = 64                    # micro-batch for the [P,D,D] flatness (memory)
N_DOCS_PPL = 8             # C4 docs for PPL (skip 4096)
S_LEN = 128
C4_SKIP = 4096
SENTINEL_SUM = 0.5         # E-34: raw P128 sum < 0.5 → sentinel (excluded)
METRIC_EVERY = 20
D = 1024
V = 248320                 # vocab


def _jcache_l23_path() -> str:
    """Resolve exp27b_jcache_L23.pt (local or Kaggle dataset)."""
    local = os.path.join(REPO, "kaggle_kernel", "results", "exp27b_jcache_L23.pt")
    if os.path.exists(local):
        return local
    for ds in ("prism-exp27b-jcache-jca", "prism-exp27b-jcache-jcb"):
        for base in ("/kaggle/input",
                     "/kaggle/input/datasets/voroninmichael"):
            p = os.path.join(base, ds, "exp27b_jcache_L23.pt")
            if os.path.exists(p):
                return p
    raise FileNotFoundError("exp27b_jcache_L23.pt not found (local or "
                            "/kaggle/input/prism-exp27b-jcache-jcb/)")


# ---------------------------------------------------------------------------
# KL leash (logsumexp trick) — only [P, K] per step, NO [P, V]
# ---------------------------------------------------------------------------

def kl_leash(H_sub: torch.Tensor,
             W_rows: torch.Tensor,
             g: torch.Tensor,
             base_k: torch.Tensor,
             lse_b: torch.Tensor) -> torch.Tensor:
    """KL(p_edit || p_base) over the full vocab, readout-only, via logsumexp.

    (E39d_fixes #12: the tok_ids param was unused — removed.)

    ROUND 8 CRITICAL #1: ACTUAL logits via the E-38e factorization (L23-only):
        logit = √D · M_K · û,   û = h/‖h‖,   M_K = W_rows · diag(g),
    where g = 1 + RMSNorm weight (frozen). The old H_sub @ W_rowsᵀ were
    pseudo-logits — off by the per-position/per-token factor ‖h‖/(√D·g) —
    so their KL was a temperature-scaled proxy, NOT comparable to
    E-39c_pre's real-forward KL. The logsumexp form was verified OFFLINE during
    development: √D·(û·(W·diag g)ᵀ) matched model.lm_head(model.model.norm(h))
    to 9.5e-6 (the old H@Wᵀ form was off by 2.69 nats). The RUNTIME checks verify
    the FORMULA against a full-softmax KL (_selftest) and the MAGNITUDE on held-out
    C4 docs (kl_f_fwd); they do not re-prove the 9.5e-6 factorization.
    base_k / lse_b are the ACTUAL base logits (from W_lm·diag g).

    Derivation (b=base logits, e=edit logits; e_v=b_v for v∉K, e_k=edit
    logits for v∈K; lse = logsumexp):
        lse_e = log( exp(lse_b) − Σ_k exp(b_k) + Σ_k exp(e_k) )
        KL    = (lse_b − lse_e) + Σ_k exp(e_k − lse_e)(e_k − b_k)
    Verified against full-softmax KL in _selftest.

    H_sub: [P, D] frozen cached hidden states.
    W_rows: [K, D] trainable lm_head rows (token set). grad flows here.
    g: [D] frozen RMSNorm (1+w), same device/dtype as W_rows.
    base_k: [P, K] base logits for the K tokens (cached, no grad).
    lse_b: [P] logsumexp of base logits over full vocab (cached, no grad).
    """
    Dd = H_sub.shape[-1]
    u = H_sub / H_sub.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # [P, D]
    SQRT_D = math.sqrt(Dd)
    e_k = SQRT_D * (u @ (W_rows * g[None, :]).t())   # [P, K] grad via W_rows
    b_k = base_k                                          # [P, K] (no grad)
    # max-trick stabilisation (round 7 §1): raw exp(e_k) overflows to Inf
    # once logit drift exceeds ~88 (observed: NaN crash at lambda=0.01 s2
    # st40). Reference m = max(lse_b, max_k e_k, max_k b_k) per position.
    m = torch.cat([lse_b[:, None], e_k, b_k], dim=1).max(dim=1).values  # [P]
    exp_e_k = torch.exp(e_k - m[:, None])
    exp_b_k = torch.exp(b_k - m[:, None])
    # lse_e = m + log( exp(lse_b - m) - Σ_k exp(b_k - m) + Σ_k exp(e_k - m) )
    lse_e = m + torch.log(torch.exp(lse_b - m) - exp_b_k.sum(1) + exp_e_k.sum(1))  # [P]
    # KL = (lse_b − lse_e) + Σ_k exp(e_k − lse_e)(e_k − b_k)
    kl_t = (lse_b - lse_e) + (torch.exp(e_k - lse_e[:, None]) * (e_k - b_k)).sum(1)
    return kl_t.mean()


def _selftest_kl(device: str) -> bool:
    """Verify the logsumexp KL against a brute-force full-softmax KL.

    ROUND 8 CRITICAL #1: exercises the ACTUAL-logit form (√D·(û·(W·diag g)ᵀ)).
    base and edit share the same Wlm/g (edit perturbs only the K rows) — the
    exact readout-only swap the loss uses.
    """
    torch.manual_seed(0)
    P, Dd, K, Vv = 32, 32, 8, 64
    H = torch.randn(P, Dd, device=device)
    Wlm = torch.randn(Vv, Dd, device=device)
    W_rows = Wlm[:K].clone() + 0.5 * torch.randn(K, Dd, device=device)
    g = torch.rand(Dd, device=device) + 1.0
    SQRT_D = math.sqrt(Dd)
    u = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    M_K_full = Wlm * g[None, :]                     # [V, D] base
    base_logits = SQRT_D * (u @ M_K_full.t())       # [P, V] actual base
    lse_b = torch.logsumexp(base_logits, dim=-1)    # [P]
    base_k = base_logits[:, :K]                     # [P, K]
    # brute force
    edit_logits = base_logits.clone()
    edit_logits[:, :K] = SQRT_D * (u @ (W_rows * g[None, :]).t())
    pe = torch.softmax(edit_logits, dim=-1)
    pb = torch.softmax(base_logits, dim=-1)
    kl_ref = (pe * (torch.log(pe + 1e-30) - torch.log(pb + 1e-30))).sum(-1).mean()
    # logsumexp trick (actual-logit form)
    kl_ours = kl_leash(H, W_rows, g, base_k, lse_b)
    ok = abs(float(kl_ref) - float(kl_ours)) < 1e-4
    print(f"[selftest] KL brute={float(kl_ref):.6f}  logsumexp={float(kl_ours):.6f}  "
          f"diff={abs(float(kl_ref)-float(kl_ours)):.2e}  {'OK' if ok else 'FAIL'}")
    # round 9 MEDIUM #2, as-filed, would ABORT main(): it forces p_topK→1
    # where exp(lse_b−m)−Σexp(b_k−m) = (1−p_topK_base)·exp(lse_b−m) → 0 in
    # float32 → log(0) → NaN. The REAL (frozen base) regime is p_topK≈0.42–
    # 0.52, where that term stays ~0.48·exp(…) > 0 and the formula is exact
    # (verified vs full-softmax KL to ~2e-5). Test THAT regime instead of
    # the cancellation edge.
    torch.manual_seed(2)
    H2 = torch.randn(16, Dd, device=device)
    Wlm2 = torch.randn(Vv, Dd, device=device)
    g2 = torch.rand(Dd, device=device) + 1.0
    u2 = H2 / H2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    base2 = SQRT_D * (u2 @ (Wlm2 * g2[None, :]).t())
    base2[:, :K] = base2[:, :K] + 5.0                     # push p_topK → ~0.5
    lse_b2 = torch.logsumexp(base2, dim=-1)
    base_k2 = base2[:, :K]
    W2 = Wlm2[:K].clone()                                  # edit = base rows
    edit2 = base2.clone()
    edit2[:, :K] = SQRT_D * (u2 @ (W2 * g2[None, :]).t())
    pe2 = torch.softmax(edit2, dim=-1)
    pb2 = torch.softmax(base2, dim=-1)
    mass = float(pb2[:, :K].sum(-1).mean())
    kl_ref2 = (pe2 * (torch.log(pe2 + 1e-30) - torch.log(pb2 + 1e-30))).sum(-1).mean()
    kl_ours2 = kl_leash(H2, W2, g2, base_k2, lse_b2)
    ok2 = (abs(float(kl_ref2) - float(kl_ours2)) < 1e-4
           and bool(torch.isfinite(kl_ours2)))
    print(f"[selftest] KL p_topK~0.5 (real L23 regime): mass={mass:.3f}  "
          f"brute={float(kl_ref2):.6f} logsumexp={float(kl_ours2):.6f}  "
          f"{'OK' if ok2 else 'FAIL'}")
    return ok and ok2


# ---------------------------------------------------------------------------
# Flatness loss (differentiable, eigh path, micro-batched for memory)
# ---------------------------------------------------------------------------

def flatness_loss(M_K: torch.Tensor, H_sub: torch.Tensor,
                  r: int = R, mb: int = MB) -> torch.Tensor:
    """L_flat = E_t ‖P_t − P̄‖_F² via micro-batch accumulation.

    M_K: [K, D] trainable.  H_sub: [P, D] frozen.
    J_t = M_K − (M_K û_t) û_tᵀ;  P_t = V_t V_tᵀ (V_t = top-r right SVs of J_t).
    Accumulates sum_t ‖P_t‖² (scalar) and SPP = Σ_t P_t ([D,D]); then
    E_t ‖P_t − P̄‖² = (Σ‖P_t‖² − SPP:SPP/P)/P.  Avoids holding [P,D,D].

    E39d_fixes #1a/#1b/#9 (CRITICAL): eigh returns eigenvalues in ASCENDING
    order with eigenvectors as COLUMNS U[:, :, j]. The v1 code took
    w.flip(1)[:, -r:] (the r SMALLEST values) and U.flip(1) (a ROW reversal
    that detaches eigenvectors from eigenvalues) — Vt was then a garbage
    mixture (bottom-r values, row-reversed vectors, ~σ_top/σ_bottom gain),
    and L_flat optimized a different subspace than the κ' metric (which uses
    the correct top-r via SVD). Correct: top-r = the LAST r of the ascending
    eigenvalues, eigenvectors = the last r COLUMNS, no flip.
    """
    u = H_sub / H_sub.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # [P, D] (round 8 LOW: clamp)
    P = H_sub.shape[0]
    sum_norm_sq = torch.zeros((), device=M_K.device, requires_grad=True)
    SPP = torch.zeros(D, D, device=M_K.device, requires_grad=True)
    for i in range(0, P, mb):
        uh = u[i:i+mb]                                        # [m, D]
        Ju = torch.einsum("pd,rd->pr", uh, M_K)               # [m, K]
        J = M_K.unsqueeze(0) - Ju.unsqueeze(-1) * uh.unsqueeze(1)  # [m,K,D]
        JJt = torch.bmm(J, J.transpose(1, 2))                 # [m,K,K]
        w, U = torch.linalg.eigh(JJt)                          # w:[m,K] ascending
        s = w[:, -r:].clamp(min=1e-20).sqrt()          # [m,r] top-r (fix #1a, #9; round 8 #4: min 1e-20 → no 0-division if M_K degrades to rank < r)
        U16 = U[:, :, -r:]                                      # [m,K,r] top-r eigenvectors (fix #1b)
        Vt = torch.bmm(J.transpose(1, 2), U16) / s[:, None, :]  # [m,D,r]
        Pt = torch.bmm(Vt, Vt.transpose(1, 2))                # [m,D,D]
        sum_norm_sq = sum_norm_sq + (Pt ** 2).sum()           # scalar (grad)
        SPP = SPP + Pt.sum(0)                                  # [D,D] (grad)
    return (sum_norm_sq - (SPP * SPP).sum() / P) / P


def _selftest_flatness(device: str) -> bool:
    """E39d_fixes #1: verify flatness_loss's Vt = top-r right SVs of J and
    that P_t is a rank-r projector (P²=P, tr P=r), vs a direct SVD."""
    torch.manual_seed(0)
    P, Dd, K, r = 8, 32, 16, 4
    M_K = torch.randn(K, Dd, device=device, requires_grad=True)
    H = torch.randn(P, Dd, device=device)
    u = H / H.norm(dim=-1, keepdim=True)
    Ju = torch.einsum("pd,rd->pr", u, M_K.detach())
    J = M_K.detach().unsqueeze(0) - Ju.unsqueeze(-1) * u.unsqueeze(1)  # [P,K,D]
    # reference: top-r right SVs via SVD of J
    for i in range(P):
        Uj, sj, Vh = torch.linalg.svd(J[i].double(), full_matrices=False)
        V_ref = Vh.t().double()[:, :r]                     # [D, r] right SVs
        # flatness path (same math as flatness_loss)
        JJt = J[i] @ J[i].t()                       # [K,K]
        w, U = torch.linalg.eigh(JJt.double())      # w:[K] U:[K,K]
        s = w[-r:].clamp(min=0).sqrt()
        U16 = U[:, -r:]
        Vt = (J[i].t().double() @ U16) / s[None, :]
        # subspaces must match (basis signs may differ): projectors equal
        P_ref = V_ref @ V_ref.t()
        P_ours = Vt @ Vt.t()
        d = (P_ref - P_ours).norm().item()
        if d > 1e-5:  # float64 eigh precision
            print(f"[selftest-flatness] FAIL pos {i}: ||P_ref - P_ours||={d:.2e}")
            return False
        # projector sanity: P²=P, tr P = r
        idem = (P_ours @ P_ours - P_ours).norm().item()
        tr = P_ours.diagonal().sum().item()
        if idem > 1e-5 or abs(tr - r) > 1e-5:
            print(f"[selftest-flatness] FAIL pos {i}: idem={idem:.2e} tr={tr}")
            return False
        # energy ordering sanity: LARGEST eigenvalue of JJt = sigma_1^2
        # (s = w[-r:] in ASCENDING order, so s[-1] is the top-r max)
        if abs(s[-1].item() - sj[0].item()) > 1e-6:
            print(f"[selftest-flatness] FAIL pos {i}: top σ {s[-1].item()} vs {sj[0].item()}")
            return False
    # round 9 LOW #7: the degenerate rank case is exactly what round-9
    # MEDIUM #3 warns about (a rank-deficient M_K shrinks ‖P_t‖² and would
    # game the surrogate). flatness_loss must at least stay FINITE there
    # (the clamp(min=1e-20) on σ prevents 0-division → NaN). The surrogate
    # *gaming* itself is a known limitation, guarded only by the KL leash —
    # documented in the run output, not fixable in the loss.
    Dd2, K2 = D, 32   # Dd2 MUST == the module D: flatness_loss's SPP buffer is [D, D]
    H2 = torch.randn(8, Dd2, device=device)
    Mk0 = torch.zeros(K2, Dd2, device=device).clone().requires_grad_(True)
    lf0 = float(flatness_loss(Mk0, H2, r, MB))                    # rank 0
    Mk1 = torch.nn.functional.pad(
        torch.randn(r, Dd2, device=device), (0, 0, 0, K2 - r)
    ).clone().requires_grad_(True)                                 # rank r
    lf1 = float(flatness_loss(Mk1, H2, r, MB))
    if not (math.isfinite(lf0) and math.isfinite(lf1)):
        print(f"[selftest-flatness] FAIL degenerate: rank0={lf0} rankr={lf1} not finite")
        return False
    print(f"[selftest-flatness] degenerate rank cases finite "
          f"(rank0={lf0:.4f}, rankr={lf1:.4f}): OK")
    print("[selftest-flatness] Vt = top-r right SVs, P_t rank-r projector: OK")
    return True


# ---------------------------------------------------------------------------
# C4 docs for PPL
# ---------------------------------------------------------------------------

def _c4_docs(n_docs: int, s_len: int, tok, skip: int = C4_SKIP):
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True).skip(skip)
    # E39d_fixes #2 (CRITICAL): next(iter(ds)) created a FRESH iterator each
    # time — all 8 "docs" were the same first document after skip.
    ds_iter = iter(ds)
    texts = [next(ds_iter)["text"] for _ in range(n_docs)]
    enc = tok(texts, truncation=True, max_length=s_len, padding="max_length",
              return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


def _ppl(model, ids, attn) -> float:
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attn)
        logits = out.logits
    sl = logits[:, :-1, :].contiguous()
    st = ids[:, 1:].contiguous()
    sa = attn[:, 1:].contiguous()
    loss = F.cross_entropy(sl.view(-1, sl.size(-1)), st.view(-1),
                           reduction='none').view(st.size())
    return float(math.exp((loss * sa).sum() / sa.sum().item()))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(results_dir: str | None = None, device: str | None = None) -> dict:
    if results_dir is None:
        results_dir = os.path.join(REPO, "results")
    os.makedirs(results_dir, exist_ok=True)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    t_start = time.time()

    # selftest the KL formula
    if not _selftest_kl(device):
        raise RuntimeError("KL logsumexp selftest failed — aborting")
    # E39d_fixes #1: selftest the flatness top-r eigenvector selection
    if not _selftest_flatness(device):
        raise RuntimeError("flatness_loss selftest failed — aborting")

    print("loading model...", flush=True)
    # E39d_fixes #11: transformers <4.35 uses torch_dtype, >=4.35 uses dtype
    _ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    _dtype_kw = ({"dtype": torch.float32} if _ver >= (4, 35)
                 else {"torch_dtype": torch.float32})
    model = AutoModelForCausalLM.from_pretrained(MODEL, **_dtype_kw)
    model.eval()
    model.to(device)
    W_lm = model.lm_head.weight.detach().clone()          # [V, D]
    # RMSNorm weight (frozen g = 1 + w)
    norm_mod = getattr(model.model, "norm", None)
    if norm_mod is not None and hasattr(norm_mod, "weight"):
        g_base = (1.0 + norm_mod.weight.float()).detach().clone()
    else:
        g_base = torch.ones(D)
    print(f"V={W_lm.shape[0]} D={D}  g_base norm={g_base.norm():.4f}")

    # ---- load jcache L23, group by doc (dom,si) for the κ' dry-run ----
    jcache_path = _jcache_l23_path()
    print(f"loading jcache: {jcache_path}", flush=True)
    ck = torch.load(jcache_path, map_location="cpu", weights_only=False)  # fix #5
    H_dict = ck["H"]
    U128 = [int(t) for t in ck["U128"]]
    assert ck["D"] == D
    tok_ids = torch.tensor(U128, dtype=torch.long, device=device)
    K = len(U128)
    S = ck["S"]

    # per-doc grouping: keys (dom, si, t)
    docs = {}
    for (dom, si, t) in H_dict:
        docs.setdefault((dom, si), {})[int(t)] = H_dict[(dom, si, t)]
    # build flat H_all (all positions) + per-doc [N, S, D] for the dry-run
    all_keys = sorted(H_dict.keys())
    H_all = torch.stack([H_dict[k].float() for k in all_keys]).to(device)
    N_total = H_all.shape[0]
    print(f"H_all: {tuple(H_all.shape)}  ({N_total} positions, {len(docs)} docs)")

    # per-doc u64 [N_docs, S, D] (float64 normalized) for the κ' dry-run
    doc_ids = sorted(docs.keys())
    u64_docs = []
    for did in doc_ids:
        seq = docs[did]
        if len(seq) != S:   # E39d_fixes #10: informative failure
            raise ValueError(f"doc {did}: {len(seq)} positions, expected {S}")
        arr = torch.stack([seq[t].float() for t in range(S)])
        # round 9 #2: clamp the norm so a (pathological) zero-norm h_t gives
        # 0/0 = NaN → _kappa_dryrun κ' = NaN → the whole sweep is poisoned.
        arr = arr / arr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        u64_docs.append(arr.double().cpu())
    u64_docs = torch.stack(u64_docs).to(device)           # [N_docs, S, D] (round 8 #2: on device — _kappa_dryrun's einsum(M_kp, u_t) crashed cuda×cpu on T4)
    N_docs = u64_docs.shape[0]

    # ---- base logits cache (one-time): ACTUAL logits via the E-38e
    # factorization (round 8 CRITICAL #1). L23-only:
    #   logit_v = √D · (W_lm·diag g_base)[v,:] · û_t,   û_t = h_t/‖h_t‖.
    # The old H@W_lmᵀ were pseudo-logits (off by ‖h‖/(√D·g), position- and
    # token-dependent) → temperature-scaled KL + biased sentinel mask.
    # Verified against the real model.lm_head(model.model.norm(h)) to 9.5e-6.
    print("computing base logits cache (actual logits, lse_b, base_k, "
          "sentinel)...", flush=True)
    with torch.no_grad():
        u_all = H_all / H_all.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # [N, D]
        M_K_full = (W_lm * g_base.to(device)[None, :]).to(device)          # [V, D] frozen base
    base_k_all = torch.empty(N_total, K, dtype=torch.float32, device=device)
    lse_b_all = torch.empty(N_total, dtype=torch.float32, device=device)
    raw_sum_all = torch.empty(N_total, dtype=torch.float32, device=device)
    B = 256
    SQRT_D = math.sqrt(D)
    for i in range(0, N_total, B):
        Hb = H_all[i:i+B]
        bl = SQRT_D * (u_all[i:i+B] @ M_K_full.t())      # [B, V] actual logits
        lse_b_all[i:i+B] = torch.logsumexp(bl, dim=-1)
        base_k_all[i:i+B] = bl[:, tok_ids]
        # E-34 sentinel: SUB-STOCHASTIC mass Σ_k p_k over U128 (= P128_RAW row
        # sum; the 0.5 threshold was calibrated on it), NOT the raw numerator
        # Σ exp(logit). exp(logit−lse) = p keeps the E-34 scale.
        raw_sum_all[i:i+B] = torch.exp(bl[:, tok_ids] - lse_b_all[i:i+B, None]).sum(1)
        del bl
    # E-34 sentinel mask: sub-stochastic P128 sum < SENTINEL_SUM → exclude
    mask = (raw_sum_all >= SENTINEL_SUM)                  # [N] True=keep
    n_sent = int((~mask).sum())
    keep_idx = torch.where(mask)[0]
    print(f"sentinel excluded: {n_sent}/{N_total} "
          f"({100*n_sent/N_total:.1f}%)  [raw sum < {SENTINEL_SUM}]")
    if keep_idx.numel() == 0:
        keep_idx = torch.arange(N_total, device=device)   # fallback: keep all

    # ---- C4 PPL docs ----
    tok = AutoTokenizer.from_pretrained(MODEL)
    ppl_ids, ppl_attn = _c4_docs(N_DOCS_PPL, S_LEN, tok)
    ppl_ids, ppl_attn = ppl_ids.to(device), ppl_attn.to(device)
    ppl_base = _ppl(model, ppl_ids, ppl_attn)
    print(f"PPL base: {ppl_base:.4f}")

    # ---- baseline κ' (EXACT E-39a dry-run) ----
    W_rows_base = W_lm[tok_ids].to(device).clone()
    M_K_base = (W_rows_base * g_base.to(device)[None, :])
    kap_base = _kappa_dryrun(M_K_base.double(), u64_docs, N_docs, S, R)
    kappa_base = kap_base["kappa"]
    print(f"κ' base (E-39a dry-run): {kappa_base:.6f}  "
          f"(CI {kap_base['kappa_ci']})  target <{KAPPA_TARGET}")

    # flatness normalizer (O(1) scale for the λ sweep): l_flat at base M_K.
    # l_flat ∈ [0, r] = [0, 16] by construction (rank-r projectors:
    # ‖P_t‖_F² = tr P_t = r). The ACTUAL base value is ~0.19 (O(1)), and
    # l_kl is O(0.01–0.1) on the same data, so WITHOUT this the spec λ-grid
    # {0.01..10} would weight the two terms at very different scales and the
    # sweep would be meaningless. Normalising makes λ a relative weight.
    # (The "~1e3" figure in the pre-round-8 docstring referred to the v1
    # garbage Vt where P_t was not a projector — not the current code.)
    # E39d_fixes #8: normalizer on a RANDOM P_STEP subset (same distribution
    # as training), not the first P_STEP positions (≈ first 4 docs only).
    # torch.manual_seed(SEEDS[0]): deterministic norm_idx across runs (round 9 #6).
    torch.manual_seed(SEEDS[0])
    with torch.no_grad():
        norm_idx = keep_idx[torch.randint(0, keep_idx.numel(), (P_STEP,),
                                          device=device)]
        l_flat_base = float(flatness_loss(
            (W_rows_base * g_base.to(device)[None, :]).detach(),
            H_all[norm_idx], R, MB))
    print(f"flatness normalizer (base l_flat): {l_flat_base:.4f}")

    results = {
        "experiment": "E-39d: L23 readout training with flatness regularizer",
        "tz": "docs/E39.md §4",
        "params": {
            "model": MODEL, "r": R, "kappa_target": KAPPA_TARGET,
            "kl_pass": KL_PASS, "lambda_kl": LAMBDA_KL,
            "lambda_grid": LAMBDA_GRID, "n_steps": N_STEPS, "lr": LR,
            "seeds": SEEDS, "p_step": P_STEP, "micro_batch": MB,
            "n_total_pos": N_total, "n_docs": N_docs, "S": S,
            "vocab": W_lm.shape[0], "k_tokens": K, "device": str(device),
            "c4_skip": C4_SKIP, "n_docs_ppl": N_DOCS_PPL,
            "sentinel_excluded": n_sent,
            "trainable": "W_lm[token_set] (128×1024); g (RMSNorm) FROZEN",
        },
        "kappa_base": kappa_base,
        "kappa_base_ci": kap_base["kappa_ci"],
        "ppl_base": ppl_base,
        "runs": [],
    }

    for lam in LAMBDA_GRID:
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            W_rows = W_lm[tok_ids].to(device).detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([W_rows], lr=LR)
            run = {"lambda": lam, "seed": seed, "history": []}
            t0 = time.time()

            for step in range(1, N_STEPS + 1):
                opt.zero_grad()
                M_K = W_rows * g_base.to(device)[None, :]     # [K, D]

                # stochastic subset (non-sentinel positions)
                idx = keep_idx[torch.randint(0, keep_idx.numel(), (P_STEP,),
                                             device=device)]
                H_sub = H_all[idx]                              # [P, D]

                l_flat = flatness_loss(M_K, H_sub, R, MB) / l_flat_base
                l_kl = kl_leash(H_sub, W_rows, g_base.to(device), base_k_all[idx], lse_b_all[idx])
                loss = lam * l_flat + LAMBDA_KL * l_kl
                # NaN/Inf guard BEFORE backward/step (round 8 MEDIUM #3):
                # a non-finite loss must not corrupt W_rows / Adam state via
                # opt.step() before it is detected.
                if not torch.isfinite(loss.detach()):
                    run["diverged_at_step"] = step
                    print(f"  λ={lam} s{seed} st{step}: loss non-finite — "
                          f"run diverged, stopping (round 8 #3 pre-step NaN-guard)",
                          flush=True)
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_([W_rows], 1.0)
                opt.step()
                # post-step re-check: grads can still produce NaN params even
                # from a finite loss.
                if not torch.isfinite(W_rows.detach()).all():
                    run["diverged_at_step"] = step
                    print(f"  λ={lam} s{seed} st{step}: params non-finite after "
                          f"step — run diverged, stopping (round 8 #3)",
                          flush=True)
                    break

                if step % METRIC_EVERY == 0 or step == N_STEPS:
                    M_Kd = (W_rows.detach() * g_base.to(device)[None, :]).double()
                    kap = _kappa_dryrun(M_Kd, u64_docs, N_docs, S, R)["kappa"]
                    with torch.no_grad():
                        old_rows = model.lm_head.weight.data[tok_ids].clone()  # fix #4
                        model.lm_head.weight.data[tok_ids] = W_rows.detach()
                        ppl = _ppl(model, ppl_ids, ppl_attn)
                        model.lm_head.weight.data[tok_ids] = old_rows
                    run["history"].append({
                        "step": step, "loss": float(loss.item()),
                        "l_flat": float(l_flat.item()),
                        "l_kl": float(l_kl.item()),
                        "kappa": kap, "ppl": ppl,
                    })
                    print(f"  λ={lam} s{seed} st{step}: loss={loss.item():.6f} "
                          f"flat={l_flat.item():.6f} kl={l_kl.item():.6f} "
                          f"κ'={kap:.6f} PPL={ppl:.4f} "
                          f"({(time.time()-t0):.0f}s)", flush=True)

            # final
            if run.get("diverged_at_step") is not None:
                run["final"] = {
                    "diverged": True, "kappa": None, "ppl": None,
                    "ppl_delta_pct": None,
                    "kappa_improvement": None,
                    "delta_W_rel": float("nan"),
                    "kl_full": None,
                    "kl_last": run["history"][-1]["l_kl"] if run["history"] else None,
                    "elapsed_s": time.time() - t0,
                }
                run["verdict"] = "DIVERGED"
                run["verdict_detail"] = {"kappa_target_met": None,
                                         "ppl_within_2pct": None,
                                         "kl_within_budget": None}
                results["runs"].append(run)
                print(f"λ={lam} s{seed}: DIVERGED at step "
                      f"{run['diverged_at_step']} "
                      f"({run['final']['elapsed_s']:.0f}s)", flush=True)
                continue
            M_Kf = (W_rows.detach() * g_base.to(device)[None, :]).double()
            kap_f = _kappa_dryrun(M_Kf, u64_docs, N_docs, S, R)["kappa"]
            with torch.no_grad():
                old_rows = model.lm_head.weight.data[tok_ids].clone()  # fix #4
                model.lm_head.weight.data[tok_ids] = W_rows.detach()
                ppl_f = _ppl(model, ppl_ids, ppl_attn)
                model.lm_head.weight.data[tok_ids] = old_rows
            # E39d_fixes #6: verdict KL on the FULL dataset, not the last
            # 512-position minibatch (noise ±20-50% would flip the verdict).
            # ROUND 8 CRITICAL #1: with ACTUAL base logits (√D·M_K·û, not the
            # old pseudo H@W_lmᵀ), the readout-only edit leaves the hidden
            # states unchanged, so the logsumexp-trick KL computed from the
            # cached H is EXACTLY the full-vocab KL of the edited model vs the
            # base model — not a proxy (cross-checked below against a real
            # 2-forward KL, and vs the 2.69-nat pseudo-logit offset).
            with torch.no_grad():
                kl_f = float(kl_leash(H_all[keep_idx], W_rows.detach(),
                                      g_base.to(device),
                                      base_k_all[keep_idx],
                                      lse_b_all[keep_idx]).item())
            # Independent verification (first run only): a REAL 2-forward
            # full-vocab KL on the C4 PPL docs (base forward vs edited
            # forward). This is a SEPARATE population (C4 vs the jcache train
            # positions used for kl_f), so it is a same-ORDER magnitude check,
            # not an equality — the threshold below (max(0.05, 20%)) reflects
            # that, not a 1e-3 formula match. (round 9 LOW #5: the two numbers
            # are kept distinct so the paper reports them on their own data.)
            kl_f_fwd = None
            if lam == LAMBDA_GRID[0] and seed == SEEDS[0]:
                with torch.no_grad():
                    o_b = model(input_ids=ppl_ids, attention_mask=ppl_attn)
                    old = model.lm_head.weight.data[tok_ids].clone()
                    model.lm_head.weight.data[tok_ids] = W_rows.detach()
                    o_e = model(input_ids=ppl_ids, attention_mask=ppl_attn)
                    model.lm_head.weight.data[tok_ids] = old
                    lp_e = F.log_softmax(o_e.logits[:, :-1, :].float(), dim=-1)
                    lp_b = F.log_softmax(o_b.logits[:, :-1, :].float(), dim=-1)
                    del o_b, o_e
                    kl_tok = (lp_e.exp() * (lp_e - lp_b)).sum(-1)   # [N, S-1]
                    sa = ppl_attn[:, 1:].contiguous()
                    kl_f_fwd = float((kl_tok * sa).sum() / sa.sum().item())
                print(f"  KL cross-check (real 2-forward, C4 docs): "
                      f"{kl_f_fwd:.6f} vs cached-H verdict {kl_f:.6f} "
                      f"(diff {abs(kl_f_fwd - kl_f):.2e})", flush=True)
            if kl_f_fwd is not None and abs(kl_f_fwd - kl_f) > max(0.05, 0.2 * max(kl_f, kl_f_fwd)):
                print(f"  WARN: 2-forward KL diverges from cached-H KL — "
                      f"readout-only assumption violated? (round 8 #1)", flush=True)
            dw = (W_rows.detach() - W_lm[tok_ids].to(device)).norm() / \
                 W_lm[tok_ids].to(device).norm()
            run["final"] = {
                "kappa": kap_f, "ppl": ppl_f,
                "ppl_delta_pct": (ppl_f / ppl_base - 1) * 100,
                "kappa_improvement": kappa_base / kap_f if kap_f > 0 else float("inf"),
                "delta_W_rel": float(dw.item()),
                "kl_full": kl_f,                        # fix #6; round 8 #1: ACTUAL-logit full-dataset KL
                "kl_full_2forward_check": kl_f_fwd,     # round 8 #1: real 2-forward cross-check (first run only)
                "kl_last": run["history"][-1]["l_kl"],   # fix #13 (was "kl_mean")
                "elapsed_s": time.time() - t0,
            }
            kap_ok = kap_f < KAPPA_TARGET
            ppl_ok = ppl_f <= ppl_base * 1.02
            kl_ok = kl_f < KL_PASS
            # E39d_fixes #3: SUCCESS requires ALL THREE criteria
            run["verdict"] = ("SUCCESS" if (kap_ok and ppl_ok and kl_ok)
                              else "PARTIAL" if kap_ok else "FAIL")
            run["verdict_detail"] = {
                "kappa_target_met": kap_ok, "ppl_within_2pct": ppl_ok,
                "kl_within_budget": kl_ok,
            }
            results["runs"].append(run)
            print(f"λ={lam} s{seed}: κ'={kap_f:.6f} PPL={ppl_f:.4f} "
                  f"({run['final']['ppl_delta_pct']:+.2f}%) "
                  f"KL_full={kl_f:.4f} "
                  f"verdict={run['verdict']} ({run['final']['elapsed_s']:.0f}s)",
                  flush=True)

    # ---- summary ----
    # E39d_fixes #7: "best" = best among SUCCESS runs (all three criteria);
    # also report the lowest-κ' run overall for reference.
    n_div = sum(1 for r in results["runs"] if r["verdict"] == "DIVERGED")
    n_success = sum(1 for r in results["runs"] if r["verdict"] == "SUCCESS")
    n_partial = sum(1 for r in results["runs"] if r["verdict"] == "PARTIAL")
    n_fail = sum(1 for r in results["runs"] if r["verdict"] == "FAIL")
    success = [r for r in results["runs"] if r["verdict"] == "SUCCESS"]
    finite = [r for r in results["runs"]
              if r["final"]["kappa"] is not None and r["final"]["kappa"] > 0]
    best = min(success, key=lambda r: r["final"]["kappa"]) if success else None
    best_overall = min(finite, key=lambda r: r["final"]["kappa"]) if finite else None
    if best is not None:
        summary = {
            "best_kappa": best["final"]["kappa"],
            "best_lambda": best["lambda"], "best_seed": best["seed"],
            "best_ppl": best["final"]["ppl"],
            "best_ppl_delta_pct": best["final"]["ppl_delta_pct"],
            "best_kl_full": best["final"]["kl_full"],
            "best_kappa_improvement": best["final"]["kappa_improvement"],
            "n_success": n_success, "n_partial": n_partial,
            "n_fail": n_fail, "n_diverged": n_div,
            "total_time_s": time.time() - t_start,
            "interpretation": (
                f"SUCCESS: κ'={best['final']['kappa']:.6f} "
                f"(base={kappa_base:.6f}, "
                f"improvement={best['final']['kappa_improvement']:.1f}×) "
                f"at λ={best['lambda']}, seed={best['seed']}, "
                f"PPL={best['final']['ppl']:.4f} "
                f"({best['final']['ppl_delta_pct']:+.2f}%), "
                f"KL_full={best['final']['kl_full']:.4f}<{KL_PASS}. "
                f"E-39c (algebraic) was REFUTED on KL (6.31 nats = 126× budget): "
                f"training beats pure algebra on the flatness/KL trade-off."
            ),
        }
    elif best_overall is not None:
        summary = {
            "best_kappa": best_overall["final"]["kappa"],
            "best_lambda": best_overall["lambda"],
            "best_seed": best_overall["seed"],
            "best_ppl": best_overall["final"]["ppl"],
            "best_ppl_delta_pct": best_overall["final"]["ppl_delta_pct"],
            "best_kl_full": best_overall["final"]["kl_full"],
            "best_kappa_improvement": best_overall["final"]["kappa_improvement"],
            "n_success": n_success, "n_partial": n_partial,
            "n_fail": n_fail, "n_diverged": n_div,
            "total_time_s": time.time() - t_start,
            "interpretation": (
                f"No run met all three criteria. Lowest κ' "
                f"={best_overall['final']['kappa']:.6f} "
                f"(base={kappa_base:.6f}) at λ={best_overall['lambda']}, "
                f"seed={best_overall['seed']} (PPL "
                f"{best_overall['final']['ppl_delta_pct']:+.2f}%, KL_full "
                f"{best_overall['final']['kl_full']:.4f}). "
                f"Verdicts: {n_success} SUCCESS / {n_partial} PARTIAL / "
                f"{n_fail} FAIL / {n_div} DIVERGED."
            ),
        }
    else:
        summary = {
            "best_kappa": None, "best_lambda": None, "best_seed": None,
            "best_ppl": None, "best_ppl_delta_pct": None,
            "best_kl_full": None, "best_kappa_improvement": None,
            "n_success": n_success, "n_partial": n_partial,
            "n_fail": n_fail, "n_diverged": n_div,
            "total_time_s": time.time() - t_start,
            "interpretation": (
                f"All {n_div} runs diverged (non-finite loss) — training "
                f"unstable at these λ/lr. No finite κ' measured."
            ),
        }
    results["summary"] = summary
    out_path = os.path.join(results_dir, "exp39d_l23_training.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    print(f"Total: {(time.time()-t_start)/60:.1f} min")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    main(results_dir=args.results_dir, device=args.device)
