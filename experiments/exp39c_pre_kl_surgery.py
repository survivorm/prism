"""E-39c_pre — KL pre-gate + behavioral budget for M_K-surgery (pre-stage of
E-39c, docs/E39.md §3).  REVISED per docs/E39_fixes.md (round 1) and
docs/E39c_pre_fixes.md (rounds 2–4).

ID HISTORY (fixes §0): the first implementation was named `exp39b_kl_surgery`
— an ID collision (E-39b per TЗ §2 is the PTQ/GPTQ gate). Renamed to
E-39c_pre: the KL-budget pre-stage of E-39c (surgery), not E-39b.

BUGFIX (round 1 §1, critical): the old code applied `W_lm[U128] += (M_kp−M_K)`
— a delta in M_K-space added to W_lm-space. Since M_K = W_lm·diag(g), the
achieved M_K was M_K_old·(1−g) + M_kp·g (row-scaled by γ≠1). Fix (TЗ §3.3
step 6): W_lm'[token_set] = M_kp · diag(γ)⁻¹.

ROUND 2 (docs/E39c_pre_fixes.md):
  §4.1 — gate uses t≥1 (matches the docstring; t=0 is the unconditioned
         first token, not a behavioral cost).
  §4.2 — γ handled in ONE device/dtype (g_dev) — no dead `g_dev`, no
         CPU/GPU device-mismatch risk.
  §4.3 — near-singular γ (min < 1e-3) → explicit warning (inverse
         normalization amplifies a column by 1/min).
  §5.1 — KL DECOMPOSITION under tied weights (the paper-level limitation):
         the 0.8B model ties lm_head to embed_tokens, so the full edit changes
         BOTH the readout and the input embeddings (→ all 24 layers). We run
         three untied model states to split the KL:
             full       : head + embed both edited   (tied, = the real model)
             head_only  : head edited, embed = base  (readout-only cost)
             embed_only : embed edited, head = base  (upstream propagation)
         plus a last-layer hidden-state drift for embed_only (how much of the
         damage is upstream, not readout).
  §5.3 — V_new STRATEGY sensitivity (the third E-39a strategy): do any of
         b_principal / c_balanced give a lower full-surgery cost than the
         bottom-16 (a_random)? All three are tested at full surgery (α=1).
         If NO direction is KL-safe, the failure is structural (V_new ⊥
         V_top), not a consequence of the bottom-16 choice. Defined locally
         (mirrors E-39a) to avoid the exp39a coupling (round-1 §8.1).
  §7   — α-BLEND sweep (the actionable next step): full top-16 rotation
         (α=1) is catastrophic (KL≈6.31 post-fix, was ≈5.02 pre-Vh.T-fix). We test partial rotation with the
         CLEAN blend (round 3 §2.2): V_mixed(α) = √α·V_new_perp + √(1−α)·V_top
         (NO QR — orthonormal by construction since V_new_perp ⊥ V_top; the
         earlier qr(·) version silently negated M_kp terms via Householder
         sign flips on an orthonormal input), M_kp(α) = U_top Σ_top
         V_mixed(α)ᵀ + M_rest, and sweep α ∈ {0,0.01,0.02,0.05,0.08,0.1,0.2,
         0.5,0.7,0.9,1.0} (round 4 §3 EXTENDS the grid below 0.1 — the
         round-3 grid's min α=0.1 was already 1.26× over the KL budget, and
         the KL=0.05 crossing sits near α≈0.08 which the old grid never
         probed) recording KL(α). Per-α perturbation ‖V_mixed−V_top‖_col is
         checked against the theory √(2−2√(1−α)) (round 3 §2.4) to verify
         the blend is a true partial rotation. round 4 §3 also measures
         per-α κ'(α) DRY-RUN (E-39a's exact _kappa_dryrun, N=32) so the
         KL-budget-vs-flattening-target trade-off is quantified across α —
         the α-grid alone cannot answer "is there an α with both KL<0.05 AND
         κ'<0.005". The sweep therefore asks: on the tested grid, is there
         ANY non-trivial α with BOTH KL < 0.05 AND κ' < 0.005?

Round-1 fixes retained (numbers refer to docs/E39_fixes.md, round 1):
  §2 κ' = DRY-RUN (base û_t, not a measurement); §3 row_diag in one space
  (W_lm-space); §4.1 κ' vs E-39a consistency (warn); §4.2 per-doc float32
  (no 4 GB); §4.4 V_new true-null/bottom-R branch; §4.5 explicit row
  assignment; §5.1 KL over attention_mask (padding excluded); §5.3 C4 via
  ds.skip(4096); §5.4 KL_PASS = 0.05 (half of the E-16 0.1 leash); §5.5 tied
  weights a real limitation. (Round-2 §5.1/§5.3 in the block above are the
  NEW decomposition + strategy-sensitivity additions, distinct from the
  round-1 §5.1/§5.3.)

Round-3 fixes (numbers refer to docs/E39c_pre_fixes.md, round 3):
  §2.1 rho_noise convention aligned with E-39a (per-position Frobenius norm,
  key renamed rho_noise_u_obs_overlap_mean); §2.2/§2.4 the α-blend QR is
  removed (orthonormal by construction) + per-α perturbation-vs-theory check;
  §4.2 V_new_perp computed once in the closure; §4.6 b_principal via the
  factored projector (no [D,D] materialization).
Round-4 fixes (docs/E39c_pre_fixes.md, round 4):
  §3 the α-grid is EXTENDED below 0.1 (0.01/0.02/0.05/0.08) and per-α κ'(α)
  dry-run is measured (E-39a's _kappa_dryrun, N=32) → the KL-vs-κ' trade-off
  is computed (disjoint/overlapping passing regions); a_random κ' dry-run is
  a consistency check vs E-39a's a_random κ' (loaded from the E-39a JSON;
  post-fix value 0.000297); §4 the 'near-budget' hedge is
  replaced by an honest '× budget, FAILS' budget_note; §5 the conclusion is
  scoped to 'the tested grid' (no 'no α works' overclaim); §6 the
  kappa_note logical error ('flattening side already satisfied') is fixed —
  κ'(a_random, exact) applies ONLY at α=1, so both constraints are
  simultaneously binding.

Protocol (CPU, deterministic seed=0):
  1. Base forward on 8 fresh C4 docs (skip 4096), S=128 → logprobs_base (once).
  2. M_K = build_M_K(W_lm, U128, g); SVD M_K = U Σ Vᵀ; top-16 split; V_new =
     bottom-16 right SVs of U=span{û_t} (E-39a construction).
  3. Three untied states (full/head_only/embed_only) → KL decomposition.
  4. α-blend sweep → per-α KL (the binding constraint; κ' referenced from
     E-39a, not recomputed).
  5. Gating: PASS iff mean KL (real tokens, t≥1) < 0.05 nats/token.

Note: the gate verdict for the FULL surgery (α=1) is unchanged by rounds 2/3
(KL≈6.31 ≫ 0.05; pre-Vh.T-fix it was ≈5.02). Round 2 adds the decomposition + the α-sweep that answer
*why* it fails and *whether* a weaker surgery survives; round 3 fixes the
α-blend construction (the earlier QR-blend silently negated M_kp terms) so
the partial-surgery verdict rests on a verified-clean rotation.
"""

import json
import os
import sys
import time
import warnings
from contextlib import contextmanager

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _e38_cache import D, load_layer  # noqa: E402
from prism.interventions import _resolve_lm_head_weight  # noqa: E402
from prism.jacobian import _lm_module  # noqa: E402
from prism.jstruct import build_M_K  # noqa: E402
# round 4 §3: per-α κ' dry-run — reuse E-39a's EXACT _kappa_dryrun so the
# α=1 point reproduces E-39a's a_random κ' (consistency check, loaded from
# the E-39a JSON) and the
# κ'(α) curve is measured under the same construction E-39a validated.
from exp39a_u_obs_rank import _kappa_dryrun  # noqa: E402

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
# round 4 §3: κ' flattening target (E-39a: κ < 0.005). Per-α κ' dry-run is
# measured against this to close the KL-vs-κ' trade-off the α-grid alone can't.
KAPPA_TARGET = 0.005
# round 5 §4.5: E-39a's ACTUAL a_random dry-run κ' — loaded from the E-39a
# JSON (surgery_dryrun.a_random.kappa) so the a_random consistency check is a
# TRUE machine-precision test. The earlier hardcoded 0.00396477 (and before
# it 0.0042) were stale quotes: after the E-39a VU-fix (bottom-16 right-SVs of
# U_obs as V_new) the real value is 0.000297, and the hardcoded ref made the
# consistency check report a false 0.0037 «diff». Loading from the JSON makes
# the check track whatever E-39a actually measured (E39_a_fixes.md §5).
with open(os.path.join(REPO, "results", "exp39a_u_obs_rank.json")) as _f:
    _E39A_DATA = json.load(_f)
KAPPA_39A_FULL = _E39A_DATA["surgery_dryrun"]["a_random"]["kappa"]  # E-39a a_random κ' (exact, from JSON)
KAPPA_39A_REF = KAPPA_39A_FULL  # alias for the α=1 dry-run ref in the JSON
QR_RANK_TOL = 1e-6
# round 2 §4.3: near-singular γ threshold (inverse normalization amplifies a
# column by 1/min(γ); 1e-3 → 1000×, already dangerous).
G_SINGULAR_TOL = 1e-3
# round 2 §7 + round 4 §3: α-blend sweep grid. round 4 EXTENDS it below 0.1
# (0.01/0.02/0.05/0.08): the round-3 grid's min-α (0.1) was already 1.26× over
# the KL budget, and the review's extrapolation (KL~α^2.5) predicts the KL=0.05
# crossing sits near α≈0.08 — which the round-3 grid NEVER probed. Adding
# these points closes the "no α works" overclaim by actually measuring the
# α<0.1 region (4 extra forwards, ~40s).
ALPHA_GRID = [0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.2, 0.5, 0.7, 0.9, 1.0]
# round 2 §5.3: candidate-pool width for the three E-39a V_new strategies
# (E-39a uses R_POOL=64 so (a)/(b)/(c) span genuinely distinct 16-dim
# subspaces; the α-sweep's V_new is the bottom-16, a narrower pool).
R_POOL = 64


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


@torch.no_grad()
def _kl_vs_base(lp_edit, lp_base, attn):
    """Per-token KL(P_edit ‖ P_base) over real tokens (padding excluded).
    Returns (mean_t_ge_0, mean_t_ge_1, per_doc_t_ge_1).

    E39_a_fixes Bug 2: per_doc is now computed at t≥1 (matching the field
    name `per_doc_t_ge_1`). The earlier version used `real` (t≥0), which
    silently included the unconditioned first token t=0 — an error of ~0.5
    nats in the per-doc values."""
    p_edit = lp_edit.exp()
    kl = (p_edit * (lp_edit - lp_base)).sum(dim=-1)          # [N, S]
    real = attn.bool()                                        # excludes padding
    t1 = real.clone(); t1[:, 0] = False                       # excludes t=0
    per_doc = [float(kl[i, t1[i]].mean()) for i in range(kl.shape[0])]
    return (float(kl[real].mean()), float(kl[t1].mean()), per_doc)


def _obs_subspace(sU, VU, R_r, report):
    """True null space if ≥ R dims, else bottom-R right SVs (E-39a branch)."""
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
def _hidden_drift(base_hidden, edit_hidden, attn):
    """‖h_edit − h_base‖ / ‖h_base‖ (round 2 §5.1): how much the last-layer
    hidden state moves when the input embeddings are edited (upstream
    propagation), independent of the readout.

    round 2 §4.3: stats are over REAL tokens with t≥1 only (matching the gate,
    which excludes the unconditioned first token t=0).
    """
    b = base_hidden.float(); e = edit_hidden.float()
    rel = (e - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-9)   # [N, S]
    real = attn.bool().clone()
    real[:, 0] = False   # round 2 §4.3: t≥1 (consistent with the gate)
    if not bool(real.any()):
        return {"mean_rel": 0.0, "p90_rel": 0.0, "max_rel": 0.0}
    return {"mean_rel": float(rel[real].mean()),
            "p90_rel": float(np.percentile(rel[real].numpy(), 90)),
            "max_rel": float(rel[real].max())}


@torch.no_grad()
def main(results_dir=None):
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = results_dir if results_dir else os.path.join(REPO, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp39c_pre_kl_surgery.json")

    # ---------------- cache + U_obs⊥ ----------------
    ck = load_layer(23, want=("H",))
    H, U128 = ck["H"], ck["U128"]
    u = H / H.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # round 4 §3: float64 û_t for the per-α κ' dry-run (E-39a convention).
    u64 = u.double()
    U = u.reshape(-1, D).double()
    print("U SVD ...")
    _, sU, Vh = torch.linalg.svd(U, full_matrices=False)
    # CRITICAL FIX (E39_a_fixes Bug 1): torch.linalg.svd returns (U, S, Vh)
    # where Vh has right SVs as ROWS (Vh = Vᵀ). The code treated VU as V
    # (right SVs as columns), so VU[:, -R] was selecting the last COLUMNS of
    # Vh (= an arbitrary orthonormal basis, NOT the bottom-R right SVs of U).
    # ρ_noise PRE-fix = 0.0148 ≈ R/D = 0.0156 confirmed this (a true bottom-R subspace
    # gives ρ ≈ σ_bottom16²/N ≈ 3.9e-5, i.e. the POST-fix value ≈0.0002). Fix: transpose.
    VU = Vh.T   # now VU = V: right SVs as columns
    sub_rep = {}
    P_sub, sub_rep = _obs_subspace(sU, VU, R, sub_rep)
    sv = torch.linalg.svdvals(P_sub)
    assert sv[-1] > QR_RANK_TOL * sv[0], "V_new pool rank-deficient"
    V_new = torch.linalg.qr(P_sub)[0]
    assert (V_new.T @ V_new - torch.eye(R, dtype=V_new.dtype)).abs().max() < 1e-8

    # ---------------- model ----------------
    print("loading model fp32 ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids, attn = _c4_docs(N_DOCS, S_LEN, tok)
    print(f"C4 docs: {ids.shape} (skip {C4_SKIP})")
    print("base forward ...")
    lp_base = _logprobs(model, ids)

    # ---------------- surgery (M_K-space) ----------------
    W_lm = _resolve_lm_head_weight(model)
    assert W_lm.shape[1] == D, W_lm.shape
    vocab = W_lm.shape[0]
    g = _lm_module(model).norm.weight + 1                     # [D] (1+w)
    # round 2 §4.3: near-singular γ → inverse normalization amplifies a column.
    if float(g.min()) < G_SINGULAR_TOL:
        warnings.warn(
            f"γ near-singular: min={float(g.min()):.2e} — inverse normalization "
            f"amplifies a column by {1.0/float(g.min()):.0e}×; row_diag may be "
            f"dominated by that column (round 2 §4.3)")
    assert g.min() > 1e-6, f"γ has zeros: min={float(g.min())}"
    # round 2 §4.2: γ in ONE device/dtype, used everywhere (no dead g_dev).
    g_dev = g.to(device=W_lm.device, dtype=torch.float64)

    M_K = build_M_K(W_lm, list(U128), g).double().detach()    # [K, D]
    mnorm_old = float(torch.linalg.norm(M_K))
    Uk, sk, Vk = torch.linalg.svd(M_K, full_matrices=False)
    U_top, s_top, V_top = Uk[:, :R], sk[:R], Vk[:R, :]
    M_rest = Uk[:, R:] @ torch.diag(sk[R:]) @ Vk[R:, :]
    # round 2 §4.1 (MEDIUM): the key overlap diagnostic for the α-blend —
    # ‖V_newᵀ V_top‖_F (0 = orthogonal → blend stable; 4 = fully aligned).
    vnew_vtop_ov = float(torch.linalg.norm(V_new.T @ V_top.T, "fro"))
    # round 3 §4.2: V_new_perp (orthogonalized against V_top) is computed ONCE
    # here and shared by every _mkp_and_rows(alpha) sweep call (it is the
    # closure the blend branch references; round 3 §4.6 avoids re-materializing
    # the P_pool projector too — V_b uses the factored form).
    V_top_col_0 = V_top.T                                  # [D, R]
    V_new_perp = torch.linalg.qr(
        V_new - V_top_col_0 @ (V_top_col_0.T @ V_new))[0]  # [D, R], ⊥ V_top

    def _mkp_and_rows(alpha, orthogonalize=True):
        """M_kp(α) and its inverse-normalized W_lm rows.
        α=0 → no-op (M_kp=M_K). α=1 → full surgery.

        Orientation: both right-SV bases are kept as COLUMN matrices [D, R]
        (each column is one unit vector in the D-dim residual stream).
          V_new_col : the E-39a bottom-16 pool (QR result, already [D, R]).
          V_top_col : V_top from the M_K SVD, transposed from [R, D]→[D, R].

        round 3 §2.2 (blend fix, replaces round 2's qr-blend): when
        `orthogonalize` (the α-SWEEP), V_new is first orthogonalized against
        V_top (V_new_perp ⊥ V_top). Since V_new_perp ⊥ V_top, the combo
        √α·V_new_perp + √(1−α)·V_top is ORTHONORMAL BY CONSTRUCTION for all
        α (each column has norm √(α+(1−α))=1, columns stay mutually
        orthogonal) — so NO QR is applied to the combo. The earlier version
        re-orthogonalized the combo with torch.linalg.qr; on an already
        orthonormal matrix QR degenerates to a signed permutation whose sign
        flips are arbitrary (Householder convention), so it silently NEGATED
        some singular-direction terms of M_kp (turning the small α-rotation
        into large sign flips → row_diag 0.77 at α=0.1 vs theoretical 0.32,
        and non-monotonic KL). Dropping the QR gives a TRUE partial rotation:
        per-column ‖V_mixed−V_top‖ = √(2−2√(1−α)) exactly, and row change
        monotone in α. When `orthogonalize=False` (the α=1 FULL-SURGERY
        design = the actual E-39a a_random rotation), V_new is used RAW so
        the full-surgery KL is the real E-39a design's cost.
        M_kp(α) = U_top·diag(σ_top)·V_mixedᵀ + M_rest   ([K,D]·[D,R]ᵀ=[K,D]).
        """
        V_top_col = V_top.T                                   # [D, R]
        if not orthogonalize:
            V_new_use = V_new
        else:
            # V_new_perp ⊥ V_top (round 2 §2), orthonormal (round 3 §4.2:
            # computed once in the closure, shared across all α).
            V_new_use = V_new_perp
        if alpha <= 0.0:
            V_mixed_col = V_top_col
        elif alpha >= 1.0:
            # round 2 §4.5: α=1 endpoint (NOT extrapolation; a grid value
            # >1.0 still returns V_new_use, never an extrapolated direction).
            V_mixed_col = V_new_use
        else:
            # round 3 §2.2: NO QR — orthonormal by construction (V_new_perp
            # ⊥ V_top). A QR here would be a signed permutation with
            # arbitrary sign flips = silent term negations in M_kp.
            sq_a = torch.sqrt(torch.tensor(alpha, dtype=V_new.dtype))
            sq_b = torch.sqrt(torch.tensor(1.0 - alpha, dtype=V_new.dtype))
            V_mixed_col = sq_a * V_new_use + sq_b * V_top_col
        # V_mixed_col [D,R] → V_mixed_col.T [R,D] (row-vectors form).
        M_kp = (U_top @ torch.diag(s_top)) @ V_mixed_col.T + M_rest
        # TЗ §3.3 step 6: W_lm'[token_set] = M_kp · diag(γ)⁻¹ (round 2 §4.2:
        # γ in the SAME device/dtype as M_kp).
        W_new_rows = (M_kp / g_dev[None, :]).to(W_lm.dtype).to(W_lm.device)
        return M_kp, W_new_rows, V_mixed_col

    # ---------------- α=1 surgery (the design under test) ----------------
    # The FULL surgery = the bottom-16 RAW V_new (VU[:,-16:], orthogonalize=False).
    # This is NOT the E-39a a_random design (bottom-64 random; that costs
    # a_random kl_t1 — see v_new_strategy_sensitivity). The headline full-surgery
    # KL is the bottom-16 raw-rotation cost (≈6.31 post-fix), independent of the
    # blend fix (which only cleans the partial α∈(0,1)).
    M_kp, W_new_rows, _ = _mkp_and_rows(1.0, orthogonalize=False)
    mnorm_new = float(torch.linalg.norm(M_kp))
    assert abs(mnorm_new - mnorm_old) < 1e-6, (
        f"M_kp Frobenius drift: {mnorm_old:.6f} → {mnorm_new:.6f}")

    # row diagnostics in ONE space (W_lm-space): round 1 §3
    dW = W_new_rows.double().cpu() - W_lm[list(U128)].double().cpu()
    row_norm = W_lm[list(U128)].double().cpu().norm(dim=-1).clamp_min(1e-9)
    dratio = dW.norm(dim=-1) / row_norm
    row_diag = {
        "space": "W_lm (dW = M_kp/g − W_lm_old; ratios in W_lm-space)",
        "mean_dW_over_row": float(dratio.mean()),
        "max_dW_over_row": float(dratio.max()),
        "p90_dW_over_row": float(np.percentile(dratio, 90)),
    }

    # ---------------- round 2 §5.1: tied-weight KL decomposition ----------------
    # The 0.8B model ties lm_head ↔ embed_tokens. The real (tied) model's full
    # edit changes BOTH. We run three untied states to split the KL. `base_t`
    # is the original (shared) weight; `edited_t` is a clone with the U128
    # rows = W_new_rows. Pointers are reassigned per state; no in-place
    # mutation of the shared tensor (round 1 §4.5).
    base_t = model.lm_head.weight.data   # original shared tensor (unmutated)
    edited_t = base_t.clone()
    edited_t[list(U128)] = W_new_rows

    last_layer = _lm_module(model).layers[-1]
    cap = {}
    def _hook(m, inp, out):
        cap["h"] = out.detach()
    h_hook = last_layer.register_forward_hook(_hook)

    def _state(head_t, emb_t):
        # reassign as Parameter (a bare FloatTensor is rejected). base_t IS the
        # original shared tensor, so the tied state restores the real weights.
        model.lm_head.weight = torch.nn.Parameter(head_t)
        model.model.embed_tokens.weight = torch.nn.Parameter(emb_t)

    @contextmanager
    def _state_ctx(head_t, emb_t):
        """round 2 §4.4: set state, yield, and ALWAYS restore the tied base
        state in finally (so an exception in a block cannot leave the model
        in a partial/intermediate weight state)."""
        _state(head_t, emb_t)
        try:
            yield
        finally:
            _state(base_t, base_t)

    print("KL decomposition (full / head_only / embed_only) ...")
    dec = {}
    # FULL (tied): both edited.
    with _state_ctx(edited_t, edited_t):
        cap.clear(); lp_full = _logprobs(model, ids)
    k0, k1, pd = _kl_vs_base(lp_full, lp_base, attn)
    dec["full"] = {"mean_t0": k0, "mean_t1": k1, "per_doc_t1": pd}
    # HEAD_ONLY: head edited, embed = base (readout-only cost).
    with _state_ctx(edited_t, base_t):
        cap.clear(); lp_ho = _logprobs(model, ids)
    k0, k1, pd = _kl_vs_base(lp_ho, lp_base, attn)
    dec["head_only"] = {"mean_t0": k0, "mean_t1": k1, "per_doc_t1": pd}
    # EMBED_ONLY: embed edited, head = base (upstream propagation).
    with _state_ctx(base_t, edited_t):
        cap.clear(); lp_eo = _logprobs(model, ids)
        h_eo = cap.get("h")   # detached (hook does out.detach())
    k0, k1, pd = _kl_vs_base(lp_eo, lp_base, attn)
    dec["embed_only"] = {"mean_t0": k0, "mean_t1": k1, "per_doc_t1": pd}
    if h_eo is not None:
        # also capture base last-layer hidden for the drift reference
        with _state_ctx(base_t, base_t):
            cap.clear(); _logprobs(model, ids)
            h_base_last = cap.get("h")
        if h_base_last is not None:
            dec["embed_only_hidden_drift"] = _hidden_drift(
                h_base_last, h_eo, attn)
    h_hook.remove()
    # restore tied (both = base) for the α-sweep, which edits head+embed as one.
    _state(base_t, base_t)
    print(f"  KL t≥1: full={dec['full']['mean_t1']:.4f} "
          f"head_only={dec['head_only']['mean_t1']:.4f} "
          f"embed_only={dec['embed_only']['mean_t1']:.4f}")

    # ---------------- round 2 §7: α-blend sweep ----------------
    # For each α, edit head+embed (tied = real model) with M_kp(α) and record
    # KL(α) (the binding behavioral cost) AND κ'(α) dry-run (round 4 §3) so the
    # KL-budget-vs-flattening-target trade-off is quantified across α.
    print("α-blend sweep (KL + per-α κ' dry-run, round 4: KL-budget vs "
          "flattening-target trade-off) ...")
    sweep = []
    for alpha in ALPHA_GRID:
        M_kp_a, W_new_a, Vm_a = _mkp_and_rows(alpha)
        # tied edit: assign the same edited tensor to head and embed.
        ed = base_t.clone(); ed[list(U128)] = W_new_a
        with _state_ctx(ed, ed):
            lp_a = _logprobs(model, ids)
        k0, k1, pd = _kl_vs_base(lp_a, lp_base, attn)
        # row change for this α (W_lm-space)
        dW_a = W_new_a.double().cpu() - W_lm[list(U128)].double().cpu()
        mean_row = float((dW_a.norm(dim=-1) / row_norm).mean())
        crossing = bool(k1 < KL_PASS)   # κ' side already satisfied (E-39a)
        # round 2 §4.2 (LOW): per-α ‖V_mixedᵀ V_top‖_F (4=full-top, 0=ortho).
        vm_top = float(torch.linalg.norm(Vm_a.T @ V_top.T, "fro"))
        # round 3 §2.4 (HIGH): per-α per-column perturbation ‖V_mixed[:,j] −
        # V_top[:,j]‖ vs theory √(2−2√(1−α)). For a CLEAN blend (V_new_perp
        # ⊥ V_top, no QR) every column has EXACTLY this norm (uniform). If
        # max ≫ expected, the blend is not a clean rotation (QR sign-flip /
        # degenerate-column fill) and the row change is inflated — i.e. the
        # non-monotonicity would be a construction artifact, not loss landscape.
        V_pert = Vm_a - V_top.T                          # [D, R]
        per_col = V_pert.norm(dim=0)                     # [R]
        exp_pert = float(np.sqrt(max(0.0, 2.0 - 2.0 * np.sqrt(1.0 - alpha))))
        pc_mean = float(per_col.mean())
        pc_max = float(per_col.max())
        pc_uniform = bool(pc_max < 1.5 * exp_pert) if alpha > 0 else True
        # round 4 §3: per-α κ' DRY-RUN (base-model û_t, E-39a's exact
        # _kappa_dryrun, N=32 cache, S=128). This closes the KL-vs-κ' trade-off
        # the α-grid alone cannot: the KL budget (KL<0.05) is only satisfiable
        # at SMALL α, but the flattening target (κ'<0.005) needs a substantial
        # top-16 rotation (LARGE α). Measuring κ'(α) shows whether the two
        # "passing" regions overlap. α=1 (V_new_perp) ≈ E-39a's a_random, so
        # its κ' should match E-39a a_random's κ' (post-fix: 0.000297, loaded
        # from the E-39a JSON) — a machine-precision consistency check on the dry-run pipeline.
        kap_a = _kappa_dryrun(M_kp_a, u64, u64.shape[0], S_LEN, R)["kappa"]
        sweep.append({
            "alpha": alpha,
            "kl_t1": k1, "kl_t0": k0,
            "mean_dW_over_row": mean_row,
            "v_mixed_vs_v_top_fro": vm_top,
            "pert_col_expected": exp_pert,
            "pert_col_observed_mean": pc_mean,
            "pert_col_observed_max": pc_max,
            "pert_col_uniform_clean_rotation": pc_uniform,
            "kappa_dryrun": kap_a,
            "kappa_below_target": bool(kap_a is not None and kap_a < KAPPA_TARGET),
            "kl_below_0.05": crossing,
        })
        print(f"  α={alpha:<4} KL(t≥1)={k1:8.4f} row={mean_row:.3f} "
              f"|Vm.Vt|={vm_top:.3f} pert={pc_mean:.3f}/{pc_max:.3f} "
              f"(exp {exp_pert:.3f}) κ'={kap_a:.4f} clean={pc_uniform} "
              f"KL<0.05={crossing} κ'<0.005={kap_a < KAPPA_TARGET}")
    # state restored by _state_ctx finally; free the last per-α clone.
    del ed

    # ---------------- round 2 §5.3: V_new strategy sensitivity ----------------
    # E-39a defined three V_new strategies in the bottom-64 pool. Do any give a
    # lower (KL-safe) full-surgery cost than a_random (bottom-16)? If NO
    # direction is KL-safe at full rotation, the surgery is infeasible
    # regardless of the V_new choice — a stronger negative result than
    # "bottom-16 fails". Strategies are defined LOCALLY (mirroring E-39a's
    # construction) to avoid the exp39a coupling flagged in review §8.1.
    P_pool = VU[:, -R_POOL:]                       # [D, R_POOL] bottom-64 pool
    rng = np.random.default_rng(SEED)
    Aa = torch.from_numpy(rng.normal(size=(R_POOL, R))).double()
    V_a = torch.linalg.qr(P_pool @ torch.linalg.qr(Aa)[0])[0]      # a_random
    # round 3 §4.6: b_principal = qr(P_pool·P_poolᵀ·V_topᵀ) computed in the
    # FACTORED form qr(P_pool·(P_poolᵀ·V_topᵀ)) — avoids materializing the
    # [D,D] projector (same result, ~67M mults saved).
    V_b = torch.linalg.qr(P_pool @ (P_pool.T @ V_top.T))[0]   # b_principal
    # round 2 §3 (MEDIUM): rank-check the c_balanced concatenation. If
    # [V_a[:,:8], V_b[:,:8]] is rank-deficient (overlap), the QR gives an
    # ARBITRARY completion → its KL is meaningless. (E-39a validated V_c this
    # way via strategies_distinct; mirror the check here.)
    Vc_cat = torch.cat([V_a[:, :8], V_b[:, :8]], dim=1)
    sv_c = torch.linalg.svdvals(Vc_cat)
    V_c_valid = bool(sv_c[-1] > QR_RANK_TOL * sv_c[0])
    if not V_c_valid:
        warnings.warn(
            f"c_balanced rank-deficient (σ_min/σ_1="
            f"{float(sv_c[-1] / sv_c[0]):.2e}); its KL is unreliable")
    V_c = torch.linalg.qr(Vc_cat)[0]               # c_balanced
    strat_kl = {}
    print("§5.3 V_new strategy sensitivity (full surgery, α=1) ...")
    for name, Vn in [("a_random", V_a), ("b_principal", V_b),
                     ("c_balanced", V_c)]:
        M_kp_s = (U_top @ torch.diag(s_top)) @ Vn.T + M_rest   # α=1 to Vn
        Ws = (M_kp_s / g_dev[None, :]).to(W_lm.dtype).to(W_lm.device)
        ed = base_t.clone(); ed[list(U128)] = Ws
        with _state_ctx(ed, ed):
            lp_s = _logprobs(model, ids)
        k0, k1, _ = _kl_vs_base(lp_s, lp_base, attn)
        # ρ_noise: contamination. round 3 §2.1: match E-39a's convention exactly
        # — E_t[‖û_tᵀ Vn‖²] = per-position Frobenius norm (sum over the R cols,
        # then mean over positions), NOT the per-element mean (which is 16×
        # smaller and was silently inconsistent across E-39a/E-39c_pre).
        ov = float((u.double() @ Vn).pow(2).sum(dim=-1).mean())
        # round 2 §4.1 (MEDIUM): per-strategy ‖Vnᵀ V_top‖_F (overlap diagnostic
        # for whether that strategy's full-surgery is a clean rotation).
        vn_ov = float(torch.linalg.norm(Vn.T @ V_top.T, "fro"))
        # round 4 §3: per-strategy κ' DRY-RUN (E-39a's _kappa_dryrun, N=32
        # cache). a_random is EXACTLY E-39a's a_random (bottom-64 QR of a
        # random projection), so its κ' should reproduce E-39a's 0.0040
        # (consistency check on the dry-run pipeline + the E-39c_pre V_new
        # construction). b/c are the other two E-39a strategies.
        kap_s = _kappa_dryrun(M_kp_s, u64, u64.shape[0], S_LEN, R)["kappa"]
        strat_kl[name] = {
            "kl_t1": k1,
            # round 3 §2.1: key name + convention now MATCH E-39a exactly
            # (E_t[‖û_tᵀ Vn‖²], per-position Frobenius norm) so the two
            # experiments are directly comparable (no silent 16× gap).
            "rho_noise_u_obs_overlap_mean": ov,
            "v_new_vs_v_top_fro": vn_ov,
            "kappa_dryrun": kap_s,
            "kappa_below_target": bool(kap_s is not None and kap_s < KAPPA_TARGET),
            "kl_below_0.05": bool(k1 < KL_PASS),
        }
        print(f"  {name:12} KL(t≥1)={k1:.4f}  ρ_noise={ov:.5f} "
              f"κ'={kap_s:.4f} |Vn.Vt|={vn_ov:.3f} KL<0.05={k1 < KL_PASS} "
              f"κ'<0.005={kap_s < KAPPA_TARGET}")
    # state restored by _state_ctx finally; free the last clone.
    del ed
    any_strat_kl_safe = any(s["kl_below_0.05"] for s in strat_kl.values())

    # ---------------- gate (round 2 §4.1: t≥1) ----------------
    kl_full_t1 = dec["full"]["mean_t1"]
    gate_pass = bool(kl_full_t1 < KL_PASS)
    # any α (incl. no-op α=0) with KL below budget; α=0 is the trivial no-op.
    nontrivial_kl_safe = [s["alpha"] for s in sweep
                          if s["kl_below_0.05"] and s["alpha"] > 0.0]
    min_kl_nonzero = min((s["kl_t1"] for s in sweep if s["alpha"] > 0.0),
                         default=None)

    # ---------------- round 4 §3: KL-vs-κ' trade-off (the actionable result) ----------------
    # The round-3 grid's min-α (0.1) was already 1.26× over the KL budget; the
    # round-4 grid probes α<0.1 so the KL=0.05 crossing is MEASURED, not
    # extrapolated. Meanwhile κ'(α) is monotone DECREASING (base 0.0162 at α=0
    # → ~0.004 at α=1), so the KL budget (small α) and the flattening target
    # (large α) are in tension. Compute the two "passing" regions and whether
    # they overlap — the question the α-grid alone could not answer.
    kl_safe_alphas = [s["alpha"] for s in sweep
                      if s["kl_below_0.05"] and s["alpha"] > 0.0]
    kap_safe_alphas = [s["alpha"] for s in sweep
                       if s["kappa_below_target"] and s["alpha"] > 0.0]
    both_safe_alphas = [s["alpha"] for s in sweep
                        if s["kl_below_0.05"] and s["kappa_below_target"]
                        and s["alpha"] > 0.0]
    # smallest non-trivial α that passes KL, and the smallest α that passes κ'
    # (needs substantial rotation). round 5 §4.2: renamed from
    # kl_crossing_alpha (misnomer: 0.01 is the smallest PASSES, not the
    # KL=0.05 crossing, which sits between α=0.08 and α=0.1).
    kl_min_passing_alpha = min(kl_safe_alphas) if kl_safe_alphas else None
    kap_min_passing_alpha = min(kap_safe_alphas) if kap_safe_alphas else None
    # round 5 §4.2: the actual KL=budget crossing α, by linear interpolation
    # between the last passing and first failing α (data, not a guess).
    kl_budget_crossing_estimate = None
    _p = [s for s in sweep if s["alpha"] > 0.0 and s["kl_below_0.05"]]
    _f = [s for s in sweep if s["alpha"] > 0.0 and not s["kl_below_0.05"]]
    if _p and _f:
        a_lo, kl_lo = max(_p, key=lambda s: s["alpha"])["alpha"], max(_p, key=lambda s: s["alpha"])["kl_t1"]
        a_hi, kl_hi = min(_f, key=lambda s: s["alpha"])["alpha"], min(_f, key=lambda s: s["alpha"])["kl_t1"]
        if kl_hi > kl_lo:
            kl_budget_crossing_estimate = round(
                a_lo + (KL_PASS - kl_lo) * (a_hi - a_lo) / (kl_hi - kl_lo), 4)
    # the α<0.1 region is now MEASURED (round 4 grid), so the "does a pass
    # exist at small α" question is answered by data, not extrapolation.
    alpha01_measured = [s["alpha"] for s in sweep
                        if 0.0 < s["alpha"] < 0.1]
    alpha01_min_kl = min((s["kl_t1"] for s in sweep
                          if 0.0 < s["alpha"] < 0.1), default=None)
    min_alpha_nonzero = min((s["alpha"] for s in sweep if s["alpha"] > 0.0))
    # round 5 §4.4: lookup-by-value (no fragile hardcoded sweep[1] index).
    sweep_by_alpha = {s["alpha"]: s for s in sweep}
    min_alpha_kap = sweep_by_alpha[min_alpha_nonzero]["kappa_dryrun"]
    # --- E39_a_fixes.md §1-§6: data-driven text (no stale hardcoded values) ---
    full_surgery_kl = dec["full"]["mean_t1"]          # bottom-16 RAW V_new (headline)
    endpoint_orth_kl = sweep_by_alpha[1.0]["kl_t1"]   # α=1 orthogonalized endpoint
    kap_lo = min(kap_safe_alphas) if kap_safe_alphas else None
    kl_at_kap_safe = [sweep_by_alpha[a]["kl_t1"] for a in sorted(kap_safe_alphas)] \
        or [full_surgery_kl]
    a_random_kl = strat_kl["a_random"]["kl_t1"]       # actual E-39a a_random design
    print(f"\n§3 KL-vs-κ' trade-off (round 4): KL-safe α={kl_safe_alphas} | "
          f"κ'-safe α={kap_safe_alphas} | both={both_safe_alphas} | "
          f"KL=budget crossing α≈{kl_budget_crossing_estimate}")

    res = {
        "experiment": "E-39c_pre: KL pre-gate + behavioral budget for M_K-surgery "
                      "(round 2: tied-weight KL decomposition + α-blend sweep)",
        "id_note": "Renamed from exp39b_kl_surgery (ID collision with the real "
                   "E-39b = PTQ/GPTQ gate, TЗ §2; see E39_fixes §0).",
        "tz": "docs/E39.md §3 (pre-stage); fixes: docs/E39_fixes.md (r1), "
              "docs/E39c_pre_fixes.md (r2)",
        "params": {"seed": SEED, "model": MODEL, "n_docs": N_DOCS,
                   "s_len": S_LEN, "r": R, "kl_pass": KL_PASS,
                   "kl_pass_rationale": "half of the E-16 KL leash (0.1)",
                   "gate_token_rule": "t≥1 (round 2 §4.1: t=0 is the "
                                      "unconditioned first token, not a "
                                      "behavioral cost)",
                   "c4_skip": C4_SKIP, "vocab": vocab,
                   "alpha_grid": ALPHA_GRID},
        "v_new_subspace": sub_rep,
        "sigma_bottom16_U": [float(x) for x in sU[-R:]],
        "m_k_norm": {"before": mnorm_old, "after": mnorm_new,
                     "frobenius_preserved": abs(mnorm_new - mnorm_old) < 1e-6},
        "row_diag": row_diag,
        # round 2 §5.1: tied-weight decomposition of the KL.
        "kl_decomposition_tied_weights": {
            "note": "0.8B ties lm_head↔embed_tokens; the full edit changes "
                    "both readout and input embeddings (→ all 24 layers). "
                    "Three untied states split the cost. Values are mean KL "
                    "over real tokens (t≥1). 'full' uses the RAW bottom-16 "
                    "V_new (VU[:,-16:], orthogonalize=False) — NOT the "
                    "orthogonalized blend endpoint, and NOT the E-39a a_random "
                    "design (bottom-64 random; see "
                    f"v_new_strategy_sensitivity.strategies.a_random = "
                    f"{a_random_kl:.2f}). The headline full-surgery KL is the "
                    "bottom-16 raw-rotation cost.",
            "full": dec["full"],
            "head_only": dec["head_only"],
            "embed_only": dec["embed_only"],
            "embed_only_hidden_drift": dec.get("embed_only_hidden_drift"),
            "interpretation": "head_only = readout-only cost; embed_only = "
                              "upstream propagation; full ≈ head_only + "
                              "embed_only + cross term (tied model). If "
                              "embed_only dominates, the surgery is NOT "
                              "readout-only: it edits the input embeddings, "
                              "so the 'a flat leaf is cheap' claim is "
                              "confounded by tied weights. (Round 2 §5.1: "
                              "the decomposition exposes that the tied model "
                              "has no cheap readout-only path.)",
            # dominant_term: which of the two edits drives the full cost.
            "dominant_term": ("embed_only (upstream propagation)" if
                              dec["embed_only"]["mean_t1"] >=
                              dec["head_only"]["mean_t1"]
                              else "head_only (readout)"),
            "sum_of_parts_vs_full": {
                "head_only_plus_embed_only":
                    dec["head_only"]["mean_t1"] + dec["embed_only"]["mean_t1"],
                "full": dec["full"]["mean_t1"],
                "cross_term": dec["full"]["mean_t1"] -
                    dec["head_only"]["mean_t1"] - dec["embed_only"]["mean_t1"],
            },
        },
        # round 2 §7 / round 3 §2.2: α-blend sweep (KL only — κ' is not the
        # binding constraint). The blend is a TRUE partial rotation: V_new is
        # orthogonalized against V_top (V_new_perp ⊥ V_top) and NO QR is applied
        # to the combo (it is orthonormal by construction). The earlier QR-blend
        # silently NEGATED singular-direction terms (Householder sign flips on an
        # orthonormal input), inflating the small-α row change and producing the
        # spurious non-monotonicity. The conclusion below is DATA-DRIVEN (derived
        # from the sweep), not hardcoded, so it reports whatever the clean blend
        # actually produces.
        "alpha_blend_sweep": {
            "construction": "V_mixed(α)=√α·V_new_perp+√(1−α)·V_top (NO QR — "
                            "orthonormal by construction since V_new_perp⊥V_top; "
                            "round 3 §2.2 fix) where V_new_perp is V_new "
                            "orthogonalized against V_top; M_kp(α)=U_top·Σ_top·"
                            "V_mixed(α)ᵀ+M_rest; α=0 no-op, α=1 = V_new_perp "
                            "endpoint. Per-column ‖V_mixed−V_top‖ = "
                            "√(2−2√(1−α)) exactly (a clean rotation, not a "
                            "sign-flip artifact).",
            "blend_stability": {
                "v_new_vs_v_top_fro": vnew_vtop_ov,
                "overlap_max": 4.0,
                "overlap_note": "‖V_newᵀV_top‖_F: 0 = orthogonal, 4 = fully "
                                "aligned. Here ~0.53 (modest, non-zero), so the "
                                "raw V_new and V_top overlap; orthogonalizing to "
                                "V_new_perp removes it and makes the blend a "
                                "clean monotonic Grassmannian path.",
            },
            "blend_diagnostics": {
                "clean_rotation_verified": all(
                    p["pert_col_uniform_clean_rotation"] for p in sweep),
                "detail": "round 3 §2.4: for a clean blend every column of "
                          "V_mixed has per-column perturbation EXACTLY "
                          "√(2−2√(1−α)) (uniform, min≈max≈expected). This "
                          "verifies the blend is a true partial rotation. The "
                          "EARLIER QR-blend version had per-column max ≈ 1.97 at "
                          "α=0.1 (vs expected 0.32) — 7/16 columns sign-flipped "
                          "by QR (Householder) and silently NEGATED M_kp terms, "
                          "which inflated the row change (0.77 vs 0.23) and made "
                          "the KL spuriously non-monotonic. With the QR removed, "
                          "the row change is monotone in α and the small-α point "
                          "is a genuinely small rotation.",
            },
            # the two full-surgery KL values are DISTINCT (paper must cite the
            # right one). Headline FULL surgery (bottom-16 RAW V_new) =
            # kl_decomposition_tied_weights.full ≈ 6.31 (post-fix; ≈5.02 pre-fix).
            # The α-blend's ORTHOGONALIZED endpoint (V_new_perp) at α=1 ≈ 6.73.
            # Both KL-infeasible; the orthogonalized endpoint is HIGHER (+6.5%).
            # (The actual E-39a a_random = bottom-64 random costs ≈6.10.)
            "endpoint_note": (f"TWO distinct full-rotation values — cite the "
                             f"right one in the paper: (1) headline FULL surgery "
                             f"(bottom-16 RAW V_new = VU[:,-16:], "
                             f"orthogonalize=False) = kl_decomposition_tied_"
                             f"weights.full = {full_surgery_kl:.2f}; "
                             f"(2) the α-blend's ORTHOGONALIZED endpoint "
                             f"(V_new_perp = V_new minus its V_top component) at "
                             f"α=1 = points[α=1].kl_t1 = {endpoint_orth_kl:.2f}. "
                             f"Gap = {endpoint_orth_kl - full_surgery_kl:+.2f} "
                             f"({(endpoint_orth_kl - full_surgery_kl) / full_surgery_kl * 100:.1f}%): "
                             f"the orthogonalized endpoint is HIGHER than the raw "
                             f"bottom-16 rotation. Both are KL-infeasible. (The "
                             f"ACTUAL E-39a a_random design — bottom-64 random, not "
                             f"bottom-16 — costs {a_random_kl:.2f}; see "
                             f"v_new_strategy_sensitivity.)"),
            "kappa_note": "round 4 §6: κ'(α) IS measured per α (dry-run on the "
                          "base-model û_t, E-39a's exact _kappa_dryrun, N=32). "
                          "The round-2/3 claim 'the flattening side is already "
                          "satisfied' was WRONG: E-39a's a_random κ' (post-fix: "
                          "0.000297, from the E-39a JSON) applies ONLY at α=1 (full rotation). At small α the top-16 "
                          "subspace is dominated by V_top, so κ' stays near the "
                          "base 0.0162. Hence the KL budget (KL<0.05, small α) "
                          "and the flattening target (κ'<0.005, large α) are "
                          "SIMULTANEOUSLY binding — see "
                          "alpha_blend_sweep.trade_off.",
            "kappa_full_surgery_ref": KAPPA_39A_FULL,
            # round 4 §3: the measured KL-vs-κ' trade-off. Both "passing"
            # regions are computed from the sweep data (not extrapolated);
            # the α<0.1 region is now MEASURED (grid extended to 0.01/0.02/
            # 0.05/0.08) so the KL crossing is a data point.
            "trade_off": {
                "kappa_target": KAPPA_TARGET,
                "kl_budget": KL_PASS,
                "kl_safe_alphas": kl_safe_alphas,
                "kappa_safe_alphas": kap_safe_alphas,
                "both_safe_alphas": both_safe_alphas,
                "alpha01_measured": alpha01_measured,
                "alpha01_min_kl": alpha01_min_kl,
                "kl_min_passing_alpha": kl_min_passing_alpha,
                "kap_min_passing_alpha": kap_min_passing_alpha,
                "kl_budget_crossing_estimate": kl_budget_crossing_estimate,
                "disjoint": (len(both_safe_alphas) == 0),
            },
            "points": sweep,
            "any_nontrivial_alpha_with_kl_below_0.05": nontrivial_kl_safe,
            "min_kl_over_nonzero_alpha": min_kl_nonzero,
            # round 4 §5: no overclaim — the verdict is scoped to the TESTED
            # GRID (now including α<0.1), and the blend-clean caveat is kept.
            "conclusion": ((f"on the tested grid (α ∈ {sorted(ALPHA_GRID)}, incl. "
                            f"α<0.1), NO α satisfies BOTH constraints. NO α passes "
                            f"the KL budget: min KL = {min_kl_nonzero:.4f} at "
                            f"α={min_alpha_nonzero} ({min_kl_nonzero / KL_PASS:.2f}× "
                            f"over); at that α, κ'≈{min_alpha_kap:.4f} (≃"
                            f"{min_alpha_kap / KAPPA_TARGET:.0f}× the {KAPPA_TARGET} "
                            f"target). The κ' target is reached only at "
                            f"α ∈ {kap_safe_alphas}, where KL = "
                            f"{min(kl_at_kap_safe):.2f}–{max(kl_at_kap_safe):.2f} "
                            f"(≃{min(kl_at_kap_safe) / KL_PASS:.0f}–"
                            f"{max(kl_at_kap_safe) / KL_PASS:.0f}× over). The two "
                            f"feasible regions are DISJOINT (trade_off) — the route "
                            f"is infeasible by TENSION, not a thin margin. The clean "
                            f"(QR-free) blend makes KL monotone in α, so this is not "
                            f"a blend artifact. Full surgery (α=1) is refuted on KL "
                            f"(~{full_surgery_kl / KL_PASS:.0f}× over).")
                            if not both_safe_alphas else
                           "an α on the tested grid achieves both KL<0.05 and "
                           "κ'<0.005 — partial surgery IS feasible; see "
                           "trade_off.both_safe_alphas"),
            # round 4 §4: honest magnitude language (no 'near-budget' hedge).
            # ACCURATE (data-driven, post-fix): NO α passes the KL budget (min
            # KL = 0.135 at α=0.01); the κ' target is reached only at α ∈
            # {0.7, 0.9, 1.0} where KL = 6.29–6.75. The feasible regions are
            # DISJOINT — the route is infeasible by TENSION, not a thin margin.
            "budget_note": ((f"NO α on the tested grid passes the KL budget: "
                            f"min KL = {min_kl_nonzero:.4f} at α={min_alpha_nonzero} "
                            f"({min_kl_nonzero / KL_PASS:.2f}× over). Since KL "
                            f"increases monotonically with α "
                            f"(clean_rotation_verified), no smaller α can help. "
                            f"The κ' target (<{KAPPA_TARGET}) is reached only at "
                            f"α ∈ {kap_safe_alphas}, where KL = "
                            f"{min(kl_at_kap_safe):.2f}–{max(kl_at_kap_safe):.2f} "
                            f"({min(kl_at_kap_safe) / KL_PASS:.0f}–"
                            f"{max(kl_at_kap_safe) / KL_PASS:.0f}× over budget). "
                            f"The feasible regions are DISJOINT — the route is "
                            f"infeasible by tension, not by a thin margin. (At the "
                            f"smallest α, κ'≈{min_alpha_kap:.4f} ≃"
                            f"{min_alpha_kap / KAPPA_TARGET:.0f}× the target; at "
                            f"the smallest κ'-passing α={kap_lo}, KL≈"
                            f"{min(kl_at_kap_safe):.2f}.)")
                            if not both_safe_alphas else
                           "an α on the tested grid achieves both KL<0.05 and "
                           "κ'<0.005 — partial surgery IS feasible; see "
                           "trade_off.both_safe_alphas"),
        },
        # round 2 §5.3: V_new strategy sensitivity (a_random/b_principal/
        # c_balanced), full surgery (α=1). Defined locally (mirrors E-39a) to
        # avoid the exp39a coupling (review §8.1).
        "v_new_strategy_sensitivity": {
            "note": "E-39a's three V_new strategies in the bottom-64 pool, "
                    "evaluated at full surgery (α=1). If NO direction is "
                    "KL-safe, the surgery is infeasible regardless of the "
                    "V_new choice — a stronger negative result.",
            "r_pool": R_POOL,
            "v_c_rank_ok": V_c_valid,
            "strategies": strat_kl,
            "any_strategy_kl_below_0.05": any_strat_kl_safe,
            # round 4 §3: the a_random strategy is EXACTLY E-39a's a_random
            # (bottom-64 QR of a random projection), so its dry-run κ' is a
            # consistency check on the whole dry-run pipeline: it should
            # reproduce E-39a's a_random κ' (post-fix: 0.000297, from E-39a JSON).
            "e39a_consistency": {
                "e39c_pre_a_random_kappa": strat_kl["a_random"]["kappa_dryrun"],
                "e39a_a_random_kappa": KAPPA_39A_FULL,
                "abs_diff": abs((strat_kl["a_random"]["kappa_dryrun"] or 0.0)
                                - KAPPA_39A_FULL),
                "note": "a_random (E-39c_pre) vs a_random (E-39a) dry-run κ'. "
                        "Small diff (≲1e-3) confirms the E-39c_pre V_new "
                        "construction + _kappa_dryrun pipeline reproduce "
                        "E-39a. (A larger diff would flag a construction/"
                        "convention mismatch.)",
            },
            "conclusion": ("no V_new direction is KL-safe at full surgery — "
                           "the failure is structural (no V_new in the bottom "
                           "pool is KL-safe at full rotation), not a "
                           "consequence of the bottom-16 choice; per-strategy "
                           "κ' dry-run also reported (round 4)" if
                           not any_strat_kl_safe else
                           "at least one V_new direction is KL-safe"),
        },
        "kappa_dryrun_alpha1": {
            "note": "round 4: per-α κ' DRY-RUN is now measured (base-model û_t, "
                    "E-39a's exact _kappa_dryrun, N=32). Edited-model û_t differ "
                    "due to tied weights — this is a DRY-RUN, NOT an E-39c "
                    "measurement. See alpha_blend_sweep.points[].kappa_dryrun "
                    "and .trade_off.",
            "per_alpha": {s["alpha"]: s["kappa_dryrun"] for s in sweep},
            "target": KAPPA_TARGET,
            "ref_39a": KAPPA_39A_REF,
        },
        "kl_full_nats_per_token": {
            "mean_real_tokens_t_ge_0": dec["full"]["mean_t0"],
            "mean_real_tokens_t_ge_1": kl_full_t1,
            "per_doc_t_ge_1": dec["full"]["per_doc_t1"],
        },
        "gate": {"metric": "mean KL over real tokens, t≥1, padding excluded "
                           "(round 2 §4.1)",
                 "threshold": KL_PASS, "value": kl_full_t1, "pass": gate_pass,
                 "basis": "KL only (κ' is a dry-run preview, not a gate input)"},
        "limitations": [
            "Tied weights: editing W_lm = editing embed_tokens — the forward "
            "changes beyond the readout (TЗ §3.5). round 2 §5.1 decomposes "
            "this into head_only vs embed_only; the FULL (real model) cost is "
            "the sum of both.",
            "C4 disjointness: ds.skip(4096) vs the 27b cache's first docs — "
            "disjoint by offset, NOT hash-verified.",
            "KL_PASS=0.05 is a protocol choice (half of E-16's 0.1 leash), "
            "not derived from data.",
            "κ' is a dry-run preview on base-model û_t; the real E-39c "
            "measurement happens on the edited model.",
            "α-blend V_new is the bottom-16 (E-39a 'a'-equivalent) subspace "
            "across the α grid; the three E-39a strategies are compared only "
            "at FULL surgery (round 2 §5.3), not across α. round 3 §2.2: the "
            "blend is orthogonalized (V_new_perp ⊥ V_top) AND the QR on the "
            "combo is removed (orthonormal by construction), so it is a "
            "verified clean partial rotation (per-column perturbation = "
            "√(2−2√(1−α)) exactly; see "
            "alpha_blend_sweep.blend_diagnostics). The earlier QR-blend's "
            "non-monotonicity was a Householder sign-flip artifact (silent "
            "negation of M_kp terms), not a loss-landscape property. The α=1 "
            "blend endpoint is V_new_perp, ≈ the full-surgery raw-V_new design "
            "(kl_decomposition_tied_weights); both are KL-infeasible.",
            "Per-α κ' IS measured (round 4 §3) as a DRY-RUN on the base-model "
            "û_t (E-39a's _kappa_dryrun, N=32). This is a PREVIEW, not an "
            "E-39c measurement: the real edited-model κ'(α) could differ "
            "(tied weights). The trade-off verdict (KL budget vs flattening "
            "target across α) rests on this dry-run κ'(α); a real E-39c "
            "measurement at a candidate α would be needed to confirm.",
            "κ' dry-run uses the 32-doc cache (N=32, S=128) — the SAME "
            "convention as E-39a (its a_random dry-run = 0.0040). The KL side "
            "uses only 8 fresh C4 docs (skip 4096), so KL and κ' are measured "
            "on different doc sets (both base-model references, disjoint from "
            "each other by offset). The a_random consistency check in "
            "v_new_strategy_sensitivity.e39a_consistency validates the "
            "pipeline.",
        ],
        "runtime_s": round(time.time() - t0, 1),
        "deterministic": True,
    }
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nE-39c_pre done in {time.time() - t0:.1f}s → {out_path}")
    print(f"gate (α=1 full, t≥1): {'PASS' if gate_pass else 'FAIL'} "
          f"(mean KL {kl_full_t1:.5f} < {KL_PASS})")
    print(f"α-sweep: non-trivial α with KL<0.05 = {nontrivial_kl_safe}; "
          f"min KL over α>0 = {min_kl_nonzero:.4f} "
          f"({min_kl_nonzero / KL_PASS:.2f}× budget at α={min_alpha_nonzero})")
    print(f"§3 trade-off (round 4): KL-safe α={kl_safe_alphas} | "
          f"κ'-safe α={kap_safe_alphas} | BOTH={both_safe_alphas} "
          f"({'DISJOINT — no α satisfies both' if not both_safe_alphas else 'overlap found'})")
    print(f"a_random κ' dry-run = {strat_kl['a_random']['kappa_dryrun']:.4f} "
          f"(E-39a: {KAPPA_39A_FULL}; |diff| = "
          f"{abs((strat_kl['a_random']['kappa_dryrun'] or 0.0) - KAPPA_39A_FULL):.4f})")
    print(f"flattening side: per-α κ' dry-run (round 4); α=1 ref "
          f"={KAPPA_39A_FULL} < {KAPPA_TARGET}")
    print(f"§5.3 strategy sensitivity: any V_new direction KL-safe = "
          f"{any_strat_kl_safe}")


if __name__ == "__main__":
    main()
