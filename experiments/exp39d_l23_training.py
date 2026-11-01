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
  - No bridge (self-bridge degenerate); KL leash preserves behavior.
  - λ sweep {0.01,0.1,1,10}, 100 Adam steps, lr=1e-3, 3 seeds, P=256 pos/step.
  - Metrics every 10 steps: κ' (EXACT E-39a _kappa_dryrun, so κ_base=0.0162
    is comparable) + PPL (C4 held-out, skip 4096).

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
             base_k: torch.Tensor,
             lse_b: torch.Tensor,
             tok_ids: torch.Tensor) -> torch.Tensor:
    """KL(p_edit || p_base) over the full vocab, readout-only, via logsumexp.

    Derivation (b=base logits, e=edit logits; e_v=b_v for v∉K, e_k=H·W_rows_k
    for v∈K; lse = logsumexp):
        lse_e = log( exp(lse_b) − Σ_k exp(b_k) + Σ_k exp(e_k) )
        KL    = (lse_b − lse_e) + Σ_k exp(e_k − lse_e)(e_k − b_k)
    Verified against full-softmax KL in _selftest.

    H_sub: [P, D] frozen cached hidden states.
    W_rows: [K, D] trainable lm_head rows (token set). grad flows here.
    base_k: [P, K] base logits for the K tokens (cached, no grad).
    lse_b: [P] logsumexp of base logits over full vocab (cached, no grad).
    """
    e_k = H_sub @ W_rows.t()                      # [P, K] (grad via W_rows)
    b_k = base_k                                  # [P, K] (no grad)
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
    """Verify the logsumexp KL against a brute-force full-softmax KL."""
    torch.manual_seed(0)
    P, Dd, K, Vv = 32, 32, 8, 64
    H = torch.randn(P, Dd, device=device)
    Wlm = torch.randn(Vv, Dd, device=device)
    W_rows = torch.randn(K, Dd, device=device)
    toks = torch.arange(K, device=device)
    base_logits = H @ Wlm.t()                    # [P, V]
    lse_b = torch.logsumexp(base_logits, dim=-1)  # [P]
    base_k = base_logits[:, toks]                 # [P, K]
    # brute force
    edit_logits = base_logits.clone()
    edit_logits[:, toks] = H @ W_rows.t()
    pe = torch.softmax(edit_logits, dim=-1)
    pb = torch.softmax(base_logits, dim=-1)
    kl_ref = (pe * (torch.log(pe + 1e-30) - torch.log(pb + 1e-30))).sum(-1).mean()
    # logsumexp trick
    kl_ours = kl_leash(H, W_rows, base_k, lse_b, toks)
    ok = abs(float(kl_ref) - float(kl_ours)) < 1e-4
    print(f"[selftest] KL brute={float(kl_ref):.6f}  logsumexp={float(kl_ours):.6f}  "
          f"diff={abs(float(kl_ref)-float(kl_ours)):.2e}  {'OK' if ok else 'FAIL'}")
    return ok


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
    """
    u = H_sub / H_sub.norm(dim=-1, keepdim=True)              # [P, D]
    P = H_sub.shape[0]
    sum_norm_sq = torch.zeros((), device=M_K.device, requires_grad=True)
    SPP = torch.zeros(D, D, device=M_K.device, requires_grad=True)
    for i in range(0, P, mb):
        uh = u[i:i+mb]                                        # [m, D]
        Ju = torch.einsum("pd,rd->pr", uh, M_K)               # [m, K]
        J = M_K.unsqueeze(0) - Ju.unsqueeze(-1) * uh.unsqueeze(1)  # [m,K,D]
        JJt = torch.bmm(J, J.transpose(1, 2))                 # [m,K,K]
        w, U = torch.linalg.eigh(JJt)                          # w:[m,K] U:[m,K,K]
        s = w.flip(1)[:, -r:].sqrt()                           # [m,r]
        U16 = U.flip(1)[:, :, -r:]                             # [m,K,r]
        Vt = torch.bmm(J.transpose(1, 2), U16) / s[:, None, :]  # [m,D,r]
        Pt = torch.bmm(Vt, Vt.transpose(1, 2))                # [m,D,D]
        sum_norm_sq = sum_norm_sq + (Pt ** 2).sum()           # scalar (grad)
        SPP = SPP + Pt.sum(0)                                  # [D,D] (grad)
    return (sum_norm_sq - (SPP * SPP).sum() / P) / P


# ---------------------------------------------------------------------------
# C4 docs for PPL
# ---------------------------------------------------------------------------

def _c4_docs(n_docs: int, s_len: int, tok, skip: int = C4_SKIP):
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True).skip(skip)
    texts = [next(iter(ds))["text"] for _ in range(n_docs)]
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

    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
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
    ck = torch.load(jcache_path, map_location="cpu")
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
        arr = torch.stack([seq[t].float() for t in range(S)])
        arr = arr / arr.norm(dim=-1, keepdim=True)
        u64_docs.append(arr.double().cpu())
    u64_docs = torch.stack(u64_docs)                     # [N_docs, S, D]
    N_docs = u64_docs.shape[0]

    # ---- base logits cache (one-time): lse_b [N], base_k [N,K], sentinel mask
    print("computing base logits cache (lse_b, base_k, sentinel)...", flush=True)
    base_k_all = torch.empty(N_total, K, dtype=torch.float32, device=device)
    lse_b_all = torch.empty(N_total, dtype=torch.float32, device=device)
    raw_sum_all = torch.empty(N_total, dtype=torch.float32, device=device)
    B = 256
    for i in range(0, N_total, B):
        Hb = H_all[i:i+B]
        bl = Hb @ W_lm.t()                                # [B, V]
        lse_b_all[i:i+B] = torch.logsumexp(bl, dim=-1)
        base_k_all[i:i+B] = bl[:, tok_ids]
        raw_k = torch.exp(bl[:, tok_ids])                 # raw (unnormalised) mass
        raw_sum_all[i:i+B] = raw_k.sum(1)
        del bl
    # E-34 sentinel mask: raw P128 sum < SENTINEL_SUM → exclude
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
    # l_flat is ~1e3 on this data while l_kl is ~1e-1, so WITHOUT this the
    # spec λ-grid {0.01..10} would have flat dominate at every λ (KL leash
    # never binds). Normalising makes λ a meaningful relative weight.
    with torch.no_grad():
        l_flat_base = float(flatness_loss(
            (W_rows_base * g_base.to(device)[None, :]).detach(),
            H_all[keep_idx[:P_STEP]], R, MB))
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
                l_kl = kl_leash(H_sub, W_rows, base_k_all[idx], lse_b_all[idx],
                                tok_ids)
                loss = lam * l_flat + LAMBDA_KL * l_kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_([W_rows], 1.0)
                opt.step()

                # NaN/Inf guard: a diverged seed must not kill the whole
                # sweep — record the divergence point and stop this run.
                if not torch.isfinite(loss.detach()):
                    run["diverged_at_step"] = step
                    print(f"  λ={lam} s{seed} st{step}: loss non-finite — "
                          f"run diverged, stopping (round 7 §1 NaN-guard)",
                          flush=True)
                    break

                if step % METRIC_EVERY == 0 or step == N_STEPS:
                    M_Kd = (W_rows.detach() * g_base.to(device)[None, :]).double()
                    kap = _kappa_dryrun(M_Kd, u64_docs, N_docs, S, R)["kappa"]
                    with torch.no_grad():
                        old = model.lm_head.weight.data
                        model.lm_head.weight.data[tok_ids] = W_rows.detach()
                        ppl = _ppl(model, ppl_ids, ppl_attn)
                        model.lm_head.weight.data = old
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
                    "kl_mean": run["history"][-1]["l_kl"] if run["history"] else None,
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
                old = model.lm_head.weight.data
                model.lm_head.weight.data[tok_ids] = W_rows.detach()
                ppl_f = _ppl(model, ppl_ids, ppl_attn)
                model.lm_head.weight.data = old
            dw = (W_rows.detach() - W_lm[tok_ids].to(device)).norm() / \
                 W_lm[tok_ids].to(device).norm()
            run["final"] = {
                "kappa": kap_f, "ppl": ppl_f,
                "ppl_delta_pct": (ppl_f / ppl_base - 1) * 100,
                "kappa_improvement": kappa_base / kap_f if kap_f > 0 else float("inf"),
                "delta_W_rel": float(dw.item()),
                "kl_mean": run["history"][-1]["l_kl"],
                "elapsed_s": time.time() - t0,
            }
            kap_ok = kap_f < KAPPA_TARGET
            ppl_ok = ppl_f <= ppl_base * 1.02
            kl_ok = run["history"][-1]["l_kl"] < KL_PASS
            run["verdict"] = ("SUCCESS" if (kap_ok and ppl_ok)
                              else "PARTIAL" if kap_ok else "FAIL")
            run["verdict_detail"] = {
                "kappa_target_met": kap_ok, "ppl_within_2pct": ppl_ok,
                "kl_within_budget": kl_ok,
            }
            results["runs"].append(run)
            print(f"λ={lam} s{seed}: κ'={kap_f:.6f} PPL={ppl_f:.4f} "
                  f"({run['final']['ppl_delta_pct']:+.2f}%) "
                  f"KL={run['history'][-1]['l_kl']:.4f} "
                  f"verdict={run['verdict']} ({run['final']['elapsed_s']:.0f}s)",
                  flush=True)

    # ---- summary ----
    valid = [r for r in results["runs"]
             if r["final"]["kappa"] is not None and r["final"]["kappa"] > 0]
    n_div = sum(1 for r in results["runs"] if r["verdict"] == "DIVERGED")
    n_success = sum(1 for r in results["runs"] if r["verdict"] == "SUCCESS")
    n_partial = sum(1 for r in results["runs"] if r["verdict"] == "PARTIAL")
    n_fail = sum(1 for r in results["runs"] if r["verdict"] == "FAIL")
    best = min(valid, key=lambda r: r["final"]["kappa"]) if valid else None
    if best is not None:
        summary = {
            "best_kappa": best["final"]["kappa"],
            "best_lambda": best["lambda"], "best_seed": best["seed"],
            "best_ppl": best["final"]["ppl"],
            "best_ppl_delta_pct": best["final"]["ppl_delta_pct"],
            "best_kappa_improvement": best["final"]["kappa_improvement"],
            "n_success": n_success, "n_partial": n_partial,
            "n_fail": n_fail, "n_diverged": n_div,
            "total_time_s": time.time() - t_start,
            "interpretation": (
                f"Best κ'={best['final']['kappa']:.6f} "
                f"(base={kappa_base:.6f}, "
                f"improvement={best['final']['kappa_improvement']:.1f}×) "
                f"at λ={best['lambda']}, seed={best['seed']}. "
                f"PPL={best['final']['ppl']:.4f} "
                f"({best['final']['ppl_delta_pct']:+.2f}%). "
                f"E-39c (algebraic) was REFUTED on KL (6.31 nats = 126× budget). "
                f"If E-39d reaches κ'<{KAPPA_TARGET} with KL<{KL_PASS}, "
                f"training beats pure algebra on the KL trade-off."
            ),
        }
    else:
        summary = {
            "best_kappa": None, "best_lambda": None, "best_seed": None,
            "best_ppl": None, "best_ppl_delta_pct": None,
            "best_kappa_improvement": None,
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
