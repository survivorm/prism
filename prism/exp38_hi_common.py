"""Shared math for E-38h/i: L23 readout-Jacobian factorization.

Jacobian of the FINAL LAYER readout at a response token t, input = L23
pre-final-norm hidden h_t (V15 contract), output = next-token logit scores:

    J(t) = (sqrt(D)/r_t) * M_K (I - u u^T)          (E-35 lemma [M-1])

with r_t = ||h_t||_2,  u = h_t/r_t,  M_K = W_lm[U128] * gamma (gamma = 1+w,
RMSNorm weight).  This is an EXACT identity for the last-layer readout
Jacobian for ANY W_lm:

    J_norm(t) = (sqrt(D)/r_t) * diag(gamma) (I - u u^T),   J(t) = W_lm @ J_norm(t)

E-38i uses this: a READOUT-only edit changes W (and hence J(s), M_K(s)) but
leaves h_t, r_t, u, J_norm, sqrt(D)/r_t invariant  =>  the identity holds
for every edited W; the residual is structurally zero and the test measures
the numeric floor + the leaf drift (J rows, M_K rows).

E-38h uses the two-channel (left = 256-token space / right = D=1024 hidden
space) SVD split of Delta_W across edit steps.

Data (local, deterministic — no model forward):
  - snapshots: results/exp16_deltas/exp16_dW_snapshots.pt
               {'W_orig' [256,1024] fp16, 'snapshots' {step: [256,1024]},
                'token_set' [256]}  (100 steps, 10..1000)
  - U128 + H/JR: kaggle_kernel/results/exp27b_jcache_L23.pt
  - model:    Qwen3.5-0.8B safetensors — tied embed_tokens rows (== W_lm)
              and model.language_model.norm.weight (final RMSNorm gamma=1+w)
"""
from __future__ import annotations

import os
import torch

# ---------- constants / paths ----------
U128_PATH = "kaggle_kernel/results/exp27b_jcache_L23.pt"
SNAPSHOTS = "results/exp16_deltas/exp16_dW_snapshots.pt"
EPS = 1e-6  # Qwen3.5-0.8B config rms_norm_eps (verified)
D = 1024

# Pre-registered subsample of the 100-step grid (10..1000), bracketing the
# E-16 knee (knee_step=84, knee_dose≈100, results/exp16_lm_head_knee_info.pt).
K_STEPS = [10, 20, 40, 80, 90, 100, 200, 400, 800, 1000]

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _p(rel):
    return os.path.join(REPO, rel)


# ---------- data loading ----------
def load_u128():
    """(U128 long tensor, L23 dense cache dict with H[(dom,si,t)] etc.)."""
    ck = torch.load(_p(U128_PATH), map_location="cpu", weights_only=False)
    U = torch.tensor(ck["U128"], dtype=torch.long)
    return U, ck


def load_snapshots():
    d = torch.load(_p(SNAPSHOTS), map_location="cpu", weights_only=True)
    snaps = {int(k): v.to(torch.float64) for k, v in d["snapshots"].items()}
    return snaps, d


def load_model_tensors():
    """(embed, gamma): tied embed_tokens weight and final RMSNorm weight
    (1+w). Verified: embed[U128] reproduces cached J23 via the identity."""
    from safetensors import safe_open
    import glob

    root = os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots"
    )
    f = sorted(glob.glob(root + "/*/model*.safetensors"))[0]
    with safe_open(f, framework="pt") as sf:
        emb = sf.get_tensor("model.language_model.embed_tokens.weight")
        gamma = sf.get_tensor("model.language_model.norm.weight")
    return emb.to(torch.float64), gamma.to(torch.float64)


# ---------- factorization ----------
def j_raw(W, h, gamma):
    """Exact final-layer readout Jacobian for ONE position:
    h [D] -> J [n_tokens, D].  J(t) = (sqrt(D)/r) * M_K (I - u u^T),
    M_K = W*gamma, u = h/||h||.  VERIFIED: cos vs cached exp27b JR (fp32)
    = 1.0000 (E-35)."""
    r = h.norm()
    u = h / r
    M = W * gamma[None, :]
    c = float(D) ** 0.5
    return (c / r) * (M - torch.outer(M @ u, u))


def factor_residual(W, h, gamma):
    """Independent check: dense  W @ J_norm(t)  vs closed form, ONE position.
    J_norm(t) = (sqrt(D)/r) * diag(gamma)(I - u u^T).
    Residual is structural zero for ANY W; measures numeric floor only."""
    r = h.norm()
    u = h / r
    c = float(D) ** 0.5
    I = torch.eye(W.shape[1], dtype=W.dtype)
    P = I - torch.outer(u, u)
    # gamma scales OUTPUT directions (rows of J_norm): J_norm = (sqrt(D)/r) diag(gamma)(I-uu^T)
    Jnorm = (c / r) * (P * gamma[:, None])
    Jd = W @ Jnorm
    M = W * gamma[None, :]
    Jc = (c / r) * (M - torch.outer(M @ u, u))
    return float((Jd - Jc).norm() / Jc.norm())


# ---------- SVD / subspace helpers (float64) ----------
def top_r_subspace(A, r):
    """Column space of A's top-r right singular vectors (D x r, orthonormal)."""
    _, _, Vt = torch.linalg.svd(A, full_matrices=False)
    return Vt[:r].t().contiguous()


def top_r_subspaces(A, r):
    _, s, Vt = torch.linalg.svd(A, full_matrices=False)
    return Vt[:r].t().contiguous(), s[:r]


def overlap(Q1, Q2, r):
    """cos^2 of principal angles (mean) between two r-dim subspaces."""
    s = (Q1.t() @ Q2).abs().clamp(max=1.0)
    return float(s.pow(2).mean())


def procrustes_gap(Q1, Q2):
    """||Q1 - Q2 O||_F / r over orthogonal O (frame alignment)."""
    U, _, Vt = torch.linalg.svd(Q1.t() @ Q2, full_matrices=False)
    O = U @ Vt
    return float((Q1 - Q2 @ O).norm() / Q1.shape[1])
