"""E-39b: PTQ-пертурбация листа — GPTQ 4-bit, что ломает квантование (docs/E39.md §2).

Qwen3.5-0.8B: lm_head.weight → M_K → J_t, û_t заморожен (H из jcache 27b).
Уровни: fp16 baseline (неквантованный W_lm), 8-bit RTN, 4-bit GPTQ.
Метрики на L in {6,12,18,19,22,23}: kappa(L) r=16 (E-38d), beta/sigma_1(M_K)
(E-38e), cos(u_g, 1/sqrt(K)) (E-37), J_rel_drift (E-38i-аналог).
PPL на C4 val held-out.

C4-сплиты (disjoint, §2.2): GPTQ Hessian — train skip 8192, 128 док x 128 tok;
PPL — train skip 16384, 32 док x 128 tok (val split недоступен в streaming).

Вывод: results/exp39b_ptq_perturbation.json
Детерминировано: GPTQ без RNG (damp=0.01, desc_act, group=128, канонический
алгоритм IST-DASLab/gptq). J — fp64, SVD — fp32 (T4), H_012 — fp64.
"""
import json
import os
import sys
import time

import numpy as np
import torch

try:
    _HERE = os.path.abspath(os.path.dirname(__file__))
except NameError:  # inlined into a notebook cell
    _HERE = os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))

from _e38_cache import LAYERS, D, S, load_layer  # noqa: E402
from prism.interventions import _resolve_lm_head_weight  # noqa: E402
from prism.json_utils import dump as json_dump  # noqa: E402
from prism.jacobian import _lm_module  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
SEED = 0
R = 16
GROUP = 128
DAMP = 0.01
KAP_BASE = 0.018
C4_CAL_SKIP = 8192
C4_PPL_SKIP = 16384
C4_CAL_DOCS = 128
C4_PPL_DOCS = 32
UNIF = 1.0 / np.sqrt(128)


def _c4_docs(n_docs, s_len, tokenizer, skip):
    from datasets import load_dataset
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.skip(skip).take(n_docs)
    out = []
    for row in ds:
        toks = tokenizer(row["text"])["input_ids"][:s_len]
        if len(toks) == s_len:
            out.append(toks)
        if len(out) >= n_docs:
            break
    return np.array(out, dtype=np.int64)


def gptq_quantize(W, H, bits=4, group_size=128, damp=0.01, actorder=True):
    """Канонический GPTQ (port IST-DASLab/gptq fasterquant): H = (2/n) X X^T
    (fp32, X — реальные входы lm_head), damp = 0.01*mean(diag), desc_act,
    per-group scale/zero из CURRENT (error-compensated) значений,
    err1 = (w-q)/d, in-block W1[:,i:] -= err1*Hinv1[i,i:], cross-block
    W[:,i2:] -= Err1*Hinv[i1:i2,i2:], unpermute → dequantized fp32.
    Без RNG — детерминировано.
    """
    dev = W.device
    W = W.float().clone()
    n, m = W.shape
    H = H.to(dev=dev, dtype=torch.float32)
    Hinv = torch.cholesky(H, upper=False).inverse()
    Hinv = Hinv.triangular_(upper=True).transpose(0, 1)
    Hinv = torch.cholesky(Hinv, upper=True)
    perm = torch.argsort(torch.diagonal(H).cpu(), dim=0, descending=actorder).to(dev)
    invperm = torch.argsort(perm)
    H = H[perm][:, perm]
    W = W[:, perm]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    maxq = 2 ** bits - 1
    zero = (maxq + 1) / 2
    Err1 = torch.zeros(m, m, device=dev)
    for i1 in range(0, m, group_size):
        i2 = min(i1 + group_size, m)
        Hinv1 = Hinv[i1:i2, i1:i2]
        W1 = W[:, i1:i2].clone()
        for i in range(i1, i2):
            w = W1[:, i - i1]
            d = Hinv1[i - i1, i - i1]
            if i == i1:
                group = W1[:, i - i1:i2 - i1]
                xmax = group.max(dim=1, keepdim=True).values
                xmin = group.min(dim=1, keepdim=True).values
                scale = (xmax - xmin) / maxq
                scale = torch.where(scale == 0, 1.0, scale)
            q = torch.clamp(torch.round(w / scale + zero), 0, maxq)
            W1[:, i - i1] = scale * (q - zero)
            err1 = (w - W1[:, i - i1]) / d
            W1[:, i - i1 + 1:] -= err1.unsqueeze(1) * Hinv1[i - i1, i - i1 + 1:].unsqueeze(0)
            W[:, i + 1:] -= err1.unsqueeze(1) * Hinv[i + 1:i2, i].unsqueeze(1)
        W[:, i1:i2] = W1
        Err1[:, i1:i2] = torch.zeros(m, i2 - i1, device=dev)
    W = W[:, invperm]
    return W.cpu()


def rtn_int8(W):
    """8-bit RTN: per-group (128) min-max симметричный scale, round-to-nearest,
    без Hessian/компенсации (по ТЗ — «стандартное 8-bit», контраст к GPTQ)."""
    n, m = W.shape
    Wf = W.float()
    groups = m // GROUP
    Wg = Wf.view(n, groups, GROUP)
    s = Wg.abs().amax(dim=2, keepdim=True).clamp(min=1e-9) / 127.0
    Q = (Wg / s).round()
    return (Q * s).view(n, m).cpu()


def _recon_J(M_K, H, U, c):
    """J[n,k,d] = c[n]*(M_K[k,d] - u[n,d]*(M_K u)[k,n]) в fp64, док-чанки."""
    n = H.shape[0]
    outs = []
    for b0 in range(0, n, 8):
        b = min(b0 + 8, n)
        u = torch.from_numpy(U[b0:b]).double()
        c = torch.from_numpy(c[b0:b]).double()
        mk_u = M_K @ u.T            # [K, cb]
        j = (M_K[None] - (mk_u.T * u)).reshape(b - b0, 128, D) * c[:, None, None]
        outs.append(j.cpu())
    return torch.cat(outs)


def _V16(J):
    """V-базисы top-16: [n_docs*S, 1024, 16] fp32 (SVD fp32, T4)."""
    Jf = J.float()
    ndocs, S_, K_, D_ = Jf.shape
    Vh = torch.linalg.svd(Jf.view(-1, K_, D_), full_matrices=False)[2]
    return Vh.view(ndocs, S_, 16, D_)


def _kappa(V):
    """H_012 = R_01*R_12*R_20 по triples (t,t+1,t+2), kappa = ||H_012 - I||_F
    (конвенция E-38d: M = V_b^T V_a, R = Qt^T P^T, flat f0 = i*S + t)."""
    nd, S_ = V.shape[0], V.shape[1]
    M = torch.bmm(V[:, 1:].transpose(1, 2), V[:, :-1]).double()
    P, s, Qt = torch.linalg.svd(M, full_matrices=False)
    R = torch.bmm(Qt.transpose(1, 2), P.transpose(1, 2))   # [nd, S-1, 16, 16]
    H012 = torch.bmm(R[:, :-1], R[:, 1:])
    I = torch.eye(16, device=H012.device)
    kap = torch.norm(H012 - I[None, None], p=2, dim=(-2, -1))
    f0 = np.arange(nd) * S_
    med = float(np.median(kap.view(-1).cpu().numpy()))
    return {"kappa_median": med, "kappa_vs_base_pct": 100.0 * (med - KAP_BASE) / KAP_BASE}


def _base_ratio(M_K):
    """median beta_t / sigma_1(M_K) по всем (doc,t); baseline ~0.9 ± 0.15 (E-38e)."""
    n = M_K.shape[0]
    w = M_K.T @ np.full(128, UNIF, dtype=np.float64)
    sig1 = float(np.linalg.svd(M_K, full_matrices=False)[1][0])
    num = np.empty(n, dtype=np.float64)
    for b0 in range(0, n, 8):
        b = min(b0 + 8, n)
        u = M_K[b0:b]
        alpha = u @ w
        proj = alpha[:, None] * u
        num[b0:b] = np.linalg.norm(w[None, :] - proj, axis=1)
    return float(np.median(num) / sig1)


def _cos_ug(M_K, U, c, uhat):
    """cos(u_g, 1/sqrt(K)), u_g = J^T 1 / ||J^T 1||; J^T 1 = c*(sqrt(K) M_K^T 1
    - (M_K^T u))(по строкам J — токены)."""
    n = M_K.shape[0]
    mt1 = M_K @ np.full(128, UNIF, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for b0 in range(0, n, 8):
        b = min(b0 + 8, n)
        mk_u = M_K @ uhat[b0:b].T
        jt1 = c[b0:b] * (128.0 ** 0.5 * mt1[None] - mk_u.T)
        out[b0:b] = (jt1 @ np.full(128, UNIF, dtype=np.float64)) / np.linalg.norm(jt1, axis=1)
    return float(np.median(out))


def _drift(J_b, J_q):
    num = torch.norm(J_q - J_b, p=2)
    den = torch.norm(J_b, p=2)
    return float(num / den)


def main(results_dir=None, device="auto"):
    t0 = time.time()
    np.random.seed(SEED)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"E-39b: device={device}")

    if results_dir:
        out_dir = results_dir
    elif os.path.isdir("/kaggle/working"):
        out_dir = os.path.join("/kaggle/working", "results")
    else:
        out_dir = os.path.join(REPO, "results")
    os.makedirs(out_dir, exist_ok=True)
    OUT_PATH = os.path.join(out_dir, "exp39b_ptq_perturbation.json")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B", dtype=torch.float16, attn_implementation="eager").to(device)
    model.eval()
    W_lm = _resolve_lm_head_weight(model).float()          # [151936, 1024] fp32
    g = (_lm_module(model).norm.weight + 1.0).cpu().numpy().astype(np.float64)

    cal = _c4_docs(C4_CAL_DOCS, S, tokenizer, C4_CAL_SKIP)
    with torch.no_grad():
        X = model.lm_head(torch.from_numpy(cal).to(device)).detach().float()
    X = X.view(-1, D).cpu().numpy()
    n = X.shape[0]
    H = (2.0 / n) * (X @ X.T)
    print(f"C4 calib: {n} Hessian vectors (skip {C4_CAL_SKIP})")
    H = torch.from_numpy(H)
    W_int8 = rtn_int8(W_lm)
    W_gptq4 = gptq_quantize(W_lm, H, bits=4, group_size=GROUP, damp=DAMP, actorder=True)

    # PPL per level (tied weights → logits вручную: model.model.norm(h) @ W.T, fp32)
    ppl = {}
    tok_ppl = torch.from_numpy(_c4_docs(C4_PPL_DOCS, S, tokenizer, C4_PPL_SKIP)).to(device)
    for name, W in [("base", W_lm), ("int8", W_int8), ("gptq4", W_gptq4)]:
        with torch.no_grad():
            h = model.model(input_ids=tok_ppl).last_hidden_state
            logits = (model.model.norm(h)).float() @ W.T.to(device)
            logp = torch.log_softmax(logits, dim=-1)
            lp = logp.gather(1, tok_ppl[:, 1:].unsqueeze(1))
        ppl[name] = float(-lp.mean())
        print(f"PPL[{name}] = {ppl[name]:.4f}")

    levels = {"base": torch.from_numpy(W_lm.numpy()),
              "int8": torch.from_numpy(W_int8.numpy()),
              "gptq4": torch.from_numpy(W_gptq4.numpy())}
    per_layer = {}
    for l in LAYERS:
        ck = load_layer(l)
        J_ref, Hh, U = ck["J"], ck["H"], ck["U128"]
        nd = J_ref.shape[0]
        h = Hh.astype(np.float64)
        c = np.sqrt(D) / np.linalg.norm(h, axis=1)
        uhat = h / c[:, None]
        base_MK = torch.from_numpy(W_lm.numpy()) @ torch.diag(g)
        J_b = _recon_J(base_MK, h, uhat, c)
        if l == 23:
            err = ((J_b - J_ref.double()).norm(p=2) / J_ref.double().norm(p=2))
            errv = err.view(nd, S, -1)
            self_check = {"rel_median": float(np.median(errv.numpy())),
                          "rel_max": float(errv.max())}
            print(f"self-check L23: rel_err median={self_check['rel_median']:.2e} "
                  f"max={self_check['rel_max']:.2e}")
        else:
            self_check = None
        per_layer[str(l)] = {"self_check_base_vs_cached_J": self_check, "levels": {}}
        for name in ("base", "int8", "gptq4"):
            MK = (levels[name] @ torch.diag(torch.from_numpy(g))).double()
            J = _recon_J(MK, h, uhat, c)
            d = _drift(J_b, J)
            m = _kappa(_V16(J))
            m["base_ratio"] = _base_ratio(MK.numpy())
            m["cos_ug_uniform"] = _cos_ug(MK, uhat, c, uhat)
            m["J_rel_drift"] = d
            per_layer[str(l)]["levels"][name] = m
            print(f"L{l} [{name}]: kappa={m['kappa_median']:.5f} "
                  f"beta/sig1={m['base_ratio']:.4f} cos={m['cos_ug_uniform']:.4f} "
                  f"drift={d:.5f}")
        del J_ref, Hh, U, h, J_b
        if device != "cpu":
            torch.cuda.empty_cache()

    k4 = per_layer["23"]["levels"]["gptq4"]["kappa_median"]
    gate = ("pass" if k4 < 0.02 else
            "moderate" if k4 <= 0.10 else "catastrophic")
    result = {
        "exp": "E-39b", "spec": "docs/E39.md §2",
        "model": "Qwen/Qwen3.5-0.8B", "dtype": "fp16 (J — fp64, SVD — fp32)",
        "levels": {"base": "unquantized fp32 W_lm",
                   "int8": "RTN per-group(128) min-max symmetric, no Hessian",
                   "gptq4": f"GPTQ bits=4 group={GROUP} desc_act damp={DAMP}"},
        "gptq": {"calib_c4_skip": C4_CAL_SKIP, "hessian_vectors": int(n),
                 "damp": DAMP, "group": GROUP, "actorder": True,
                 "algo": "IST-DASLab/gptq fasterquant port (deterministic, no RNG)"},
        "ppl": ppl, "layers": per_layer,
        "gate": {"kappa_L23_gptq4": k4, "gate": gate,
                 "thresholds": {"pass": "<0.02", "moderate": "0.02-0.10",
                                "catastrophic": ">0.10"},
                 "base_ref": KAP_BASE},
        "timing": {"total_s": round(time.time() - t0, 1)},
        "meta": {"seed": SEED, "r": R,
                 "note": "J = readout-linearized W_lm[S]·diag(g)·(I−ûûᵀ)·c из H/U128 "
                         "jcache 27b; base vs cached J self-check на L23; "
                         "û заморожен (каши H) — PTQ меняет только M_K"}
    }
    json_dump(result, OUT_PATH)
    print(f"\nGATE: kappa_L23[gptq4] = {k4:.5f} → {gate}")
    print(f"saved {OUT_PATH}")


if __name__ == "__main__":
    main(results_dir="results")
