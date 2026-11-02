"""E-39f — Полный tied-weight GPTQ-4bit: сдвигает ли квантование эмбеддингов лист L23.

ТЗ: docs/E39f.md. E-39b показал: readout-only GPTQ-4bit не двигает κ(L23) (ratio 0.988,
PPL +0.90%) — но квантовал только `lm_head.weight`, а Qwen3.5-0.8B **tied**
(`lm_head.weight is embed_tokens.weight`), то есть настоящий PTQ квантует одну матрицу
и как readout, и как эмбеддинг. E-39f считает полный tied-PTQ и раскладывает Δκ(L23):

    Δ_total    = κ_full − κ_fp16
    Δ_readout  = κ_readout − κ_fp16      (канал E-39b: только M_K)
    Δ_upstream = κ_full − κ_readout      (новое: û_t сдвинут квантованным эмбеддингом)
    share_upstream = |Δ_upstream| / (|Δ_readout| + |Δ_upstream|)

Режимы κ (L23, r=16; алгебра и конвенции — вербатим E-39b/E-38d):
  κ_fp16    : M_K(base),   û из кэша exp27b         → baseline, assert ≈ 0.01623
  κ_control : M_K(base),   û из СВЕЖЕГО forward     → шумовое поле протокола (без него
              Δ_upstream нельзя отличить от шума порядка редукций)
  κ_readout : M_K(W_q),    û из кэша                → assert: воспроизведение E-39b
  κ_full    : M_K(W_q),    û из квантованного forward → новый объект
Режимы квантования: g=128 (стандарт, сравним с E-39b) и g=32 (harsh probe ТЗ §4.2/§5.2).
Диагностики: h_drift, û_cos, β/σ₁(M_K), J_rel_drift, PPL (fp16 / readout-only / full
tied), null-модель (16 seeds Haar û; ТЗ §4.3: κ_null < 1.0 ⇒ декомпозиция невалидна).

û для κ_full обязана сниматься ДО финальной RMSNorm: в факторе E-35
J = c_t·W_K·diag(g)·(I − û ûᵀ) вектор û — направление output'а слоя 23 (кэш exp27b берёт
`out.hidden_states[l+1]`, см. exp27b_dense._forward_dense), тогда как
`model.model(...).last_hidden_state` — уже post-norm. Поэтому pre-norm состояние снимается
хуком на последнем decoder-слое; post-norm берётся из того же forward для PPL.

Разрывы с ТЗ (записаны явно, D-номера попадают в JSON):
  D1 (ТЗ §4.2 шаг 0). `h_fresh` vs кэш: структурная проверка — `u_cos_min ≥ 0.9999`
     (те же документы, тот же порядок позиций) + репорт max_rel; жёсткий порог 1e-5
     держится только при fp32. При fp16 (T4) rel-ошибка hidden states ~1e-3 по
     определению, её роль берёт на себя контроль κ_control.
  D2 (ТЗ §4.1 dtype). dtype = fp16 на CUDA / fp32 на CPU (как E-39b N3): fp16-форвард на
     CPU в 5–10× медленнее. GPTQ от этого сдвигается относительно E-39b, поэтому гейт
     воспроизведения — по drift (±5%), а κ_readout — только loose-assert (±2e-3).
  D3 (ТЗ §4.2 шаг 4). PPL считается matmul'ом по post-norm h из того же forward'а,
     а не отдельными forward'ами; NLL-маска и формула — вербатим E-39b.
  D4. J_rel_drift берётся аналитически из ΔM_K (ΔJ = c(ΔM_K − (ΔM_K û) ûᵀ)) — та же
     формула, что в E-39b, без удержания двух [32,128,128,1024] тензоров.
  D5 (ТЗ §10.4 п.3). QA-accuracy не измерен: офлайн-датасетов нет; downstream = PPL.

Выход: results/exp39f_full_tied.json (+ results/exp39f_pool.pt — кэш документов пула).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from prism.calibration import load_calibration_data
from transformers import AutoModelForCausalLM, AutoTokenizer

from _e38_cache import D, load_layer  # noqa: E402  (на Kaggle инлайнится --extra-local)

# --- inlined prism helpers (канон: prism/{jacobian,interventions,jstruct}.py) ---
def _lm_module(model):
    """Text model с .layers/.norm (prism/jacobian.py:_lm_module)."""
    mm = model.model
    return mm if hasattr(mm, "layers") else mm.language_model


def _resolve_lm_head_weight(model) -> torch.Tensor:
    """model.lm_head.weight с фолбэком на get_output_embeddings()."""
    lm = getattr(model, "lm_head", None)
    if lm is not None and getattr(lm, "weight", None) is not None:
        return lm.weight
    out = model.get_output_embeddings()
    if out is None or getattr(out, "weight", None) is None:
        raise AttributeError("у модели нет lm_head.weight / output embeddings")
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


def _M_K_rows(W, token_set, g):
    """M_K по K строкам без fp64-копии всей [V,D] матрицы (экономия ~2 ГБ на режим).

    build_M_K применяется уже к вырезке строк (валидация сохраняется), арифметика
    в fp64 — как у build_M_K(W_lm64, ...) в E-39b.
    """
    ts = torch.as_tensor(token_set, dtype=torch.long)
    return build_M_K(W[ts].double(), list(range(ts.numel())), g).float()


MODEL = "Qwen/Qwen3.5-0.8B"
SEED = 0
R = 16
S_PREFIX = 128                 # то же окно, что у exp27b/кэша
L_POOL = 256
DOMAINS = ("wiki", "math", "code", "chat")
N_PER_DOMAIN = 8
L_TARGET = 23                  # единственный слой с точной факторизацией (E-35)

GPTQ_BITS = 4
GPTQ_GROUP = 128
GPTQ_GROUP_HARSH = 32
GPTQ_DAMP = 0.01
C4_CAL_SKIP = 8192             # GPTQ Hessian: 128 docs × 128 tok (сплит E-39b)
N_CAL, S_CAL = 128, 128
C4_PPL_SKIP = 24576
N_PPL, S_PPL = 32, 128

# References (E-39b v12 / E-38d / E-38e): воспроизведение — часть гейта
KAPPA_BASE_REF, KAPPA_BASE_TOL = 0.01623, 0.005
KAPPA_READOUT_REF, KAPPA_READOUT_TOL = 0.016043, 2e-3
DRIFT_REF, DRIFT_REL_TOL = 0.12047, 0.05
BETA_BASE_REF, BETA_BASE_TOL = 0.882, 0.05
KAPPA_PASS, KAPPA_MODERATE = 0.02, 0.10
NULL_KAPPA_MIN, N_NULL_SEEDS = 1.0, 16
DELTA_MIN = 1e-3               # ТЗ §2.3: доли определены только при |Δ_total| > 1e-3
UCOS_MIN = 0.9999
BOOT = 1000
POOL_ART = "exp39f_pool.pt"


# ---------------------------------------------------------------- документы
def _c4_docs(n_docs, s_len, tokenizer, skip=0, split="train"):
    """n_docs документов C4, токенизированных полным текстом до s_len токенов (E-39b)."""
    ds = load_dataset("allenai/c4", "en", split=split, streaming=True)
    texts, n_skip = [], 0
    for doc in ds:
        if n_skip < skip:
            n_skip += 1
            continue
        if doc["text"]:
            texts.append(doc["text"])
        if len(texts) >= n_docs:
            break
    if len(texts) < n_docs:
        raise RuntimeError(f"C4 streaming yielded {len(texts)}/{n_docs} docs")
    enc = tokenizer(texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=s_len)
    return enc["input_ids"], enc["attention_mask"]


def _load_pool(model_name, domains, n, L=L_POOL):
    """Пулы документов — вербатим exp27b_dense._load_pool (те же документы, что у кэша)."""
    per: dict[str, list] = {}
    for dom in domains:
        big, data0 = [], []
        for att in range(5):
            big = load_calibration_data(model_name, max_length=L, min_length=1,
                                        n_samples=n * 64, domain=dom)
            data0 = [s for s in big if s["input_ids"].shape[1] >= L]
            if len(data0) >= n or len(big) >= n * 64:
                break
            print(f"  pool {dom}: stream gave {len(data0)} >=L, retry {att + 1}...",
                  flush=True)
            time.sleep(10 * (att + 1))
        data = data0
        if len(data) < n:            # короткоисточные домены: склейка окон
            tok = AutoTokenizer.from_pretrained(model_name)
            buf, out, consumed = "", [], 0
            for t in big:
                buf += " " + tok.decode(t["input_ids"][0],
                                        skip_special_tokens=True).strip()
                words = buf.split(" ")
                while len(out) < n and len(words) - consumed >= 64:
                    ids = tok(" ".join(words[consumed:consumed + 320]),
                              max_length=L, truncation=True, return_tensors="pt")
                    if ids["input_ids"].shape[1] >= L:
                        out.append(ids)
                        consumed += 320
                        continue
                    break
            data = data + out
            seen, ded = set(), []
            for x in data:
                k = tuple(x["input_ids"][0].tolist())
                if k not in seen:
                    seen.add(k)
                    ded.append(x)
            data = (ded + data)[:n]
        else:
            data = data[:n]
        print(f"  pool {dom}: {len(data)} @ {L}", flush=True)
        per[dom] = data
    return per


def pool_fingerprint(pool):
    """sha256 пула — та же схема, что exp27b_dense (сверка сплита)."""
    return {d: hashlib.sha256(b"".join(
        x["input_ids"].flatten().numpy().tobytes()
        for x in pool[d])).hexdigest()[:12] for d in pool}


def get_pool(model_name, results_dir, use_cache=True):
    """Документы пула (domain-major) + fingerprint.

    Артефакт-кэш есть только локально; на Kaggle пул стримится заново, а идентичность
    документов всё равно проверяется assert'ом по h против кэша (Шаг 0).
    """
    art = os.path.join(results_dir, POOL_ART)
    if use_cache and os.path.exists(art):
        blob = torch.load(art, map_location="cpu", weights_only=False)
        print(f"pool: артефакт {art} fp={blob['pool_fp']}", flush=True)
        return {"ids": blob["ids"], "attn": blob["attn"]}, blob["pool_fp"]
    pool = _load_pool(model_name, list(DOMAINS), N_PER_DOMAIN)
    fp = pool_fingerprint({d: pool[d] for d in DOMAINS})
    docs = [pool[d][i] for d in DOMAINS for i in range(N_PER_DOMAIN)]   # domain-major
    out = {"ids": torch.cat([x["input_ids"] for x in docs]),
           "attn": torch.cat([x["attention_mask"] for x in docs])}
    try:
        torch.save({**out, "pool_fp": fp, "L": L_POOL}, art)
        print(f"pool: кэш → {art} fp={fp}", flush=True)
    except OSError as e:
        print(f"pool: артефакт не сохранён ({e})", flush=True)
    return out, fp


# ---------------------------------------------------------------- линейная алгебра
def _procrustes_batched(V_a, V_b):
    """Procrustes для [B,D,r]: M = V_bᵀV_a = PΣQᵀ, R = QPᵀ; res = 2r − 2Σσ."""
    M = torch.bmm(V_b.transpose(1, 2), V_a).double()
    P, s, Qt = torch.linalg.svd(M, full_matrices=False)
    Rmat = torch.bmm(Qt.transpose(1, 2), P.transpose(1, 2))
    res = (2.0 * V_a.shape[2] - 2.0 * s.sum(dim=-1)).clamp(min=0.0)
    return Rmat, res


def _kappa_per_position(J, r, device, chunk_docs=4):
    """κ по тройкам (t,t+1,t+2) → массив [n, S−2]; медиана по нему = κ E-39b.

    Вербатим exp39b._kappa_from_J (SVD в fp32, Procrustes/гоономия в fp64), но
    сохраняется полный массив: нужны per-doc медианы и bootstrap по документам.
    """
    n, S, K, Dd = J.shape
    n_tr = S - 2
    Jf = J.float().to(device)
    V_all = torch.empty(n * S, Dd, r, device=device, dtype=torch.float32)
    for c0 in range(0, n, chunk_docs):
        c1 = min(c0 + chunk_docs, n)
        Jb = Jf[c0:c1].reshape((c1 - c0) * S, K, Dd)
        _, s, Vh = torch.linalg.svd(Jb, full_matrices=False)
        V_all[c0 * S:c1 * S] = Vh[:, :r, :].transpose(1, 2)
        del Jb, s, Vh
        if device.type == "cuda":
            torch.cuda.empty_cache()
    f0 = (torch.arange(n, device=device).repeat_interleave(n_tr) * S
          + torch.arange(n_tr, device=device).repeat(n))
    out = torch.empty(n * n_tr, device=device, dtype=torch.float64)
    n_flat = n * n_tr
    chunk = max(1, min(n_flat, (1500 * 10**6) // (3 * Dd * r * 4 + 48 * r * r)))
    I = torch.eye(r, device=device, dtype=torch.float64)
    for b0 in range(0, n_flat, chunk):
        b1 = min(b0 + chunk, n_flat)
        fa = f0[b0:b1]
        R01, _ = _procrustes_batched(V_all[fa], V_all[fa + 1])
        R12, _ = _procrustes_batched(V_all[fa + 1], V_all[fa + 2])
        R20, _ = _procrustes_batched(V_all[fa + 2], V_all[fa])
        Holo = torch.einsum("bij,bjk,bkl->bil", R01, R12, R20)
        out[b0:b1] = torch.linalg.norm(Holo - I, ord="fro", dim=[1, 2])
        del R01, R12, R20, Holo
        if device.type == "cuda":
            torch.cuda.empty_cache()
    del V_all
    return out.reshape(n, n_tr).cpu().numpy()


def _hidden_to_u(H):
    """û = h/‖h‖ [n,S,D] и c = √D/‖h‖ [n,S] из pre-norm hidden states."""
    hn = H.norm(dim=2).clamp(min=1e-12)
    return H / hn.unsqueeze(2), (float(H.shape[-1]) ** 0.5) / hn


def _J_from_u(M_K, U_hat, c, chunk_docs=8):
    """J = c_t·M_K·(I − û_t û_tᵀ), чанками по документам (память — как E-39b)."""
    n, S, Dd = U_hat.shape
    K = M_K.shape[0]
    Jout = torch.empty(n, S, K, Dd, device=U_hat.device, dtype=torch.float32)
    for c0 in range(0, n, chunk_docs):
        c1 = min(c0 + chunk_docs, n)
        U = U_hat[c0:c1]
        MK_Ut = M_K @ U.transpose(1, 2)                       # [nc,K,S]
        cross = MK_Ut.permute(0, 2, 1).unsqueeze(3) * U.unsqueeze(2)
        out = Jout[c0:c1]
        torch.sub(M_K.unsqueeze(0).unsqueeze(0), cross, out=out)
        out.mul_(c[c0:c1].unsqueeze(2).unsqueeze(3))
        del U, MK_Ut, cross, out
        if U_hat.device.type == "cuda":
            torch.cuda.empty_cache()
    return Jout


def _beta_sigma1(M_K, U_hat):
    """β/σ₁(M_K) (E-38e): w = M_Kᵀ(1/√K), α_t = ûᵀw, β_t = ‖w − α_t û_t‖."""
    K = M_K.shape[0]
    w = M_K.T @ (torch.ones(K, device=M_K.device, dtype=M_K.dtype) / np.sqrt(K))
    s1 = torch.linalg.svdvals(M_K.float())[0].item()
    alpha = U_hat @ w                                  # [n,S]
    beta = (w.view(1, 1, -1) - alpha.unsqueeze(2) * U_hat).norm(dim=2)
    v = (beta / s1).flatten().cpu().numpy()
    v = v[np.isfinite(v)]
    return (float(np.median(v)) if v.size else float("nan")), int(v.size)


def _drift_dMK(M_base, M_q, U_hat, c, chunk_docs=8):
    """J_rel_drift = ‖ΔJ‖_F/‖J‖_F по позициям, ΔJ = c(ΔM − (ΔM û)ûᵀ) — аналитически (D4)."""
    n = U_hat.shape[0]
    dM = M_q - M_base
    num, den = [], []
    for c0 in range(0, n, chunk_docs):
        c1 = min(c0 + chunk_docs, n)
        U = U_hat[c0:c1]
        cc = c[c0:c1].unsqueeze(2).unsqueeze(3)
        for W, acc in ((dM, num), (M_base, den)):
            WU = W @ U.transpose(1, 2)
            J = (W.unsqueeze(0).unsqueeze(0)
                 - WU.permute(0, 2, 1).unsqueeze(3) * U.unsqueeze(2)) * cc
            acc.append(torch.linalg.norm(J, ord="fro", dim=[2, 3]))
            del WU, J
        del U
        if U_hat.device.type == "cuda":
            torch.cuda.empty_cache()
    ratio = (torch.cat(num) / torch.cat(den).clamp(min=1e-12)).cpu().numpy()
    del num, den
    return ratio


def _stats(v):
    v = np.asarray(v, dtype=float).flatten()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {"median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def _boot_ci_over_docs(per_doc, b, rng):
    """bootstrap CI медианы по документам."""
    per_doc = np.asarray(per_doc, dtype=float)
    n = len(per_doc)
    vals = [float(np.median(per_doc[rng.integers(0, n, n)])) for _ in range(b)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _quantize_gptq(W, X, bits, group, damp_frac, device):
    """GPTQ (порт IST-DASLab/gptq, вербатим E-39b). W: [V,D], X: [n,D] → декевант fp32.

    H = (2/n)Σxxᵀ, dead-columns → 0, damp = damp_frac·mean(diag H), desc_act,
    Hinv = chol(chol_inv(chol(H)), upper=True), per-group scale/zero, внутриблочные
    и межблочные компенсации ошибок, unpermute в конце. Детерминирован.
    """
    V, Dd = W.shape
    W = W.to(device).clone().float()
    n = X.shape[0]
    H = (2.0 / n) * (X.float().T @ X.float())
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
    Lc = torch.linalg.cholesky(Hp)
    Hinv = torch.cholesky_inverse(Lc)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(Wp)
    group_scales, group_zeros = [], []
    for i1 in range(0, Dd, group):
        i2 = min(i1 + group, Dd)
        W1 = Wp[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        xmax = W1.max(dim=1, keepdim=True).values
        xmin = W1.min(dim=1, keepdim=True).values
        scale = ((xmax - xmin) / maxq).squeeze(1)
        zero = (-xmin / scale.clamp(min=1e-12).unsqueeze(1)).squeeze(1)
        group_scales.append(scale.clone())
        group_zeros.append(zero.clone())
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i].clamp(min=1e-12)
            q = torch.clamp(torch.round(w / scale.clamp(min=1e-12) + zero), 0, maxq)
            Q1[:, i] = q
            err1 = (w - scale * (q - zero)) / d
            W1[:, i:] -= err1.unsqueeze(1) * Hinv1[i, i:]
            Err1[:, i] = err1
        Q[:, i1:i2] = Q1
        if i2 < Dd:
            Wp[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    Wq_p = torch.zeros_like(Wp)
    for gi in range(len(group_scales)):
        j0 = gi * group
        j1 = min(j0 + group, Dd)
        Wq_p[:, j0:j1] = (group_scales[gi].unsqueeze(1)
                          * (Q[:, j0:j1] - group_zeros[gi].unsqueeze(1)))
    Wq = torch.zeros(V, Dd, device=device, dtype=torch.float32)
    Wq[:, perm] = Wq_p
    return Wq


# ---------------------------------------------------------------- модель
def _tied_pair(model):
    """(lm_head.weight, embed_tokens.weight); не tied ⇒ abort (ТЗ §4.3, abort #1)."""
    lm_w = _resolve_lm_head_weight(model)
    emb = model.get_input_embeddings()
    if emb is None or getattr(emb, "weight", None) is None:
        raise RuntimeError("E-39f abort: у модели нет input embeddings — tied-PTQ "
                           "не определён")
    if lm_w.data_ptr() != emb.weight.data_ptr():
        raise RuntimeError(
            f"E-39f abort: lm_head и embed_tokens — разные тензоры ({lm_w.data_ptr()} "
            f"!= {emb.weight.data_ptr()}); эксперимент про tied-PTQ не имеет смысла")
    return lm_w, emb.weight


def _forward_states(model, ids, attn, last_layer, chunk=8):
    """(pre-norm, post-norm) hidden states [n,S,D] fp32 CPU за один проход по чанкам.

    pre-norm — output последнего decoder-слоя (хук), post-norm — last_hidden_state
    (он же вход lm_head, проверено в E-39b P0.4). Оба из одного forward ⇒ одна и та же
    (квантованная или базовая) модель.
    """
    pre_parts, post_parts = [], []

    def hook(_m, _inp, out):
        pre_parts.append((out[0] if isinstance(out, tuple) else out)
                         .detach().float().cpu())

    h = last_layer.register_forward_hook(hook)
    try:
        for c0 in range(0, ids.shape[0], chunk):
            c1 = min(c0 + chunk, ids.shape[0])
            n_before = len(pre_parts)
            with torch.no_grad():
                o = model(input_ids=ids[c0:c1], attention_mask=attn[c0:c1])
            if len(pre_parts) != n_before + 1:
                raise RuntimeError("hook не сработал (слой не вызывался)")
            pre_parts[-1] = pre_parts[-1].reshape(c1 - c0, ids.shape[1], -1)
            post_parts.append(o.last_hidden_state.float().cpu())
            del o
            if ids.device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        h.remove()
    return torch.stack(pre_parts), torch.stack(post_parts)


def main(device=None, results_dir=None, small_model=None, pool_cache=True):
    t0 = time.time()
    dev = (torch.device(device) if isinstance(device, str)
           else device or torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    out_dir = results_dir or os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "exp39f_full_tied.json")
    MODEL_P = small_model or MODEL
    model_dtype = torch.float16 if dev.type == "cuda" else torch.float32
    print(f"E-39f full tied PTQ: device={dev} model={MODEL_P} dtype={model_dtype}",
          flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_P, torch_dtype=model_dtype, attn_implementation="eager").to(dev).eval()
    tok = AutoTokenizer.from_pretrained(MODEL_P)
    lm_w, emb_w = _tied_pair(model)
    layers = _lm_module(model).layers
    n_layers = len(layers)
    if L_TARGET >= n_layers:
        raise RuntimeError(f"L_TARGET={L_TARGET} >= n_layers={n_layers}")
    last_layer = layers[n_layers - 1]
    W_base = lm_w.detach().clone().cpu()                     # model dtype, для restore
    g = _lm_module(model).norm.weight.float().cpu() + 1      # [D] = 1 + w (не квантуется)
    W32 = W_base.float()
    print(f"tied OK: W {tuple(W32.shape)}, layers={n_layers}, g_min={g.min():.3f}",
          flush=True)

    # ---------------- Шаг 0: кэш, пул, self-check ----------------
    cache = load_layer(L_TARGET, want=("H",))
    H_cache = cache["H"]
    U128 = cache["U128"]
    n_docs, S = H_cache.shape[:2]
    if S != S_PREFIX:
        raise RuntimeError(f"S кэша {S} != {S_PREFIX}")
    pool, pool_fp = get_pool(MODEL_P, out_dir, use_cache=pool_cache)
    if pool["ids"].shape[0] != n_docs:
        raise RuntimeError(f"пул: {pool['ids'].shape[0]} документов, кэш: {n_docs}")
    ids = pool["ids"][:, :S_PREFIX].to(dev)
    attn = pool["attn"][:, :S_PREFIX].to(dev)

    print("fresh base forward (32 docs × 128) ...", flush=True)
    h_pre_base, _ = _forward_states(model, ids, attn, last_layer)
    h_pre_base_t = h_pre_base.flip(1).contiguous()   # кэш: t считается ОТ КОНЦА префикса
    rel = ((h_pre_base_t - H_cache).norm(dim=2)
           / H_cache.norm(dim=2).clamp(min=1e-12)).flatten().numpy()
    ucos_base = torch.nn.functional.cosine_similarity(
        h_pre_base_t.flatten(0, 1), H_cache.flatten(0, 1), dim=1).numpy()
    hcheck = {"max_rel": float(np.nanmax(rel)), "median_rel": float(np.nanmedian(rel)),
              "u_cos_median": float(np.nanmedian(ucos_base)),
              "u_cos_min": float(np.nanmin(ucos_base))}
    print(f"  h_fresh vs кэш: max_rel={hcheck['max_rel']:.2e} "
          f"u_cos_min={hcheck['u_cos_min']:.8f}", flush=True)
    assert hcheck["u_cos_min"] >= UCOS_MIN, (
        f"Шаг 0 abort: u_cos_min = {hcheck['u_cos_min']:.8f} < {UCOS_MIN} — другие "
        "документы или порядок позиций (пул не совпадает с кэшем)")
    if model_dtype == torch.float32:
        assert hcheck["max_rel"] < 1e-5, (
            f"Шаг 0 abort (fp32): max_rel = {hcheck['max_rel']:.2e} >= 1e-5 — forward "
            "расходится с производителем кэша")

    Hc = H_cache.to(dev)
    U_cache, c_cache = _hidden_to_u(Hc)
    M_base = _M_K_rows(W32, U128, g).to(dev)

    def kappa_of(M_K, U_hat, c, label):
        J = _J_from_u(M_K, U_hat, c)
        kp = _kappa_per_position(J, R, dev)
        del J
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        med_doc = np.median(kp, axis=1)
        res = {"kappa": float(np.median(kp)),
               "kappa_mean": float(np.mean(kp)),
               "per_doc_median": [float(x) for x in med_doc],
               "boot_ci95": _boot_ci_over_docs(med_doc, BOOT, rng),
               "positions": _stats(kp)}
        res["beta_sigma1"], res["beta_n_valid"] = _beta_sigma1(M_K, U_hat)
        print(f"  κ[{label}] = {res['kappa']:.5f}  β/σ₁ = {res['beta_sigma1']:.4f}",
              flush=True)
        return res

    kap_fp16 = kappa_of(M_base, U_cache, c_cache, "fp16/cache")
    assert abs(kap_fp16["kappa"] - KAPPA_BASE_REF) < KAPPA_BASE_TOL, (
        f"Шаг 0 abort: κ_fp16 = {kap_fp16['kappa']:.5f} != {KAPPA_BASE_REF} "
        f"(±{KAPPA_BASE_TOL}) — baseline не воспроизводится")
    assert abs(kap_fp16["beta_sigma1"] - BETA_BASE_REF) < BETA_BASE_TOL, (
        f"β/σ₁ fp16 = {kap_fp16['beta_sigma1']:.3f} != {BETA_BASE_REF} (E-38e)")
    U_fresh, c_fresh = _hidden_to_u(h_pre_base_t.to(dev))
    kap_control = kappa_of(M_base, U_fresh, c_fresh, "control/fresh-base")
    del U_fresh, c_fresh

    # ---------------- Шаг 1: GPTQ-калибровка ----------------
    print(f"GPTQ Hessian: C4 train skip {C4_CAL_SKIP} ...", flush=True)
    ids_cal, attn_cal = _c4_docs(N_CAL, S_CAL, tok, skip=C4_CAL_SKIP)
    ids_cal, attn_cal = ids_cal.to(dev), attn_cal.to(dev)
    caps = []
    for c0 in range(0, N_CAL, 16):
        c1 = min(c0 + 16, N_CAL)
        with torch.no_grad():
            o = model.model(input_ids=ids_cal[c0:c1], attention_mask=attn_cal[c0:c1])
        caps.append(o.last_hidden_state.float().cpu())
        del o
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    X_all = torch.cat(caps, dim=0)
    del caps
    X = X_all[attn_cal.cpu().bool()]        # padding в Гессиан не входит (E-39b P0.2)
    del X_all, ids_cal, attn_cal
    print(f"  входов Гессиана: {tuple(X.shape)} реальных токенов", flush=True)
    X_dev = X.to(dev)
    del X
    t_g = time.time()
    # GPTQ внутри делает .float() — вход fp32 побитово тот же, что fp64-вход E-39b.
    W_q128 = _quantize_gptq(W32, X_dev, GPTQ_BITS, GPTQ_GROUP, GPTQ_DAMP, dev).cpu()
    print(f"  GPTQ g=128: {time.time() - t_g:.1f}s", flush=True)
    t_g = time.time()
    W_q32 = _quantize_gptq(W32, X_dev, GPTQ_BITS, GPTQ_GROUP_HARSH, GPTQ_DAMP, dev).cpu()
    print(f"  GPTQ g=32 (harsh): {time.time() - t_g:.1f}s", flush=True)
    del X_dev
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # PPL-документы стримятся ОДИН раз и используются во всех режимах.
    try:
        ids_ppl, attn_ppl = _c4_docs(N_PPL, S_PPL, tok, skip=0, split="val")
        ppl_split = "val"
    except Exception as e:                       # split 'val' в C4 отсутствует (E-39b)
        print(f"  C4 val недоступен ({e}) → train skip {C4_PPL_SKIP}", flush=True)
        ids_ppl, attn_ppl = _c4_docs(N_PPL, S_PPL, tok, skip=C4_PPL_SKIP)
        ppl_split = f"train skip {C4_PPL_SKIP}"
    ids_ppl, attn_ppl = ids_ppl.to(dev), attn_ppl.to(dev)
    print("base forward на PPL-документах ...", flush=True)
    _, h_post_base = _forward_states(model, ids_ppl, attn_ppl, last_layer)

    def ppl_from(post, W, label):
        """PPL = exp(ΣNLL/n_real) по готовым post-norm h (matmul, D3; маска E-39b)."""
        Wd = W.float().to(dev)
        tot, n_real = 0.0, 0
        for c0 in range(0, post.shape[0], 8):
            c1 = min(c0 + 8, post.shape[0])
            lp = torch.log_softmax(post[c0:c1].to(dev) @ Wd.T, dim=-1)
            tgt = ids_ppl[c0:c1, 1:]
            m = attn_ppl[c0:c1, 1:].bool()
            lp_t = lp[:, :-1, :].gather(2, tgt.unsqueeze(2)).squeeze(2)
            tot += float((-lp_t[m]).sum().item())
            n_real += int(m.sum().item())
            del lp, lp_t, tgt, m
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        del Wd
        v = float(np.exp(tot / max(n_real, 1)))
        print(f"  PPL[{label}] = {v:.4f} (n_real={n_real})", flush=True)
        return v, n_real

    # ---------------- Шаги 2–5: режимы ----------------
    def run_regime(tag, W_q):
        M_q = _M_K_rows(W_q, U128, g).to(dev)
        kap_read = kappa_of(M_q, U_cache, c_cache, f"readout/{tag}")
        drift = _stats(_drift_dMK(M_base, M_q, U_cache, c_cache))
        print(f"  drift[{tag}] median={drift['median']:.5f}", flush=True)

        with torch.no_grad():
            lm_w.data.copy_(W_q.to(lm_w.dtype).to(lm_w.device))
        assert lm_w.data_ptr() == emb_w.data_ptr(), "tied разрушен после replace"
        print(f"[{tag}] quantized full forward (пул + PPL-документы) ...", flush=True)
        h_pre_q, _ = _forward_states(model, ids, attn, last_layer)
        _, h_post_q = _forward_states(model, ids_ppl, attn_ppl, last_layer)
        with torch.no_grad():
            lm_w.data.copy_(W_base.to(lm_w.device))
        if not torch.equal(lm_w[:8].detach().cpu(), W_base[:8]):
            raise RuntimeError("восстановление базового tied-веса не удалось")
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        h_pre_q_t = h_pre_q.flip(1).contiguous().to(dev)
        U_q, c_q = _hidden_to_u(h_pre_q_t)
        kap_full = kappa_of(M_q, U_q, c_q, f"full/{tag}")
        h_drift = (h_pre_q_t - Hc).norm(dim=2) / Hc.norm(dim=2).clamp(min=1e-12)
        u_cos = (U_q * U_cache).sum(dim=2)
        diag = {
            "h_drift": _stats(h_drift), "u_cos": _stats(u_cos),
            "h_drift_per_doc_median": [float(x) for x in h_drift.median(dim=1).values],
            "u_cos_per_doc_median": [float(x) for x in u_cos.median(dim=1).values],
        }
        print(f"  h_drift median={diag['h_drift']['median']:.4f} "
              f"u_cos median={diag['u_cos']['median']:.6f}", flush=True)
        del h_drift, u_cos, U_q, c_q, h_pre_q_t, M_q
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return {"kappa_readout": kap_read, "kappa_full": kap_full,
                "drift_J": drift, **diag}, h_post_q

    reg128, h_post_q128 = run_regime("g128", W_q128)
    reg32, h_post_q32 = run_regime("g32", W_q32)

    ppl = {}
    ppl["fp16"], n_real = ppl_from(h_post_base, W32, "fp16")
    ppl["readout_only_g128"], _ = ppl_from(h_post_base, W_q128, "readout_only_g128")
    ppl["full_tied_g128"], _ = ppl_from(h_post_q128, W_q128, "full_tied_g128")
    ppl["full_tied_g32"], _ = ppl_from(h_post_q32, W_q32, "full_tied_g32")

    # ---------------- Шаг 6: null-модель (Haar û) ----------------
    print(f"null-модель: Haar û, {N_NULL_SEEDS} seeds ...", flush=True)
    null_k = []
    for s in range(N_NULL_SEEDS):
        gs = torch.Generator().manual_seed(1000 + s)
        Ur = torch.randn(n_docs, S, D, generator=gs).to(dev)
        Ur = Ur / Ur.norm(dim=2, keepdim=True).clamp(min=1e-12)
        null_k.append(kappa_of(M_base, Ur, c_cache, f"null/{s}")["kappa"])
        del Ur
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    kappa_null = float(np.median(null_k))
    assert kappa_null >= NULL_KAPPA_MIN, (
        f"Шаг 6 abort: κ_null = {kappa_null:.4f} < {NULL_KAPPA_MIN} — κ не видит "
        "û-канал, декомпозиция невалидна")

    # ---------------- декомпозиция, гейты, сверка ----------------
    def decomp(reg):
        k_fp = kap_fp16["kappa"]
        k_ro = reg["kappa_readout"]["kappa"]
        k_fu = reg["kappa_full"]["kappa"]
        d_ro, d_up, d_tot = k_ro - k_fp, k_fu - k_ro, k_fu - k_fp
        den = abs(d_ro) + abs(d_up)
        return {"kappa_fp16": k_fp, "kappa_readout": k_ro, "kappa_full": k_fu,
                "delta_readout": float(d_ro), "delta_upstream": float(d_up),
                "delta_total": float(d_tot),
                "share_upstream": float(abs(d_up) / den) if den > 0 else float("nan"),
                "shares_defined": bool(abs(d_tot) > DELTA_MIN),
                "per_doc_delta_upstream": [a - b for a, b in zip(
                    reg["kappa_full"]["per_doc_median"],
                    reg["kappa_readout"]["per_doc_median"])],
                "noise_floor_delta_control": float(kap_control["kappa"] - k_fp)}

    dec = {"g128": decomp(reg128), "g32": decomp(reg32)}
    d128, d32 = dec["g128"], dec["g32"]
    for tag, d in dec.items():
        print(f"[{tag}] Δ_readout={d['delta_readout']:+.5f} "
              f"Δ_upstream={d['delta_upstream']:+.5f} "
              f"share_up={d['share_upstream']:.3f}", flush=True)

    if not d128["shares_defined"]:
        gate_main = "both_channels_small"
    elif d128["share_upstream"] >= 0.5:
        gate_main = "upstream_dominates"
    elif d128["share_upstream"] < 0.25:
        gate_main = "readout_dominates"
    else:
        gate_main = "both_channels"
    ratio32 = d32["kappa_full"] / kap_fp16["kappa"]
    gate_harsh = ("leaf_robust_even_harsh" if d32["kappa_full"] < KAPPA_PASS
                  and ratio32 <= 1.25 else
                  "moderate_harsh" if d32["kappa_full"] < KAPPA_MODERATE
                  else "catastrophic_harsh")
    dpct = (ppl["full_tied_g128"] - ppl["fp16"]) / ppl["fp16"] * 100
    gate_ppl = ("ppl_robust" if dpct < 5 else
                "ppl_moderate" if dpct <= 20 else "ppl_catastrophic")

    drift128 = reg128["drift_J"]["median"]
    assert abs(drift128 - DRIFT_REF) <= DRIFT_REL_TOL * DRIFT_REF, (
        f"GPTQ-пайплайн расходится с E-39b: drift median = {drift128:.5f}, "
        f"ожидалось {DRIFT_REF} ±{DRIFT_REL_TOL:.0%}")
    assert abs(d128["kappa_readout"] - KAPPA_READOUT_REF) < KAPPA_READOUT_TOL, (
        f"κ_readout = {d128['kappa_readout']:.5f} != {KAPPA_READOUT_REF} "
        f"(±{KAPPA_READOUT_TOL})")

    result = {
        "experiment": "E-39f full tied-weight PTQ (декомпозиция readout vs upstream)",
        "spec": "docs/E39f.md",
        "model": MODEL_P, "device": str(dev), "dtype": str(model_dtype),
        "tied_verified": True, "layer": L_TARGET, "r": R,
        "n_layers": int(n_layers), "D": int(D), "n_docs": int(n_docs), "S": int(S),
        "kappa_modes": {"fp16": kap_fp16, "control_fresh_base": kap_control,
                        "g128_readout": reg128["kappa_readout"],
                        "g128_full": reg128["kappa_full"],
                        "g32_readout": reg32["kappa_readout"],
                        "g32_full": reg32["kappa_full"]},
        "null_model": {"kappa_median": kappa_null, "kappa_seeds": null_k,
                       "n_seeds": N_NULL_SEEDS, "threshold": NULL_KAPPA_MIN},
        "decomposition": dec,
        "channel_diagnostics": {
            t: {k: reg[k] for k in ("h_drift", "u_cos", "h_drift_per_doc_median",
                                    "u_cos_per_doc_median")}
            for t, reg in (("g128", reg128), ("g32", reg32))},
        "drift_J": {"g128": reg128["drift_J"], "g32": reg32["drift_J"],
                    "e39b_reference": DRIFT_REF},
        "ppl": ppl, "ppl_split": ppl_split, "ppl_n_real": int(n_real),
        "ppl_delta_pct_full_tied_g128": float(dpct),
        "ppl_note": (
            "readout_only_g128 = базовые эмбеддинги + квантованный head (это же E-39b); "
            "full_tied = оба конца (реальный PTQ). full_tied с ppl_gptq4 из E-39b "
            "напрямую не сравнивается — разные объекты (ТЗ §7.4)."
        ),
        "gate": {"main": gate_main, "harsh": gate_harsh, "ppl": gate_ppl,
                 "harsh_ratio_g32": float(ratio32),
                 "thresholds": {"share_upstream_hi": 0.5, "share_upstream_lo": 0.25,
                                "delta_min": DELTA_MIN, "kappa_pass": KAPPA_PASS,
                                "kappa_moderate": KAPPA_MODERATE}},
        "selfchecks": {
            "tied_data_ptr": True,
            "h_fresh_vs_cache": hcheck,
            "h_rel_tol_applies_fp32_only": bool(model_dtype == torch.float32),
            "kappa_fp16_ref": KAPPA_BASE_REF, "kappa_fp16": kap_fp16["kappa"],
            "kappa_control_noise_floor": kap_control["kappa"],
            "kappa_readout_ref": KAPPA_READOUT_REF,
            "kappa_readout": d128["kappa_readout"],
            "drift_ref": DRIFT_REF, "drift": drift128,
            "beta_sigma1_fp16": kap_fp16["beta_sigma1"], "beta_ref": BETA_BASE_REF,
            "kappa_null": kappa_null, "pool_fp": pool_fp,
        },
        "deviations_from_spec": {
            "D1": "h-vs-cache: порог 1e-5 только при fp32; при fp16 — u_cos_min ≥ 0.9999 "
                  "+ контроль κ_control",
            "D2": "dtype = fp16 на CUDA / fp32 на CPU (E-39b N3); воспроизведение E-39b "
                  "проверяется по drift (±5%), κ_readout — loose ±2e-3",
            "D3": "PPL — matmul по сохранённым post-norm h (тот же forward), NLL-маска "
                  "вербатим E-39b",
            "D4": "J_rel_drift аналитически из ΔM_K (та же формула, меньше памяти)",
            "D5": "QA-accuracy (ТЗ §10.4 п.3) не измерен — нет офлайн-датасетов",
        },
        "limitations": [
            "только L23: при L<23 формула readout-линеаризована (E-39b) и реальной "
            "leaf-кривизны E-28 не даёт",
            "û для κ_full — из forward'а квантованной модели в dtype "
            f"{model_dtype}; относительный шум против кэша {hcheck['max_rel']:.1e} "
            "учтён контролем κ_control",
            "веса RMSNorm g не квантуются (стандарт GPTQ); KV-cache и activation "
            "quantization не тестировались",
            "κ — функционал от Procrustes-углов топ-r базиса, инвариантный к амплитуде: "
            "часть устойчивости принадлежит метрике (E-39b round-3)",
            "N=32 документа × 128 позиций, один чекпойнт, микс C4/math/code/chat",
            "GPTQ-калибровка на активациях базовой модели (ТЗ §7.2), не итеративная",
        ],
        "wall_sec": time.time() - t0,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\ngate: main={gate_main} harsh={gate_harsh} ppl={gate_ppl} | share_up "
          f"g128={d128['share_upstream']:.3f} g32={d32['share_upstream']:.3f}")
    print(f"output: {out_path}  wall: {time.time() - t0:.0f}s")
    return result


if __name__ == "__main__":
    main()
