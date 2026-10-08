#!/usr/bin/env python3
"""Exp 31 — B9: полная корреляционная функция C(Δpos) поля J вдоль траектории.

docs/Exp31.md. Post-processing на кэше exp27b (JR raw J, P128, P128_RAW) —
без GPU, без новых форвардов. Плюс один лёгкий шаг загрузки данных
(rebuild pool + offset_mapping): пул exp27b детерминирован (streaming HF,
фильтр >=256 токенов), pool_fp из кэша верифицирует идентичность побайтово.

Срезы (per layer L ∈ {6,12,18,22,23}, S=128, K=128, 32 документа):
  1. C(Δpos) = mean_pairs cos_F(J_t, J_{t+Δ}), Δ=1..32 — Frobenius-cosinus
     ПОЛНОЙ матрицы J (одна конвенция; «cos по строкам в порядке ранга»
     E-0.1 — другая величина, не используется). Per-документ сначала,
     затем медиана/IQR по 32 документам (протокол E-34).
  2. C_top-16(Δpos) = cos(max principal angle) между топ-16 правыми
     подпространствами (та же метрика, что в E-28: max, не mean/RMS).
  3. Фит формы: log C ~ a - Δ/ξ (экспонента, нить) vs
     log C ~ a - α log Δ (степенная, лист); AIC.
  4. Позиционно-разрешённая карта C(t, 16) — локализация разрывов.
  5. Сегментно-усреднённое C(Δ | intra/inter) по границам предложений.

Позиционная конвенция (фикс ТЗ): в dense-кэше t=0 — последняя позиция,
t=127 — первая; token_idx = S-1-t. Δpos определён в кэш-t; границы
предложений маппятся через seq_pos = S-1-t.

Pre-flight (ТЗ): пики dP(t) = ||p_{t+1}-p_t|| (MAD-детектор E-28) vs
границы предложений — внешняя проверка сегментно-усреднённого среза.

Sentinel: пара (t, t+Δ) исключается, если хотя бы один конец sentinel
(P128_RAW sum < 0.5, E34.SENTINEL_SUM); доля исключённых фиксируется.

Артефакты: results/exp31_correlation_function.json
           results/exp31_pool_offsets.pt (кэш пула с offset_mapping)
"""
import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import exp27b_dense as E27B
import exp34_p1_coupling as E34

MODEL = "Qwen/Qwen3.5-0.8B"
DOMAINS = ["wiki", "math", "code", "chat"]
LAYERS = [6, 12, 18, 22, 23]
N_SAMPLES = 8
S = 128
K = 128
R_TOP = 16          # топ-r подпространство для C_top-r
DELTA_MAX = 32      # C(Δ) для Δ = 1..32
DELTA_MAP = 16      # позиционно-разрешённая карта при «среднем» Δ
SENTINEL_SUM = E34.SENTINEL_SUM   # 0.5
CKPT_DIR = '/data/prism_kernels/exp27b/results'
OUT_JSON = 'results/exp31_correlation_function.json'
OUT_POOL = 'results/exp31_pool_offsets.pt'

_MATH_FORMULA = re.compile(
    r'\$\$[^$]+\$\$|\$[^$]+\$|\\\([^()]*\\\)|\\\[.*?\\\]', re.DOTALL)


# ---------------- pool + offset_mapping ----------------

def token_spans(tok, ids: list[int]):
    """Char-spans per token через инкрементальный decode (prefix property):
    span_t = decode(ids[:t+1])[len(decode(ids[:t])):]. Для Qwen byte-fallback
    декодер append-only. None — если property нарушена (fallback на
    roundtrip-encode)."""
    prev = ""
    spans = []
    for t in range(len(ids)):
        cur = tok.decode(ids[:t + 1], skip_special_tokens=False)
        if not cur.startswith(prev):
            return None
        spans.append(cur[len(prev):])
        prev = cur
    return spans


def rebuild_pool(model: str, n_samples: int = N_SAMPLES,
                 domains: list = DOMAINS, L: int = 256) -> dict:
    """Тот же детерминированный пул, что в exp27b_dense (streaming HF,
    фильтр >=256, дедуп). Per (dom, si): input_ids [S], offset_mapping
    первых S токенов, decoded text, полные 256-ids (для fingerprint).

    Offset_mapping: инкрементальный decode (robust к байт-фолбэк-токенам);
    roundtrip decode->encode — только fallback, если prefix property
    нарушена. Токенайзеру нельзя передать list[int] напрямую (transformers
    5.x трактует его как pre-tokenized текст)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    pool = E27B._load_pool(model, domains, n_samples, L)
    out = {}
    n_fallback = 0
    for dom in domains:
        for si, s in enumerate(pool[dom]):
            ids256 = s["input_ids"][0]
            ids = ids256[:S]
            text = tok.decode(ids, skip_special_tokens=False)
            spans = token_spans(tok, ids.tolist())
            if spans is not None:
                c, om = 0, []
                for sp in spans:
                    om.append((c, c + len(sp)))
                    c += len(sp)
                assert c == len(text), f'{dom}[{si}]: spans sum != len(text)'
            else:
                reenc = tok(text, add_special_tokens=False,
                            return_offsets_mapping=True)
                assert reenc["input_ids"][:S - 1] == ids[:S - 1].tolist(), \
                    f'{dom}[{si}]: roundtrip mismatch (no fallback left)'
                om = reenc["offset_mapping"][:S]
                n_fallback += 1
            out[(dom, si)] = {
                "input_ids": ids.cpu(), "input_ids_256": ids256.cpu(),
                "offset_mapping": np.array(om), "text": text,
            }
    if n_fallback:
        print(f'rebuild_pool: {n_fallback} docs via roundtrip fallback')
    return out


def pool_fp(pool: dict, ck) -> dict:
    """Fingerprint 256-токенового пула (как в exp27b_dense) vs pool_fp кэша."""
    return {dom: hashlib.sha256(b"".join(
        pool[(dom, si)]["input_ids_256"].flatten().numpy().tobytes()
        for si in range(ck["n_samples"]))).hexdigest()[:12]
        == ck["pool_fp"][dom] for dom in ck["domains"]}


# ---------------- segment boundaries ----------------

def _spans_to_tokens(spans, om):
    """Char-spans [(s, e)] -> token-index boundaries (start of each span)."""
    bounds = []
    for s_, e_ in spans:
        idx = [i for i, (a, b) in enumerate(om) if a >= s_ and (b <= e_ or b == 0)]
        if not idx:
            i0 = min((i for i, (a, b) in enumerate(om) if a >= s_),
                     default=len(om) - 1)
            idx = [i0]
        bounds.append(int(idx[0]))
    return sorted(set(b for b in bounds if 0 < b < S))


def sentence_spans(domain: str, text: str) -> list:
    """Char-spans сегментов по домену (ТЗ, подводный камень 5):
    wiki/chat — nltk.sent_tokenize; math — формулы как единицы + межформульные
    куски; code — границы \\n\\n (если отсутствуют — один сегмент)."""
    n = len(text)
    if domain in ("wiki", "chat"):
        from nltk.tokenize import sent_tokenize
        spans = []
        for sent in sent_tokenize(text):
            i = text.find(sent)
            if i >= 0:
                spans.append((i, i + len(sent)))
        return spans
    if domain == "math":
        spans = [(m.start(), m.end()) for m in _MATH_FORMULA.finditer(text)]
        pos = 0
        for m in _MATH_FORMULA.finditer(text):
            if m.start() > pos:
                spans.append((pos, m.start()))
            pos = m.end()
        if pos < n:
            spans.append((pos, n))
        return spans
    parts = re.split(r'\n\n+', text)
    spans, pos = [], 0
    for p in parts:
        i = text.find(p, pos)
        if i >= 0:
            spans.append((i, i + len(p)))
            pos = i + len(p)
    return spans or [(0, n)]


# ---------------- metrics ----------------

def vtop16(J, r=R_TOP):
    """Топ-r ПРАВОЕ подпространство каждого J_t (в R^D).
    Через eigh(грам J_t J_t^T [K,K]): U_r = топ-r левых сингулярных
    векторов, V_r = J_t^T U_r / sigma_i — span равен топ-r правому
    (v_i = J^T u_i / sigma_i). Батч bmm: ~5-10с на документ против
    ~21с полного SVD [S,K,D].
    J [S,K,D] -> V [S,D,r] (столбцы по убыванию sigma)."""
    G = torch.bmm(J, J.transpose(1, 2))        # [S,K,K] = J_t J_t^T
    w, u = torch.linalg.eigh(G)                # по возрастанию
    wr = w[..., -r:]                           # [S,r]
    Ur = u[..., -r:]                           # [S,K,r]
    V = torch.bmm(J.transpose(1, 2), Ur)       # [S,D,r]
    V = V / wr.unsqueeze(1).clamp_min(1e-30).sqrt()
    return V.contiguous()


def frob_cos(Jt, Jd, nt=None, nd=None):
    """cos_F(J_t, J_{t+Δ}) для пар: Jt, Jd [n, K, D] -> [n] (per-pair).
    nt/nd — опциональные предвычисленные нормы [n]."""
    num = (Jt * Jd).sum(dim=(-1, -2))
    na = nt if nt is not None else Jt.flatten(1).norm(dim=1)
    nb = nd if nd is not None else Jd.flatten(1).norm(dim=1)
    if torch.is_tensor(na):
        na, nb = na.numpy(), nb.numpy()
    return (num.numpy() / np.maximum(na * nb, 1e-30))


def principal_angle_max(Va, Vb):
    """Max principal angle [n, D, r] x [n, D, r] -> [n] (рад)."""
    M = Va.transpose(-1, -2) @ Vb
    s = torch.linalg.svdvals(M).clamp(max=1.0 - 1e-12)
    return torch.acos(s).max(-1).values.numpy()


def doc_curves(J, Vtop, sent, deltas, Jn):
    """C(Δ) и C_top-r(Δ) одного документа: per-Δ mean по валидным парам.
    J [S,K,D], Vtop [S,D,r], sent [S] bool, Jn [S], Jn [S] (нормы ||J_t||_F)."""
    T = J.shape[0]
    C = np.full(len(deltas), np.nan)
    Ctop = np.full(len(deltas), np.nan)
    for di, d in enumerate(deltas):
        v = ~sent[:T - d] & ~sent[d:]
        if not v.any():
            continue
        C[di] = frob_cos(J[:T - d][v], J[d:][v], Jn[:T - d][v], Jn[d:][v]).mean()
        Ctop[di] = np.cos(principal_angle_max(Vtop[:T - d][v],
                                              Vtop[d:][v])).mean()
    return C, Ctop


def doc_map(J, sent, d=DELTA_MAP, Jn=None):
    """C(t, Δ=16) per t (NaN где пара невалидна)."""
    T = J.shape[0]
    out = np.full(T - d, np.nan)
    v = ~sent[:T - d] & ~sent[d:]
    if v.any():
        out[v] = frob_cos(J[:T - d][v], J[d:][v],
                          Jn[:T - d][v] if Jn is not None else None,
                          Jn[d:][v] if Jn is not None else None)
    return out


def seg_split(J, sent, bounds_t, d, Jn):
    """C(Δ) intra vs inter для одного Δ: bounds_t — границы в кэш-t.
    Пара (t, t+Δ) intra, если на (t, t+Δ) нет границы.
    -> (C_intra, n_intra, C_inter, n_inter)."""
    T = J.shape[0]
    v = ~sent[:T - d] & ~sent[d:]
    if not v.any():
        return np.nan, 0, np.nan, 0
    cos = frob_cos(J[:T - d][v], J[d:][v], Jn[:T - d][v], Jn[d:][v])
    tv = np.where(v)[0]
    inb = np.zeros(T, bool)
    for b in bounds_t:
        if 0 < b < T:
            inb[b] = True
    # пара (t, t+d) пересекает границу m при t+1 <= m <= t+d
    through = inb[tv[:, None] + np.arange(1, d + 1)].any(axis=1)
    return (float(cos[~through].mean()) if (~through).any() else np.nan,
            int((~through).sum()),
            float(cos[through].mean()) if through.any() else np.nan,
            int(through.sum()))


def fit_decay(deltas, c):
    """Экспоненциальный (ξ) vs степенной (α) фит на медианной кривой.
    AIC = n*ln(RSS/n) + 2k. Только точки с c > 0."""
    ok = (c > 0) & np.isfinite(c)
    d = np.array(deltas, dtype=float)[ok]
    y = np.log(c[ok])
    out = {'n_pts': int(ok.sum())}
    if len(d) < 4:
        out['note'] = 'not enough positive points'
        return out
    Xe = np.column_stack([np.ones_like(d), d])
    be, res_e, *_ = np.linalg.lstsq(Xe, y, rcond=None)
    rss_e = float(res_e @ res_e) if res_e.size else float(((y - Xe @ be) ** 2).sum())
    Xp = np.column_stack([np.ones_like(d), np.log(d)])
    bp, res_p, *_ = np.linalg.lstsq(Xp, y, rcond=None)
    rss_p = float(res_p @ res_p) if res_p.size else float(((y - Xp @ bp) ** 2).sum())
    n = len(d)
    sst = float(((y - y.mean()) ** 2).sum())
    aic_e = n * np.log(max(rss_e, 1e-300) / n) + 4
    aic_p = n * np.log(max(rss_p, 1e-300) / n) + 4
    out.update({
        'xi_exp': float(-1.0 / be[1]) if be[1] < 0 else None,
        'alpha_power': float(-bp[1]),
        'aic_exp': float(aic_e), 'aic_power': float(aic_p),
        'r2_exp': float(1 - rss_e / sst) if sst > 0 else None,
        'r2_power': float(1 - rss_p / sst) if sst > 0 else None,
        'shape': 'exponential' if aic_e < aic_p else 'power',
    })
    return out


def dP_peaks(P, sent):
    """dP(t) = ||p_{t+1}-p_t|| (P128 normalized), MAD-пиковый детектор E-28.
    Пик на t — переход (t -> t+1) в кэш-t."""
    T = P.shape[0]
    dp = (P[1:] - P[:-1]).norm(dim=1).numpy()
    v = ~sent[:-1] & ~sent[1:]
    dn = dp[v]
    med = float(np.median(dn))
    mad = float(np.median(np.abs(dn - med)))
    thr = med + max(2.0 * 1.4826 * mad, 1e-6)
    peaks = set(int(t) for t in np.where(v & (dp > thr))[0])
    return dp, peaks, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt-dir', default=CKPT_DIR)
    ap.add_argument('--out', default=OUT_JSON)
    ap.add_argument('--pool', default=OUT_POOL)
    ap.add_argument('--rebuild-pool', action='store_true',
                    help='пересобрать пул + offset_mapping (нужен HF)')
    ap.add_argument('--layers', type=int, nargs='+', default=LAYERS)
    ap.add_argument('--deltas', type=int, nargs='+',
                    default=list(range(1, DELTA_MAX + 1)))
    ap.add_argument('--selfcheck', action='store_true')
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()

    ck0 = torch.load(os.path.join(a.ckpt_dir, 'exp27b_jcache_L22.pt'),
                     map_location='cpu', weights_only=False)
    if a.rebuild_pool or not os.path.exists(a.pool):
        print('rebuilding pool with offset_mapping (HF streaming)...')
        pool = rebuild_pool(MODEL)
        fp = pool_fp(pool, ck0)
        print('pool fingerprint vs cache:', fp)
        assert all(fp.values()), f'pool mismatch: {fp}'
        os.makedirs(os.path.dirname(a.pool) or '.', exist_ok=True)
        torch.save(pool, a.pool)
        print(f'wrote {a.pool}')
    else:
        pool = torch.load(a.pool, map_location='cpu', weights_only=False)
        fp = pool_fp(pool, ck0)
        assert all(fp.values()), f'pool fingerprint mismatch: {fp}'

    # границы предложений (token-idx, 0 = первый токен последовательности)
    bounds_tok, n_bounds = {}, {}
    for (dom, si), item in pool.items():
        spans = sentence_spans(dom, item['text'])
        b = _spans_to_tokens(spans, item['offset_mapping'])
        bounds_tok[(dom, si)] = b
        n_bounds[dom] = n_bounds.get(dom, 0) + len(b)
    print('n sentence boundaries per domain:', n_bounds)
    for d in DOMAINS:
        print(f'  {d} sample 0:', bounds_tok[(d, 0)][:12])

    # pre-flight: dP пики vs границы (P128 слой-инвариантен, берём L22;
    # совпадение с L23 проверяется явно)
    ck23 = torch.load(os.path.join(a.ckpt_dir, 'exp27b_jcache_L23.pt'),
                      map_location='cpu', weights_only=False)
    same_p = all(torch.equal(ck0['P128'][k], ck23['P128'][k])
                 for k in list(ck0['P128'])[:16])
    print(f'P128 L22==L23 (16 probe keys): {same_p}')
    pf = {}
    for d in ck0['domains']:
        n_near = n_peak = 0
        for si in range(ck0['n_samples']):
            P = torch.stack([ck0['P128'][(d, si, t)] for t in range(ck0['S'])])
            Pr = torch.stack([ck0['P128_RAW'][(d, si, t)]
                              for t in range(ck0['S'])])
            sent = (Pr.sum(-1) < SENTINEL_SUM).numpy()
            _, peaks, _ = dP_peaks(P, sent)
            near = set()
            for j in bounds_tok[(d, si)]:
                bt = S - 1 - j  # граница token-j -> кэш-t; переход bt-1
                for dt in (-2, -1, 0, 1, 2):
                    near.add(bt - 1 + dt)
            n_near += len(peaks & near)
            n_peak += len(peaks)
        pf[d] = {'n_peaks': int(n_peak), 'n_peaks_near_boundary': int(n_near),
                 'frac_peaks_near_boundary': n_near / max(n_peak, 1)}
    print('pre-flight dP peaks vs sentence boundaries:',
          {d: f"{v['frac_peaks_near_boundary']:.2f} "
              f"({v['n_peaks_near_boundary']}/{v['n_peaks']})"
           for d, v in pf.items()})

    deltas = sorted(a.deltas)
    rep = {
        'experiment': 'E-31 B9: C(dpos) coherence length',
        'layers': a.layers, 'S': S, 'K': K, 'r_top': R_TOP,
        'preflight_p128_l22_eq_l23': bool(same_p),
        'deltas': deltas, 'delta_map': DELTA_MAP,
        'convention': 't=0 last token, token_idx=S-1-t; dpos in cache-t; '
                      'seq_pos=S-1-t for boundaries',
        'sentinel': f'P128_RAW sum < {SENTINEL_SUM}',
        'aggregation': 'per-doc mean over valid pairs -> median/IQR over '
                       '32 docs (E-34 protocol)',
        'c_metric': 'C_F = <J_t,J_{t+d}>_F/(||J_t|| ||J_{t+d}||) full matrix; '
                    'C_top-r = cos(max principal angle), top-r right subspaces',
        'pool_fp': pool_fp(pool, ck0),
        'preflight_dP_vs_boundaries': pf,
        'n_bounds_per_domain': n_bounds,
    }
    for l in a.layers:
        ck = ck0 if l == 22 else torch.load(
            os.path.join(a.ckpt_dir, f'exp27b_jcache_L{l}.pt'),
            map_location='cpu', weights_only=False)
        docs = []
        for dom in ck['domains']:
            for si in range(ck['n_samples']):
                J = torch.stack([ck['JR'][(dom, si, t)] for t in range(S)])
                Jn = J.flatten(1).norm(dim=1).numpy()
                Pr = torch.stack([ck['P128_RAW'][(dom, si, t)]
                                  for t in range(S)])
                sent = (Pr.sum(-1) < SENTINEL_SUM).numpy()
                Vtop = vtop16(J)
                C, Ctop = doc_curves(J, Vtop, sent, deltas, Jn)
                m = doc_map(J, sent, Jn=Jn)
                bt = [S - 1 - j for j in bounds_tok[(dom, si)]
                      if 0 < S - 1 - j < S]
                intra, inter, ni, ne = [], [], [], []
                for d in deltas:
                    ci, n_i, ce, n_e = seg_split(J, sent, bt, d, Jn)
                    intra.append(ci)
                    inter.append(ce)
                    ni.append(n_i)
                    ne.append(n_e)
                sent_pair = [float((sent[:S - d] | sent[d:]).sum() / (S - d))
                             for d in deltas]
                docs.append({'dom': dom, 'si': si, 'C': C, 'C_top': Ctop,
                             'map16': m, 'intra': intra, 'inter': inter,
                             'n_intra': ni, 'n_inter': ne,
                             'sent_pair_frac': sent_pair,
                             'n_sent_pos': int(sent.sum())})
                del J, Vtop
        rep[str(l)] = _aggregate(docs, deltas)
        Lr = rep[str(l)]
        print(f"L{l}: C(1)={Lr['C_med'][0]:.3f} C(32)={Lr['C_med'][-1]:.3f} "
              f"Ctop(1)={Lr['C_top_med'][0]:.3f} xi={Lr['fit'].get('xi_exp')} "
              f"shape={Lr['fit'].get('shape')} breaks={Lr['map16_breakpoints']}")
    rep['source'] = 'experiments/exp31_correlation.py'
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(_clean(rep), f, indent=2, ensure_ascii=False)
    print(f'wrote {a.out}')


def _aggregate(docs, deltas):
    C = np.array([d['C'] for d in docs])
    Ct = np.array([d['C_top'] for d in docs])
    maps = np.array([d['map16'] for d in docs])
    intr = np.array([d['intra'] for d in docs])
    inter = np.array([d['inter'] for d in docs])
    spf = np.array([d['sent_pair_frac'] for d in docs])
    med = lambda x: np.nanmedian(x, axis=0)
    q = lambda x: (np.nanpercentile(x, 25, axis=0), np.nanpercentile(x, 75, axis=0))
    c1, c3 = q(C)
    tc1, tc3 = q(Ct)
    mc1, mc3 = q(maps)
    ic1, ic3 = q(intr)
    ec1, ec3 = q(inter)
    fit = fit_decay(deltas, med(C))
    # разрывы на карте: позиции, где C(t,16) << медиана (MAD снизу, 3σ)
    mm = med(maps)
    dn = mm[np.isfinite(mm)]
    mad = float(np.median(np.abs(dn - np.median(dn))))
    low = np.where(np.isfinite(mm) &
                   (mm < np.median(dn) - max(3.0 * 1.4826 * mad, 1e-6)))[0]
    return {
        'C_med': med(C).tolist(), 'C_q1': c1.tolist(), 'C_q3': c3.tolist(),
        'C_top_med': med(Ct).tolist(), 'C_top_q1': tc1.tolist(),
        'C_top_q3': tc3.tolist(),
        'map16_med': med(maps).tolist(), 'map16_q1': mc1.tolist(),
        'map16_q3': mc3.tolist(),
        'map16_breakpoints': low.tolist(),
        'intra_med': med(intr).tolist(), 'intra_q1': ic1.tolist(),
        'intra_q3': ic3.tolist(),
        'inter_med': med(inter).tolist(), 'inter_q1': ec1.tolist(),
        'inter_q3': ec3.tolist(),
        'n_intra_med': [float(np.median(x)) for x in
                        [[d['n_intra'][i] for d in docs] for i in range(len(deltas))]],
        'n_inter_med': [float(np.median(x)) for x in
                        [[d['n_inter'][i] for d in docs] for i in range(len(deltas))]],
        'sent_pair_frac_med': med(spf).tolist(),
        'n_sent_pos_med': float(np.median([d['n_sent_pos'] for d in docs])),
        'fit': fit,
        'n_docs': len(docs),
    }


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)) and not np.isfinite(o):
        return None
    if isinstance(o, np.integer):
        return int(o)
    return o


# ---------------- selfcheck ----------------

def _fake_cache(S_=16, K_=4, D_=8, n=1, seed=0):
    g = np.random.default_rng(seed)
    ck = {'JR': {}, 'P128': {}, 'P128_RAW': {}, 'domains': ['wiki'],
          'n_samples': n, 'S': S_, 'D': D_, 'U128': list(range(K_)),
          'layers': [23]}
    Jbase = torch.from_numpy(g.standard_normal((S_, K_, D_)).astype(np.float32))
    for si in range(n):
        for t in range(S_):
            ck['JR'][('wiki', si, t)] = Jbase[t]
            p = torch.from_numpy(np.full(K_, 1.0 / K_, np.float32))
            ck['P128'][('wiki', si, t)] = p
            ck['P128_RAW'][('wiki', si, t)] = p.clone()
    return ck, Jbase


def selfcheck():
    deltas = [1, 2, 4, 8]
    # 1) постоянное поле (ранг 1 -> топ-1 подпространство определено) ->
    #    C = 1, C_top = 1
    ck, Jb = _fake_cache()
    J = Jb[0:1].expand(16, 4, 8).contiguous()
    sent = np.zeros(16, bool)
    Jn = J.flatten(1).norm(dim=1).numpy()
    Vtop = vtop16(J, r=1)
    C, Ct = doc_curves(J, Vtop, sent, deltas, Jn)
    assert np.allclose(C, 1.0, atol=1e-5), C
    assert np.allclose(Ct, 1.0, atol=1e-5), Ct
    # 1b) vtop16 эквивалентен SVD-топу (на случайном полноранговом поле)
    Vtop16 = vtop16(Jb, r=2)
    Vr = torch.linalg.svd(Jb, full_matrices=False)[2][:, :2].transpose(-1, -2)
    for t in range(0, 16, 5):
        ang = float(torch.acos(torch.linalg.svdvals(
            Vtop16[t].t() @ Vr[t]).clamp(max=1 - 1e-12)).max())
        assert ang < 1e-3, ang  # float32, крошечная 4x8
    # 2) sentinel: обнулённый конец исключает пару (поле константное -> 1.0)
    sent2 = sent.copy()
    sent2[5] = True
    C2, _ = doc_curves(J, Vtop, sent2, [1], Jn)
    assert abs(C2[0] - 1.0) < 1e-5
    # 3) map: NaN только там, где пара (t, t+2) касается sentinel t=5:
    #    (3,5) и (5,7) невалидны; (4,6) — валидна (endpoint-only конвенция)
    m2 = doc_map(J, sent2, d=2, Jn=Jn)
    assert np.isnan(m2[3]) and np.isfinite(m2[4]) and np.isnan(m2[5])
    assert np.isfinite(m2[0])
    # 4) fit: экспоненциальная -> exponential, xi ~ 5; степенная -> power,
    #    alpha ~ 0.7
    d = np.arange(1, 33)
    f = fit_decay(d.tolist(), np.exp(-d / 5.0))
    assert f['shape'] == 'exponential' and abs(f['xi_exp'] - 5.0) < 0.5, f
    fp = fit_decay(d.tolist(), d ** (-0.7))
    assert fp['shape'] == 'power' and abs(fp['alpha_power'] - 0.7) < 0.05, fp
    # 5) seg_split: граница bt=4; Δ=2: пары (2,4),(3,5) пересекают bt=4
    #    -> 12 intra + 2 inter из 14
    ci, ni, ce, ne = seg_split(J, np.zeros(16, bool), [4], 2, Jn)
    assert ni == 12 and ne == 2, (ni, ne)
    # 6) _spans_to_tokens
    om = np.array([(0, 3), (3, 6), (6, 9), (9, 12), (12, 15)])
    assert _spans_to_tokens([(0, 6), (6, 15)], om) == [2]
    # 7) sentence_spans: math — $-формулы как единицы; здесь 2 формулы +
    # 3 межформульных куска
    s = sentence_spans('math', 'text here $(x+1)$ more $[a=b]$ end')
    assert len(s) == 5, s
    # 8) dP_peaks: один выброс -> пики на переходах (4->5) и (5->6)
    P = torch.from_numpy(np.stack([np.full(4, 0.25) for _ in range(10)]
                                  ).astype(np.float32))
    P[5] = torch.from_numpy(np.array([0.4, 0.4, 0.1, 0.1], np.float32))
    dp, peaks, thr = dP_peaks(P, np.zeros(10, bool))
    assert {4, 5} <= peaks, (peaks, thr)
    # 9) token_spans: spans покрывают текст без дыр, sum == len
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The quick brown fox. Jumps over the lazy dog! "
              "\u00e9\u4f60\u00e7", add_special_tokens=False, 
             return_offsets_mapping=False)["input_ids"]
    text = tok.decode(ids, skip_special_tokens=False)
    spans = token_spans(tok, list(ids))
    assert spans is not None
    c = 0
    for sp in spans:
        assert text[c:c + len(sp)] == sp, (c, sp)
        c += len(sp)
    assert c == len(text)
    print('selfcheck OK')


if __name__ == '__main__':
    main()
