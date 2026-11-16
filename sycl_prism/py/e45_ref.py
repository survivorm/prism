"""numpy-эталон формул E-45 (base_pre / dense_metrics / sparse-путь) + selftest.

Зачем. Мой первый torch-зонд был отдан без проверки и упал на первой же строке
(`tok_ids.shape[1]`), причём семантику U128 я переписал по памяти, а не по коду.
Теперь источник истины — код репозитория:
  experiments/exp45_damage_vs_structure.py:149-216, 243-258, 327-368
  experiments/exp45c_gauge_null.py:9-16
Эталон используется в двух ролях:
  1) как источник истины для `probe/probe_xpu_pipeline.py --selftest`
     (torch-путь обязан совпасть с ним на маленькой форме);
  2) как самостоятельная проверка: здесь же перенесён selftest репозитория
     (sparse-формулы против грубого softmax, допуск 1e-9) и аналитический
     контроль E-45c (KL = TV = flip = 0 при W → W + 1·cᵗ).

Запуск:  python3 e45_ref.py [--selftest]
"""
from __future__ import annotations

import numpy as np

K_TOK = 128          # |U128| — как в exp45 (константа там: K_TOK = 128)
CH = 128             # позиций в чанке — как в exp45 (CH = 128)


def logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def softmax(x, axis=-1):
    e = np.exp(x - logsumexp(x, axis=axis)[:, None])
    return e / e.sum(axis=1)[:, None]


def tok_col_map(tk: np.ndarray, V: int) -> np.ndarray:
    """TOK_COL: [V] → номер столбца U128 либо -1 (exp45:339-340)."""
    col = np.full(V, -1, dtype=np.int64)
    col[tk] = np.arange(len(tk), dtype=np.int64)
    return col


def base_pre(Hq, W_base, tk, dtype=np.float64):
    """Эталон exp45:188-215.

    Hq: (N, D), W_base: (V, D), tk: (K_TOK,) int — общий набор U128 для всех
    позиций. Возвращает b_k (N,K_TOK) fp64, lse (N,), mx_nonK (N,), argK (N,) —
    номер столбца U128 для top-1 либо -1.

    dtype=np.float64 (по умолчанию) — ТОЧНЫЙ эталон: сводки считаются по
    fp64-произведению. dtype=np.float32 — тот же путь, что и в E-45 (fp32 GEMM,
    fp64 накопление); нужен, чтобы измерить расхождение fp32-пути с точным
    значением. **Сравнивать fp32-путь с fp32-эталоном нельзя**: разные BLAS
    дают расхождение до ~1e-7 по логитам (проверено: 4.47e-08 на torch
    2.13+cu130 против numpy).
    """
    Hq = np.asarray(Hq, dtype=dtype)
    W_base = np.asarray(W_base, dtype=dtype)
    tk = np.asarray(tk, dtype=np.int64)
    N = Hq.shape[0]
    V = W_base.shape[0]
    col = tok_col_map(tk, V)

    b_k = np.empty((N, len(tk)), dtype=np.float64)
    lse = np.empty(N, dtype=np.float64)
    mx_nonK = np.empty(N, dtype=np.float64)
    argK = np.empty(N, dtype=np.int64)

    for c0 in range(0, N, CH):
        c1 = min(c0 + CH, N)
        lb = (Hq[c0:c1] @ W_base.T).astype(np.float64)   # GEMM в dtype, накопление fp64
        lse[c0:c1] = logsumexp(lb, 1)
        b_k[c0:c1] = lb[:, tk]
        ik = lb.argmax(1)
        lbm = lb.copy()
        lbm[:, tk] = -np.inf
        mx_nonK[c0:c1] = lbm.max(1)
        c = col[ik]
        argK[c0:c1] = np.where(c >= 0, c, -1)
    return b_k, lse, mx_nonK, argK


def dense_metrics(Hq, W_edit, W_base, dtype=np.float64):
    """Эталон exp45:218-241 (плотный путь; сводки — fp64)."""
    Hq = np.asarray(Hq, dtype=dtype)
    lb = (Hq @ np.asarray(W_base, dtype=dtype).T).astype(np.float64)
    le = (Hq @ np.asarray(W_edit, dtype=dtype).T).astype(np.float64)
    lb_e, le_e = logsumexp(lb, 1), logsumexp(le, 1)
    lpb, lpe = lb - lb_e[:, None], le - le_e[:, None]
    pb, pe = np.exp(lpb), np.exp(lpe)
    kl_be = np.clip((pb * (lpb - lpe)).sum(1), 0.0, None)
    kl_eb = np.clip((pe * (lpe - lpb)).sum(1), 0.0, None)
    tv = 0.5 * (np.abs(pb - pe)).sum(1)
    flip = (lb.argmax(1) != le.argmax(1)).astype(np.float64)
    return {"kl_be": kl_be, "kl_eb": kl_eb, "tv": tv, "flip": flip}


# ── разреженный путь (exp45:149-186) ────────────────────────────────────────
def lse_from_sparse(b_k, e_k, lse_b):
    m = np.max(np.concatenate([lse_b[:, None], e_k, b_k], axis=1), axis=1)
    return (m + np.log(np.exp(lse_b - m)
                       - np.exp(b_k - m[:, None]).sum(1)
                       + np.exp(e_k - m[:, None]).sum(1)))


def sparse_terms(b_k, e_k, lse_b, lse_e):
    db = lse_b - lse_e
    pb = np.exp(b_k - lse_b[:, None])
    pe = np.exp(e_k - lse_e[:, None])
    dk = e_k - b_k
    kl_be = -(pb * dk).sum(1) - db
    kl_eb = (pe * dk).sum(1) + db
    tv = 0.5 * ((np.abs(pe - pb)).sum(1)
                + (1.0 - pb.sum(1)) * np.abs(np.exp(db) - 1.0))
    return np.clip(kl_be, 0.0, None), np.clip(kl_eb, 0.0, None), np.clip(tv, 0.0, None)


def sparse_flip(e_k, b_max_nonK, argK):
    """exp45:175-186: смена top-1 при правке только K столбцов."""
    m_e = e_k.max(1)          # torch: e_k.max(1) -> (values, indices)
    j_e = e_k.argmax(1)
    new_in_K = m_e > b_max_nonK
    was_in_K = argK >= 0
    kept = np.where(new_in_K, argK == j_e, ~was_in_K)
    return (~kept).astype(np.float64)


# ── selftest ────────────────────────────────────────────────────────────────
def selftest(verbose: bool = True) -> dict:
    """Порт selftest репозитория (exp45:327-368) + контроль E-45c."""
    out = {}
    rng = np.random.default_rng(0)
    m, V, K = 40, 700, K_TOK
    lb = rng.standard_normal((m, V)) * 3.0
    tk = rng.permutation(V)[:K]
    col = tok_col_map(tk, V)

    b_k, lse_b = lb[:, tk], logsumexp(lb, 1)
    ik = lb.argmax(1)
    lbm = lb.copy()
    lbm[:, tk] = -np.inf
    mxn = lbm.max(1)
    argK = np.where(col[ik] >= 0, col[ik], -1)

    for scale, tag in ((2.0, "слабая"), (15.0, "сильная")):
        eb = lb.copy()
        eb[:, tk] += rng.standard_normal((m, K)) * scale
        e_k = eb[:, tk]
        lse_e = lse_from_sparse(b_k, e_k, lse_b)
        d_lse = float(np.abs(lse_e - logsumexp(eb, 1)).max())
        assert d_lse < 1e-10, f"lse, {tag}: {d_lse:.2e}"
        kl_be, kl_eb, tv = sparse_terms(b_k, e_k, lse_b, lse_e)
        pb, pe = softmax(lb, 1), softmax(eb, 1)
        refs = {
            "kl_be": (pb * np.log(pb / pe)).sum(1),
            "kl_eb": (pe * np.log(pe / pb)).sum(1),
            "tv": 0.5 * np.abs(pb - pe).sum(1),
        }
        for nm, a, r in (("kl_be", kl_be, refs["kl_be"]),
                         ("kl_eb", kl_eb, refs["kl_eb"]),
                         ("tv", tv, refs["tv"])):
            d = float(np.abs(a - r).max())
            assert d < 1e-9, f"{tag} {nm}: {d:.2e}"
            out[f"{tag}_{nm}_maxdiff"] = d
            if verbose:
                print(f"  [selftest {tag}] {nm}: max|Δ| = {d:.2e}")
        fl = sparse_flip(e_k, mxn, argK)
        d = float(np.abs(fl - (eb.argmax(1) != ik)).max())
        assert d == 0.0, f"{tag} flip: {d}"
        out[f"{tag}_flip_maxdiff"] = d
        if verbose:
            print(f"  [selftest {tag}] flip: max|Δ| = {d:.2e}")

    # ── контроль E-45c: W → W + 1·cᵗ ⇒ softmax инвариантен ⇒ KL = TV = flip = 0
    rng2 = np.random.default_rng(1)
    N, D, V2 = 64, 32, 512
    Hq = (rng2.standard_normal((N, D)) * 0.5).astype(np.float32)
    W = (rng2.standard_normal((V2, D)) * 0.02).astype(np.float32)
    c = (rng2.standard_normal(D) * 0.1).astype(np.float32)
    tk2 = rng2.permutation(V2)[:K_TOK].astype(np.int64)
    W_edit = (W + c[None, :]).astype(np.float32)
    mtr = dense_metrics(Hq, W_edit, W)
    out["gauge_max_kl"] = float(mtr["kl_be"].max())
    out["gauge_max_tv"] = float(mtr["tv"].max())
    out["gauge_flip_sum"] = float(mtr["flip"].sum())
    assert out["gauge_max_kl"] < 1e-4, out["gauge_max_kl"]
    assert out["gauge_max_tv"] < 1e-4, out["gauge_max_tv"]
    assert out["gauge_flip_sum"] == 0.0, out["gauge_flip_sum"]

    # base_pre против прямого softmax: lse и mx_nonK
    b_k2, lse2, mxn2, argK2 = base_pre(Hq, W, tk2, dtype=np.float64)
    lb2 = (Hq.astype(np.float64) @ W.astype(np.float64).T)
    out["base_pre_lse_maxdiff"] = float(np.abs(lse2 - logsumexp(lb2, 1)).max())
    lbm2 = lb2.copy()
    lbm2[:, tk2] = -np.inf
    out["base_pre_mxnonk_maxdiff"] = float(np.abs(mxn2 - lbm2.max(1)).max())
    out["base_pre_bk_maxdiff"] = float(np.abs(b_k2 - lb2[:, tk2]).max())
    assert out["base_pre_lse_maxdiff"] < 1e-10
    assert out["base_pre_mxnonk_maxdiff"] < 1e-10
    assert out["base_pre_bk_maxdiff"] < 1e-10

    # sanity: argK указывает на столбец U128 именно того токена, который есть top-1
    ik2 = lb2.argmax(1)
    for i in range(N):
        if argK2[i] >= 0:
            assert tk2[argK2[i]] == ik2[i], f"argK mismatch в строке {i}"
    out["argK_consistent"] = True

    if verbose:
        print("  [контроль E-45c] maxKL=%.3e maxTV=%.3e flip=%.0f"
              % (out["gauge_max_kl"], out["gauge_max_tv"], out["gauge_flip_sum"]))
        print("selftest формул E-45 (numpy-эталон): OK")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    res = selftest(verbose=True)
    for k, v in res.items():
        print(f"  {k} = {v}")
