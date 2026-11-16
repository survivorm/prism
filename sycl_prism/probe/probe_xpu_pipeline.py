"""Зонд переноса E-45 на XPU: dense-путь (base_pre + dense_metrics) и sparse-путь.

Источник истины — код репозитория, а не память:
  experiments/exp45_damage_vs_structure.py:149-216 (base_pre), 218-241 (dense_metrics),
      243-258 (sparse_metrics), 327-368 (selftest формул)
  experiments/exp45c_gauge_null.py:9-16   (аналитическая свобода W → W + 1·cᵗ)
Семантика U128 (важно, в первой версии зонда она была переписана неверно):
  tk       — (K_TOK,) общий набор из 128 id (не per-position!);
  TOK_COL  — карта [V] → номер столбца U128, либо -1;
  argK     — номер столбца U128, в котором лежит ТОП-1 (vocab-id!), иначе −1.
             (Ошибочная версия была `argK = ik < K_TOK` — сравнение vocab-id с 128.)

Самопроверка (обязательна, `--selftest`): torch-путь сверяется с numpy-эталоном
`py/e45_ref.py` на одинаковых входах; эталон, в свою очередь, проходит портированный
selftest репозитория (sparse-формулы против грубого softmax, допуск 1e-9) и аналитический
контроль E-45c (KL = TV = flip = 0).

Запуск (из корня репозитория, venv репозитория):
    ../.venv/bin/python ./probe/probe_xpu_pipeline.py --selftest
    ../.venv/bin/python ./probe/probe_xpu_pipeline.py --V 248000 --N 4096

Про окружение: `torch.xpu` существует ТОЛЬКО в XPU-сборке torch. Обычная CUDA-сборка
(например +cu130) модуля `torch.xpu` не содержит — тогда зонд честно печатает, чего
не хватает, и работает как CPU-база. Установка XPU-сборки:
    pip install torch --index-url https://download.pytorch.org/whl/xpu
Альтернатива, не требующая torch.xpu: собственный SYCL-прибор (C ABI + ctypes,
`../src/prism_sycl.cpp`, сборка `icpx -fsycl`) — он работает с ЛЮБЫМ torch.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time

import numpy as np
import torch

K_TOK = 128      # |U128| — как в exp45
CH = 128         # позиций в чанке dense-пути — как в exp45
SP_CH = 4 * CH   # чанк разреженного пути (нет [m,V]) — как в exp45

_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "py", "e45_ref.py")


def _load_ref():
    """numpy-эталон лежит рядом: sycl_prism/py/e45_ref.py (часть архива)."""
    path = os.path.abspath(_REF_PATH)
    if not os.path.exists(path):
        raise SystemExit(
            f"НЕ НАЙДЕН эталон: {path}\n"
            "Это не ошибка зонда, а неполная копия архива: положите рядом\n"
            "  sycl_prism/py/e45_ref.py  (и tests/, run_local.sh).\n"
            "Быстрее всего: unzip -o sycl_prism.zip -d <репозиторий>/\n"
            "Проверить, что копия свежая: sha256sum "
            "sycl_prism/probe/probe_xpu_pipeline.py"
        )
    spec = importlib.util.spec_from_file_location("e45_ref", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def self_sha256() -> str:
    """Отпечаток самого скрипта — сразу видно, что запущена устаревшая копия."""
    import hashlib

    try:
        with open(os.path.abspath(__file__), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return "?"


def _sync(dev: str) -> None:
    if dev == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


# ──────────────────────────── torch-реализация (по коду exp45) ────────────────────────────
def tok_col_map(tk: torch.Tensor, V: int) -> torch.Tensor:
    """exp45:339-340 — [V] → номер столбца U128 либо −1."""
    col = torch.full((V,), -1, dtype=torch.long)
    col[tk] = torch.arange(len(tk), dtype=torch.long)
    return col


def base_pre(Hq, W_base, tk, col=None, v_chunk=None):
    """exp45:188-215: логиги U128, lse, лучший логик вне U128, столбец top-1 внутри U128.

    Hq: (N, D) fp32; W_base: (V, D) fp32; tk: (K_TOK,) long — общий набор U128.

    v_chunk: если задан, словарь режется на куски (память ~ N×v_chunk×8 вместо
    N×V×8). На форме E-45 (V=248k) это единственный способ уложиться в 3-4 ГБ RAM:
    сам W уже занимает V×D×4 = 1 ГБ, а копия W_edit — ещё 1 ГБ (её здесь нет,
    калибровочный сдвиг добавляется к логитам, см. dense_metrics(shift=...)).

    Чанковый режим меняет только ПОРЯДОК СУММИРОВАНИЯ (logaddexp по кускам
    словаря и отдельный GEMM для 128 строк U128) ⇒ значения совпадают с
    целочисленным вариантом с точностью fp32-GEMM (~1e-7 по логитам).
    """
    N = Hq.shape[0]
    V = W_base.shape[0]
    if col is None:
        col = tok_col_map(tk, V)
    b_k = torch.empty(N, len(tk), dtype=torch.float64, device=Hq.device)
    lse = torch.empty(N, dtype=torch.float64, device=Hq.device)
    mx_nonK = torch.empty(N, dtype=torch.float64, device=Hq.device)
    argK = torch.empty(N, dtype=torch.long, device=Hq.device)

    if not v_chunk or v_chunk >= V:
        for c0 in range(0, N, CH):
            c1 = min(c0 + CH, N)
            lb = (Hq[c0:c1] @ W_base.t()).double()
            lse[c0:c1] = torch.logsumexp(lb, 1)
            b_k[c0:c1] = lb[:, tk]
            _, ik = lb.max(1)
            lbm = lb.clone()
            lbm[:, tk] = float("-inf")
            mx_nonK[c0:c1] = lbm.max(1).values
            c = col[ik]
            argK[c0:c1] = torch.where(c >= 0, c, torch.full_like(c, -1))
            del lb, lbm, ik, c
        return b_k, lse, mx_nonK, argK

    # ── чанковый по словарю режим ──
    lse_all = torch.full((N,), float("-inf"), dtype=torch.float64, device=Hq.device)
    mx_all = torch.full((N,), float("-inf"), dtype=torch.float64, device=Hq.device)
    best_val = torch.full((N,), float("-inf"), dtype=torch.float64, device=Hq.device)
    best_idx = torch.zeros(N, dtype=torch.long, device=Hq.device)
    for v0 in range(0, V, v_chunk):
        v1 = min(v0 + v_chunk, V)
        Wc = W_base[v0:v1]
        local = tk[(tk >= v0) & (tk < v1)] - v0
        for c0 in range(0, N, CH):
            c1 = min(c0 + CH, N)
            lb = (Hq[c0:c1] @ Wc.t()).double()          # (CH, v1-v0)
            lse_all[c0:c1] = torch.logaddexp(
                lse_all[c0:c1], torch.logsumexp(lb, 1))
            lbm = lb.clone()
            if local.numel():
                lbm[:, local] = float("-inf")
            mx_all[c0:c1] = torch.maximum(mx_all[c0:c1], lbm.max(1).values)
            vmax, vidx = lb.max(1)
            upd = vmax > best_val[c0:c1]
            best_val[c0:c1] = torch.where(upd, vmax, best_val[c0:c1])
            best_idx[c0:c1] = torch.where(upd, vidx + v0, best_idx[c0:c1])
            del lb, lbm, vmax, vidx
    lse[:] = lse_all
    mx_nonK[:] = mx_all
    c = col[best_idx]
    argK[:] = torch.where(c >= 0, c, torch.full_like(c, -1))
    b_k[:] = (Hq @ W_base[tk].t()).double()
    return b_k, lse, mx_nonK, argK


def dense_metrics(Hq, W_edit, W_base, shift=None):
    """exp45:218-241 — KL/TV/flip по всем позициям; сводки в fp64.

    shift: (N,) — добавляется к логитам отредактированной матрицы. Эквивалентно
    замене W → W + 1·cᵗ при shift = Hq @ c, но БЕЗ копии W (экономия 1 ГБ при
    V=248k, D=1024) и без второго GEMM.
    """
    acc = {k: [] for k in ("kl_be", "kl_eb", "tv", "flip")}
    N = Hq.shape[0]
    for c0 in range(0, N, CH):
        c1 = min(c0 + CH, N)
        lb = Hq[c0:c1] @ W_base.t()
        le = Hq[c0:c1] @ W_edit.t() if shift is None else lb + shift[c0:c1, None]
        lb_e, le_e = torch.logsumexp(lb, 1), torch.logsumexp(le, 1)
        lpb, lpe = lb - lb_e[:, None], le - le_e[:, None]
        pb, pe = torch.exp(lpb), torch.exp(lpe)
        acc["kl_be"].append((pb * (lpb - lpe)).sum(1, dtype=torch.float64).clamp_min(0.0))
        acc["kl_eb"].append((pe * (lpe - lpb)).sum(1, dtype=torch.float64).clamp_min(0.0))
        acc["tv"].append((0.5 * (pb - pe).abs()).sum(1, dtype=torch.float64))
        acc["flip"].append((lb.argmax(1) != le.argmax(1)).double())
        del lb, le, pb, pe, lpb, lpe
    return {k: torch.cat(v) for k, v in acc.items()}


def sparse_metrics(Hq, W_rows_new, pre):
    """exp45:243-258 — метрики при правке ТОЛЬКО U128-строк (без [m,V]).

    Hq: (N, D); W_rows_new: (K_TOK, D) — новые строки U128;
    pre = (b_k, lse_b, mx_nonK, argK) из base_pre.
    """
    b_k, lse_b, mx_nonK, argK = pre
    Wr = W_rows_new.double()
    N = Hq.shape[0]
    acc = {k: [] for k in ("kl_be", "kl_eb", "tv", "flip")}
    for c0 in range(0, N, SP_CH):
        c1 = min(c0 + SP_CH, N)
        bk = b_k[c0:c1]
        e_k = Hq[c0:c1].double() @ Wr.t()
        lse_e = _lse_from_sparse_t(bk, e_k, lse_b[c0:c1])
        kl_be, kl_eb, tv = _sparse_terms_t(bk, e_k, lse_b[c0:c1], lse_e)
        acc["kl_be"].append(kl_be)
        acc["kl_eb"].append(kl_eb)
        acc["tv"].append(tv)
        acc["flip"].append(_sparse_flip_t(e_k, mx_nonK[c0:c1], argK[c0:c1]))
    return {k: torch.cat(v) for k, v in acc.items()}


def _lse_from_sparse_t(b_k, e_k, lse_b):
    """exp45:149-154."""
    m = torch.cat([lse_b[:, None], e_k, b_k], 1).max(1).values
    return m + torch.log(torch.exp(lse_b - m)
                         - torch.exp(b_k - m[:, None]).sum(1)
                         + torch.exp(e_k - m[:, None]).sum(1))


def _sparse_terms_t(b_k, e_k, lse_b, lse_e):
    db = lse_b - lse_e
    pb = torch.exp(b_k - lse_b[:, None])
    pe = torch.exp(e_k - lse_e[:, None])
    dk = e_k - b_k
    kl_be = -(pb * dk).sum(1) - db
    kl_eb = (pe * dk).sum(1) + db
    tv = 0.5 * ((pe - pb).abs().sum(1)
                + (1.0 - pb.sum(1)) * (torch.exp(db) - 1.0).abs())
    return kl_be.clamp_min(0.0), kl_eb.clamp_min(0.0), tv.clamp_min(0.0)


def _sparse_flip_t(e_k, b_max_nonK, argK):
    m_e, j_e = e_k.max(1)
    new_in_K = m_e > b_max_nonK
    was_in_K = argK >= 0
    kept = torch.where(new_in_K, argK == j_e, ~was_in_K)
    return (~kept).double()


def timed(fn, dev: str, n: int = 3) -> float:
    fn()
    _sync(dev)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        _sync(dev)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


# ──────────────────────────── окружение ────────────────────────────
def torch_build_info() -> str:
    v = torch.__version__
    if "xpu" in v.lower():
        return f"{v} — XPU-сборка"
    if "+cu" in v:
        return (f"{v} — CUDA-сборка ⇒ модуля torch.xpu в ней НЕТ "
                f"(XPU-сборка: pip install torch --index-url "
                f"https://download.pytorch.org/whl/xpu)")
    return f"{v} — сборка без метки XPU/CUDA"


def selftest(dev: str = "cpu") -> bool:
    """Сверка torch-пути с numpy-эталоном на маленькой форме + контроль E-45c."""
    ref = _load_ref()
    print("── numpy-эталон (портированный selftest репозитория + контроль E-45c) ──")
    r = ref.selftest(verbose=True)
    print(f"   эталон: maxKL={r['gauge_max_kl']:.3e} maxTV={r['gauge_max_tv']:.3e} "
          f"flip={r['gauge_flip_sum']:.0f}")

    print("── сверка torch ↔ numpy на одинаковых входах ──")
    g = torch.Generator().manual_seed(1)
    N, D, V = 64, 32, 512
    Hq = torch.randn(N, D, generator=g) * 0.5
    W = torch.randn(V, D, generator=g) * 0.02
    tk = torch.randperm(V, generator=g)[:K_TOK].sort().values
    c = torch.randn(D, generator=g) * 0.1

    Hq_d = Hq.to(dev)
    W_d = W.to(dev)
    tk_d = tk.to(dev)
    col = tok_col_map(tk_d, V).to(dev)

    b_k_t, lse_t, mxn_t, argk_t = base_pre(Hq_d, W_d, tk_d, col)
    b_k_n, lse_n, mxn_n, argk_n = ref.base_pre(Hq.numpy(), W.numpy(), tk.numpy())
    d = {
        "b_k": float((b_k_t.cpu().numpy() - b_k_n).max()),
        "lse": float((lse_t.cpu().numpy() - lse_n).max()),
        "mx_nonK": float((mxn_t.cpu().numpy() - mxn_n).max()),
    }
    same_arg = bool((argk_t.cpu().numpy() == argk_n).all())
    W_edit = (W + c[None, :]).to(dev)
    m_t = dense_metrics(Hq_d, W_edit, W_d)
    m_n = ref.dense_metrics(Hq.numpy(), (W + c[None, :]).numpy(), W.numpy())
    d["kl_be"] = float((m_t["kl_be"].cpu().numpy() - m_n["kl_be"]).max())
    d["tv"] = float((m_t["tv"].cpu().numpy() - m_n["tv"]).max())
    d["flip"] = float((m_t["flip"].cpu().numpy() - m_n["flip"]).sum())

    # Допуски — по АРИФМЕТИКЕ, а не по тому, что случайно сошлось на одной машине.
    # fp32-GEMM даёт относительную ошибку ~1e-7 (разные BLAS суммируют по-разному;
    # наблюдалось 4.47e-08 по логитам между torch 2.13+cu130 и numpy) ⇒
    # непрерывные поля сравниваем ОТНОСИТЕЛЬНО масштаба поля, с запасом ×100;
    # дискретные (flip, argK) — точно: они ловят ошибку семантики, а не шум.
    sc_bk = max(1e-30, float(np.abs(b_k_n).max()))
    sc_mx = max(1e-30, float(np.abs(mxn_n).max()))
    sc_lse = max(1e-30, float(np.abs(lse_n).max()))
    scale_kl = max(1e-30, float(np.abs(m_n["kl_be"]).max()))
    scale_tv = max(1e-30, float(np.abs(m_n["tv"]).max()))
    tol = {
        "b_k": 1e-5 * sc_bk,
        "mx_nonK": 1e-5 * sc_mx,
        "lse": 1e-6 * sc_lse,
        "kl_be": max(1e-6, 1e-4 * scale_kl),
        "tv": max(1e-6, 1e-4 * scale_tv),
    }
    print("   масштабы: |b_k|max=%.3e |mx_nonK|max=%.3e |lse|max=%.3e "
          "|KL|max=%.3e |TV|max=%.3e"
          % (sc_bk, sc_mx, sc_lse, scale_kl, scale_tv))
    bad = []
    for k, lim in tol.items():
        mark = "OK " if d[k] <= lim else "ХУЖЕ ДОПУСКА"
        if d[k] > lim:
            bad.append(f"{k}: {d[k]:.3e} > {lim:.3e}")
        print(f"   {k}: max|Δ| = {d[k]:.3e}  (допуск {lim:.1e})  {mark}")
    if d["flip"] != 0:
        bad.append(f"flip: {d['flip']:.0f} расхождений")
    if not same_arg:
        bad.append("argK разошёлся")
    # чанковый по словарю режим против целочисленного: совпадать должны
    # в пределах шума fp32-GEMM (порядок суммирования другой)
    b_k_c, lse_c, mxn_c, argk_c = base_pre(Hq_d, W_d, tk_d, col,
                                            v_chunk=max(1, V // 3))
    d_chunk = {
        "b_k": float((b_k_c.cpu().numpy() - b_k_n).max()),
        "lse": float((lse_c.cpu().numpy() - lse_n).max()),
        "mx_nonK": float((mxn_c.cpu().numpy() - mxn_n).max()),
    }
    same_arg_c = bool((argk_c.cpu().numpy() == argk_n).all())
    print("── чанковый режим (v_chunk) против целочисленного ──")
    for k, v in d_chunk.items():
        print(f"   {k}: max|Δ| = {v:.3e}")
    print(f"   argK совпал: {same_arg_c}")
    if not same_arg_c:
        bad_chunk = ["argK разошёлся в чанковом режиме"]
    else:
        bad_chunk = []
    for k, lim in (("b_k", 1e-5), ("lse", 1e-9), ("mx_nonK", 1e-5)):
        if d_chunk[k] > lim:
            bad_chunk.append(f"{k} (чанк): {d_chunk[k]:.3e} > {lim:.1e}")
    for b in bad_chunk:
        print(f"   ✗ {b}")

    print(f"   flip: расхождений {d['flip']:.0f} | argK совпал: {same_arg}")
    if bad:
        for b in bad:
            print(f"   ✗ {b}")
    ok = not bad
    print(f"── САМОПРОВЕРКА: {'OK' if ok else 'FAIL'} ──")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4096, help="позиций (кэш exp27b: 4096)")
    ap.add_argument("--D", type=int, default=1024, help="hidden size")
    ap.add_argument("--V", type=int, default=32768, help="размер словаря (в E-45: ~248k)")
    ap.add_argument("--selftest", action="store_true",
                    help="сверка torch ↔ numpy-эталон + контроль E-45c")
    ap.add_argument("--skip-forward", action="store_true")
    ap.add_argument("--model-dir", default=os.environ.get(
        "PRISM_MODEL_DIR", "/data/Qwen/Qwen3.5-0.8B"))
    ap.add_argument("--v-chunk", type=int, default=0,
                    help="размер куска словаря (0 = весь); ограничивает память")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    print(f"зонд: {self_sha256()}  ({os.path.basename(__file__)}, "
          f"{os.path.abspath(__file__)})")
    print("torch:", torch_build_info())
    selftest_ok = selftest("cpu")
    if not selftest_ok:
        print("САМОПРОВЕРКА НЕ ПРОЙДЕНА — дальше идти бессмысленно.")
        return 2

    devs = ["cpu"]
    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                devs.append("xpu")
                print("xpu:", torch.xpu.get_device_name(0))
            else:
                print("xpu: torch.xpu есть, но is_available() = False")
        except Exception as e:  # noqa: BLE001
            print("xpu недоступен:", repr(e))
    if "xpu" not in devs:
        print("!!! torch.xpu недоступен — ниже только CPU-база.")
        print("    XPU-сборка torch: pip install torch --index-url "
              "https://download.pytorch.org/whl/xpu")
        print("    Либо собственный SYCL-прибор (icpx -fsycl, C ABI + ctypes) — "
              "ему torch.xpu не нужен.")

    g = torch.Generator().manual_seed(1)
    N, D, V = args.N, args.D, args.V
    Hq_cpu = torch.randn(N, D, generator=g) * 0.5
    W_cpu = torch.randn(V, D, generator=g) * 0.02
    tk = torch.randperm(V, generator=g)[:K_TOK].sort().values
    cvec = torch.randn(D, generator=g) * 0.1        # калибровочный вектор E-45c

    print("\n=== 1. dense-путь E-45: base_pre (GEMM [N,D]x[D,V] + softmax) ===")
    print(f"    форма: N={N} D={D} V={V} | v_chunk={args.v_chunk or 'весь словарь'} "
          f"| CH={CH} (fp32 GEMM → fp64 сводки)")
    res = {}
    for dev in devs:
        Hq = Hq_cpu.to(dev)
        W = W_cpu.to(dev)
        tk_d = tk.to(dev)
        col = tok_col_map(tk_d, V).to(dev)
        v_chunk = args.v_chunk or None
        t = timed(lambda: base_pre(Hq, W, tk_d, col, v_chunk), dev, args.n)
        # калибровочная правка E-45c: W → W + 1·cᵗ ≡ сдвиг логитов на Hq@c
        shift = Hq @ cvec.to(dev)
        t2 = timed(lambda: dense_metrics(Hq, None, W, shift=shift), dev, args.n)
        m = dense_metrics(Hq, None, W, shift=shift)
        kl = float(m["kl_be"].max())
        tv = float(m["tv"].max())
        flip = float(m["flip"].sum())
        res[dev] = (t, t2, kl, tv, flip)
        print(f"[{dev}] base_pre={t*1e3:9.1f} мс | dense_metrics={t2*1e3:9.1f} мс "
              f"| контроль E-45c: maxKL={kl:.3e} maxTV={tv:.3e} flip={flip:.0f}")
    if "xpu" in res and "cpu" in res:
        print(f"ускорение: base_pre ×{res['cpu'][0]/max(res['xpu'][0],1e-9):.2f} | "
              f"dense_metrics ×{res['cpu'][1]/max(res['xpu'][1],1e-9):.2f}")

    print("\n=== 2. аналитический контроль E-45c (ground truth) ===")
    for dev in devs:
        Hq = Hq_cpu.to(dev)
        W = W_cpu.to(dev)
        shift = Hq @ cvec.to(dev)
        m = dense_metrics(Hq, None, W, shift=shift)
        kl, tv, flip = (float(m["kl_be"].max()), float(m["tv"].max()),
                        float(m["flip"].sum()))
        good = (kl < 1e-4) and (tv < 1e-4) and (flip == 0)
        print(f"[{dev}] KL={kl:.3e} TV={tv:.3e} flip={flip:.0f} -> "
              f"{'ИНВАРИАНТ ДЕРЖИТСЯ' if good else 'ИНВАРИАНТ НАРУШЕН'}")
    if "xpu" in res and "cpu" in res:
        ck = res["xpu"][2] - res["cpu"][2]
        print(f"разница GPU-CPU по maxKL: {ck:.3e} (должна быть ≤ шума fp32)")

    if not args.skip_forward:
        print("\n=== 3. teacher-forced форвард + per-position hidden_states (форма E-45b) ===")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokz = AutoTokenizer.from_pretrained(args.model_dir)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_dir, dtype=torch.float32)
            model.eval()
            ids = tokz("The capital of France is Paris, and the capital of "
                       "Germany is", return_tensors="pt").input_ids
            for dev in devs:
                m2 = model.to(dev)
                with torch.no_grad():
                    out = m2(ids.to(dev), output_hidden_states=True, use_cache=False)
                    _sync(dev)
                    t0 = time.perf_counter()
                    for _ in range(3):
                        out = m2(ids.to(dev), output_hidden_states=True,
                                 use_cache=False)
                    _sync(dev)
                    dt = (time.perf_counter() - t0) / 3
                n_tok = int(ids.shape[1])
                print(f"[{dev}] форвард {n_tok} ток. = {dt*1e3:.1f} мс "
                      f"({n_tok/dt:.1f} ток/с) | hidden_states[24]: "
                      f"{tuple(out.hidden_states[24].shape)}")
        except Exception as e:  # noqa: BLE001
            print(f"форвард пропущен: {e!r}")
            print(f"  (подсказка: модель в {args.model_dir}; задай PRISM_MODEL_DIR)")

    print("\nГотово. Сверка с эталоном — в начале (--selftest), она обязательна.")
    return 0 if selftest_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
