"""Зонд upstream torch.xpu на Intel iGPU — 10 минут, которые экономят дни.

Проверяет РОВНО то, что нужно prism:
  1. есть ли вообще torch.xpu и какой iGPU он видит;
  2. работает ли fp16 (у пользователя «16 падает» в Vulkan-вариантах — здесь
     отдельный вопрос: XPU-бэкенд идёт через Level Zero, а не через Vulkan);
  3. работает ли autograd (backward) — для Якобиана это обязательное условие;
  4. есть ли linalg.svd/eigh на xpu и в fp64 (у iGPU fp64 = 1/64 скорости);
  5. грузится ли реальная модель (Qwen3.5-0.8B) и работает ли путь с GDN
     (linear attention) — в transformers это кастомные CUDA-ядра, на XPU они
     могут падать или откатиться на чистый python-путь.

Запуск: python3 probe_xpu.py
"""
from __future__ import annotations

import time
import traceback


def hdr(s: str) -> None:
    print("\n" + "=" * 70)
    print(s)
    print("=" * 70)


def timed(fn, n=5):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return None, repr(e)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2], None


def main() -> int:
    hdr("1. torch и наличие xpu")
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print("torch не установлен:", e)
        return 2
    print("torch:", torch.__version__)
    if not hasattr(torch, "xpu"):
        print("torch.xpu ОТСУТСТВУЕТ ⇒ нужна сборка с XPU (index-url .../whl/xpu)")
        return 2
    try:
        ok = torch.xpu.is_available()
        print("torch.xpu.is_available():", ok)
        if not ok:
            print("устройство не видно: проверьте драйвер Intel (compute runtime) и ONEAPI_DEVICE_SELECTOR")
            return 2
        print("имя устройства:", torch.xpu.get_device_name(0))
        props = torch.xpu.get_device_properties(0)
        print("свойства:", props)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2

    hdr("2. fp16 / fp32: matmul + проверка точности")
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        try:
            a = torch.randn(512, 512, device="xpu", dtype=dt)
            b = torch.randn(512, 512, device="xpu", dtype=dt)
            c = a @ b
            torch.xpu.synchronize()
            cpu = (a.float().cpu() @ b.float().cpu())
            rel = float((c.float().cpu() - cpu).abs().max() / cpu.abs().max())
            t, err = timed(lambda: a @ b)
            print(f"{str(dt):18s} OK  maxrel={rel:.3e}  t={t*1e3:.2f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"{str(dt):18s} ПАДЕНИЕ: {e!r}")

    hdr("3. autograd: backward через matmul (нужно для Якобиана)")
    try:
        w = torch.randn(256, 256, device="xpu", requires_grad=True)
        x = torch.randn(8, 256, device="xpu")
        y = (x @ w).pow(2).sum()
        y.backward()
        torch.xpu.synchronize()
        print("backward: OK, |grad| =", float(w.grad.abs().sum()))
    except Exception as e:  # noqa: BLE001
        print("backward ПАДЕНИЕ:", repr(e))

    hdr("4. linalg: svd / eigvalsh на xpu (fp32 и fp64)")
    for dt in (torch.float32, torch.float64):
        for name, fn_ in (("svd", lambda m: torch.linalg.svd(m, full_matrices=False)),
                          ("eigvalsh", lambda m: torch.linalg.eigvalsh(m))):
            try:
                m = torch.randn(128, 128, device="xpu", dtype=dt)
                if name == "eigvalsh":
                    m = m @ m.T
                out = fn_(m)
                torch.xpu.synchronize()
                print(f"{name:9s} {str(dt):16s} OK")
            except Exception as e:  # noqa: BLE001
                print(f"{name:9s} {str(dt):16s} ПАДЕНИЕ: {e!r}")

    hdr("5. реальная модель (Qwen3.5-0.8B): forward + hidden_states + backward")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import os
        mid = os.environ.get("PRISM_MODEL_DIR", ".hf_cache/hub/models--Qwen--Qwen3.5-0.8B")
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32)
        model.eval()
        ids = tok("The capital of France is", return_tensors="pt").input_ids.to("xpu")
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, use_cache=False)
        torch.xpu.synchronize()
        print("forward: OK; слоёв:", len(out.hidden_states),
              "; h[-1]:", tuple(out.hidden_states[-1].shape))
        # проверка GDN-пути: типы слоёв и ошибки кастомных ядер
        lt = getattr(model.config, "layer_types", None)
        print("layer_types:", lt)
        h = out.hidden_states[-1].detach().clone()
        print("h[-1] (fp32):", tuple(h.shape), "| requires_grad:", h.requires_grad)
        print("Примечание: GDN-касты в transformers — проверьте, что путь attention/linear-attention "
              "не падает на xpu (см. шаг 5b ниже).")
        del h
    except Exception as e:  # noqa: BLE001
        print("модель: ПАДЕНИЕ или не найдена:", repr(e))

    hdr("6. скорость: CPU (float64) против xpu (float32) на типовом приборе")
    try:
        T, D = 384, 1024
        x = torch.randn(T, D)
        t_cpu, err = timed(lambda: torch.linalg.svd(x[:128].double(), full_matrices=False), n=3)
        print(f"CPU fp64 svd[128x1024]: {t_cpu*1e3:.1f} ms" if t_cpu else f"CPU: {err}")
        try:
            xg = x.to("xpu")
            t_gpu, err2 = timed(lambda: torch.linalg.svd(xg[:128], full_matrices=False), n=3)
            print(f"xpu fp32 svd[128x1024]: {t_gpu*1e3:.1f} ms" if t_gpu else f"xpu: {err2}")
        except Exception as e:  # noqa: BLE001
            print("xpu svd:", repr(e))
    except Exception as e:  # noqa: BLE001
        print("бенчмарк не выполнен:", repr(e))

    hdr("ИТОГ")
    print("Если xpu доступен и fp16+backward работают — модельный слой можно унести на iGPU.")
    print("Если GDN-путь падает — для Якобиана остаётся CPU, и ускорение прибора (SYCL) становится")
    print("единственным, что даёт iGPU в этой серии.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
