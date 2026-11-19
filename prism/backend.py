"""prism.backend — core-диспетчер CPU (torch) / SYCL (prism_sycl) бэкендов.

TZ_sycl_new_ops.md §2.6 (по допуску пользователя). Механика:
  BACKEND = os.environ.get("PRISM_BACKEND", "cpu")
  B.name  = "cpu" | "sycl"
  B.matmul(A, B):  cpu → A @ B (torch, бит-в-бит);
                   sycl → 3D [b,m,k]@[k,n] host-reshape в 2D → ps.gemm →
                   reshape обратно; f64-вход → f32 + one-shot warning;
                   вывод — torch (cpu: как есть; sycl: from_numpy f32).
  B.eigh(M):       cpu → torch.linalg.eigh(M) ((w, V), бит-в-бит);
                   sycl → ps.eigjac → (from_numpy vals f64, vecs f32).
                   ВАЖНО: vals f64 в sycl (cpu f32) — задокументировано, на
                   1e-3-совпадение не влияет; n > 256 → BackendError с
                   подсказкой (torch fallback невозможен по контракту
                   «один вызов» — эксперимент решает явно).
  B.svd_top(A, r): cpu → torch.linalg.svd(A, full_matrices=False) →
                   top-r срезы ((U[:, :r], S[:r], Vh[:r]));
                   sycl → ps.svd_top → torch.
Исключения: ошибки ps (PS_ERR_*) → BackendError с сообщением (операция,
форма, причина). Lazy-import sycl: в cpu-режиме библиотека не поднимается.

Импорт: как prism/field.py — по файлу (importlib), НЕ `import prism`
(тяжёлый __init__: datasets/torch).
"""
from __future__ import annotations

import os

import numpy as np

import torch

__all__ = ["Backend", "BackendError", "B"]


class BackendError(RuntimeError):
    """Ошибка бэкенда (операция, форма, причина)."""


class Backend:
    """Диспетчер: один вызов — оба режима (веток по режиму в местах вызова нет).

    name — "cpu" | "sycl" (из env PRISM_BACKEND, дефолт cpu).
    """

    def __init__(self, backend: str | None = None):
        if backend is None:
            backend = os.environ.get("PRISM_BACKEND", "cpu")
        self._name = backend.strip().lower()
        if self._name not in ("cpu", "sycl"):
            raise BackendError(
                f"PRISM_BACKEND: неизвестный режим '{backend}' (допустимо: cpu, sycl)")
        self._ps = None            # lazy: sycl-библиотека поднимается только в sycl-режиме
        self._f64_warned = False   # one-shot warning: f64-вход → f32

    @property
    def name(self) -> str:
        return self._name

    # --------------------------- внутреннее ---------------------------

    def _pslib(self):
        if self._ps is None:
            import importlib.util
            import sys
            from pathlib import Path
            root = Path(__file__).resolve().parents[1]   # корень репо
            for cand in (root / "sycl_prism" / "py",
                         Path(os.environ.get("PRISM_SYCL_PY", "")).parent
                         if os.environ.get("PRISM_SYCL_PY") else None):
                if cand is not None and (cand / "prism_sycl.py").exists():
                    spec = importlib.util.spec_from_file_location(
                        "prism_sycl", str(cand / "prism_sycl.py"))
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules["prism_sycl"] = mod
                    spec.loader.exec_module(mod)
                    self._ps = mod
                    break
            if self._ps is None:
                raise BackendError(
                    "sycl-режим: не найден sycl_prism/py/prism_sycl.py "
                    "(укажите PRISM_SYCL_PY=/путь/к/prism_sycl.py)")
        return self._ps

    def _warn_f64(self, op: str) -> None:
        if not self._f64_warned:
            self._f64_warned = True
            print(f"[backend] WARNING ({op}): f64-вход кастится в f32 "
                  f"(SYCL-библиотека — только fp32)", flush=True)

    def _to_f32(self, t: torch.Tensor, op: str) -> torch.Tensor:
        if t.dtype == torch.float32:
            return t
        if t.dtype == torch.float64:
            self._warn_f64(op)
            return t.float()
        raise BackendError(f"{op}: вход должен быть float32/float64 (получено {t.dtype})")

    # --------------------------- операции ---------------------------

    def matmul(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """C = A @ B. 2D [m,k]@[k,n] или 3D [b,m,k]@[k,n] → [b,m,n].

        cpu: torch, бит-в-бит. sycl: ps.gemm (fp32; 3D — host-reshape в 2D).
        """
        if A.ndim not in (2, 3) or B.ndim != 2:
            raise BackendError(
                f"matmul: формы (A {A.ndim}D, B {B.ndim}D) — поддерживаются 2D@2D "
                f"и 3D[batch]@[k,n] (батч только у A)")
        if A.shape[-1] != B.shape[0]:
            raise BackendError(
                f"matmul: K не сошёлся (A {tuple(A.shape)}, B {tuple(B.shape)})")
        if self._name == "cpu":
            return A @ B
        # sycl
        ps = self._pslib()
        a = self._to_f32(A, "matmul").detach().cpu().numpy()
        b = self._to_f32(B, "matmul").detach().cpu().numpy()
        was_3d = a.ndim == 3
        if was_3d:
            bb, m, k = a.shape
            a2 = a.reshape(bb * m, k)
        else:
            bb, m, k = 1, a.shape[0], a.shape[1]
            a2 = a          # само-аудит r4: в 2D-ветке a2 НЕ присваивался ⇒
            # UnboundLocalError на ps.gemm (2D-matmul через sycl не работал НИКОГДА;
            # канарейка — tests/test_backend.py::test_sycl_matmul_2d, был КРАСНЫЙ)
        n = b.shape[1]
        c2 = ps.gemm(a2, b)
        c = c2.reshape(bb, m, n) if was_3d else c2
        return torch.from_numpy(c)

    def eigh(self, M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(w, V) — полный спектр симметричной 2D [n,n] / 3D [b,n,n] матрицы.

        cpu: torch.linalg.eigh (w f32, V f32), бит-в-бит.
        sycl: ps.eigjac (n ≤ 256): w f64 (задокументированное отличие), V f32.
        """
        if M.ndim not in (2, 3) or M.shape[-1] != M.shape[-2]:
            raise BackendError(f"eigh: форма {tuple(M.shape)} — нужна [n,n] или [b,n,n]")
        n = M.shape[-1]
        if self._name == "cpu":
            return torch.linalg.eigh(M)
        ps = self._pslib()
        if n > 256:
            raise BackendError(
                f"eigh: sycl-режим поддерживает n ≤ 256 (получено n={n}); "
                f"torch-fallback невозможен по контракту «один вызов» — "
                f"эксперимент решает явно (напр., torch.linalg.eigh на хосте)")
        m = self._to_f32(M, "eigh").detach().cpu().numpy()
        was_2d = m.ndim == 2
        m3 = np.ascontiguousarray(m[None, :, :]) if was_2d else np.ascontiguousarray(m)
        vals, vecs, _ = ps.eigjac(m3)
        if was_2d:
            return torch.from_numpy(vals[0]), torch.from_numpy(vecs[0])
        return torch.from_numpy(vals), torch.from_numpy(vecs)

    def svd_top(self, A: torch.Tensor, r: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """top-r SVD A [m,n]: (U [m,r], S [r], Vh [r,n]) — конвенция torch (S desc).

        cpu: torch.linalg.svd, full_matrices=False → top-r срезы.
        sycl: ps.svd_top (m,n ≤ 1024 **и** min(m,n) ≤ 256 — малая сторона идёт
        в eigjac, его лимит n≤256; большая — в gemm ≤1024). Выходы sycl-
        ветки C-contiguous без отрицательных шагов (torch.from_numpy не
        принимает отрицательные шаги — само-аудит r4, M1/M2).
        Dtype-отличие (задокументировано, как в eigh): sycl → S f64, cpu → f32
        (эталон λ/σ считается в f64 на хосте).
        """
        if A.ndim != 2:
            raise BackendError(f"svd_top: форма {tuple(A.shape)} — нужен [m,n]")
        m, n = A.shape
        if not (1 <= r <= min(m, n)):
            raise BackendError(f"svd_top: r={r} вне [1, min(m,n)] (форма {m}x{n})")
        if self._name == "cpu":
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            return U[:, :r], S[:r], Vh[:r, :]
        ps = self._pslib()
        a = self._to_f32(A, "svd_top").detach().cpu().numpy()
        U, S, Vh = ps.svd_top(a, int(r))
        return torch.from_numpy(U), torch.from_numpy(S), torch.from_numpy(Vh)


# Экземпляр по умолчанию (режим из env на момент создания).
B = Backend()
