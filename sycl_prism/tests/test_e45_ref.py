"""pytest-тесты для numpy-эталона формул E-45 (`py/e45_ref.py`).

Зачем они в репозитории. Мой первый зонд переноса на XPU был отдан без проверки и
упал на первой строке (`tok_ids.shape[1]`), причём семантику U128 я переписал по
памяти: вместо номера столбца U128 для top-1 (`TOK_COL[ik]`) стояло `ik < K_TOK`.
Обе ошибки ловятся этими тестами за секунды, без GPU.

Тесты numpy-only ⇒ идут в CI репозитория (`pytest tests -q`).
Путь к модулю ищется относительно этого файла, поэтому тест работает и внутри
`sycl_prism/tests/`, и после копирования в `tests/` репозитория.
"""

import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "..", "py", "e45_ref.py"),
    os.path.join(HERE, "..", "sycl_prism", "py", "e45_ref.py"),
    os.path.join(HERE, "..", "..", "sycl_prism", "py", "e45_ref.py"),
]


def _load():
    for path in CANDIDATES:
        real = os.path.abspath(path)
        if os.path.exists(real):
            spec = importlib.util.spec_from_file_location("e45_ref", real)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("e45_ref.py не найден")


ref = _load()


def test_repo_selftest_formulas():
    """Порт selftest репозитория (exp45:327-368): sparse-формулы против softmax."""
    out = ref.selftest(verbose=False)
    for tag in ("слабая", "сильная"):
        assert out[f"{tag}_kl_be_maxdiff"] < 1e-9
        assert out[f"{tag}_kl_eb_maxdiff"] < 1e-9
        assert out[f"{tag}_tv_maxdiff"] < 1e-9
        assert out[f"{tag}_flip_maxdiff"] == 0.0


def test_gauge_invariance_E45c():
    """E-45c: W → W + 1·cᵗ ⇒ softmax инвариантен ⇒ KL = TV = flip = 0."""
    out = ref.selftest(verbose=False)
    assert out["gauge_max_kl"] < 1e-4
    assert out["gauge_max_tv"] < 1e-4
    assert out["gauge_flip_sum"] == 0.0


def test_base_pre_matches_direct_softmax():
    """base_pre обязан совпадать с прямым softmax: lse, max вне U128, b_k."""
    out = ref.selftest(verbose=False)
    assert out["base_pre_lse_maxdiff"] < 1e-10
    assert out["base_pre_mxnonk_maxdiff"] < 1e-10
    assert out["base_pre_bk_maxdiff"] < 1e-10


def test_argK_is_column_of_top1_inside_U128():
    """argK = номер СТОЛБЦА U128, где лежит top-1 (иначе −1) — не vocab-id.

    Регрессия на ошибку `argK = ik < K_TOK` (сравнение vocab-id с 128).
    """
    rng = np.random.default_rng(7)
    N, D, V, K = 32, 16, 256, ref.K_TOK
    Hq = (rng.standard_normal((N, D)) * 0.5).astype(np.float32)
    W = (rng.standard_normal((V, D)) * 0.02).astype(np.float32)
    tk = rng.permutation(V)[:K].astype(np.int64)
    b_k, lse, mxn, argK = ref.base_pre(Hq, W, tk)
    lb = (Hq @ W.T).astype(np.float64)
    ik = lb.argmax(1)
    for i in range(N):
        if ik[i] in tk:
            assert argK[i] >= 0, f"строка {i}: top-1 ∈ U128, но argK = −1"
            assert tk[argK[i]] == ik[i], (
                f"строка {i}: argK указывает не на столбец top-1 "
                f"(argK={argK[i]}, tk[argK]={tk[argK[i]]}, top1={ik[i]})"
            )
        else:
            assert argK[i] == -1, f"строка {i}: top-1 вне U128, но argK = {argK[i]}"


def test_tok_col_map_is_inverse_of_tk():
    V, K = 512, ref.K_TOK
    tk = np.random.default_rng(3).permutation(V)[:K].astype(np.int64)
    col = ref.tok_col_map(tk, V)
    assert (col < 0).sum() == V - K
    for j, t in enumerate(tk):
        assert col[t] == j
