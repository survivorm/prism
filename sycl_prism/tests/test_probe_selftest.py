"""pytest-тест для зонда переноса (`probe/probe_xpu_pipeline.py`).

Гоняет собственную самопроверку зонда: numpy-эталон (портированный selftest
репозитория) + сверка torch ↔ numpy на одинаковых входах. Ловит именно тот класс
ошибок, который стоил мне одного упавшего зонда: неверная форма/семантика U128.

Тест требует torch (в репозитории он есть ⇒ идёт в CI). Если torch не импортируется —
тест пропускается, а не падает.
"""

import ast
import importlib.util
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE_CANDIDATES = [
    os.path.join(HERE, "..", "probe", "probe_xpu_pipeline.py"),
    os.path.join(HERE, "..", "sycl_prism", "probe", "probe_xpu_pipeline.py"),
    os.path.join(HERE, "..", "..", "sycl_prism", "probe", "probe_xpu_pipeline.py"),
]


def _find_probe():
    for path in PROBE_CANDIDATES:
        real = os.path.abspath(path)
        if os.path.exists(real):
            return real
    return None


def test_probe_selftest_passes():
    probe = _find_probe()
    if probe is None:
        pytest.skip("probe_xpu_pipeline.py не найден")
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch не импортируется")
    rc = subprocess.run(
        [sys.executable, probe, "--selftest", "--V", "512", "--N", "64",
         "--skip-forward"],
        capture_output=True, text=True, timeout=600,
    )
    out = rc.stdout + rc.stderr
    assert "САМОПРОВЕРКА: OK" in out, (
        "самопроверка зонда не прошла:\n" + out[-4000:]
    )
    assert rc.returncode == 0, f"зонд вернул код {rc.returncode}:\n" + out[-4000:]


def test_probe_has_u128_semantics():
    """В коде зонда не должно быть сравнения `ik < K_TOK` (регрессия argK).

    Проверка идёт по AST, а не по тексту: объяснение ошибки в docstring — это
    не ошибка (такой ложный срабатыватель и был первой версией теста).
    """
    probe = _find_probe()
    if probe is None:
        pytest.skip("probe_xpu_pipeline.py не найден")
    with open(probe, encoding="utf-8") as f:
        src = f.read()
    assert "TOK_COL" in src or "tok_col_map" in src
    tree = ast.parse(src)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Name) and left.id == "ik"):
            continue
        for cmp_node in node.comparators:
            if isinstance(cmp_node, ast.Name) and cmp_node.id == "K_TOK":
                bad.append(node.lineno)
    assert not bad, (
        "в зонде вернулась ошибочная семантика argK (сравнение vocab-id с K_TOK) "
        f"в строках {bad}"
    )


def test_probe_imports_cleanly():
    probe = _find_probe()
    if probe is None:
        pytest.skip("probe_xpu_pipeline.py не найден")
    spec = importlib.util.spec_from_file_location("probe_xpu_pipeline", probe)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.K_TOK == 128
    assert mod.CH == 128
    assert mod.SP_CH == 4 * mod.CH
