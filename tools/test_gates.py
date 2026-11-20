"""Дизайн-тесты гейтов: каждый гейт обязан ПАДАТЬ на своём дефекте и проходить после allowlist.

Почему так, а не «проверим, что скрипт запускается»: гейт без исполняемого негативного теста —
это тавтология (класс дефектов E-39j из аудита). Здесь на каждый гейт есть designed-сценарий:
вход с точно известным дефектом → RC=1 и имя файла в выводе; тот же вход в allowlist → RC=0.

Гейты читают REPO из модульной глобали, поэтому tmp-каталог подставляется monkeypatch'ем:
сканируется только tmp, репозиторий не трогается. Вложенный pytest запускается с
`-p no:cacheprovider` (внутри `gate_collect.collect_one`) и с cwd=tmp — рекурсии нет.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gate_artifacts  # noqa: E402
import gate_collect  # noqa: E402
import gate_paths  # noqa: E402
from artifact_io import dump_json_atomic  # noqa: E402


# ---------------------------------------------------------------- G1: тест-файл без тестов
def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_g1_fails_on_collection_error_and_empty_test_file(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "test_good.py", "def test_ok():\n    assert True\n")
    _write(tmp_path / "test_broken.py", "import definitely_no_such_module_xyz\n\n\ndef test_x():\n    pass\n")
    _write(tmp_path / "script_tests.py", "def main():\n    return 0\n\n\nif __name__ == '__main__':\n    main()\n")

    monkeypatch.setattr(gate_collect, "REPO", str(tmp_path))
    monkeypatch.setattr(gate_collect, "AREAS", (".",))
    rc = gate_collect.main()
    out = capsys.readouterr().out

    assert rc == 1, out
    assert "DEAD_TEST test_broken.py" in out      # RC=2: collection error
    assert "DEAD_TEST script_tests.py" in out     # RC=5: no tests collected
    assert "DEAD_TEST test_good.py" not in out    # рабочий тест не тронут

    # тот же вход, оба дефекта в allowlist -> RC=0
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "gate_allowlist.txt").write_text(
        "G1\ttest_broken.py\tизвестный дефект (дизайн-тест)\n"
        "G1\tscript_tests.py\tизвестный дефект (дизайн-тест)\n",
        encoding="utf-8",
    )
    rc2 = gate_collect.main()
    assert rc2 == 0, capsys.readouterr().out


# ---------------------------------------------------------------- G2: путь от __file__
def _probe_body(n_dirname, extra=""):
    """Файл, который строит путь от __file__ c ровно n_dirname вызовами dirname."""
    expr = "os.path.abspath(__file__)"
    for _ in range(n_dirname):
        expr = f"os.path.dirname({expr})"
    return f"import os\n\nOUT = os.path.join({expr}, 'results'{extra})\n"


def test_g2_number_of_dirname_is_counted_exactly(tmp_path, monkeypatch, capsys):
    """Глубина файла фиксирована, меняется ЧИСЛО dirname; каждый кейс — в своём корне."""
    cases = {}
    for n in (1, 2, 3):
        root = tmp_path / f"k{n}"
        f = _write(root / "a" / "b" / "gate_probe.py", _probe_body(n))
        cases[n] = (root, f)

    # кейс 2 -> <root>/a/results: цель создана заранее -> ссылка не считается битой (OK)
    (cases[2][0] / "a" / "results").mkdir(parents=True)

    seen_targets = set()
    for n, (root, f) in cases.items():
        monkeypatch.setattr(gate_paths, "REPO", str(root))
        refs = gate_paths.scan_file(str(f))
        if n == 2:
            assert refs == [], refs               # цель существует -> не дефект
            continue
        assert len(refs) == 1, (n, refs)
        seen_targets.add(refs[0]["resolved"])
        # родитель существует (a/b и k3) -> класс FILE_MISSING, RC не меняется
        assert refs[0]["kind"] == "file_missing", (n, refs)
    assert len(seen_targets) == 2, seen_targets   # 1 и 3 dirname дают РАЗНЫЕ цели


def test_g2_dir_missing_blocks_and_allowlist_releases(tmp_path, monkeypatch, capsys):
    body = ("import os\n\nHERE = os.path.dirname(os.path.abspath(__file__))\n"
            "OUT = os.path.join(HERE, 'nope', 'results')\n")
    f = _write(tmp_path / "a" / "b" / "gate_probe.py", body)
    monkeypatch.setattr(gate_paths, "REPO", str(tmp_path))
    monkeypatch.setattr(gate_paths, "AREAS", (".",))
    refs = gate_paths.scan_file(str(f))
    assert [r["kind"] for r in refs] == ["dir_missing"], refs

    rc = gate_paths.main()
    assert rc == 1
    assert "DIR_MISSING a/b/gate_probe.py" in capsys.readouterr().out

    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "gate_allowlist.txt").write_text(
        "G2\ta/b/gate_probe.py\tlinks=1\tдизайн-тест: известный дефект\n", encoding="utf-8")
    assert gate_paths.main() == 0, capsys.readouterr().out

    # links не совпало -> STALE и RC=1 (запись не легализует будущие поломки)
    (tmp_path / "tools" / "gate_allowlist.txt").write_text(
        "G2\ta/b/gate_probe.py\tlinks=7\tдизайн-тест: устаревшее links\n", encoding="utf-8")
    assert gate_paths.main() == 1, capsys.readouterr().out


def test_g2_alias_form_is_visible(tmp_path, monkeypatch):
    """Регрессия на слепоту r1: путь через алиас-переменную обязан быть виден."""
    body = ("import os\n\n"
            "_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
            "ART = os.path.join(_ROOT, 'results', 'x.json')\n")
    f = _write(tmp_path / "alias_root" / "a" / "b" / "gate_probe.py", body)
    monkeypatch.setattr(gate_paths, "REPO", str(tmp_path))
    refs = gate_paths.scan_file(str(f))
    assert len(refs) == 1, refs
    # dirname(dirname(file)) для <root>/alias_root/a/b/gate_probe.py = <root>/alias_root/a
    assert refs[0]["resolved"] == os.path.join("alias_root", "a", "results", "x.json")


# ---------------------------------------------------------------- G6: артефакт о себе
def test_g6_three_reasons_and_gzip(tmp_path, monkeypatch, capsys):
    import gzip
    import json

    _write(tmp_path / "results" / "ok.json", json.dumps({"a": 1}))
    _write(tmp_path / "results" / "nan.json", '{"a": NaN}')
    _write(tmp_path / "results" / "nrows.json", json.dumps({"meta": {"n_rows": 2}, "rows": [1]}))
    _write(tmp_path / "results" / "broken.json", '{"a": 1')
    with gzip.open(tmp_path / "results" / "nan_gz.json.gz", "wt", encoding="utf-8") as fh:
        fh.write('{"a": NaN}')

    monkeypatch.setattr(gate_artifacts, "REPO", str(tmp_path))
    rc = gate_artifacts.main()
    out = capsys.readouterr().out
    assert rc == 1, out
    assert "INVALID results/broken.json" in out
    assert "NONFINITE results/nan.json" in out
    assert "NONFINITE results/nan_gz.json.gz" in out     # gzip распакован, а не «битый»
    assert "NROWS results/nrows.json" in out

    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "gate_allowlist.txt").write_text(
        "G6\tresults/broken.json\tдизайн-тест\n"
        "G6\tresults/nan.json\tдизайн-тест\n"
        "G6\tresults/nan_gz.json.gz\tдизайн-тест\n",
        encoding="utf-8")
    # nrows не легализуется allowlist'ом — он всегда блокирующий
    assert gate_artifacts.main() == 1


def test_g6_store_excluded_from_canon(tmp_path, monkeypatch):
    import json
    _write(tmp_path / "results" / "AT0" / "store" / "blob.json", json.dumps({"a": 1}))
    _write(tmp_path / "results" / "AT0" / "keep.json", json.dumps({"a": 1}))
    monkeypatch.setattr(gate_artifacts, "REPO", str(tmp_path))
    files = [os.path.relpath(f, str(tmp_path)) for f in gate_artifacts.canon()]
    assert files == [os.path.join("results", "AT0", "keep.json")], files


# ---------------------------------------------------------------- artifact_io
def test_dump_json_atomic_rejects_nan_and_keeps_target_untouched(tmp_path):
    target = tmp_path / "art.json"
    dump_json_atomic({"a": 1}, str(target))
    assert target.exists()
    before = target.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        dump_json_atomic({"a": float("nan")}, str(target))
    assert target.read_text(encoding="utf-8") == before
    assert [p.name for p in tmp_path.iterdir()] == ["art.json"]     # tmp-файл убран
