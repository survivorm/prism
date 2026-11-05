"""Пины calibration-пула в Kaggle-ядро (очередь R1b, `docs/regen_queue.md`).

Держит два правила инлайнера `kaggle_kernel/build_kernel.py`:
1. у домена может быть НЕСКОЛЬКО пинов (разные протоколы: min32 и min64), и в ядро должны
   ложиться все — выбор по ключу делает `prism.calibration._pin_path`;
2. домена без пинов быть не может — сборка обязана упасть громко, а не собрать ядро, которое
   молча поедет на streaming-потоке с Hub (урок R1: тот же код, другой пул, C_b сдвинулся ×31).
"""

import importlib.util
import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]


def _loader():
    spec = importlib.util.spec_from_file_location(
        "prism_build_kernel", REPO / "kaggle_kernel" / "build_kernel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pin_cells(nb_path: pathlib.Path) -> list[str]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    return [
        "".join(c["source"])
        for c in nb["cells"]
        if "pinned calibration pool" in "".join(c["source"])
    ]


def test_all_pins_of_domain_are_inlined(tmp_path):
    mod = _loader()
    mod.build_kernel(
        exp_file=str(REPO / "experiments" / "exp01_coherence.py"),
        out_dir=str(tmp_path),
        kernel_id="voroninmichael/test-exp01-pins",
        calib_pins=["wiki"],
    )
    cells = _pin_cells(tmp_path / "kernel.ipynb")
    joined = "\n".join(cells)
    pins = sorted(p.name for p in (REPO / "data" / "calib_pins").glob("wiki__*.json"))
    assert len(pins) >= 2, f"тест рассчитывает на >1 пина wiki, нашлись {pins}"
    assert len(cells) == len(pins), f"ожидалось {len(pins)} pin-cell, собрано {len(cells)}"
    for name in pins:
        assert repr(name) in joined, f"пин {name} не записан в ядро"
    # REQUIRE_PIN обязан стоять в каждом пин-селе: промах по ключу должен падать, а не ехать с Hub
    assert joined.count("PRISM_CALIB_REQUIRE_PIN") >= len(pins)


def test_domain_without_pin_fails_loud(tmp_path):
    mod = _loader()
    with pytest.raises(SystemExit):
        mod.build_kernel(
            exp_file=str(REPO / "experiments" / "exp01_coherence.py"),
            out_dir=str(tmp_path),
            kernel_id="voroninmichael/test-exp01-pins",
            calib_pins=["nosuchdomain"],
        )
