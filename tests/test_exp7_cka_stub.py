"""Смоук E-7 на подставных моделях (CPU, секунды).

Зачем: путь `main()` в `exp7_cka_baseline.py` исполняется только на GPU-ядре Kaggle,
а ровно этот класс дефекта («код дошёл до ядра и упал на второй минуте») уже трижды стоил
прогонов в E-39m. Проверка держает то, что ломалось и что ломается обычно: сбор скрытых
состояний одним проходом (индексы слоёв), вызов CKA на кросс-размерной паре, наличие у
метрики нулей на ТОЙ ЖЕ форме (правило docs/paper.md §6 п. 29) и запись артефакта.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

_spec = importlib.util.spec_from_file_location(
    "exp7_under_test", ROOT / "experiments" / "exp7_cka_baseline.py"
)
e7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e7)

V, D_L, D_S, N_L, N_S = 200, 96, 48, 12, 8


class _Out:
    def __init__(self, hidden_states, logits):
        self.hidden_states = hidden_states
        self.logits = logits


class _Cfg:
    def __init__(self, n):
        self.num_hidden_layers = n


class _Emb:
    def __init__(self, D):
        self.weight = torch.randn(V, D)


class FakeModel:
    """Отдаёт все слои за один вызов; скрытое состояние последнего токена = номер образца."""

    def __init__(self, n_layers, D, tag):
        self.config = _Cfg(n_layers)
        self.device = torch.device("cpu")
        self.D = D
        self._g = torch.Generator().manual_seed(D + n_layers)
        self._emb = _Emb(D)
        self.calls = 0
        self.tag = tag

    def eval(self):
        return self

    def parameters(self):
        return iter([torch.zeros(1)])

    def get_input_embeddings(self):
        return self._emb

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=False):
        self.calls += 1
        n = input_ids.shape[0]
        s = input_ids.shape[1]
        hs = []
        for l in range(self.n_layers_call() + 1):
            # детерминированный «сигнал по образцам» + шум, чтобы Gram-матрицы различались
            base = torch.arange(n, dtype=torch.float32).view(n, 1, 1) / 10.0
            hs.append(base + 0.05 * torch.randn(n, s, self.D, generator=self._g))
        logits = torch.randn(n, 1, V, generator=self._g)
        return _Out(hs, logits)

    def n_layers_call(self):
        return self.config.num_hidden_layers


def test_exp7_stub(tmp_path, monkeypatch):
    samples = [
        {"input_ids": torch.randint(0, V, (1, 8)), "attention_mask": torch.ones(1, 8, dtype=torch.long)}
        for _ in range(6)
    ]
    models = {}

    def fake_from_pretrained(name, **kw):
        m = FakeModel(N_L, D_L, "L") if "4B" in str(name) or name == "L" else FakeModel(N_S, D_S, "S")
        models[str(name)] = m
        return m

    monkeypatch.setattr(e7, "load_calibration_data", lambda *a, **k: samples)
    monkeypatch.setattr(e7, "AutoModelForCausalLM", type("M", (), {"from_pretrained": staticmethod(fake_from_pretrained)}))
    monkeypatch.setattr(e7, "extract_embeddings", lambda m: m.get_input_embeddings().weight)
    monkeypatch.setattr(
        e7, "compute_jacobian_for_tokens",
        lambda model, layer, ids, attn, token_set: torch.randn(len(token_set), model.D),
    )
    monkeypatch.setattr(e7, "jacobian_gap", lambda *a, **k: 0.5)

    e7.main(large_model="L", small_model="S", n_samples=6, K=8,
            max_length=8, min_length=1, results_dir=str(tmp_path))

    art = json.loads((tmp_path / "exp7_cka_baseline.json").read_text(encoding="utf-8"))
    assert art["cka_form"] == "standard", "запись мёртвой legacy-формы запрещена"
    assert len(art["cka"]) == len(art["cka_nulls"]) > 3
    for pair, nul in art["cka_nulls"].items():
        assert nul["n_objects"] == 6, f"{pair}: CKA считается не по образцам"
        assert nul["dims"] == [D_L, D_S], f"{pair:12} кросс-размерная пара потеряна"
        for key in ("perm", "gauss", "control_shared_factor"):
            assert 0.0 <= nul[key] <= 1.0 + 1e-6, f"{pair}: {key} вне [0, 1]"
    # сбор одним проходом: по одному форварду на образец (а не на пару слоёв); у большой
    # модели плюс один вызов — единый проход за token_set для jacobian_gap
    for name, m in models.items():
        expect = len(samples) + (1 if name == "L" else 0)
        assert m.calls == expect, f"{name}: {m.calls} форвардов вместо {expect}"

    # --- вторая ветка: gap не пересчитывается, а берётся из записанного артефакта ---
    prev = tmp_path / "prev_gap.json"
    prev.write_text(json.dumps({"gap": art["gap"]}), encoding="utf-8")
    out2 = tmp_path / "run2"
    e7.main(large_model="L", small_model="S", n_samples=6, K=8, max_length=8, min_length=1,
            results_dir=str(out2), dtype="float32", reuse_gap_from=str(prev))
    a2 = json.loads((out2 / "exp7_cka_baseline.json").read_text(encoding="utf-8"))
    assert a2["gap_source"] == str(prev) and list(a2["gap"]) == list(a2["cka"]), \
        "Spearman по CKA и gap должен считаться по одним и тем же парам в одном порядке"
