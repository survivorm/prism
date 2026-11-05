#!/usr/bin/env python
"""R11 (очередь `docs/regen_queue.md`): почему `basis_overlap_fresh` = 43.75 %.

Мина ревью (`docs/E46+_review.md`, разбор был в `docs/E46+_answers.md`, удалён в `07c27df` —
читать из git history; `docs/audit_40plus.md` жив):
свежепересчитанный базис U128 пересёкся с законным кэша на 43.75 %, и это объяснили «уехал
streaming-пул». Альтернативы не были исключены. Здесь проверяются три, и все — без единого
Джейкобиана, только логиты последней позиции (ровно то, из чего базис и строится):

H1 «уехал поток»: если бы дело было только в содержимом пула, два пула ОДНОГО протокола и
   ОДНОГО размера из соседних документов давали бы близкое 100 % пересечение.
H2 «размер пула»: 8 документов против 32.
H3 «хвост частотного отбора весь в ничьих»: при 8 документах и K=128 голосов 8×128 = 1024, а
   слотов 128 — если у большинства отобранных токенов голос ровно 1, то порядок внутри хвоста
   задаётся не данными, а порядком разрешения ничьих (`Counter.most_common` против
   `sorted(key=-count)`), и перекрытие обязано быть далеко от 100 % при ЛЮБОМ пуле.

Якорь. Точное число 43.75 % из `results/exp47b_gen_kinematics.json` здесь не воспроизводится в
принципе: у того прогона был streaming-пул при max_length=256, а тут закреплённые тексты пина
(отбор при max_length=64) — пулы разные по построению. Проверяется вместо этого сам код отбора
(синтетическая самопроверка с заранее известным ответом) и то, что overlap попадает в
правдоподобную полосу.

Запуск: `cd experiments && ../.venv/bin/python exp_r11_basis_tie.py` (CPU, ~3 мин).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _e38_cache  # noqa: E402

MODEL = "Qwen/Qwen3.5-0.8B"
K = 128          # размер базиса exp27b
MAX_LEN = 256    # S=256, как в сборщике U128 ядра exp27b
PIN_DIR = Path(__file__).resolve().parents[1] / "data" / "calib_pins"
ANCHOR_OVERLAP = 0.4375   # results/exp47b_gen_kinematics.json : basis_overlap_fresh
DOMS = ("wiki", "math", "code", "chat")


def pin_texts(dom: str) -> list[str]:
    hits = sorted(PIN_DIR.glob(f"{dom}__*__n8__min64__max64__plain.json"))
    assert len(hits) == 1, f"{dom}: ожидался ровно один пин n8/min64/max64, найдено {len(hits)}"
    texts = json.loads(hits[0].read_text(encoding="utf-8"))["texts"]
    assert len(texts) == 8, f"{dom}: в пине {len(texts)} текстов, ожидалось 8"
    return texts


def votes(model, tok, texts: list[str]) -> list[list[int]]:
    """Топ-128 последней позиции на документ — ровно тот голос, из которого строится U128."""
    out = []
    for t in texts:
        enc = tok(t, max_length=MAX_LEN, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"],
                           attention_mask=enc["attention_mask"]).logits[0, -1].float()
        out.append(logits.topk(K).indices.tolist())
        print(f"  vote done, {len(out)}", flush=True)
    return out


def basis(vs: list[list[int]], tie: str = "heap") -> tuple[list[int], Counter]:
    c: Counter[int] = Counter()
    for v in vs:
        c.update(v)
    if tie == "heap":                       # как prism.calibration.build_global_token_set
        sel = [t for t, _ in c.most_common(K)]
    else:                                   # как ядро exp27b: sorted(key=-count) + sorted(id)
        sel = [t for t, _ in sorted(c.items(), key=lambda x: -x[1])[:K]]
    return sorted(sel), c


def overlap(a: list[int], b: list[int]) -> float:
    return len(set(a) & set(b)) / len(b)


def selfcheck() -> None:
    """Отбор базиса на синтетике с заранее известным ответом (без модели)."""
    vs = [[1, 2, 3], [1, 2, 9], [4, 5, 6]]
    c: Counter[int] = Counter()
    for v in vs:
        c.update(v)
    assert c[1] == 2 and c[9] == 1 and c[4] == 1
    sel, _ = basis(vs, tie="sorted")
    assert set(t for t, _ in sorted(c.items(), key=lambda x: -x[1])[:2]) == {1, 2}
    assert sorted(sel) == sel, "базис обязан быть отсортирован по id (как в обоих сборщиках)"


def main() -> None:
    selfcheck()
    torch.set_num_threads(os.cpu_count() or 4)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()

    texts = {d: pin_texts(d) for d in DOMS}
    print("[r11] голоса по доменам...", flush=True)
    vs = {d: votes(model, tok, texts[d]) for d in DOMS}
    allv = [v for d in DOMS for v in vs[d]]

    ck_path = _e38_cache.jcache_path(23)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False, mmap=True)
    U_cache = sorted(int(t) for t in ck["U128"])
    assert len(U_cache) == K, f"U128 в кэше имеет размер {len(U_cache)}"

    b_wiki, c_wiki = basis(vs["wiki"])
    # совпадение с числом exp47b не ожидается (другой пул), но значение обязано быть в
    # правдоподобной полосе: ~1 или ~0 означали бы, что протокол здесь собран не тот
    anchor = overlap(b_wiki, U_cache)
    print(f"[r11] overlap(wiki-8 пин, U128 кэша) = {anchor:.4f} "
          f"(у exp47b на streaming-пуле было {ANCHOR_OVERLAP})", flush=True)
    assert 0.2 < anchor < 0.9, f"overlap вне правдоподобной полосы: {anchor:.4f}"

    out: dict = {"anchor": {"overlap_wiki8pin_vs_cache": anchor,
                            "exp47b_overlap_fresh_streaming": ANCHOR_OVERLAP,
                            "pool_note": "пулы разные по построению: у exp47b streaming при "
                                         "max_length=256, здесь тексты пина (отбор при 64)",
                            "jcache": os.path.basename(ck_path)},
                 "protocol": {"model": MODEL, "K": K, "max_length": MAX_LEN,
                              "votes": "top-128 логитов последней позиции, по документу",
                              "pool_source": "data/calib_pins (закреплённые тексты)"}}

    # H3: плотность ничьих в хвосте отбора
    sel_counts = [c_wiki[t] for t in b_wiki]
    out["tie_density_wiki8"] = {
        "docs": 8, "K": K, "votes_total": 8 * K, "distinct_voted_tokens": len(c_wiki),
        "selected_with_count_1": sum(1 for x in sel_counts if x == 1),
        "selected_with_count_ge2": sum(1 for x in sel_counts if x >= 2),
        "selected_with_count_ge4": sum(1 for x in sel_counts if x >= 4),
        "count_of_128th": min(sel_counts),
        "ties_at_cut": sum(1 for v in c_wiki.values() if v == min(sel_counts)),
    }

    # H1/H2: перекрытия пулов одного протокола (permutation-of-content контроль)
    b = {d: basis(vs[d])[0] for d in DOMS}
    b_all, c_all = basis(allv)
    h1 = {f"{x}__{y}": overlap(b[x], b[y]) for i, x in enumerate(DOMS) for y in DOMS[i + 1:]}
    half_a, half_b = basis(vs["wiki"][:4]), basis(vs["wiki"][4:])
    out["overlaps_same_protocol_same_size"] = {
        "междоменные_8x8": h1,
        "wiki_4docs_vs_wiki_4docs_disjoint": overlap(half_a[0], half_b[0]),
        "wiki8_vs_32docs_global": overlap(b_all, b["wiki"]),
        "cache32_vs_wiki8": anchor,
        "tie_break_matters_heap_vs_sorted": overlap(
            basis(vs["wiki"], "heap")[0], basis(vs["wiki"], "sorted")[0]),
    }
    out["global_basis_size32"] = {
        "docs": 32, "votes_total": 32 * K, "distinct_voted_tokens": len(c_all),
        "selected_with_count_1": sum(1 for t in b_all if c_all[t] == 1),
        "overlap_with_cache": overlap(b_all, U_cache),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    dst = Path(__file__).resolve().parents[1] / "results" / "exp_r11_basis_tie.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to {dst}", flush=True)


if __name__ == "__main__":
    main()
