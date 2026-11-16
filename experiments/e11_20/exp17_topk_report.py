"""Exp17 C5: top-k report — does the LoRA-tuned 0.8B propose real tokens?

Three models, same prompts:
  1. 0.8B base (no deltas)
  2. 0.8B + cumulative lm_head deltas from all lora_{pair}.pt files
     (--keep-o-proj also applies each pair's o_proj delta to its l_S)
  3. 4B fp16 (loaded at report time)

per prompt/category/k:
  agreement    = |topk(m2) ∩ (topk(m1) ∪ topk(m3))| / k
  garbage_rate = 1 - agreement

Output: results/exp17/c5_topk_report.json
  {k, category, agreement, garbage_rate, examples[3], note}

Model 3 OOM policy: try in-kernel fp16 load; on OOM unload the current
0.8B and retry; on second failure skip the 4B and report with a note
(model-2 agreement then measured against the 0.8B base only).

Heavy: 20 prompts × 4 categories × 3 models. Run on GPU (Kaggle), not
local CPU.
"""

import argparse
import gc
import json
import os

import torch
from transformers import (  # type: ignore[import-not-found]
    AutoModelForCausalLM,
    AutoTokenizer,
)

from prism.calibration import load_calibration_data

CATEGORIES = ["wiki", "math", "code", "chat"]
K_SWEEP = (1, 5, 10)


def _load(small_model: str, large_model: str, device: str, which: str):
    if which == "small":
        return AutoModelForCausalLM.from_pretrained(
            small_model, torch_dtype=torch.float32, low_cpu_mem_usage=True
        ).to(device)  # type: ignore[arg-type]
    return AutoModelForCausalLM.from_pretrained(
        large_model, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)  # type: ignore[arg-type]


def _free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _topk_all(model, prompt, k_list, device) -> list[list[int]]:
    ids = prompt["input_ids"].to(device)
    attn = prompt["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=attn).logits[0, -1, :]
    return [topk.indices.tolist() for topk in (torch.topk(logits, k) for k in k_list)]


def main(
    small_model: str = "Qwen/Qwen3.5-0.8B",
    large_model: str = "/data/Qwen/Qwen3.5-4B",
    results_dir: str = "results",
    lora_dir: str = "",
    n_prompts: int = 20,
    keep_o_proj: bool = False,
    device: str = "auto",
) -> None:
    torch.manual_seed(42)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    lora_dir = lora_dir or os.path.join(results_dir, "exp17")
    lora_files = sorted(
        f for f in os.listdir(lora_dir) if f.startswith("lora_") and f.endswith(".pt")
    )
    if not lora_files:
        raise FileNotFoundError(f"no lora_{{pair}}.pt files in {lora_dir}")
    print(f"C5: {len(lora_files)} lora files, {n_prompts} prompts × {CATEGORIES}, "
          f"k={list(K_SWEEP)}, keep_o_proj={keep_o_proj}, device={device}")

    # -- Prompts (one shared set: generated once, used by all 3 models) ----
    prompts: dict[str, list[dict]] = {}
    for cat in CATEGORIES:
        data = load_calibration_data(
            small_model,
            max_length=512,
            min_length=256,
            n_samples=n_prompts,
            domain=cat,
        )
        prompts[cat] = data[:n_prompts]
        print(f"  {cat}: {len(prompts[cat])} prompts")

    # -- Model 1: base 0.8B ----------------------------------------------
    print("Loading 0.8B base...")
    tokenizer = AutoTokenizer.from_pretrained(small_model)
    m_base = _load(small_model, large_model, device, "small")
    base_topk = {
        cat: [_topk_all(m_base, p, list(K_SWEEP), device) for p in prompts[cat]]
        for cat in CATEGORIES
    }
    _free_model(m_base)

    # -- Model 2: 0.8B + cumulative deltas --------------------------------
    print("Loading 0.8B + deltas...")
    m_ed = _load(small_model, large_model, device, "small")
    with torch.no_grad():
        for lf in lora_files:
            ld = torch.load(
                os.path.join(lora_dir, lf), map_location="cpu", weights_only=True
            )
            # lm_head: same head for every pair -> cumulative sum
            m_ed.lm_head.weight.data += ld["B_lh"].to(m_ed.lm_head.weight) @ ld[
                "A_lh"
            ].to(m_ed.lm_head.weight)
            if keep_o_proj:
                a_attn = ld.get("A_attn")
                b_attn = ld.get("B_attn")
                l_s = int(ld["l_S"])
                if a_attn is not None and b_attn is not None:
                    layer_mod = m_ed.model.layers[l_s]
                    o_mod = getattr(getattr(layer_mod, "self_attn", None), "o_proj", None)
                    if o_mod is not None:
                        o_mod.weight.data += b_attn.to(o_mod.weight) @ a_attn.to(
                            o_mod.weight
                        )
    ed_topk = {
        cat: [_topk_all(m_ed, p, list(K_SWEEP), device) for p in prompts[cat]]
        for cat in CATEGORIES
    }
    _free_model(m_ed)

    # -- Model 3: 4B fp16 (OOM fallback per spec) -------------------------
    print("Loading 4B fp16...")
    m_4b = None
    note = None
    try:
        m_4b = _load(small_model, large_model, device, "large")
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"  4B OOM ({e}); empty_cache + retry once")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            m_4b = _load(small_model, large_model, device, "large")
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e2:
            note = f"4B skipped: {e2}"
            print(f"  [note] {note}")
    if m_4b is not None:
        l4b_topk = {
            cat: [_topk_all(m_4b, p, list(K_SWEEP), device) for p in prompts[cat]]
            for cat in CATEGORIES
        }
        _free_model(m_4b)
    else:
        l4b_topk = None

    # -- Report ------------------------------------------------------------
    report: dict = {
        "description": (
            "C5 top-k report: model2 = 0.8B + cumulative lm_head LoRA deltas "
            "(+ o_proj deltas if keep_o_proj) vs model1 = 0.8B base and "
            "model3 = 4B fp16. agreement = |topk(m2) ∩ (topk(m1) ∪ topk(m3))| / k."
        ),
        "small_model": small_model,
        "large_model": large_model,
        "keep_o_proj": keep_o_proj,
        "lora_files": lora_files,
        "n_prompts": n_prompts,
        "k_sweep": list(K_SWEEP),
        "note": note,
        "categories": {},
    }
    all_agreements = []
    for cat in CATEGORIES:
        n_cat = len(prompts[cat])
        cat_report: dict = {}
        for i, k in enumerate(K_SWEEP):
            agrees = []
            examples = []
            for pi in range(n_cat):
                t2 = set(ed_topk[cat][pi][i])
                union = set(base_topk[cat][pi][i])
                if l4b_topk is not None:
                    union |= set(l4b_topk[cat][pi][i])
                a = len(t2 & union) / k
                agrees.append(a)
                if pi < 3:
                    examples.append(
                        {
                            "prompt_tail": tokenizer.decode(
                                prompts[cat][pi]["input_ids"][0][-8:]
                            ),
                            "topk_m2": [
                                tokenizer.decode([t]) for t in ed_topk[cat][pi][i]
                            ],
                            "topk_m1": [
                                tokenizer.decode([t]) for t in base_topk[cat][pi][i]
                            ],
                            "topk_m3": (
                                [
                                    tokenizer.decode([t])
                                    for t in l4b_topk[cat][pi][i]
                                ]
                                if l4b_topk is not None
                                else None
                            ),
                            "agreement": a,
                        }
                    )
            mean = sum(agrees) / n_cat
            cat_report[f"k={k}"] = {
                "agreement": mean,
                "garbage_rate": 1.0 - mean,
                "examples": examples,
            }
            all_agreements.append(mean)
        report["categories"][cat] = cat_report
    report["overall"] = {
        "agreement": sum(all_agreements) / len(all_agreements),
        "garbage_rate": 1.0 - sum(all_agreements) / len(all_agreements),
    }

    path = os.path.join(results_dir, "exp17", "c5_topk_report.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {path}")
    print(f"overall agreement={report['overall']['agreement']:.3f}, "
          f"garbage_rate={report['overall']['garbage_rate']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp17 C5: top-k report")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--lora-dir", default="")
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument(
        "--keep-o-proj",
        action="store_true",
        help="also apply each pair's o_proj delta to its l_S (default: lm_head only)",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    main(**vars(args))
