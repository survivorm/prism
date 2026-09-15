"""
TEST: доза-ответ LORA по ВЕЛИЧИНЕ (прокси «ранних чекпоинтов»).
exp16 edit линейна: logit(h; W_orig + s·ΔW) = base_logit + s·(h·ΔWᵀ).
s=1 = финальный чекпоинт (bridge 0.921, ‖ΔW‖=128.8); s<1 = «меньше обучения».
4B-логит считаются 1× на запрос (s-независимы) → все s почти бесплатны.

Считаем agreement/garbage vs 4B-учитель по s. Вопрос: меньше edit = меньше
damage (линейно), или damage нелинеен (есть «сладкая точка»/порог)?

1 бэкаунд-процесс, нити=2 (не зациклить комп).
Запуск:  .venv/bin/python tests/test_lora_dose.py --n 6
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "/data/Qwen/Qwen3.5-0.8B"
LARGE = "/data/Qwen/Qwen3.5-4B"
LMHEAD = "/data/prism_ckpt/exp16_lm_head_final.pt"
HOLDOUT = "/data/prism_holdout_c4.json"
S_VALUES = (0.1, 0.25, 0.5, 0.75, 1.0)


def _cap_threads():
    import os
    n = int(os.environ.get("PRISM_THREADS", "2"))
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass
    print(f"[threads] capped to {n}", flush=True)


def topk_set(logits, k):
    return set(logits.topk(k).indices.tolist())


def main(n=6, kset=(1, 5, 10), plausible_k=200, maxlen=24, device="cpu"):
    t0 = time.time()
    _cap_threads()
    tok = AutoTokenizer.from_pretrained(BASE)
    queries = json.load(open(HOLDOUT))[:n]
    d = torch.load(LMHEAD, map_location="cpu", weights_only=False)
    dW = d["delta_W"]
    print(f"[edit] ‖ΔW‖F={d['dose']:.1f}, bridge@end(s=1)={d['bridge']:.3f}", flush=True)

    m08 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    m08.eval()
    m4 = AutoModelForCausalLM.from_pretrained(LARGE, torch_dtype=torch.bfloat16)
    m4.eval()
    print(f"[loaded] {time.time()-t0:.0f}s", flush=True)

    per_s = {s: [] for s in S_VALUES}
    for i, q in enumerate(queries):
        ids = tok(q["text"])["input_ids"][:maxlen]
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            o08 = m08(input_ids=x, output_hidden_states=True)
            h = o08.hidden_states[-1][0, -1]
            base_logit = o08.logits[0, -1].float()      # h·W_orig.T
            delta_logit = h @ dW.T                      # h·ΔW.T  (edit at s=1)
        with torch.no_grad():
            t_logit = m4(input_ids=x).logits[0, -1].float()
        t_top = topk_set(t_logit, plausible_k)
        t_p = F.softmax(t_logit, dim=-1)
        t_order = torch.argsort(t_p, descending=True)
        n_tp = max(1, int(torch.cumsum(t_p[t_order], 0).ge(0.9).nonzero()[0].item()) + 1)
        t_top_p = t_order[:n_tp]  # teacher's top-p(0.9) set
        base_p = F.softmax(base_logit, dim=-1)
        for s in S_VALUES:
            edit = base_logit + s * delta_logit
            e_p = F.softmax(edit, dim=-1)
            rec = {"i": i, "s": s}
            for k in kset:
                rec[f"agr_{k}"] = len(topk_set(edit, k) & topk_set(t_logit, k)) / k
                rec[f"garb_{k}"] = len(topk_set(edit, k) - t_top) / k
                rec[f"agr_base_{k}"] = len(topk_set(base_logit, k) & topk_set(t_logit, k)) / k
                rec[f"garb_base_{k}"] = len(topk_set(base_logit, k) - t_top) / k
            rec["ret"] = float(e_p[t_top_p].sum())          # top-p retention: model mass in teacher's top-p region
            rec["ret_base"] = float(base_p[t_top_p].sum())
            per_s[s].append(rec)
        print(f"  [q{i}] done ({time.time()-t0:.0f}s elapsed)", flush=True)

    def avg(vals, key):
        return sum(r[key] for r in vals) / len(vals)

    print("\n" + "=" * 74)
    print(f"ДОЗА-ОТВЕТ: s (доля exp16 ΔW) → agreement/garbage vs 4B, n={n}")
    base_row = f"{'s=0':>5}  "
    for k in kset:
        base_row += f"{avg(per_s[S_VALUES[0]], f'agr_base_%d' % k):.3f}  "
    for k in kset:
        base_row += f"{avg(per_s[S_VALUES[0]], f'garb_base_%d' % k):.3f} "
    print(base_row.rstrip() + "   <- base (no LORA)")
    for s in S_VALUES:
        row = f"{s:>5}  "
        for k in kset:
            row += f"{avg(per_s[s], f'agr_%d' % k):.3f}  "
        for k in kset:
            row += f"{avg(per_s[s], f'garb_%d' % k):.3f} "
        print(row.rstrip())
    print("=" * 74)

    g = [avg(per_s[s], "garb_10") for s in S_VALUES]
    a = [avg(per_s[s], "agr_10") for s in S_VALUES]
    g_base = avg(per_s[S_VALUES[0]], "garb_base_10")
    a_base = avg(per_s[S_VALUES[0]], "agr_base_10")
    mono = all(g[i] <= g[i + 1] for i in range(len(g) - 1))
    print(f"\nTREND (k=10):  s        " + "  ".join(f"{s:.2f}" for s in S_VALUES))
    print(f"               agr    " + "  ".join(f"{x:.3f}" for x in a) + f"  (base {a_base:.3f})")
    print(f"               garb   " + "  ".join(f"{x:.3f}" for x in g) + f"  (base {g_base:.3f})")
    print(f"\nVERDICT: garbage {'растёт монотонно' if mono else 'НЕ монотонно'} с s. "
          f"{'Нет «сладкой точки»: damage монотонен по величине edit — меньше edit = меньше damage.' if mono else 'Есть нелинейность/порог — существует «сладкая точка».'}")
    print(f"\nTOP-P RETENTION (масса модели в top-p(0.9) регионе учителя):  base={avg(per_s[S_VALUES[0]], 'ret_base'):.3f}  "
          + "  ".join(f"s{s}={avg(per_s[s], 'ret'):.3f}" for s in S_VALUES))
    print(f"\n[total] {time.time()-t0:.0f}s")

    json.dump({"n": n, "s": list(S_VALUES), "kset": list(kset),
               "agr10": {f"s{s}": avg(per_s[s], "agr_10") for s in S_VALUES},
               "garb10": {f"s{s}": avg(per_s[s], "garb_10") for s in S_VALUES},
               "ret": {f"s{s}": avg(per_s[s], "ret") for s in S_VALUES},
               "ret_base": avg(per_s[S_VALUES[0]], "ret_base"),
               "base": {"agr10": a_base, "garb10": g_base}},
              open("/data/prism_lora_dose.json", "w"), ensure_ascii=False, indent=1)
    print("saved /data/prism_lora_dose.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--maxlen", type=int, default=24)
    a = ap.parse_args()
    main(n=a.n, maxlen=a.maxlen)
