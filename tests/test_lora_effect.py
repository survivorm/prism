"""LORA-effect test, TOP-P RETENTION metric (user-directed: top-10 is fragile).

top-p set = smallest prefix of the distribution (sorted desc) covering mass >= p.
retention: two views (both should drop if the edit moves toward garbage):
  (a) student_view = sum(teacher_prob[t] for t in student's top-p set)
       = teacher's mass that lands in the student's confident region.
  (b) teacher_view = sum(student_prob[t] for t in teacher's top-p set)
       = student's mass that stays in the teacher's confident region.
base 0.8B should score high on both; 0.8B + exp16 LORA (garbage) should score low.

Cost: 1 x 0.8B (base + edited via precomputed logits) + 1 x 4B (teacher) per query.
Run ONE process, low threads, background:
  nohup .venv/bin/python tests/test_lora_effect.py --n 6 --maxlen 24 > /data/prism_lora_effect.log 2>&1 &
"""
import argparse, json, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SMALL = "/data/Qwen/Qwen3.5-0.8B"
LARGE = "/data/Qwen/Qwen3.5-4B"

QUERIES = [
    "In a triangle, the sum of the interior angles is",
    "The capital of France is",
    "To convert Celsius to Fahrenheit you multiply by",
    "The derivative of x^2 with respect to x is",
    "In HTTP, the method that retrieves a resource is",
    "The largest planet in the solar system is",
    "Water boils at 100 degrees",
    "The square root of 144 is",
    "A function that maps each element to another is called a",
    "The speed of light in vacuum is approximately",
    "In Python, the keyword to define a function is",
    "The chemical formula for water is",
]


def top_p_set(probs, p):
    """Indices of the smallest prefix (sorted by prob desc) covering mass >= p."""
    order = torch.argsort(probs, descending=True)
    cum = torch.cumsum(probs[order], dim=0)
    n = int((cum >= p).nonzero()[0, 0].item()) + 1
    return order[:n]


def retention(model_p, model_set, teacher_p, teacher_set):
    """returns (teacher mass in model's top-p set, model mass in teacher's top-p set)."""
    teacher_in_model = teacher_p[model_set].sum()   # teacher mass landing in model's confident region
    model_in_teacher = model_p[teacher_set].sum()   # model mass staying in teacher's confident region
    return teacher_in_model, model_in_teacher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--maxlen", type=int, default=24)
    ap.add_argument("--p", type=float, default=0.9)
    ap.add_argument("--out", default="/data/prism_lora_effect.json")
    args = ap.parse_args()
    torch.set_num_threads(2)
    dev = "cpu"

    print("[1/3] loading models...", flush=True)
    tok = AutoTokenizer.from_pretrained(SMALL)
    small = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float32).to(dev).eval()
    W_orig = small.lm_head.weight
    large = AutoModelForCausalLM.from_pretrained(LARGE, dtype=torch.bfloat16).to(dev).eval()

    print("[2/3] loading LORA dW (exp16_lm_head_final.pt)...", flush=True)
    dW = torch.load("/data/prism_ckpt/exp16_lm_head_final.pt", map_location="cpu")["delta_W"].float().to(dev)
    W_edit = W_orig + dW
    print(f"  ||dW||_F={dW.norm():.1f}, ||W||_F={W_orig.float().norm():.1f}", flush=True)

    print(f"[3/3] top-p retention (p={args.p}) on {args.n} queries...", flush=True)
    rows = []
    for i, q in enumerate(QUERIES[:args.n]):
        h3 = tok(q, return_tensors="pt").to(dev)
        hs3 = torch.cat([h3["input_ids"], tok(" 12", return_tensors="pt").to(dev)["input_ids"][:, 1:]], dim=1)
        hs4 = torch.cat([tok(q, return_tensors="pt").to(dev)["input_ids"],
                         tok(" 12", return_tensors="pt").to(dev)["input_ids"][:, 1:]], dim=1)
        hs3 = hs3[:, :args.maxlen]; hs4 = hs4[:, :args.maxlen]
        with torch.no_grad():
            h = small(hs3, output_hidden_states=True).hidden_states[-1][0, -1, :]
            base_l = W_orig @ h
            edit_l = W_edit @ h
            t_l = large(hs4).logits[0, -1, :].float()
        base_p = F.softmax(base_l, dim=-1)
        edit_p = F.softmax(edit_l, dim=-1)
        teac_p = F.softmax(t_l, dim=-1)
        bs = top_p_set(base_p, args.p)
        es = top_p_set(edit_p, args.p)
        ts = top_p_set(teac_p, args.p)
        b_in_t, t_in_b = retention(base_p, bs, teac_p, ts)
        e_in_t, t_in_e = retention(edit_p, es, teac_p, ts)
        row = dict(
            i=i, n_tok=int(hs3.shape[1]), txt=q[:50],
            base_model_in_teacher=float(b_in_t), base_teacher_in_base=float(t_in_b),
            edit_model_in_teacher=float(e_in_t), edit_teacher_in_edit=float(t_in_e),
            base_set=int(len(bs)), edit_set=int(len(es)), teacher_set=int(len(ts)),
        )
        rows.append(row)
        print(f"  [{i}] p={args.p} base in-teacher={b_in_t:.3f} | edit in-teacher={e_in_t:.3f} "
              f"| sets base/edit/teacher {len(bs)}/{len(es)}/{len(ts)}", flush=True)

    import statistics
    def avg(k): return round(statistics.mean(r[k] for r in rows), 4)
    summary = {
        "p": args.p,
        "base_model_in_teacher": avg("base_model_in_teacher"), "edit_model_in_teacher": avg("edit_model_in_teacher"),
        "base_teacher_in_base": avg("base_teacher_in_base"), "edit_teacher_in_edit": avg("edit_teacher_in_edit"),
    }
    out = dict(n=args.n, maxlen=args.maxlen, p=args.p, summary=summary, rows=rows)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SUMMARY (top-p retention, p=%.2f) ===" % args.p, flush=True)
    print(f"  model mass in TEACHER's top-p region: base {summary['base_model_in_teacher']} -> edit {summary['edit_model_in_teacher']}")
    print(f"  teacher mass in MODEL's top-p region: base {summary['base_teacher_in_base']} -> edit {summary['edit_teacher_in_edit']}")
    print(f"  saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
