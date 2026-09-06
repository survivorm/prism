#!/usr/bin/env python3
"""Experiment 2 (v3): geometry of the 4B->0.8B gap across layer pairs.

Motivation: the v1/v2 grid (anchors {0,8,16,24,30}, hand-picked 3 targets
each) was a CPU compromise and left the alignment valley undersampled.
v3 probes the geometry properly:

  * dense anchor grid: every 4 layers of the 32-layer 4B model
  * adaptive target search: for each anchor start at the PROPORTIONAL
    target p = round(l_L * n_S / n_L) and walk toward deeper AND shallower
    targets while the gap keeps decreasing; stop on a rise (valley found),
    a bound, or the per-direction cap
  * Jacobian cache per (model, layer, sample): each layer's J is computed
    once and reused by every pair that needs it

Metric: GAP = 1 - cos (probability-weighted) = prism.bridge.jacobian_gap.
LOWER gap = better alignment. ("bridge"/cos = alignment, higher = better;
"gap"/1-cos = the loss, lower = better.)

Question: does each 4B anchor have a single valley in gap-vs-target-depth,
and does the valley sit at the proportional partner p(l), the end partner
e(l) = l-8, or elsewhere?

Outputs:
  results/exp2_heatmap.json   (per-anchor target traces + valley + params)
  results/exp2_geometry.png   (gap vs target, one line per anchor)
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prism.calibration import extract_embeddings, uniform_layer_mapping
from prism.bridge import jacobian_gap
from prism.jacobian import compute_jacobian_for_tokens


def end_mapping(l_large: int, offset: int = 8) -> int:
    return max(0, l_large - offset)


class JCache:
    """Per-(model, layer) Jacobian cache; reset each sample."""

    def __init__(self):
        self._d = {}

    def reset(self):
        self._d = {}

    def get(self, model, model_id: str, layer: int, input_ids, attn, token_set):
        key = (model_id, layer)
        if key not in self._d:
            self._d[key] = compute_jacobian_for_tokens(
                model, layer, input_ids, attn, token_set
            )
        return self._d[key]


def probe_anchor_targets(gap_fn, p, n_targets_max, cap, tol=0.0):
    """Ordered list of (l_S, gap) tracing the valley around proportional p.

    Walk deeper and shallower from p while gap decreases (allowing small
    fluctuations up to `tol`). Stop when gap rises by more than tol
    (monotonicity break), at a bound, or at the cap. Returns the full
    ordered trace; the caller extracts turning points from it.

    If tol > 0, a rise of <= tol is tolerated (noise), and we continue
    probing. If tol == 0, strict: stop on any non-decrease.
    """
    lo, hi = 0, n_targets_max - 1

    deeper = []
    prev = gap_fn(p)
    deeper.append((p, prev))
    x = p
    for _ in range(cap):
        if x + 1 > hi:
            break
        g = gap_fn(x + 1)
        if g <= prev + tol:  # still decreasing (or flat within tol)
            deeper.append((x + 1, g))
            x += 1
            prev = min(prev, g)  # track the running minimum
        else:
            break

    shallower = []
    prev = gap_fn(p)
    x = p
    for _ in range(cap):
        if x - 1 < lo:
            break
        g = gap_fn(x - 1)
        if g <= prev + tol:
            shallower.append((x - 1, g))
            x -= 1
            prev = min(prev, g)
        else:
            break

    # ordered ascending in l_S: shallower (reversed) then p, deeper
    return list(reversed(shallower)) + deeper


def turning_points(trace):
    """Given ordered (l_S, gap) list, return list of l_S where direction changes."""
    tps = []
    for i in range(1, len(trace) - 1):
        d_prev = trace[i][1] - trace[i-1][1]
        d_next = trace[i+1][1] - trace[i][1]
        if d_prev < 0 and d_next > 0:  # decreasing -> increasing = local min
            tps.append((trace[i][0], "min"))
        elif d_prev > 0 and d_next < 0:  # increasing -> decreasing = local max
            tps.append((trace[i][0], "max"))
    return tps


def load_samples(targets_dir, tok, n_samples, max_length, min_length):
    samples = []
    if targets_dir and os.path.isdir(targets_dir):
        for f in sorted(os.listdir(targets_dir)):
            if f.endswith(".json"):
                samples.append(json.load(open(os.path.join(targets_dir, f))))
    if not samples:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In 1969, humans first walked on the Moon.",
            "def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)",
        ]
        for t in texts:
            ids = tok.encode(t, return_tensors="pt")
            samples.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return samples[:n_samples]


def main(
    large_model: str,
    small_model: str,
    device: str,
    max_length: int,
    min_length: int,
    targets_dir,
    results_dir: str,
    anchors=None,
    cap: int = 6,
    n_samples: int = 8,
    K: int = 64,
):
    t0 = time.time()
    print(f"Loading models: {large_model}, {small_model}")
    model_L = AutoModelForCausalLM.from_pretrained(
        large_model, torch_dtype=torch.float32, device_map=device
    )
    model_S = AutoModelForCausalLM.from_pretrained(
        small_model, torch_dtype=torch.float32, device_map=device
    )
    tok = AutoTokenizer.from_pretrained(small_model)

    n_L = model_L.config.num_hidden_layers
    n_S = model_S.config.num_hidden_layers
    if anchors is None:
        anchors = list(range(0, n_L, 4))
    prop = uniform_layer_mapping(n_L, n_S)

    samples = load_samples(targets_dir, tok, n_samples, max_length, min_length)
    print(f"n_L={n_L}, n_S={n_S}, anchors={anchors}, samples={len(samples)}, K={K}")
    for a in anchors:
        print(f"  anchor {a:2d}: proportional={prop[a]}, end={end_mapping(a)}")

    E_L = extract_embeddings(model_L).to(device)
    E_S = extract_embeddings(model_S).to(device)

    jL = JCache()
    jS = JCache()
    # traces[a][l_S] -> list of per-sample gap values
    traces = {int(a): {} for a in anchors}

    for si, s in enumerate(samples):
        ids = s["input_ids"].to(device)
        attn = s["attention_mask"].to(device)
        jL.reset()
        jS.reset()

        with torch.no_grad():
            logits_L = model_L(input_ids=ids, attention_mask=attn).logits[:, -1]
        token_set = sorted(logits_L[0].topk(K, dim=-1).indices.tolist())
        p_L = F.softmax(logits_L[0][token_set].float(), dim=-1)

        def gap_fn(l_S, _lL=None):
            J_L = jL.get(model_L, "L", _lL, ids, attn, token_set)
            J_S = jS.get(model_S, "S", l_S, ids, attn, token_set)
            return jacobian_gap(J_L, J_S, E_L, E_S, p_L)

        for a in anchors:
            a = int(a)
            fn = lambda l_S, _a=a: gap_fn(l_S, _a)  # noqa: E731
            ordered = probe_anchor_targets(fn, prop[a], n_S, cap, tol=0.0)
            for l_S, g in ordered:
                traces[a].setdefault(int(l_S), []).append(g)
            trace_tps = turning_points(ordered)
            if trace_tps:
                traces[a].setdefault("_tps", []).append([t for t in trace_tps])
        print(f"  sample {si + 1}/{len(samples)} done ({time.time() - t0:.0f}s)")

    # ---- aggregate ----
    out = {
        "params": {
            "large_model": large_model,
            "small_model": small_model,
            "n_large": n_L,
            "n_small": n_S,
            "anchors": [int(a) for a in anchors],
            "n_samples": len(samples),
            "K": K,
            "max_length": max_length,
            "min_length": min_length,
            "cap": cap,
            "tol": 0.0,
            "metric": "gap = 1 - weighted cosine (jacobian_gap); lower = better alignment",
        },
        "proportional_target": {
            str(int(a)): prop[int(a)] for a in anchors
        },
        "anchors": {},
        "rows": [],
    }

    for a in anchors:
        a = int(a)
        targets = []
        for l_S in sorted(traces[a].keys()):
            gs = traces[a][l_S]
            targets.append({
                "l_S": int(l_S),
                "gap_mean": float(sum(gs) / len(gs)),
                "gap_min": float(min(gs)),
                "is_proportional": int(l_S) == prop[a],
                "is_end": int(l_S) == end_mapping(a),
            })
        valley = min(targets, key=lambda t: t["gap_mean"]) if targets else None
        tps = traces[a].get("_tps", [])
        out["anchors"][str(a)] = {
            "proportional": prop[a],
            "end": end_mapping(a),
            "targets": targets,
            "valley_l_S": valley["l_S"] if valley else None,
            "valley_gap": valley["gap_mean"] if valley else None,
            "valley_is_proportional": bool(valley and valley["l_S"] == prop[a]) if valley else None,
            "valley_is_end": bool(valley and valley["l_S"] == end_mapping(a)) if valley else None,
            "turning_points": tps[:1] if tps else [],  # sample-0 trace turning points
        }
        for t in targets:
            out["rows"].append({
                "l_L": a, "l_S": t["l_S"], "gap": t["gap_mean"],
                "is_proportional": t["is_proportional"], "is_end": t["is_end"],
            })

    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "exp2_heatmap.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")

    # ---- summary ----
    print(f"\n{'4B':>4} {'prop':>5} {'end':>4} {'valley':>7} {'gap':>8}  kind   turning_pts")
    for a in anchors:
        a = int(a)
        d = out["anchors"][str(a)]
        kind = []
        if d.get("valley_is_proportional"):
            kind.append("prop")
        if d.get("valley_is_end"):
            kind.append("end")
        kind = "+".join(kind) if kind else "other"
        tps = d.get("turning_points", [])
        tps_str = ",".join(f"L{t[0]}:{t[1]}" for t in tps) if tps else "-"
        print(f"{a:>4} {d['proportional']:>5} {d['end']:>4} "
              f"{str(d['valley_l_S']):>7} {d['valley_gap']:>8.4f}  {kind}  {tps_str}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))
        for a in anchors:
            a = int(a)
            d = out["anchors"][str(a)]
            xs = [t["l_S"] for t in d["targets"]]
            ys = [t["gap_mean"] for t in d["targets"]]
            ax.plot(xs, ys, "o-", label=f"4B L{a} (p={d['proportional']})")
        ax.set_xlabel("0.8B target layer l_S")
        ax.set_ylabel("gap = 1 - cosine (lower = better alignment)")
        ax.set_title("Exp 2 (v3): gap vs 0.8B target, per 4B anchor")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
        p2 = os.path.join(results_dir, "exp2_geometry.png")
        fig.tight_layout()
        fig.savefig(p2, dpi=120)
        print(f"Saved plot to {p2}")
    except Exception as e:  # pragma: no cover
        print(f"  (plot skipped: {e})")

    print(f"\nDone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    ap.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--min-length", type=int, default=32)
    ap.add_argument("--targets", default=None)
    ap.add_argument("--results", default="results")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--cap", type=int, default=6)
    a = ap.parse_args()
    main(a.large_model, a.small_model, a.device, a.max_length, a.min_length,
         a.targets, a.results, n_samples=a.n_samples, K=a.K, cap=a.cap)
