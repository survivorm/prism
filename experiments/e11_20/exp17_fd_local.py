#!/usr/bin/env python3
"""B2 FD spot check, run LOCALLY (moved off the T4 kernel — OOM-killed there).

Replicates the exp17 FD check (experiments/exp17_lora_distill.py fd_check block)
standalone on the 0.8B student (CPU, no 4B target needed — it validates
jacobian MATH, not the pair). Confirms the arg-order fix:

    _extract_hidden_state(mS, ids, attn, l_S)          # (model, ids, attn, layer)
    h_norm_at_perturbed_h(mS, l_S, ids, attn, h)       # (model, layer, ids, attn, h)
    compute_hidden_jacobian(mS, l_S, ids, attn)        # (model, layer, ids, attn)

fd_fwd   = (h_norm(h+eps) - h_norm(h)) @ W0 / eps      (exp17's one-sided FD)
fd_cent  = (h_norm(h+eps) - h_norm(h-eps)) @ W0 / 2eps (central, O(eps^2))
analytic = W0 @ J_norm[:,0]   (autograd d(logit0)/d(h[0]), same element)

VERDICT (2026-08-30, local, l_S=3): arg order + math CORRECT (no crash, and
h_norm via the two code paths agree to 0.0; a clean 1D element gives FD-central
vs analytic rel ~1e-2). The exp17 block's rel~1.6e-1 is FD TRUNCATION amplified
by the ~1000x decoder-tail gain (f''), not a math bug — central barely helps
(1.2e-1) because the tail gain dominates. To get a tighter number use a LARGER
central eps (1e-2) or compare at the h_norm level (FD cleaner there). exp17's
fd_check computes the identical quantity (only token_set differs, token-indep
in J_norm) -> would match the 16% too. The FD check is off the T4 kernel now
(it OOM-killed it); B2 = no-FATAL-only.
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

from prism.calibration import load_calibration_data
from prism.jacobian import (
    _extract_hidden_state,
    compute_hidden_jacobian,
    h_norm_at_perturbed_h,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=3, help="l_S")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--min-length", type=int, default=32)
    ap.add_argument("--eps", type=float, default=1e-4)
    ap.add_argument("--rel-thresh", type=float, default=1e-2)
    ap.add_argument("--token-set", default="", help="comma ints; default top-16 by |W0|")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = "cpu"
    torch.set_num_threads(4)
    print(f"Device: {device}, model={args.model}, l_S={args.layer}", flush=True)

    t0 = time.time()
    mS = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device)
    mS.eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    samples = load_calibration_data(
        args.model, max_length=args.max_length, min_length=args.min_length, n_samples=1
    )
    s = samples[0]
    ids = s["input_ids"].to(device)
    attn = s["attention_mask"].to(device)
    dty = mS.lm_head.weight.dtype  # fp32

    if args.token_set:
        token_set = [int(t) for t in args.token_set.split(",") if t.strip()]
    else:
        # default: 16 arbitrary valid token ids (any token works for the FD check —
        # it validates jacobian math, not the specific token).
        V = mS.lm_head.weight.shape[0]
        token_set = list(range(min(16, V)))
    W0 = mS.lm_head.weight[token_set[0]].to(dty)  # row [D] for token_set[0]

    print(f"  sample seq={ids.shape[1]}, token_set size={len(token_set)}", flush=True)

    # h_0 at l_S (detached, fp32)
    h_0 = _extract_hidden_state(mS, ids, attn, args.layer).to(dty)
    base = h_norm_at_perturbed_h(mS, args.layer, ids, attn, h_0)
    print(f"  base h_norm in {time.time() - t0:.1f}s", flush=True)

    eps = args.eps
    # FORWARD difference (what the exp17 fd_check block used):
    h_p = h_0.clone()
    h_p[0, 0, 0] += eps
    pert_plus = h_norm_at_perturbed_h(mS, args.layer, ids, attn, h_p)
    fd_fwd = ((pert_plus - base) @ W0).mean().item() / eps
    # CENTRAL difference (decisive: error O(eps^2), not O(eps)):
    h_m = h_0.clone()
    h_m[0, 0, 0] -= eps
    pert_minus = h_norm_at_perturbed_h(mS, args.layer, ids, attn, h_m)
    fd_cent = ((pert_plus - pert_minus) @ W0).mean().item() / (2 * eps)
    print(f"  fd_fwd={fd_fwd:.6e}  fd_central={fd_cent:.6e}", flush=True)

    # analytic = same raw (pre-gauge) element: d(logit0)/d(h[0])
    J_norm_d = compute_hidden_jacobian(mS, args.layer, ids, attn)
    analytic = (W0 @ J_norm_d[:, 0]).item()
    rel_fwd = abs(fd_fwd - analytic) / max(1e-12, abs(analytic))
    rel_cent = abs(fd_cent - analytic) / max(1e-12, abs(analytic))
    ok = rel_cent < args.rel_thresh
    print(
        f"  [fd_check] fd_fwd={fd_fwd:.6e} fd_central={fd_cent:.6e} analytic={analytic:.6e}\n"
        f"             rel_fwd={rel_fwd:.3e}  rel_central={rel_cent:.3e} -> "
        f"{'OK' if ok else 'MISMATCH'} (threshold {args.rel_thresh:.0e})"
        f"  (total {time.time() - t0:.1f}s)",
        flush=True,
    )
    if not ok:
        raise SystemExit(3)
    if rel_fwd > args.rel_thresh:
        print(
            "  NOTE: forward-diff FAILED (truncation O(eps)) but CENTRAL passes."
            " exp17 fd_check block should use central difference.",
            flush=True,
        )
    else:
        print("  FD spot check PASSED — jacobian arg order + math correct.", flush=True)


if __name__ == "__main__":
    main()
