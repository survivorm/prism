"""Experiment 17: Per-layer LoRA distillation of the bridge.

LoRA r=4 on `lm_head` + attention o_proj of each layer from the pair
sequence (v23).

Bridge (v23, matrix-free): on refresh steps (every k=100, per sample) the
J_S_p rows are built by `compute_hidden_jacobian_rows` with W_rows =
(W + B_lh@A_lh)[token_set] LIVE in the graph and create_graph=True — one
backward per row through h reaches lm_head AND o_proj (HVP, the D^3
Hessian is never materialized). Fill/stale steps keep the cheap dense
J_norm (create_graph=False, a 4MB [D,D] fp32 matrix) so the lm_head bridge
gradient keeps flowing between refreshes; o_proj is refresh-only.
J refresh OOM/exception = FATAL exit 3 (v23: no degrade — a degraded run
is a lie). B_attn is randn-init (dL/dA_attn > 0 from step 0).

After each pair BOTH lora deltas are frozen into weights (lm_head +
o_proj).

Target pairing: samples match stage0 cache entries by exact input_ids;
on mismatch the bridge for that sample is skipped (KL unaffected).

Loss = bridge(J_tgt_p, J_S_p) + kl_weight * KL(p_ed || p_orig).
Max 300 steps per pair on GPU (--steps overrides; 40 on local CPU smoke).

Two orders (GPU), late→early only (CPU).
CPU config: pairs [23, 15, 7] mapped to small model layers.
"""

import argparse
import json
import os
import re
import contextlib
import time
from collections.abc import Callable

import torch
# --- XLA / TPU support (no-op on GPU/CPU; active when device == "xla") ---
try:
    import torch_xla.core.xla_model as _xm
    _XLA = True
except Exception:
    _xm = None
    _XLA = False


@contextlib.contextmanager
def _step_cm():
    if _XLA and str(_DEVICE).startswith("xla"):
        with _xm.step():
            yield
    else:
        with contextlib.nullcontext():
            yield

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

from prism.calibration import load_calibration_data
from prism.cache_io import find_targets_dir
from prism.layer_selection import classify_pair, get_layer_types
from prism.jacobian import (
    _extract_hidden_state,
    bridge_rows_backward,
    compute_hidden_jacobian,
    compute_hidden_jacobian_rows,
    h_norm_at_perturbed_h,
    softmax_jacobian,
)


def _layer_order(pairs: list[str], device: str) -> dict[str, list[str]]:
    late_to_early = sorted(pairs, key=lambda p: int(p.split("_")[1]), reverse=True)
    if device == "cpu":
        wanted = [23, 15, 7]
        subset = [p for p in late_to_early if int(p.split("_")[1]) in wanted]
        if not subset:
            subset = late_to_early[:7]
        return {"late_to_early": subset}
    return {
        "late_to_early": late_to_early,
        "early_to_late": list(reversed(late_to_early)),
    }


def main(
    small_model: str = "Qwen/Qwen3.5-0.8B",
    device: str = "auto",
    max_length: int = 64,
    min_length: int = 32,
    targets_dir: str = "results/targets",
    results_dir: str = "results",
    steps: int = 0,
    fd_check: int = 0,
    pairs: str = "",
):
    torch.manual_seed(42)
    start = time.time()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "xla":
        assert _XLA, "torch_xla not installed"
        device = _xm.xla_device()
        torch.set_num_threads(1)
    if device == "cpu":
        torch.set_num_threads(4)
    print(f"Device: {device}")
    os.makedirs(results_dir, exist_ok=True)

    # -- Calibration data -------------------------------------------------
    n_samples = 2 if device == "cpu" else 4
    print(f"Loading {n_samples} calibration samples...")
    samples = load_calibration_data(
        small_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=n_samples,
    )

    # -- Load target cache ------------------------------------------------
    # FUSE-safe, layout-independent (prism.cache_io): preferred dir, else
    # listdir walk of /kaggle/input — no hardcoded slug/subpath.
    targets_dir = find_targets_dir(targets_dir)
    print(f"Using target cache at {targets_dir}")

    # -- Load target cache ------------------------------------------------
    cache_files = sorted(
        f
        for f in os.listdir(targets_dir)
        if f.endswith(".pt") and not f.startswith(".")
    )
    if not cache_files:
        raise FileNotFoundError(
            f"No target cache .pt files found in {targets_dir}. Run Stage 0 first."
        )

    cache_by_pair: dict[str, list[dict]] = {}
    for cf in cache_files:
        data = torch.load(os.path.join(targets_dir, cf), weights_only=True)
        pair = data["pair"]
        # Tag with the sample index stored in the filename ({pair}_{si}.pt)
        m = re.search(r"_(\d+)\.pt$", cf)
        data["_si"] = int(m.group(1)) if m else -1
        cache_by_pair.setdefault(pair, []).append(data)
    # Sort each pair's targets by sample index (filename sort breaks at si>=10)
    for pair in cache_by_pair:
        cache_by_pair[pair].sort(key=lambda t: t["_si"])

    all_pairs = sorted(cache_by_pair.keys(), key=lambda p: int(p.split("_")[1]))
    if pairs:
        allowed = {s.strip() for s in pairs.split(",") if s.strip()}
        missing = allowed - set(all_pairs)
        if missing:
            raise ValueError(f"--pairs {sorted(missing)} not in target cache")
        all_pairs = [p for p in all_pairs if p in allowed]
    orders = _layer_order(all_pairs, device)

    # C1: v23 = max 300 steps per pair (early-stop may end sooner). Local
    # CPU stays at 40 (smoke) unless --steps is given.
    if steps <= 0:
        steps = 500 if str(device).startswith(("cuda", "xla")) else 40
    r = 4
    lr = 3e-3
    kl_weight = 0.1

    per_order_results = {}

    for order_name, pair_seq in orders.items():
        print(f"\n{'=' * 60}")
        print(f"Order: {order_name} ({len(pair_seq)} pairs)")
        print(f"{'=' * 60}")

        # Load fresh model for this order
        mS = AutoModelForCausalLM.from_pretrained(
            small_model,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        ).to(device)  # type: ignore[arg-type]
        mS.eval()
        layer_types = get_layer_types(mS)

        # Cache original logits (updated after each layer)
        orig_logits_cache: list[torch.Tensor] = []
        with torch.no_grad():
            for s in samples[:n_samples]:
                ids = s["input_ids"].to(device)
                attn = s["attention_mask"].to(device)
                orig_logits_cache.append(
                    mS(input_ids=ids, attention_mask=attn).logits[:, -1, :]
                )

        per_layer_results: list[dict] = []

        for pair_key in pair_seq:
            targets = cache_by_pair[pair_key]
            # D1: pair targets to samples by EXACT input_ids (v2-cache has
            # input_ids), not filename order. stage0 and exp17 load the same
            # calibration samples (same loader/seed/lengths), so all n_samples
            # should match; on mismatch that sample's bridge is skipped (warn).
            t_by_ids = {tuple(t["input_ids"].reshape(-1).tolist()): t for t in targets}
            sample_targets: list[dict | None] = [
                t_by_ids.get(tuple(s["input_ids"].reshape(-1).tolist()))
                for s in samples[:n_samples]
            ]
            n_matched = sum(t is not None for t in sample_targets)
            if n_matched < n_samples:
                print(
                    f"    [warn] pairing: {n_matched}/{n_samples} samples matched "
                    f"by input_ids in {pair_key} — unmatched bridges skipped"
                )

            l_S = int(pair_key.split("_")[1])
            token_set = targets[0]["token_set"]
            n_use = min(n_samples, len(targets))
            print(
                f"\n  Pair {pair_key} (l_S={l_S}, {n_use} targets, "
                f"matched {n_matched}/{n_samples} by input_ids)"
            )
            layer_start = time.time()
            # Mode guard (v23): full-attn student layer has o_proj -> bridge
            # (o_proj LoRA + J refresh). GDN linear-attn has no o_proj ->
            # kl_only: the J refresh would build a 2nd-order graph that
            # reaches no trainable o_proj = pure waste. classify_pair defines
            # exactly this split; A_attn (below) is its runtime proxy.
            pair_mode = classify_pair(l_S, layer_types)

            # Freeze all model params
            for p in mS.parameters():
                p.requires_grad = False

            # LoRA on lm_head: W_eff = W + B_lh @ A_lh, B: [V,r], A: [r,D]
            V, D = mS.lm_head.weight.shape
            A_lh = torch.nn.Parameter(torch.randn(r, D, device=device) * 0.02)
            B_lh = torch.nn.Parameter(torch.zeros(r, V, device=device).T)  # [V, r]

            # LoRA on target layer's attention o_proj (full-attention layers
            # only; GDN linear-attention layers have out_proj, no o_proj).
            # Qwen3.5's o_proj is NON-square [out, in] = [1024, 2048], so the
            # LoRA delta is applied to o_proj's INPUT x [*, in] by patching
            # the Linear's forward: delta = B @ (A @ x). Dimensionally exact
            # for any in/out and immune to positional-vs-kwargs callers.
            layer_mod = mS.model.layers[l_S]
            o_proj_mod = None
            if hasattr(layer_mod, "self_attn"):
                o_proj_mod = getattr(layer_mod.self_attn, "o_proj", None)

            A_attn: torch.nn.Parameter | None = None
            B_attn: torch.nn.Parameter | None = None
            hook_remove: Callable[..., None] | None = None
            if o_proj_mod is not None:
                out_dim, in_dim = o_proj_mod.weight.shape
                A_attn = torch.nn.Parameter(
                    torch.randn(r, in_dim, device=device) * 0.02
                )
                B_attn = torch.nn.Parameter(
                    torch.randn(out_dim, r, device=device) * 0.02
                )  # v23: not zeros (dL/dA_attn>0, probe AND check)

                # This is how A_attn/B_attn connect to the grad graph: with
                # create_graph=True the J's h is built through the patched
                # o_proj, so dL/dJ_norm contracted with dJ_norm/dh reaches
                # o_proj.
                _orig_fwd = o_proj_mod.forward

                def _patched_o_proj(
                    x: torch.Tensor,
                    _b: torch.Tensor = B_attn,
                    _a: torch.Tensor = A_attn,
                    _f: Callable = _orig_fwd,
                ) -> torch.Tensor:
                    delta = (_b @ _a @ x.transpose(-1, -2)).transpose(-1, -2)
                    return _f(x) + delta

                o_proj_mod.forward = _patched_o_proj

                def _restore(
                    _m: torch.nn.Module = o_proj_mod, _f: Callable = _orig_fwd
                ) -> None:
                    _m.forward = _f

                hook_remove = _restore

            trainable = [A_lh, B_lh]
            if A_attn is not None:
                trainable += [A_attn, B_attn]  # type: ignore[list-item]
            for p in trainable:
                p.requires_grad = True

            optimizer = torch.optim.Adam(trainable, lr=lr)

            # -- v23 probe: matrix-free refresh path (same as in-loop) -----
            # k is FIXED (v22's budget-derived k removed — rows path memory
            # is per-chunk, the budget formula no longer applies).
            # v23.2: k 50->100 — refresh is 4 (n_samples) 2nd-order HVPs;
            # halving the rate halves the dominant wall-clock cost, the KL
            # leash + cheap dense lm_head bridge keep flowing between.
            # Probe: matrix-free rows, create_graph=True through the patched
            # o_proj, DIRECT cosine against J_tgt_p (no W matmul: W is live
            # in-loop, frozen throwaway here). p = re-normalized K-subset
            # gauge from the target cache (A5: bf16 p cast to fp32).
            # FATAL on OOM / dead graph (v23: no degrade — a degraded run is
            # a lie). AND-check: v23 B_attn init makes dL/dA_attn > 0 too.
            k = 100
            J_norms: dict[int, torch.Tensor] = {}
            first_s = samples[0]
            ids_0 = first_s["input_ids"].to(device)
            attn_0 = first_s["attention_mask"].to(device)
            if pair_mode == "kl_only":
                # GDN student layer: no o_proj, no LoRA there. KL-only leash
                # trains the lm_head LoRA; the J bridge / 2nd-order refresh
                # is skipped (it would reach nothing). Not a degrade — this
                # pair was never bridge-eligible (see classify_pair).
                print(f"    kl_only: no o_proj at l_S={l_S} (GDN) — J bridge skipped")
            elif A_attn is not None:
                p_0 = None
                for t in sample_targets:
                    if t is not None and "p_L_full" in t:
                        pf = t["p_L_full"].to(device).float()
                        p_0 = pf[token_set] / pf[token_set].sum()
                        break
                if p_0 is None:
                    print(f"    [FATAL] probe: no target sample on {pair_key}")
                    os._exit(3)
                W_probe = mS.lm_head.weight[token_set].detach()
                try:
                    probe_rows = compute_hidden_jacobian_rows(
                        mS,
                        l_S,
                        ids_0,
                        attn_0,
                        token_set,
                        p_0,
                        W_probe,
                        create_graph=True,
                    )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    print(f"    [FATAL] probe OOM on {pair_key}: {e}")
                    os._exit(3)
                g_tgt_0 = (
                    next(t["J_tgt_p"] for t in sample_targets if t is not None)
                    .to(device)
                    .float()
                )
                probe_loss = torch.nn.functional.cosine_similarity(
                    probe_rows.float(), g_tgt_0, dim=1
                ).mean()
                probe_loss.backward()
                gA = (
                    A_attn.grad.abs().sum().item()
                    if A_attn.grad is not None
                    else 0.0
                )
                gB = (
                    B_attn.grad.abs().sum().item()
                    if B_attn is not None and B_attn.grad is not None
                    else 0.0
                )
                ok = gA > 0 and gB > 0
                print(
                    f"    [smoke_j] dL/dA_attn={gA:.2e}, dL/dB_attn={gB:.2e} "
                    f"-> {'OK' if ok else 'graph dead'}"
                )
                if not ok:
                    print(
                        "    [FATAL] create_graph graph does not reach o_proj"
                    )
                    os._exit(3)
                del probe_rows
                # B2 finite-diff spot check (plan B2): scalar s_0 =
                # h_norm @ W_rows[0]; FD derivative at h_0 vs analytic
                # matrix-free row element J_S_p[0, 0]. Rel err < 1e-2.
                # (The plan's formula is a scalar — element (0,0) is the
                # dimensionally consistent reading of "J_S_p[0] vs that FD".)
                if fd_check > 0:
                    with torch.no_grad():
                        dty = mS.lm_head.weight.dtype
                        h_0 = _extract_hidden_state(
                            mS, ids_0, attn_0, l_S
                        ).to(dty)
                        W0 = mS.lm_head.weight[token_set[0]].to(dty)
                        base = h_norm_at_perturbed_h(
                            mS, l_S, ids_0, attn_0, h_0
                        )
                        eps = 1e-4
                        h_p = h_0.clone()
                        h_p[0, 0, 0] += eps
                        pert = h_norm_at_perturbed_h(
                            mS, l_S, ids_0, attn_0, h_p
                        )
                        fd_scalar = ((pert - base) @ W0).mean().item() / eps
                    # analytic = same raw (pre-gauge) element: J_S[0,0]
                    # = sum_k W0[k] * J_norm[k,0] (dense J column, no gauge).
                    # (The matrix-free rows path's output is gauge-adjusted
                    # — softmax_jacobian — so it can't be compared 1:1 with
                    # this scalar FD; both use the same J_norm underneath.)
                    J_norm_d = compute_hidden_jacobian(mS, l_S, ids_0, attn_0)
                    analytic = (W0 @ J_norm_d[:, 0]).item()
                    rel = abs(fd_scalar - analytic) / max(1e-12, abs(analytic))
                    ok_fd = rel < 1e-2
                    print(
                        f"    [fd_check] fd={fd_scalar:.4e} analytic="
                        f"{analytic:.4e} rel={rel:.2e} -> "
                        f"{'OK' if ok_fd else 'MISMATCH'}"
                    )
                    if not ok_fd:
                        print("    [FATAL] finite-diff spot check failed")
                        os._exit(3)
            print(
                f"    J refresh: 2nd-order (o_proj HVP) every k={k} steps; "
                f"1st-order (lm_head, dense no-HVP, h-reuse) every step (v23.2b)"
            )

            # -- C2 early-stop (v23) -------------------------------------
            # No-grad bridge eval every 50 steps (fresh dense J per sample,
            # current effective weights — cheap: no create_graph graph). Stop
            # after 3 consecutive eval points with < 0.5% relative
            # improvement. Baseline at step 0 (fresh LoRA).
            N_EVAL, P_FLAT, THRESH = 50, 3, 0.005
            eval_history: list[tuple[int, float]] = []
            flat_streak = 0
            stopped_at = steps

            def _eval_bridge() -> float:
                vals = []
                with torch.no_grad():
                    for si in range(n_use):
                        tgt = sample_targets[si]
                        if tgt is None:
                            continue
                        s = samples[si]
                        ids = s["input_ids"].to(device)
                        attn_s = s["attention_mask"].to(device)
                        W_e = mS.lm_head.weight + B_lh @ A_lh
                        J_n = compute_hidden_jacobian(mS, l_S, ids, attn_s)
                        J_S_e = W_e[token_set] @ J_n
                        J_S_e_p = softmax_jacobian(
                            J_S_e, tgt["p"].to(device).float()
                        )
                        cos = torch.nn.functional.cosine_similarity(
                            J_S_e_p.float(),
                            tgt["J_tgt_p"].to(device).float(),
                            dim=1,
                        ).mean().item()
                        vals.append(1.0 - cos)
                return sum(vals) / len(vals) if vals else 0.0

            def _eval_point(step_no: int) -> None:
                nonlocal flat_streak, stopped_at
                cur = _eval_bridge()
                eval_history.append((step_no, cur))
                prev = eval_history[-2][1] if len(eval_history) >= 2 else None
                if prev is not None and prev > 0:
                    if (prev - cur) / prev < THRESH:
                        flat_streak += 1
                    else:
                        flat_streak = 0
                print(f"    [eval] step={step_no} bridge_loss={cur:.4f}")

            _eval_point(0)

            # C3: checkpoints at 60/120/180/240/300 (cumulative LoRA +
            # optimizer state + metrics). C4: the last one is also the
            # delivered lora_{pair}.pt.
            ckpt_dir = os.path.join(results_dir, "exp17")
            os.makedirs(ckpt_dir, exist_ok=True)
            CKPT_STEPS = (60, 120, 180, 240, 300)
            last_peak_vram = 0.0

            def _save_ckpt(step_no: int, final: bool) -> None:
                payload = {
                    "pair": pair_key,
                    "l_S": l_S,
                    "order": order_name,
                    "r": r,
                    "step": step_no,
                    "A_lh": A_lh.detach().cpu(),
                    "B_lh": B_lh.detach().cpu(),
                    "A_attn": (A_attn.detach().cpu() if A_attn is not None else None),
                    "B_attn": (B_attn.detach().cpu() if B_attn is not None else None),
                    "eval_cos": 1.0 - eval_history[-1][1] if eval_history else 0.0,
                    "optimizer": optimizer.state_dict(),
                    "peak_vram_gb": last_peak_vram,
                }
                name = (
                    f"lora_{pair_key}.pt" if final else f"ckpt_{pair_key}_{step_no}.pt"
                )
                torch.save(payload, os.path.join(ckpt_dir, name))
                print(f"    [ckpt] saved {name}")

            for step in range(steps):
                with _step_cm():
                    step_start = time.time()
                    optimizer.zero_grad()
                    total_bridge = 0.0
                    total_kl = 0.0
                    n_bridge = 0

                    for si in range(n_use):
                        s = samples[si]
                        tgt: dict | None = sample_targets[si]

                        # Effective lm_head weight: W + B_lh @ A_lh. Rebuilt PER
                        # SAMPLE — each sample runs its own backward() (docs #16),
                        # so a W_eff shared across si would be freed by si=0's
                        # backward and si=1's backward would hit "second backward"
                        # (RuntimeError). Fresh graph per backward = no double-free.
                        W_eff = mS.lm_head.weight + B_lh @ A_lh  # [V, D]

                        # KL leash — full forward (includes o_proj LoRA via hook).
                        # v23.1: output_hidden_states=True so the l_S hidden state
                        # is captured here and REUSED by the one-time fill and the
                        # refresh (matrix-free rows + dense re-fill) instead of
                        # each redoing a full forward of the same deterministic
                        # hidden state (3 full forwards -> 1 per refresh sample).
                        ids_s = s["input_ids"].to(device)
                        attn_s = s["attention_mask"].to(device)
                        # use_cache=False matches the J-path forwards so the
                        # reused h_s is bit-identical to what refresh / re-fill do.
                        out = mS.model(
                            input_ids=ids_s,
                            attention_mask=attn_s,
                            output_hidden_states=True,
                            use_cache=False,
                        )
                        hidden = out.last_hidden_state[:, -1, :]  # [1, D]
                        h_s = (
                            out.hidden_states[l_S + 1][:, -1:, :]
                            if A_attn is not None
                            and out.hidden_states is not None
                            else None
                        )  # [1, 1, D] l_S hidden state (graph-connected)
                        lm_b = mS.lm_head.bias if mS.lm_head.bias is not None else 0
                        logits_ed = hidden @ W_eff.T + lm_b  # [1, V]
                        p_ed = torch.softmax(logits_ed.float(), dim=-1)
                        with torch.no_grad():
                            p_orig = torch.softmax(orig_logits_cache[si].float(), dim=-1)
                        kl_loss = (
                            p_ed * torch.log(p_ed.clamp_min(1e-12) / (p_orig + 1e-12))
                        ).mean()

                        loss = kl_weight * kl_loss
                        total_kl += kl_loss.item()

                        already_bwd = False  # True when the refresh branch
                        # already did the (KL + bridge) backward inside
                        # bridge_rows_backward -> skip the per-sample loss.backward().
                        # Bridge — only for input_ids-matched targets (D1) AND
                        # only when the student layer is full-attn (has o_proj,
                        # A_attn set). GDN/kl_only pairs never build the held
                        # dense J or the 2nd-order refresh (no o_proj to train).
                        if tgt is not None and A_attn is not None:
                            J_tgt_p = tgt["J_tgt_p"].to(device)
                            p_L = tgt["p"].to(device).float()  # v23 A5: cache p is bf16
                            refresh_step = False
                            J_S_p: torch.Tensor | None = None
                            if si not in J_norms:
                                # one-time fill: dense J (create_graph=False = a
                                # 4MB [D,D] fp32 matrix, no 2nd-order graph — cheap;
                                # v22's OOM was the create_graph refresh, not this).
                                # v23.1: reuses the KL forward's h (no full forward).
                                J_norms[si] = compute_hidden_jacobian(
                                    mS, l_S, ids_s, attn_s, h=h_s
                                )
                                refresh_step = False
                            elif step % k == 0:
                                # refresh: A4 chunked bridge backward. bridge_
                                # rows_backward forwards once (KL is the extra_loss,
                                # backproped with chunk 0) then backprops each chunk
                                # of K rows (C=8) of (1-cos).mean() immediately ->
                                # peak = C+1 2nd-order row-graphs, not K (the v22
                                # OOM was the full K=256 retained create_graph graph).
                                # Gradients reach lm_head (live W_eff rows) AND
                                # o_proj (HVP through h) this step. The returned
                                # J_S_p is detached (graph consumed) — re-fill the
                                # held dense J_norm for stale steps (v22 freshness).
                                # held is already detached from prior stale steps
                                # (no live graph) — drop the old one, new fill below.
                                del J_norms[si]
                                p_refresh = tgt["p_L_full"].to(device).float()
                                p_refresh = (
                                    p_refresh[token_set] / p_refresh[token_set].sum()
                                )
                                try:
                                    rows, br = bridge_rows_backward(
                                        mS,
                                        l_S,
                                        ids_s,
                                        attn_s,
                                        token_set,
                                        p_refresh,
                                        W_eff[token_set],
                                        J_tgt_p,
                                        extra_loss=kl_weight * kl_loss,
                                        verbose=True,
                                        h=h_s,
                                    )
                                except (
                                    torch.cuda.OutOfMemoryError,
                                    RuntimeError,
                                ) as e:
                                    print(
                                        f"    [FATAL] v23 refresh OOM {pair_key} "
                                        f"step={step} si={si}: {e}"
                                    )
                                    os._exit(3)
                                J_S_p = rows
                                bridge_loss = torch.tensor(br)
                                total_bridge += br
                                n_bridge += 1
                                # new dense fill (no 2nd-order graph) replaces held;
                                # v23.1: reuses the KL forward's h (no full re-forward)
                                J_norms[si] = compute_hidden_jacobian(
                                    mS, l_S, ids_s, attn_s, h=h_s
                                )
                                refresh_step = True
                                already_bwd = True  # skip the loss.backward()
                                # below; the per-sample optimizer.step() still runs.
                                if str(device).startswith(("cuda", "xla")):
                                    # A2: VRAM telemetry after the (chunked)
                                    # refresh — the peak-relevant moment.
                                    peak = torch.cuda.max_memory_allocated() / 2**30
                                    cur = torch.cuda.memory_allocated() / 2**30
                                    last_peak_vram = max(last_peak_vram, peak)
                                    print(
                                        f"    [vram] pair={pair_key} step={step} "
                                        f"si={si} k={k} peak={peak:.1f}G "
                                        f"cur={cur:.1f}G"
                                    )
                            else:
                                # v23.2b DUAL CADENCE: recompute the 1st-order
                                # (no-HVP) J_norm EVERY step, reusing the KL
                                # forward's h (cheap decoder-tail + matmul, no
                                # 2nd-order graph). So the lm_head bridge runs on a
                                # FRESH hidden every step (no stale-J staleness),
                                # while the expensive 2nd-order (HVP, o_proj) stays
                                # at k below.
                                refresh_step = False
                                J_norms[si] = compute_hidden_jacobian(
                                    mS, l_S, ids_s, attn_s, h=h_s
                                )
                            if refresh_step and J_S_p is not None:
                                # J_S_p already detached by bridge_rows_backward
                                # (graph consumed by the per-chunk backward). Its
                                # (1-cos) + KL backward ran there; only recompute
                                # the metric scalar here for logging.
                                cos_sim = torch.nn.functional.cosine_similarity(
                                    J_S_p.float(), J_tgt_p.float(), dim=1
                                )
                                bridge_loss = (1.0 - cos_sim).mean()
                            elif J_norms.get(si) is not None:
                                # stale: held dense J_norm @ live W_eff rows ->
                                # lm_head bridge gradient only (o_proj is refresh-only
                                # by design; the dense path's h has no create_graph
                                # graph). Same as v22 stale behavior.
                                J_S = W_eff[token_set] @ J_norms[si]
                                J_S_p = softmax_jacobian(J_S, p_L)
                                cos_sim = torch.nn.functional.cosine_similarity(
                                    J_S_p.float(), J_tgt_p.float(), dim=1
                                )
                                bridge_loss = (1.0 - cos_sim).mean()
                                loss = loss + bridge_loss
                                total_bridge += bridge_loss.item()
                                n_bridge += 1

                        if not already_bwd:
                            loss.backward()

                        # This backward consumed the grad graph. Any live
                        # create_graph J for this si is now freed; a stale-step
                        # backward through it would be a "second backward"
                        # (RuntimeError). Detach it in place so its value still
                        # drives the lm_head bridge until the next refresh
                        # rebuilds a fresh graph (the o_proj HVP is refresh-only).
                        J_cur = J_norms.get(si)
                        if J_cur is not None and J_cur.grad_fn is not None:
                            J_norms[si] = J_cur.detach()
                    optimizer.step()

                step_time = time.time() - step_start
                eta = (steps - step - 1) * step_time
                if (step + 1) % 5 == 0 or step == 0:
                    print(
                        f"    step {step + 1}/{steps}: "
                        f"bridge={total_bridge / max(n_bridge, 1):.4f}, "
                        f"kl={total_kl:.4f}, step={step_time:.0f}s, eta={eta:.0f}s"
                    )

                if (step + 1) in CKPT_STEPS:
                    _save_ckpt(step + 1, final=False)

                if (step + 1) % N_EVAL == 0:
                    _eval_point(step + 1)
                    if flat_streak >= P_FLAT:
                        stopped_at = step + 1
                        print(
                            f"    [early-stop] {pair_key} at step {step + 1} "
                            "(3 consecutive evals < 0.5% improvement)"
                        )
                        break

            # -- Freeze lora deltas into weights ---------------------------
            delta_lm_final = B_lh @ A_lh  # [V, D]
            with torch.no_grad():
                mS.lm_head.weight.data += delta_lm_final
            if o_proj_mod is not None and A_attn is not None and B_attn is not None:
                # B[out, r] @ A[r, in] = [out, in] == o_proj.weight shape.
                o_proj_mod.weight.data += B_attn @ A_attn

            # Cleanup BEFORE eval: the delta is now frozen in the weights;
            # leaving the live patch on would double it in the eval forward.
            # (A_lh/B_lh/A_attn/B_attn kept alive — _save_ckpt below reads
            # them; the next pair iteration's rebinding releases them.)
            J_norms.clear()
            if hook_remove is not None:
                hook_remove()

            # C4: delivered per-pair artifact (A/B matrices + optimizer +
            # metrics), saved AFTER the freeze so the file holds the final
            # frozen deltas and the eval below runs without the patch.
            _save_ckpt(stopped_at, final=True)

            # -- Evaluate bridge (no grad, create_graph=False) -------------
            bridge_vals = []
            with torch.no_grad():
                W_final = mS.lm_head.weight  # frozen
                for si in range(n_use):
                    s = samples[si]
                    tgt = sample_targets[si]
                    if tgt is None:
                        continue
                    ids = s["input_ids"].to(device)
                    attn_s = s["attention_mask"].to(device)
                    J_tgt_p = tgt["J_tgt_p"].to(device)
                    p_L = tgt["p"].to(device)
                    J_norm_eval = compute_hidden_jacobian(mS, l_S, ids, attn_s)
                    J_S_ed = W_final[token_set] @ J_norm_eval
                    J_S_ed_p = softmax_jacobian(J_S_ed, p_L)
                    cos = (
                        torch.nn.functional.cosine_similarity(
                            J_S_ed_p.float(), J_tgt_p.float(), dim=1
                        )
                        .mean()
                        .item()
                    )
                    bridge_vals.append(cos)

            # KL to original
            kl_sum = 0.0
            with torch.no_grad():
                for si in range(min(n_samples, len(orig_logits_cache))):
                    s = samples[si]
                    ids = s["input_ids"].to(device)
                    attn_s = s["attention_mask"].to(device)
                    logits_ed = mS(input_ids=ids, attention_mask=attn_s).logits[
                        :, -1, :
                    ]
                    p_ed = torch.softmax(logits_ed.float(), dim=-1)
                    p_orig = torch.softmax(orig_logits_cache[si].float(), dim=-1)
                    kl = (
                        (p_ed * torch.log(p_ed.clamp_min(1e-12) / (p_orig + 1e-12)))
                        .mean()
                        .item()
                    )
                    kl_sum += kl
            kl_avg = kl_sum / min(n_samples, len(orig_logits_cache))

            layer_result = {
                "pair": pair_key,
                "bridge_mean": sum(bridge_vals) / len(bridge_vals)
                if bridge_vals
                else 0,
                "KL_ed_orig": kl_avg,
                "k_refresh": k,
                "stopped_at": stopped_at,
                "eval_history": eval_history,
            }
            layer_time = time.time() - layer_start
            print(f"    Layer {pair_key} done in {layer_time:.0f}s")
            print(f"    bridge={layer_result['bridge_mean']:.4f}, KL={kl_avg:.4f}")
            per_layer_results.append(layer_result)

            # Update orig logits for next layer (cumulative edits)
            with torch.no_grad():
                for si, s in enumerate(samples[:n_samples]):
                    ids = s["input_ids"].to(device)
                    attn_s = s["attention_mask"].to(device)
                    orig_logits_cache[si] = mS(
                        input_ids=ids, attention_mask=attn_s
                    ).logits[:, -1, :]

        per_order_results[order_name] = {
            "pairs": pair_seq,
            "per_layer": per_layer_results,
        }

    # -- Save --------------------------------------------------------------
    result = {
        "description": (
            "v23: per-layer LoRA r=4 on lm_head + attention o_proj of l_S, bridge loss on "
            "matrix-free per-sample J_S_p rows (no dense [D,D] graph: refresh steps build "
            "rows with W_rows = (W+B_lh@A_lh)[token_set] live, create_graph=True, chunked "
            "backward; one backward per row reaches lm_head AND o_proj — HVP, no D^3). "
            "Fill/stale steps use the cheap dense J_norm so the lm_head bridge gradient "
            "keeps flowing; o_proj is refresh-only. k=100 fixed. J refresh OOM -> FATAL "
            "exit 3 (no degrade). Max 300 steps/pair (CPU smoke 40). Targets matched by "
            "exact input_ids; mismatch -> bridge skipped (KL unaffected). After each pair "
            "both LoRA deltas are frozen into weights. Two orders (GPU), late-to-early "
            "only (CPU)."
        ),
        "per_order": per_order_results,
        "r": r,
        "steps": steps,
        "kl_weight": kl_weight,
        "device": device,
        "elapsed_seconds": time.time() - start,
    }
    path = os.path.join(results_dir, "exp17_lora_distill.json")
    result["params"] = {
        "small_model": small_model,
        "device": device,
        "max_length": max_length,
        "min_length": min_length,
        "targets_dir": targets_dir,
    }
    try:
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
    except OSError as e:
        print(f"Failed to write {path}: {e}")
        raise

    print(f"\nSaved to {path}")
    print(f"Total time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 17: Per-layer LoRA distillation")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--steps", type=int, default=0, help=">0: max steps/pair (default: 300 cuda / 40 cpu)")
    parser.add_argument("--fd-check", type=int, default=0, help=">0: finite-diff spot check on that pair (B2)")
    parser.add_argument("--pairs", default="", help="comma list of pair keys to run (default: all in cache)")
    args = parser.parse_args()
    main(**vars(args))
