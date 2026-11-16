"""Stage 0: Compute and cache target Jacobians for Exp16–19.

For each layer pair (l_L, l_S) from uniform_layer_mapping(n_L, n_S) with
step ~7 pairs, and for n_samples calibration samples:
  - compute J_L (large model Jacobian)
  - transport to small space: J_tgt = (J_L @ E_L.T) @ E_S @ G_S_inv
  - gauge-free: J_tgt_p = softmax_jacobian(J_tgt, p_L)
  - compute J_S_p (small model gauge-free Jacobian)
  - cache {J_tgt_p, J_S_p, p, pair, l_S, token_set} to disk

This is the expensive step — run once, all experiments reuse the cache.
CPU-friendly: runs sequentially, fp32, small K.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F

from prism.calibration import (
    extract_embeddings,
    get_layer_types,
    load_calibration_data,
)
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
)
from prism.layer_selection import select_pairs

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _gpu_config(device: str) -> dict:
    """Return parameter config adapted to device."""
    if device.startswith("cuda"):
        return {
            "n_samples": 4,
            "K": 256,
            "dtype_L": torch.bfloat16,
            "dtype_S": torch.float32,
        }
    return {
        "n_samples": 2,
        "K": 128,
        "dtype_L": torch.float32,
        "dtype_S": torch.float32,
    }


def _save_checkpoint(targets_dir: str, pairs: list[tuple], done_pairs: list[tuple]):
    """Save progress checkpoint."""
    ckpt = {
        "pairs": [[a, b] for a, b in pairs],
        "done": [[a, b] for a, b in done_pairs],
    }
    with open(os.path.join(targets_dir, ".checkpoint.json"), "w") as f:
        json.dump(ckpt, f)


def main(
    large_model: str = "/data/Qwen/Qwen3.5-4B",
    small_model: str = "Qwen/Qwen3.5-0.8B",
    device_L: str = "auto",
    device_S: str = "auto",
    max_length: int = 64,
    min_length: int = 32,
    targets_dir: str = "results/targets",
    results_dir: str = "results",
):
    torch.manual_seed(42)
    start = time.time()

    if device_L == "auto":
        device_L = "cuda" if torch.cuda.is_available() else "cpu"
    if device_S == "auto":
        device_S = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = _gpu_config(device_L)  # use device_L for large model config
    if device_L == "cpu" and device_S == "cpu":
        torch.set_num_threads(4)
    print(f"Device L: {device_L}, S: {device_S}")
    print(
        f"Config: n_samples={cfg['n_samples']}, K={cfg['K']}, "
        f"dtype_L={cfg['dtype_L']}, dtype_S={cfg['dtype_S']}"
    )

    os.makedirs(targets_dir, exist_ok=True)

    # -- Load models -------------------------------------------------------
    print(f"Loading large model: {large_model}")
    mL = AutoModelForCausalLM.from_pretrained(
        large_model,
        torch_dtype=cfg["dtype_L"],
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device_L)  # type: ignore[arg-type]
    mL.eval()

    print(f"Loading small model: {small_model}")
    mS = AutoModelForCausalLM.from_pretrained(
        small_model,
        torch_dtype=cfg["dtype_S"],
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device_S)  # type: ignore[arg-type]
    mS.eval()

    n_L = mL.config.num_hidden_layers
    n_S = mS.config.num_hidden_layers
    print(f"Large: {n_L} layers, {mL.config.hidden_size} hidden")
    print(f"Small: {n_S} layers, {mS.config.hidden_size} hidden")

    # -- Re-selection (contract 0.1/0.2): per l_S = full-attn + 0_0 + last --
    layer_types = get_layer_types(mS)  # type: ignore[arg-type]
    pairs = select_pairs(n_L, n_S, layer_types)
    print(f"layer_types: {layer_types}")
    print(f"Selected pairs ({len(pairs)}): {pairs}")

    # -- Calibration data --------------------------------------------------
    print(f"Loading {cfg['n_samples']} calibration samples...")
    samples = load_calibration_data(
        small_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=cfg["n_samples"],
    )

    # -- Embeddings --------------------------------------------------------
    E_L = extract_embeddings(mL).to(device_L)  # [V, D_L]  # type: ignore[arg-type]
    E_S = extract_embeddings(mS).to(device_S)  # [V, D_S]  # type: ignore[arg-type]
    print(f"E_L: {E_L.shape}, E_S: {E_S.shape}")

    # -- G_S_inv = (E_S^T E_S)^{-1} float64 for invertibility ------------
    print("Computing G_S_inv = (E_S^T E_S)^{-1}...")
    G_S = E_S.float().T @ E_S.float()  # [D_S, D_S]
    G_S_inv = torch.linalg.pinv(G_S.double()).float()  # [D_S, D_S]
    print(f"G_S_inv shape: {G_S_inv.shape}")

    # -- Token set from first sample (large model top-K) ------------------
    ids0 = samples[0]["input_ids"].to(device_L)
    attn0 = samples[0]["attention_mask"].to(device_L)
    with torch.no_grad():
        lg0 = mL(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
    token_set = sorted(lg0.topk(cfg["K"], dim=-1).indices[0].cpu().tolist())
    print(f"Token set size: {len(token_set)}")

    # -- Check existing cache ----------------------------------------------
    # A pair is "done" only if every sample file exists AND has the v2
    # schema (p_L_full, input_ids). v1 files (pre-pairing era) are recomputed.
    done_pairs: list[tuple[int, int]] = []
    for pair in pairs:
        pair_key = f"{pair[0]}_{pair[1]}"
        ok = True
        for si in range(cfg["n_samples"]):
            path = os.path.join(targets_dir, f"{pair_key}_{si}.pt")
            if not os.path.exists(path):
                ok = False
                break
            try:
                d = torch.load(path, weights_only=True)
            except Exception:
                ok = False
                break
            if "p_L_full" not in d or "input_ids" not in d:
                ok = False
                break
        if ok:
            done_pairs.append(pair)
            print(f"  Pair {pair_key}: cached, skipping")
        else:
            print(f"  Pair {pair_key}: recompute (missing or pre-v2 cache)")

    # -- Check checkpoint --------------------------------------------------
    ckpt_path = os.path.join(targets_dir, ".checkpoint.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        for d in ckpt.get("done", []):
            dp = (d[0], d[1])
            if dp not in done_pairs:
                done_pairs.append(dp)

    remaining = [p for p in pairs if p not in done_pairs]
    print(f"Remaining pairs: {len(remaining)}/{len(pairs)}")

    if not remaining:
        print("All pairs already cached. Stage 0 complete.")
        print(f"Total time: {time.time() - start:.0f}s")
        return

    # -- Runtime guard: measure first Jacobian time ------------------------
    _guard_checked = False
    _guard_k = cfg["K"]
    _guard_n = cfg["n_samples"]

    # -- Compute targets ---------------------------------------------------
    for l_L, l_S in remaining:
        pair_key = f"{l_L}_{l_S}"
        print(f"\n=== Pair {pair_key} ===")

        # -- Runtime guard: measure first Jacobian time --------------------
        # GPU: ~0.5s per token-row is normal (K=256 -> ~135s on T4); the
        # threshold only fires on CPU or a broken CUDA environment.
        _guard_thresh = 300 if device_L.startswith("cuda") else 60
        _probe_J = None
        if not _guard_checked:
            _guard_checked = True
            print(f"  Probing J_L (layer {l_L}) timing...")
            t0 = time.time()
            _probe_J = compute_jacobian_for_tokens(mL, l_L, ids0, attn0, token_set)
            t_jL = time.time() - t0
            if t_jL > _guard_thresh:
                _probe_J = None
                print(
                    f"  WARNING: Jacobian took {t_jL:.1f}s (>{_guard_thresh}s). "
                    f"Reducing K and n_samples."
                )
                _guard_k = max(32, cfg["K"] // 2)
                _guard_n = max(1, cfg["n_samples"] // 2)
                # Rebuild token set with smaller K
                token_set = sorted(lg0.topk(_guard_k, dim=-1).indices[0].cpu().tolist())
                print(f"  New K={_guard_k}, n_samples={_guard_n}")
            else:
                print(f"  Jacobian time: {t_jL:.1f}s — OK, reused for sample 1")

        for si, s in enumerate(samples[:_guard_n]):
            ids_s = s["input_ids"].to(device_S)
            attn_s = s["attention_mask"].to(device_S)
            ids_L_s = ids_s.to(device_L)
            attn_L_s = attn_s.to(device_L)

            # -- J_L + transport for THIS sample --------------------------
            if _probe_J is not None and si == 0:
                # Probe computed exactly this (same model, same sample 0).
                J_L_s = _probe_J
                _probe_J = None
            else:
                print(f"  Computing J_L (layer {l_L}, sample {si + 1})...")
                J_L_s = compute_jacobian_for_tokens(
                    mL, l_L, ids_L_s, attn_L_s, token_set
                )

            with torch.no_grad():
                logits_L_s = mL(input_ids=ids_L_s, attention_mask=attn_L_s).logits[
                    :, -1, :
                ]
                p_s = torch.softmax(logits_L_s[0, token_set], dim=-1).to(device_S)

            print(f"  Transporting (sample {si + 1})...")
            J_L_proj_s = (
                J_L_s.float().to(device_S) @ E_L.float().to(device_S).T
            )  # [|U|, V]
            J_tgt_s = J_L_proj_s @ E_S.float() @ G_S_inv  # [|U|, D_S]
            J_tgt_p_s = softmax_jacobian(J_tgt_s, p_s)  # [|U|, D_S]

            # -- J_S_p on this sample ------------------------------------
            print(f"  Computing J_S (layer {l_S}, sample {si + 1})...")
            J_S = compute_jacobian_for_tokens(mS, l_S, ids_s, attn_s, token_set)
            J_S_p = softmax_jacobian(J_S, p_s)

            # Cache
            cache_path = os.path.join(targets_dir, f"{pair_key}_{si}.pt")
            torch.save(
                {
                    "J_tgt_p": J_tgt_p_s.cpu(),
                    "J_S_p": J_S_p.cpu(),
                    "p": p_s.cpu(),
                    # Full-vocab p_L for THIS sample: exp18 needs per-sample
                    # KL(p_ed||p_L) without a layer-mismatched J target.
                    "p_L_full": F.softmax(logits_L_s[0], dim=-1).float().cpu(),
                    "pair": pair_key,
                    "l_S": l_S,
                    "l_L": l_L,
                    "token_set": token_set,
                    "K": len(token_set),
                    # Per-sample ids: lets consumers verify sample<->target
                    # pairing across kernel runs (no silent position pairing).
                    "input_ids": ids_s.cpu(),
                },
                cache_path,
            )
            print(f"  Saved: {cache_path}")
            del J_L_s, J_L_proj_s, J_tgt_s, J_tgt_p_s, J_S, J_S_p

        done_pairs.append((l_L, l_S))
        _save_checkpoint(targets_dir, pairs, done_pairs)
        print(f"  Pair {pair_key} done. Elapsed: {time.time() - start:.0f}s")

    # -- Summary -----------------------------------------------------------
    elapsed = time.time() - start
    summary = {
        "description": "Stage 0: pre-compute and cache target Jacobians for exp16-19. Pairs = runtime re-selection (full-attention small-model layers + 0_0 + last, contract 0.1/0.2). For each layer pair (l_L, l_S) transports J_L to small-model space via J_tgt = J_L @ E_L.T @ E_S @ G_S_inv, then gauge-free variants J_tgt_p / J_S_p via softmax weighting. Each per-sample cache holds J targets, gauge-free p over token_set, p_L_full (full-vocab p_L for exp18), token_set, and input_ids (pairing verification).",
        "layer_types": layer_types,
        "pairs": [[a, b] for a, b in pairs],
        "n_samples": _guard_n,
        "K": _guard_k,
        "device_L": device_L,
        "device_S": device_S,
        "elapsed_seconds": elapsed,
    }
    summary_path = os.path.join(targets_dir, "stage0_summary.json")
    summary["params"] = {
        "large_model": large_model,
        "small_model": small_model,
        "device_L": device_L,
        "device_S": device_S,
        "max_length": max_length,
        "min_length": min_length,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Stage 0 complete ===")
    print(f"Pairs cached: {len(done_pairs)}/{len(pairs)}")
    print(f"Total time: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Compute target Jacobians")
    parser.add_argument("--large-model", default="/data/Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device-L", default="auto", dest="device_L")
    parser.add_argument("--device-S", default="auto", dest="device_S")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
