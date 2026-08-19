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

from prism.calibration import (
    extract_embeddings,
    load_calibration_data,
    uniform_layer_mapping,
)
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
)

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
            "dtype_L": torch.float16,
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
    ).to(device_L)
    mL.eval()

    print(f"Loading small model: {small_model}")
    mS = AutoModelForCausalLM.from_pretrained(
        small_model,
        torch_dtype=cfg["dtype_S"],
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device_S)
    mS.eval()

    n_L = mL.config.num_hidden_layers
    n_S = mS.config.num_hidden_layers
    print(f"Large: {n_L} layers, {mL.config.hidden_size} hidden")
    print(f"Small: {n_S} layers, {mS.config.hidden_size} hidden")

    # -- Layer mapping: ~7 pairs at uniform step --------------------------
    mapping = uniform_layer_mapping(n_L, n_S)
    step = max(1, len(mapping) // 7)
    pairs = list(mapping.items())[::step]
    print(f"Layer pairs ({len(pairs)}): {pairs}")

    # -- Calibration data --------------------------------------------------
    print(f"Loading {cfg['n_samples']} calibration samples...")
    samples = load_calibration_data(
        small_model,
        max_length=max_length,
        min_length=min_length,
        n_samples=cfg["n_samples"],
    )

    # -- Embeddings --------------------------------------------------------
    E_L = extract_embeddings(mL).to(device_L)  # [V, D_L]
    E_S = extract_embeddings(mS).to(device_S)  # [V, D_S]
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
    done_pairs: list[tuple[int, int]] = []
    for pair in pairs:
        pair_key = f"{pair[0]}_{pair[1]}"
        all_found = all(
            os.path.exists(os.path.join(targets_dir, f"{pair_key}_{si}.pt"))
            for si in range(cfg["n_samples"])
        )
        if all_found:
            done_pairs.append(pair)
            print(f"  Pair {pair_key}: cached, skipping")

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
        if not _guard_checked:
            _guard_checked = True
            print(f"  Probing J_L (layer {l_L}) timing...")
            t0 = time.time()
            _probe_J = compute_jacobian_for_tokens(mL, l_L, ids0, attn0, token_set)
            t_jL = time.time() - t0
            del _probe_J
            if t_jL > 60:
                print(
                    f"  WARNING: Jacobian took {t_jL:.1f}s (>60s). "
                    f"Reducing K and n_samples."
                )
                _guard_k = max(32, cfg["K"] // 2)
                _guard_n = max(1, cfg["n_samples"] // 2)
                # Rebuild token set with smaller K
                token_set = sorted(lg0.topk(_guard_k, dim=-1).indices[0].cpu().tolist())
                print(f"  New K={_guard_k}, n_samples={_guard_n}")
            else:
                print(f"  Jacobian time: {t_jL:.1f}s — OK")

        # p_L = softmax of large model logits on token_set (first sample)
        with torch.no_grad():
            logits_L = mL(input_ids=ids0, attention_mask=attn0).logits[:, -1, :]
            p_L = torch.softmax(logits_L[0, token_set], dim=-1).to(device_S)

        # J_L at layer l_L on first sample
        print(f"  Computing J_L (layer {l_L})...")
        J_L = compute_jacobian_for_tokens(mL, l_L, ids0, attn0, token_set)

        # Transport: J_L @ E_L.T @ E_S @ G_S_inv
        print("  Transporting...")
        J_L_proj = J_L.float().to(device_S) @ E_L.float().to(device_S).T  # [|U|, V]
        J_tgt = J_L_proj @ E_S.float() @ G_S_inv  # [|U|, D_S]

        # Gauge-free targets
        J_tgt_p = softmax_jacobian(J_tgt, p_L)  # [|U|, D_S]

        for si, s in enumerate(samples[:_guard_n]):
            ids_s = s["input_ids"].to(device_S)
            attn_s = s["attention_mask"].to(device_S)

            # J_S_p on this sample
            print(f"  Computing J_S (layer {l_S}, sample {si + 1})...")
            J_S = compute_jacobian_for_tokens(mS, l_S, ids_s, attn_s, token_set)
            # p for gauge: use large model probs on this sample
            with torch.no_grad():
                logits_L_s = mL(
                    input_ids=ids_s.to(device_L), attention_mask=attn_s.to(device_L)
                ).logits[:, -1, :]
                p_s = torch.softmax(logits_L_s[0, token_set], dim=-1).to(device_S)
            J_S_p = softmax_jacobian(J_S, p_s)

            # Cache
            cache_path = os.path.join(targets_dir, f"{pair_key}_{si}.pt")
            torch.save(
                {
                    "J_tgt_p": J_tgt_p.cpu(),
                    "J_S_p": J_S_p.cpu(),
                    "p": p_s.cpu(),
                    "pair": pair_key,
                    "l_S": l_S,
                    "l_L": l_L,
                    "token_set": token_set,
                    "K": len(token_set),
                },
                cache_path,
            )
            print(f"  Saved: {cache_path}")

        done_pairs.append((l_L, l_S))
        _save_checkpoint(targets_dir, pairs, done_pairs)
        print(f"  Pair {pair_key} done. Elapsed: {time.time() - start:.0f}s")

    # -- Summary -----------------------------------------------------------
    elapsed = time.time() - start
    summary = {
        "pairs": [[a, b] for a, b in pairs],
        "n_samples": cfg["n_samples"],
        "K": cfg["K"],
        "device_L": device_L,
        "device_S": device_S,
        "elapsed_seconds": elapsed,
    }
    summary_path = os.path.join(targets_dir, "stage0_summary.json")
    summary["params"] = {large_model: large_model, small_model: small_model, device_L: device_L, device_S: device_S, max_length: max_length, min_length: min_length, targets_dir: targets_dir}
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    main(**vars(args))
