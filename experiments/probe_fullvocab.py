"""Kaggle probe: is the full-vocab target Jacobian worth it vs the top-K we run?

User's question: "Может вообще Full pass бессмысленен, и не вносит большой вклад?"
We MEASURE it with checks on Kaggle (T4), anchored to the EXACT exp16 stage0
cache (no fresh per-token 4B loop, no HF-nondeterminism gamble):

  For the last layer (the ONLY cheap FULL case):
    exp16 K baseline : cos(J_S_p, J_tgt_p) on the cached 256 rows (0 compute).
    full-vocab       : J_S  = W_S @ J_norm_S                     (analytic, PROOF1)
                       J_tgt = W_S @ (J_norm_L @ M)              (M-FUSED, PROOF2)
                       cos over all V rows (p_L_full) AND over the same
                       256 rows (exp16 convention) — both compared to K.
    M = E_L.T @ E_S @ G_S_inv  [D_L, D_S] ~10 MiB (no V x V, no [V,D_L]).
  For mid layers: full V is INFEASIBLE (V backward passes) — report the
    exp16 K baseline only + the cost of a hypothetical full (V passes).

  Distribution side: p_L_full mass/energy captured by exp16's 256-token set.
  Speed side: full = a handful of matmuls (measured) vs exp16 K = 256 backward
    passes per target (stage0 timing).

Two PHASES on one 16 GB T4 (both models cuda:0, peak kept <14 GB):
  A: 4B fp16 (~8 GB) -> E_L, J_norm_L (last layer, analytic). Free the 4B.
  B: 0.8B fp32 (~3.2 GB) -> E_S, M, J_tgt_full (1 GB), J_S_V (1 GB).
Self-checks run FIRST (no model load) proving the M-fusion transport is exact.
"""

import json
import os
import time

import torch
import torch.nn.functional as Fn

from prism.cache_io import find_targets_dir
from prism.calibration import extract_embeddings
from prism.jacobian import (
    compute_hidden_jacobian,
    softmax_jacobian,
)

from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]


def _cuda_empty_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float(Fn.cosine_similarity(a, b, dim=0))


def _self_checks() -> dict:
    """Prove M-fusion == naive exp16 transport BEFORE trusting the matmul."""
    r: dict = {}
    D_l, D_s, Vt, Kt = 128, 64, 5000, 32
    g = torch.Generator().manual_seed(0)
    E_L = torch.randn(Vt, D_l, generator=g)
    E_S = torch.randn(Vt, D_s, generator=g)
    J = torch.randn(Kt, D_l, generator=g)
    G_inv = torch.linalg.pinv((E_S.T @ E_S).double()).float()
    M = E_L.T @ (E_S @ G_inv)  # [D_l, D_s]
    naive = (J @ E_L.T) @ (E_S @ G_inv)  # exp16 today
    fused = J @ M
    r["Mfusion_relerr"] = float(
        (naive - fused).abs().max().item() / (naive.abs().max().item() + 1e-9)
    )
    r["M_shape"] = list(M.shape)
    r["M_MiB"] = round(M.numel() * 4 / 1e6, 1)
    return r


def main(
    large_model: str = "Qwen/Qwen3.5-4B",
    small_model: str = "Qwen/Qwen3.5-0.8B",
    device_L: str = "cuda:0",
    device_S: str = "cuda:0",
    results_dir: str = "results",
    targets_dir: str = "results/targets",
    max_length: int = 64,
    n_samples: int = 1,
) -> None:
    torch.manual_seed(42)
    t_start = time.time()
    targets_dir = find_targets_dir(
        targets_dir
    )  # FUSE-safe: walk /kaggle/input if RES empty
    out: dict = {
        "self_check": _self_checks(),
        "device_L": device_L,
        "device_S": device_S,
        "last_layer": {},
        "mid_layers": [],
        "memory": {},
    }

    # ===================================================================
    # PHASE A — 4B (fp16, ~8 GB): E_L + J_norm_L (last layer, analytic)
    # ===================================================================
    print(f"PHASE A: L={large_model} (fp16, {device_L})...")
    t0 = time.time()
    mL = AutoModelForCausalLM.from_pretrained(
        large_model,
        torch_dtype=torch.float16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device_L)  # type: ignore[arg-type]
    mL.eval()
    n_L = mL.config.num_hidden_layers
    last_L = n_L - 1

    # One sample for J_norm_L (input-independent enough for the bridge cos).
    # Use the exp16 cache's input_ids so both models see the SAME tokens.
    cache_files = (
        sorted(f for f in os.listdir(targets_dir) if f.endswith(".pt"))
        if os.path.isdir(targets_dir)
        else []
    )
    ids = None
    for cf in cache_files:
        d0 = torch.load(os.path.join(targets_dir, cf), weights_only=True)
        if "input_ids" in d0:
            ids = d0["input_ids"].long().to(device_L)
            break
    if ids is None:
        from prism.calibration import load_calibration_data

        s0 = load_calibration_data(
            small_model,
            max_length=max_length,
            min_length=32,
            n_samples=1,
        )[0]
        ids = s0["input_ids"].long().to(device_L)
    attn = torch.ones_like(ids)

    with torch.no_grad():
        logits_L = mL(input_ids=ids, attention_mask=attn).logits[0, -1, :]
    p_L_full = Fn.softmax(logits_L.float(), dim=-1)  # [V] fp32 — full softmax
    E_L = extract_embeddings(mL).float()  # [V, D_L]  # type: ignore[arg-type]
    J_norm_L = compute_hidden_jacobian(mL, last_L, ids, attn)  # [D_L,D_L] analytic
    del logits_L, mL
    _cuda_empty_cache()
    out["memory"]["after_4B_freed_GiB"] = (
        round(torch.cuda.memory_allocated(device_L) / 1e9, 2)
        if torch.cuda.is_available()
        else 0
    )
    print(f"PHASE A done in {time.time() - t0:.0f}s (J_norm_L analytic)")

    # ===================================================================
    # PHASE B — 0.8B (fp32, ~3.2 GB): E_S, M, J_tgt_full, J_S_V
    # ===================================================================
    print(f"PHASE B: S={small_model} (fp32, {device_S})...")
    t0 = time.time()
    mS = AutoModelForCausalLM.from_pretrained(
        small_model,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    ).to(device_S)  # type: ignore[arg-type]
    mS.eval()
    n_S = mS.config.num_hidden_layers
    last_S = n_S - 1
    ids_S = ids.to(device_S)
    attn_S = attn.to(device_S)

    E_S = extract_embeddings(mS)  # type: ignore[arg-type]
    E_S = E_S.float().to(device_S)  # [V, D_S]
    G_inv = torch.linalg.pinv((E_S.T @ E_S).double()).float()  # [D_S, D_S]
    E_L_S = E_L.to(device_S)  # [V, D_L] (held since PHASE A)
    del E_L
    M = E_L_S.T @ (E_S @ G_inv)  # [D_L, D_S] ~10 MiB
    del E_L_S, E_S

    # full-vocab small-model Jacobian (analytic last layer, PROOF1)
    J_norm_S = compute_hidden_jacobian(mS, last_S, ids_S, attn_S)  # [D_S,D_S]
    W_S = mS.lm_head.weight.float()
    J_S_V = W_S @ J_norm_S  # [V, D_S] ~1 GB
    del J_norm_S
    # full-vocab target (M-fused, PROOF2)
    J_norm_M = J_norm_L.to(device_S) @ M  # [D_L, D_S] ~10 MiB
    del J_norm_L
    J_tgt_full = W_S @ J_norm_M  # [V, D_S] ~1 GB
    del J_norm_M, M
    _cuda_empty_cache()
    full_s = time.time() - t0
    print(f"PHASE B done in {full_s:.0f}s (J_S_V + J_tgt_full = 2x [V,D_S])")
    out["memory"]["with_full_J_GiB"] = (
        round(torch.cuda.memory_allocated(device_S) / 1e9, 2)
        if torch.cuda.is_available()
        else 0
    )

    p_Lf = p_L_full.to(device_S)  # [V]

    # ===================================================================
    # LAST LAYER — full vs K=256 (the cheap-full case; the real question)
    # ===================================================================
    last_pair_key = f"{n_L - 1}_{n_S - 1}"
    last_tgt = None
    for cf in cache_files:
        d = torch.load(os.path.join(targets_dir, cf), weights_only=True)
        if d.get("pair") == last_pair_key:
            last_tgt = d
            break
    ll = out["last_layer"]
    if last_tgt is not None:
        ts = torch.as_tensor(last_tgt["token_set"], dtype=torch.long, device=device_S)
        p_sub = last_tgt["p"].to(device_S).float()  # [K] exp16 subset-softmax
        J_Sp_k = last_tgt["J_S_p"].to(device_S).float()  # [K, D_S]
        J_tgtp_k = last_tgt["J_tgt_p"].to(device_S).float()  # [K, D_S]
        cos_k = _cos(J_Sp_k, J_tgtp_k)  # exp16's K=256 baseline (0 compute)
        ll["K"] = int(last_tgt.get("K", len(ts)))
        ll["exp16_K_cos"] = round(cos_k, 6)
        # full-vocab bridge, all V rows, p_L_full gauge-free weights
        ll["fullV_cos"] = round(
            _cos(softmax_jacobian(J_S_V, p_Lf), softmax_jacobian(J_tgt_full, p_Lf)), 6
        )
        # full-vocab restricted to the SAME K rows, exp16 weights (apples-apples)
        ll["fullV_on_Krows_cos"] = round(
            _cos(
                softmax_jacobian(J_S_V[ts], p_sub),
                softmax_jacobian(J_tgt_full[ts], p_sub),
            ),
            6,
        )
        # distribution side: mass / energy of p_L_full on exp16's token_set
        ll["pL_full_mass_on_K"] = round(float(p_Lf[ts].sum().item()), 5)
        ll["pL_full_energy_on_K"] = round(float((p_Lf[ts] ** 2).sum().item()), 6)
        print(
            f"LAST {last_pair_key}: exp16 K={ll['K']} cos={cos_k:.4f} | "
            f"fullV cos={ll['fullV_cos']:.4f} | fullV@Krows={ll['fullV_on_Krows_cos']:.4f} | "
            f"mass(K)={ll['pL_full_mass_on_K']:.3f} energy(K)={ll['pL_full_energy_on_K']:.4f}"
        )
    del J_S_V, J_tgt_full

    # ===================================================================
    # MID LAYERS — full infeasible; report exp16 K baseline only
    # ===================================================================
    V = p_Lf.shape[0]
    for cf in cache_files:
        d = torch.load(os.path.join(targets_dir, cf), weights_only=True)
        lk = d.get("pair")
        if lk == last_pair_key or lk is None:
            continue
        ts = torch.as_tensor(d["token_set"], dtype=torch.long, device=device_S)
        p_sub = d["p"].to(device_S).float()
        cos_k = _cos(d["J_S_p"].to(device_S).float(), d["J_tgt_p"].to(device_S).float())
        out["mid_layers"].append(
            {
                "pair": lk,
                "K": int(d.get("K", len(ts))),
                "exp16_K_cos": round(cos_k, 6),
                "fullV_feasible": False,
                "fullV_cost_note": f"~{V} backward passes (infeasible on T4)",
                "pL_full_mass_on_K": round(float(p_Lf[ts].sum().item()), 5),
            }
        )
    print(f"MID layers: {len(out['mid_layers'])} pairs (full infeasible)")

    out["elapsed_total_s"] = round(time.time() - t_start, 1)

    # ---- verdict --------------------------------------------------------
    ll = out["last_layer"]
    gap_K = ll.get("fullV_on_Krows_cos", 0) - ll.get("exp16_K_cos", 0)
    out["verdict"] = {
        "last_pair": last_pair_key,
        "exp16_K_cos": ll.get("exp16_K_cos"),
        "fullV_cos_allrows": ll.get("fullV_cos"),
        "fullV_cos_on_Krows": ll.get("fullV_on_Krows_cos"),
        "gap_fullV_minus_K": round(gap_K, 6),
        "mass_K": ll.get("pL_full_mass_on_K"),
        "note": (
            "gap_fullV_minus_K ~0 AND mass_K ~1.0 => exp16's K=256 already "
            "captures the bridge signal AND the distribution mass => the "
            "full-vocab Jacobian pass adds nothing worth its cost. Keep top-K."
        ),
    }
    print("\nVERDICT:", json.dumps(out["verdict"], indent=2))

    os.makedirs(results_dir, exist_ok=True)
    fp = os.path.join(results_dir, "fullvocab_probe.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] probe written {fp} in {out['elapsed_total_s']}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Probe: is the full-vocab target Jacobian worth it?"
    )
    parser.add_argument("--large-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--small-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device-L", default="cuda:0")
    parser.add_argument("--device-S", default="cuda:0")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--targets-dir", default="results/targets")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=1)
    a = parser.parse_args()
    main(**vars(a))
