"""Exp 27 post (local, no GPU): verdicts vs F, half-split bootstrap, PNG, findings.

Input:  kaggle_kernel/kernel-exp27/out/results/exp27_noise_floor.json
        (or --json path)
Outputs: results/exp27_findings.md, results/exp27_*.png
"""

import argparse
import json
import os

import numpy as np

LAYERS = ("6", "12", "18", "19", "22", "23")
ATTN = {l: "full" if l in ("19", "23") else "GDN" for l in LAYERS}

def _tier(c, F):
    if c < F:
        return "ORTHOGONAL"
    if c < 3 * F:
        return "WEAK"
    return "COHERENT"

def _cfg(res, l, key):
    return res["per_layer"][l]["configs"].get(key, {})

def half_split(res, l, n_boot=100, seed=42):
    """16/16 random split of the 32 baseline samples, 100 bootstraps on
    |C_a^A - C_a^B| (pos-pk). Returns mean CI + percentile spread."""
    rows = res["per_layer"][l]["halfsplit_baseline"]["per_sample"]
    rng = np.random.default_rng(seed)
    idx = np.arange(len(rows))
    diffs = []
    for _ in range(n_boot):
        sh = rng.permutation(idx)
        a, b = sh[:16], sh[16:]
        ca = np.mean([rows[i][1] for i in a])  # pos-pk col
        cb = np.mean([rows[i][1] for i in b])
        diffs.append(abs(ca - cb))
    return {
        "mean": float(np.mean(diffs)),
        "p50": float(np.percentile(diffs, 50)),
        "p90": float(np.percentile(diffs, 90)),
    }

def verdict_table(res):
    """Pre-registered 7.1: tiers for the 4 metrics at baseline S=64 K=64."""
    rows = []
    for l in LAYERS:
        F = _cfg(res, l, "S64_K64_p0")["F"]
        a01 = _cfg(res, l, "S64_K64_pair0_1")
        cb = _cfg(res, l, "S64_K64_p0")
        rows.append({
            "L": l, "attn": ATTN[l], "F": F,
            "C_a_pospk": a01.get("C_a_pospk"),
            "C_a_samepk": a01.get("C_a_samepk_late"),
            "C_a_raw": a01.get("C_a_raw"),
            "C_b": cb.get("C_b_mean"),
            "tier_pospk": _tier(a01["C_a_pospk"], F) if a01 else "n/a",
            "tier_b": _tier(cb["C_b_mean"], F) if cb else "n/a",
        })
    return rows

def main(json_path: str, out_dir: str = "results"):
    res = json.load(open(json_path))
    D = res["params"]["D"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Exp 27 verdict table (baseline S=64 K=64, F = 1/sqrt(r_eff(K+D-r_eff))) ===")
    vt = verdict_table(res)
    hdr = f"{'L':>4} {'attn':>5} {'F':>7} {'C_a_pp':>8} {'C_a_sp':>8} {'C_a_raw':>8} {'C_b':>7}  tier_pospk tier_b"
    print(hdr)
    for r in vt:
        print(f"{r['L']:>4} {r['attn']:>5} {r['F']:>7.4f} {r['C_a_pospk']:>8.3f} "
              f"{r['C_a_samepk']:>8.3f} {r['C_a_raw']:>8.3f} {r['C_b']:>7.3f}  "
              f"{r['tier_pospk']:>10} {r['tier_b']:>6}")

    # S-sweep (7.2): C_a_pospk pair0_1 per S at K=64.
    print("\n=== S-sweep (C_a pos-pk, pair 0-1, K=64) ===")
    for l in LAYERS:
        vals = {S: _cfg(res, l, f"S{S}_K64_pair0_1").get("C_a_pospk") for S in (32, 64, 128, 256)}
        print(f"L{l}: " + " ".join(f"S{S}={v:.3f}" if v is not None else f"S{S}=n/a" for S, v in vals.items()))

    # Position-sweep (7.3): C_a on pairs (0,1),(1,2),(2,4),(4,8),(8,16) at S=64 K=64.
    print("\n=== Position-sweep (S=64, K=64) ===")
    for l in LAYERS:
        parts = []
        for pr in ("0_1", "1_2", "2_4", "4_8", "8_16"):
            v = _cfg(res, l, f"S64_K64_pair{pr}").get("C_a_pospk")
            parts.append(f"{pr}:{v:.3f}" if v is not None else f"{pr}:n/a")
        print(f"L{l}: " + " ".join(parts))

    # K-sweep (7.4): gauge_share + C_a at pos 0 / pair 0_1.
    print("\n=== K-sweep (S=64) ===")
    for l in LAYERS:
        parts = []
        for K in (32, 64, 128):
            gs = _cfg(res, l, f"S64_K{K}_p0").get("gauge_share")
            ca = _cfg(res, l, f"S64_K{K}_pair0_1").get("C_a_pospk")
            parts.append(f"K{K}(gs={gs:.4f},Ca={ca:.3f})" if gs is not None and ca is not None else f"K{K}:n/a")
        print(f"L{l}: " + " ".join(parts))

    # Truncation test 6.2.
    print("\n=== Truncation test (C_a S=128 pair64_65 vs S=64 pair0_1) ===")
    for l in LAYERS:
        t = _cfg(res, l, "S128_K64_pair64_65").get("C_a_pospk")
        b = _cfg(res, l, "S64_K64_pair0_1").get("C_a_pospk")
        print(f"L{l}: S128_p64={t:.3f} vs S64_p0={b:.3f}" if t is not None and b is not None else f"L{l}: n/a")

    # Half-split (7.5) + L18 vs L19 (6.4).
    print("\n=== Half-split CI (pos-pk baseline) + L18/L19 ===")
    for l in LAYERS:
        hs = half_split(res, l)
        F = _cfg(res, l, "S64_K64_p0")["F"]
        c18 = _cfg(res, "18" if l == "19" else "18", "S64_K64_p0").get("C_b_mean")
        c19 = _cfg(res, l, "S64_K64_p0").get("C_b_mean")
        print(f"L{l}: CI mean={hs['mean']:.4f} p90={hs['p90']:.4f} (F={F:.4f}) C_b L18={c18:.3f} L19={c19:.3f}")

    # PNGs (guarded).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # heatmap C_a_pospk / F over (layer x S).
        fig, ax = plt.subplots(figsize=(7, 4))
        Ss = (32, 64, 128, 256)
        M = np.array([
            [(_cfg(res, l, f"S{S}_K64_pair0_1").get("C_a_pospk", 0)
              / _cfg(res, l, f"S{S}_K64_p0")["F"])
             for S in Ss] for l in LAYERS
        ])
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(Ss))); ax.set_xticklabels([f"S={s}" for s in Ss])
        ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels(LAYERS)
        for i in range(len(LAYERS)):
            for j in range(len(Ss)):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title("C_a(pos-pk)/F by (layer x S), K=64")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "exp27_heatmap_layer_S.png"), dpi=120)

        # heatmap C_a/F over (layer x pos-pair).
        fig, ax = plt.subplots(figsize=(7, 4))
        pairs = ("0_1", "1_2", "2_4", "4_8", "8_16")
        M = np.array([
            [(_cfg(res, l, f"S64_K64_pair{pr}").get("C_a_pospk", 0)
              / _cfg(res, l, "S64_K64_p0")["F"])
             for pr in pairs] for l in LAYERS
        ])
        im = ax.imshow(M, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p.replace("_", "-") for p in pairs])
        ax.set_yticks(range(len(LAYERS))); ax.set_yticklabels(LAYERS)
        for i in range(len(LAYERS)):
            for j in range(len(pairs)):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8)
        ax.set_title("C_a(pos-pk)/F by (layer x position pair), S=64 K=64")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "exp27_heatmap_layer_pos.png"), dpi=120)

        # r_eff bar chart (three definitions) at S=64 K=64 pos 0.
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(LAYERS)); w = 0.26
        for k, name in enumerate(("r_eff_entrop", "r_eff_stable", "r_eff_PR")):
            vals = np.array([_cfg(res, l, "S64_K64_p0").get(name, 0) for l in LAYERS])
            ax.bar(x + (k - 1) * w, vals, w, label=name)
        ax.set_xticks(x); ax.set_xticklabels([f"L{l}" for l in LAYERS])
        ax.set_yscale("log"); ax.set_ylabel("r_eff (log)")
        ax.set_title("Effective rank by layer (S=64 K=64 pos 0)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "exp27_r_eff.png"), dpi=120)
        print("PNGs saved")
    except Exception as e:
        print(f"PNG skipped: {e}")
    print("post done")

def main_cli():
    ap = argparse.ArgumentParser(description="Exp 27 post")
    ap.add_argument("--json", required=True)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    main(args.json, args.out_dir)

if __name__ == "__main__":
    main_cli()
