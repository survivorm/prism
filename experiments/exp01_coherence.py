"""Experiment 0.1: Coherence Length of the Jacobian field J(x).

Kill-switch for ESG / Exp 0.2: measure how smooth x -> J(x) is. If the field
is fractal, domain-conditional gramians G^d = (1/N) sum J J^T average over a
fractal and Bures/MMD on them are uninterpretable.

Metric (per docs/Exp0(1-3).md, Exp 0.1):
    C(delta) = E[ <J(x), J(x')>_F / (||J(x)|| ||J(x')||)  |  ||h(x)-h(x')|| ~ delta ]
    ell = delta at C = 0.5;  L(x,x') = ||J-J'||_F / ||h-h'||_2  (side stat).

Design (fixed in PLAN.md):
- gauge-free J: softmax_jacobian(J_raw, p), p = renormalized softmax over the
  K-subset at the sample's final position (exp16/19 convention).
- self-bridge: 0.8B only, token set = global top-K over final logits of all
  samples (4B projection dropped: gauge-free already kills common mode).
- h: last-position hidden states at layers {6,12,18,22}, one forward per
  sample gives all four (output_hidden_states).
- pair types: (a) same sample, adjacent positions (S-1,S) & (S-2,S-1);
  (b) two samples, same domain (delta ~ sigma_d); (c) cross-domain (large).
- gate: C(sigma_d) on the pooled (a)+(b) within-domain curve (v1 used (b)
  only; sigma_d < min delta(b) -> n/a).
- delta bins: log scale, >=5 pairs/bin else drop the bin.

Judgement gates: C(sigma_d) > 0.5 smooth / ~0.3 moderate / < 0.1 fractal-stop.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

from prism.calibration import (
    load_calibration_data,
)
from prism.jacobian import (
    compute_jacobian_for_tokens,
    softmax_jacobian,
)

MIN_BIN_PAIRS = 5
N_BINS = 14
TRUNC_TOKENS = 8


def _cos(u: torch.Tensor, v: torch.Tensor) -> float:
    num = (u * v).sum()
    den = u.pow(2).sum().sqrt() * v.pow(2).sum().sqrt()
    if den.item() < 1e-12:
        return float("nan")
    return (num / den).item()


def coh_curve(
    cos_deltas: list[tuple[float, float]], n_bins: int = N_BINS
) -> tuple[list[float], list[float]]:
    """Log-delta binning: (bin_centers, mean_cos). Drops bins with < MIN_BIN_PAIRS."""
    pts = [(d, c) for d, c in cos_deltas if d > 0 and c == c]
    if not pts:
        return [], []
    ds = np.array([p[0] for p in pts])
    edges = np.logspace(
        float(np.log10(ds.min())), float(np.log10(ds.max())) + 1e-6, n_bins + 1
    )
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [c for d, c in pts if lo <= d < hi]
        if len(sel) >= MIN_BIN_PAIRS:
            centers.append(float(np.sqrt(lo * hi)))
            means.append(float(np.mean(sel)))
    return centers, means


def ell_of(centers: list[float], means: list[float], level: float = 0.5) -> float | None:
    """delta where C crosses `level` (linear interp in log-delta). None = C stays above."""
    for i in range(len(means) - 1):
        a, b = means[i] - level, means[i + 1] - level
        if a * b <= 0 and a != b:
            t = a / (a - b)
            d0, d1 = np.log(centers[i]), np.log(centers[i + 1])
            return float(np.exp(d0 + t * (d1 - d0)))
    return None


def _pair_stats(js: list[torch.Tensor], hs: list[torch.Tensor]):
    """All C(n,2) pairs -> (cos_deltas, L_values)."""
    cos_deltas, ls = [], []
    n = len(js)
    for i in range(n):
        for j in range(i + 1, n):
            d = (hs[i] - hs[j]).norm().item()
            cos_deltas.append((d, _cos(js[i], js[j])))
            ls.append((js[i] - js[j]).norm().item() / max(d, 1e-12))
    return cos_deltas, ls


def _curve_from_pairs(
    cos_deltas: list[tuple[float, float]], ls: list[float], n_points: int
) -> dict:
    """C(delta) curve + ell + L stats from an EXACT list of pairs."""
    centers, means = coh_curve(cos_deltas)
    return {
        "n_points": n_points,
        "n_pairs": len(cos_deltas),
        "delta_centers": centers,
        "C": means,
        "ell_0.5": ell_of(centers, means),
        "L_mean": float(np.mean(ls)) if ls else None,
        "L_p90": float(np.percentile(ls, 90)) if ls else None,
    }


def _load_all(
    model_name: str, domains: list[str], n_samples: int, max_length: int
) -> dict[str, list[dict]]:
    """Per domain: n_samples full sequences, exactly max_length tokens."""
    per: dict[str, list[dict]] = {}
    for dom in domains:
        full = load_calibration_data(
            model_name,
            max_length=max_length,
            min_length=max_length,  # force exactly max_length (truncation active)
            n_samples=n_samples,
            domain=dom,
        )
        if len(full) < n_samples:
            full = load_calibration_data(
                model_name,
                max_length=max_length,
                min_length=1,
                n_samples=n_samples,
                domain=dom,
            )
        per[dom] = full
    return per


def _one_forward(
    m, s: dict, layers: list[int]
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """One forward: final-softmax p + last-position h at each layer."""
    ids = s["input_ids"].to(m.device)
    attn = s["attention_mask"].to(m.device)
    with torch.no_grad():
        out = m(
            input_ids=ids, attention_mask=attn, output_hidden_states=True
        )
    p = torch.softmax(out.logits[0, -1, :], dim=-1)
    hs: dict[int, list[torch.Tensor]] = {}
    for l in layers:
        hs[l] = [out.hidden_states[l + 1][0, i, :].float().cpu() for i in (-1, -2, -3)]
    return p, hs


def main(
    model="Qwen/Qwen3.5-0.8B",
    device="auto",
    results_dir="results",
    layers=(6, 12, 18, 22),
    K=64,
    n_samples=8,
    domains=("wiki", "math", "code", "chat"),
    max_length=64,
    make_png: bool = True,
):
    torch.manual_seed(42)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    from transformers import AutoModelForCausalLM

    print(f"Loading {model}...")
    m = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device)
    m.eval()
    layers = list(layers)
    domains = list(domains)

    print(f"Loading {n_samples} samples x {len(domains)} domains (+truncated)...")
    per_domain = _load_all(model, domains, n_samples, max_length)

    # Global token set: top-K final-logit tokens across all samples (0.8B).
    counts: dict[int, int] = {}
    for dom in domains:
        for s in per_domain[dom][:n_samples]:
            p, _ = _one_forward(m, s, [])
            top = p.topk(K).indices.tolist()
            for t in top:
                counts[t] = counts.get(t, 0) + 1
    token_set = sorted([t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:K]])
    print(f"Token set |U| = {len(token_set)}")

    # J-cache: per (domain, sample, layer, pos) -> gauge-free J [K, D].
    # pos 0=S, 1=S-1, 2=S-2. J at S-k = compute_jacobian_for_tokens on
    # ids[:, :-k] (causality: exact). 8 x 4 layers x 3 = 384 J-calls.
    J: dict[tuple, torch.Tensor] = {}
    H: dict[tuple, torch.Tensor] = {}
    done = 0
    total = sum(len(v) for v in per_domain.values()) * len(layers) * 3
    for dom in domains:
        for si, s in enumerate(per_domain[dom]):
            p, hs = _one_forward(m, s, layers)
            pk = p[token_set]
            pk = pk / pk.sum()
            for l in layers:
                for pos in (0, 1, 2):
                    ids_k = s["input_ids"][:, :-pos] if pos else s["input_ids"]
                    mask_k = s["attention_mask"][:, :-pos] if pos else s["attention_mask"]
                    Jp = softmax_jacobian(
                        compute_jacobian_for_tokens(
                            m, l, ids_k.to(m.device), mask_k.to(m.device), token_set,
                        ).float(),
                        pk.to(m.device),
                    ).float().cpu()
                    J[(dom, si, l, pos)] = Jp
                    H[(dom, si, l, pos)] = hs[l][pos]
                    done += 1
            print(f"  {dom} s{si}: 3pos x {len(layers)}L J done ({done}/{total})")

    # Per-layer: 3 pair-type curves + sigma_d + gate.
    per_layer: dict[str, dict] = {}
    for l in layers:
        res: dict = {}
        # (a) intra-sample adjacent positions: (S-1,S) and (S-2,S-1).
        a_cd: list[tuple[float, float]] = []
        a_ls: list[float] = []
        for dom in domains:
            for si in range(n_samples):
                for p1, p2 in ((1, 0), (2, 1)):
                    J1, J2 = J[(dom, si, l, p1)], J[(dom, si, l, p2)]
                    H1, H2 = H[(dom, si, l, p1)], H[(dom, si, l, p2)]
                    d = (H1 - H2).norm().item()
                    a_cd.append((d, _cos(J1, J2)))
                    a_ls.append((J1 - J2).norm().item() / max(d, 1e-12))
        res["intra_sample"] = _curve_from_pairs(a_cd, a_ls, 2 * n_samples * len(domains))
        res["intra_sample"]["C_raw"] = [c for _, c in a_cd]
        res["intra_sample"]["delta_raw"] = [d for d, _ in a_cd]
        # (b) intra-domain: C(n,2) sample pairs within one domain.
        intra = []
        b_cd_pool: list[tuple[float, float]] = []
        for dom in domains:
            jb = [J[(dom, si, l, 0)] for si in range(n_samples)]
            hb = [H[(dom, si, l, 0)] for si in range(n_samples)]
            cd_dom, ls_dom = _pair_stats(jb, hb)
            b_cd_pool.extend(cd_dom)
            intra.append(
                {"domain": dom, **_curve_from_pairs(cd_dom, ls_dom, len(jb))}
            )
        res["intra_domain"] = intra
        # sigma_d per spec: norm of per-dim std (unbiased=False) of full
        # samples' last-position h, mean over domains.
        stds = []
        for dom in domains:
            hs_d = torch.stack([H[(dom, si, l, 0)] for si in range(n_samples)])
            stds.append(hs_d.std(dim=0, unbiased=False).norm().item())
        res["sigma_d"] = float(np.mean(stds))
        # (c) cross-domain: all between-domain sample pairs.
        cd_cd: list[tuple[float, float]] = []
        cd_ls: list[float] = []
        n_cross = 0
        for di, d1 in enumerate(domains):
            for d2 in domains[di + 1:]:
                for si in range(n_samples):
                    for sj in range(n_samples):
                        J1, J2 = J[(d1, si, l, 0)], J[(d2, sj, l, 0)]
                        H1, H2 = H[(d1, si, l, 0)], H[(d2, sj, l, 0)]
                        d = (H1 - H2).norm().item()
                        cd_cd.append((d, _cos(J1, J2)))
                        cd_ls.append((J1 - J2).norm().item() / max(d, 1e-12))
                        n_cross += 1
        res["cross_domain"] = _curve_from_pairs(cd_cd, cd_ls, n_cross)
        # Gate: C(sigma_d) from the POOLED within-domain curve: adjacent
        # position pairs (a) + sample pairs (b) across all domains.
        # (b) alone starts at delta > sigma_d, so the gate needs (a)'s
        # small-delta range to interpolate at sigma_d.
        g_cd = a_cd + b_cd_pool
        g_centers, g_means = coh_curve(g_cd)
        if len(g_centers) >= 2 and min(g_centers) <= res["sigma_d"] <= max(g_centers):
            res["C_at_sigma_d"] = float(
                np.interp(res["sigma_d"], g_centers, g_means)
            )
        else:
            res["C_at_sigma_d"] = None
        res["gate_curve"] = {
            "delta_centers": g_centers, "C": g_means, "n_pairs": len(g_cd)
        }
        per_layer[str(l)] = res
        print(
            f"  L{l}: sigma_d={res['sigma_d']:.3f} "
            f"C@sigma_d={res['C_at_sigma_d']} "
            f"ell(b)={intra[0]['ell_0.5']} L_mean(b)={intra[0]['L_mean']}"
        )

    # Save JSON + PNG.
    all_results = {
        "layers": per_layer,
        "params": {
            "model": model,
            "n_samples": n_samples,
            "K": K,
            "max_length": max_length,
            "device": device,
            "layers": layers,
            "domains": domains,
            "trunc_tokens": TRUNC_TOKENS,
            "min_bin_pairs": MIN_BIN_PAIRS,
            "n_bins": N_BINS,
            "token_set_size": len(token_set),
        },
    }
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "exp01_coherence.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {json_path}")

    if make_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=False)
            for ax, l in zip(axes.flat, layers):
                for blk_key, label, marker in (
                    ("intra_sample", "a: intra-sample", "o"),
                    ("intra_domain", "b: intra-domain (pooled dom)", "s"),
                    ("cross_domain", "c: cross-domain", "^"),
                ):
                    if blk_key == "intra_domain":
                        pts = [
                            (b["delta_centers"], b["C"]) for b in per_layer[str(l)]["intra_domain"]
                        ]
                        for centers, means in pts:
                            ax.plot(centers, means, marker, ms=4, alpha=0.5)
                        continue
                    blk = per_layer[str(l)][blk_key]
                    ax.plot(
                        blk["delta_centers"], blk["C"],
                        marker=marker, label=label,
                    )
                sd = per_layer[str(l)]["sigma_d"]
                ax.axvline(sd, color="gray", ls="--", lw=0.8)
                ax.set_title(f"L{l} (sigma_d={sd:.3f})")
                ax.set_xscale("log")
                ax.set_xlabel("|h-h'|")
                ax.set_ylabel("C")
                ax.set_ylim(-0.1, 1.05)
                ax.legend(fontsize=7)
            fig.suptitle("Exp 0.1: Jacobian coherence curves (0.8B, gauge-free)")
            fig.tight_layout()
            png_path = os.path.join(results_dir, "exp01_coherence.png")
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            print(f"Saved to {png_path}")
        except Exception as e:
            print(f"PNG failed (non-fatal): {e}")

    # Verdict summary (gate thresholds per plan).
    for l in layers:
        g = per_layer[str(l)]["C_at_sigma_d"]
        verdict = (
            "SMOOTH (C>0.5)" if g is not None and g > 0.5
            else "MODERATE (0.3<=C<=0.5)" if g is not None and g >= 0.3
            else "FRACTAL-RISK (C<0.3)" if g is not None else "n/a"
        )
        print(f"L{l}: C(sigma_d)={g} -> {verdict}")


def _selfcheck() -> None:
    """Synthetic smooth field: C(delta) must decrease monotonically and ell found.

    Circular manifold: J(t) = [cos t, sin t, cos 2t, sin 2t, ...] — bounded,
    and <J(t),J(t')> depends only on |t-t'| and decays monotonically on [0,3].
    h(t) = t*ones -> delta = 4|t-t'| (proportional, so bins order by |dt|).
    """
    D = 16
    n = 40
    ts = torch.linspace(0.0, 3.0, n)
    Js, Hs = [], []
    for t in ts:
        # Circle: <v(t), v(t')> = 8*cos(t-t') — a pure function of |dt|, and
        # cos(d) is strictly decreasing on d in [0, 3] < pi. Norm = 4 const.
        base = torch.stack([torch.cos(t), torch.sin(t)])
        Js.append(base.repeat(D // 2))
        Hs.append(t * torch.ones(D))
    cd, ls = _pair_stats(Js, Hs)
    centers, means = coh_curve(cd)
    assert len(centers) >= 3, f"need >=3 bins, got {len(centers)}"
    assert all(means[i] >= means[i + 1] - 1e-6 for i in range(len(means) - 1)), \
        f"C must decrease monotonically: {[round(x, 4) for x in means]}"
    ell = ell_of(centers, means)
    assert ell is not None and 0 < ell < 5, f"bad ell={ell}"
    assert all(x > 0 for x in ls[:5])
    print("selfcheck OK: C monotone, ell =", round(ell, 3))


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description="Exp 0.1: Jacobian coherence length")
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--layers", default="6,12,18,22")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--domains", default="wiki,math,code,chat")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--no-png", action="store_true")
    args = parser.parse_args()
    main(
        model=args.model,
        device=args.device,
        results_dir=args.results_dir,
        layers=tuple(int(x) for x in args.layers.split(",")),
        K=args.K,
        n_samples=args.n_samples,
        domains=tuple(x for x in args.domains.split(",") if x),
        max_length=args.max_length,
        make_png=not args.no_png,
    )
