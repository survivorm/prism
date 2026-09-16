"""Experiment 0.1 v6: coherence + gauge-share / position-sensitivity diagnostics.

v6 (colleague review 3, 09-16): same pipeline as v5, diagnostics added:
  1. L_mean/L_p90 per convention (v5 shared pos-pk L for all three - bug).
  2. gauge_share = ||P(p)J||_F^2 / ||J||_F^2 per (sample, layer, pos).
  3. cos(p_t, p_{t+1}) on token-set-normalized distributions, per pair.
  4. verdict vs 1/sqrt(D) noise floor: ORTHOGONAL / WEAK / COHERENT.
  5. L18 flag in summary: C_b > 1.5*C_a.
  6. PK reproducibility: per-layer pk_hash + pk_top_token_ids (full PK in
     the in-memory cache; JSON gets hash + top-8 only).
JSON: exp01_coherence_v6.json (v5's exp01_coherence.json left intact).

v4 (colleague review, 09-16): the v3 gate POOLED (a) adjacent-position pairs
and (b) sample pairs into one curve — meaningless, since (a) C~0.3-0.9 while
(b) C~0.005-0.05 at comparable delta (10-100x). Fixes:
  1. Separate C_a(delta), C_b(delta) curves; gate = C_b(sigma_d) canonical
     (between-sample), C_a(sigma_d) reported alongside.
  2. gauge p (pk) taken at the SAME position as J (logits[0, S-1-pos]).
  3. One forward per sample (logits+hs cached; J pass = backward only).
  4. half-split ESG reliability: separate script exp01_esg_halfsplit.py.

v5 (colleague review 2, 09-16): v3's high C_a (0.33-0.92) was an artifact of
the shared final-position projector; v4's per-position gauges are the
convention-correct numbers, but to distinguish (A) field-not-smooth vs
(B) gauge-kills-signal vs (C) small-sample noise we report THREE gauge
conventions per pair, from the cached raw J:
  - pos-pk:   J^(p_t) vs J^(p_{t+1})  (one projector per position — canonical)
  - same-pk:  P(p_t) J_t vs P(p_t) J_{t+1}  (one projector on both)
  - raw:      J_t vs J_{t+1}  (no gauge)
Per-pair cos lists are saved for distribution/histogram analysis
(L18 anomaly: C_b > C_a).

NO FIELD-CHARACTER VERDICT is asserted in this script; the tiers below are
screening labels, to be recalibrated against the 1/sqrt(D) noise floor.

Metric:
    C(delta) = E[ <J,J'>_F / (||J|| ||J'||)  |  ||h-h'|| ~ delta ]
Screening labels ONLY (thresholds calibrated to v3's projector-artifact
scale — NOT a field-character verdict; recalibrate vs 1/sqrt(D) noise floor
~0.03 before interpreting): C_b>0.5 SMOOTH / >=0.1 WEAK / <0.1 & C_a>=0.3
FIBERED / <0.1 FRACTAL-RISK.
"""

import argparse
import hashlib
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
    # ponytail: edges padded by 2% so sigma_d slightly outside the pair range
    # still interpolates (v3: L6 sigma_d=1.663 vs min bin 1.67 -> None).
    edges = np.logspace(
        float(np.log10(ds.min())) - 0.01, float(np.log10(ds.max())) + 0.01, n_bins + 1
    )
    centers, means = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [c for d, c in pts if lo <= d < hi]
        if len(sel) >= MIN_BIN_PAIRS:
            centers.append(float(np.sqrt(lo * hi)))
            means.append(float(np.mean(sel)))
    return centers, means


def ell_of(centers: list[float], means: list[float], level: float = 0.5) -> float | None:
    """delta where C crosses `level` (linear interp in log-delta). None = no crossing."""
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


def _one_forward(m, s: dict, layers: list[int]) -> dict:
    """ONE forward per sample: full logits + last-3 hidden states per layer.

    v4 fix (colleague 1.2): gauge p must come from the SAME position as J,
    so the full logits tensor is kept; the J pass (backward) reuses this
    forward and does not re-run the model.
    """
    ids = s["input_ids"].to(m.device)
    attn = s["attention_mask"].to(m.device)
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    logits = out.logits[0].float().cpu()
    hs: dict[int, list[torch.Tensor]] = {}
    for l in layers:
        hs[l] = [out.hidden_states[l + 1][0, i, :].float().cpu() for i in (-1, -2, -3)]
    return {"logits": logits, "hs": hs}


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

    print(f"Loading {n_samples} samples x {len(domains)} domains...")
    per_domain = _load_all(model, domains, n_samples, max_length)

    # v4: ONE forward per sample, full logits cached (gauge p per position).
    fwd: dict[tuple, dict] = {}
    for dom in domains:
        for si, s in enumerate(per_domain[dom][:n_samples]):
            fwd[(dom, si)] = _one_forward(m, s, layers)

    # Global token set: top-K final-logit tokens across all samples (0.8B).
    counts: dict[int, int] = {}
    for dom in domains:
        for si in range(n_samples):
            p = torch.softmax(fwd[(dom, si)]["logits"][-1], dim=-1)
            top = p.topk(K).indices.tolist()
            for t in top:
                counts[t] = counts.get(t, 0) + 1
    token_set = sorted([t for t, _ in sorted(counts.items(), key=lambda x: -x[1])[:K]])
    print(f"Token set |U| = {len(token_set)}")

    # J-cache: per (domain, sample, layer, pos) -> gauge-free J [K, D] AND
    # raw J (for the three-convention analysis, v5). pos 0=S, 1=S-1, 2=S-2.
    # J at S-pos via compute_jacobian_for_tokens on ids[:, :-pos] (causality:
    # exact). Gauge pk from the SAME position.
    J: dict[tuple, torch.Tensor] = {}
    JRAW: dict[tuple, torch.Tensor] = {}
    PK: dict[tuple, torch.Tensor] = {}
    H: dict[tuple, torch.Tensor] = {}
    done = 0
    total = n_samples * len(domains) * len(layers) * 3
    for dom in domains:
        for si in range(n_samples):
            f = fwd[(dom, si)]
            for l in layers:
                for pos in (0, 1, 2):
                    ids_k = per_domain[dom][si]["input_ids"]
                    mask_k = per_domain[dom][si]["attention_mask"]
                    if pos:
                        ids_k = ids_k[:, :-pos]
                        mask_k = mask_k[:, :-pos]
                    p = torch.softmax(f["logits"][-(pos + 1)], dim=-1)
                    pk = p[token_set]
                    pk = (pk / pk.sum()).to(m.device)
                    Jr = compute_jacobian_for_tokens(
                        m, l, ids_k.to(m.device), mask_k.to(m.device), token_set,
                    ).float()
                    Jp = softmax_jacobian(Jr, pk).float()
                    J[(dom, si, l, pos)] = Jp.cpu()
                    JRAW[(dom, si, l, pos)] = Jr.cpu()
                    PK[(dom, si, l, pos)] = pk.cpu()
                    H[(dom, si, l, pos)] = f["hs"][l][pos]
                    done += 1
            print(f"  {dom} s{si}: {len(layers)}L x 3pos J done ({done}/{total})")

    # Per-layer: separate (a)/(b) curves + sigma_d + split gates + ESG half-split.
    per_layer: dict[str, dict] = {}
    for l in layers:
        res: dict = {}
        # (a) intra-sample adjacent positions: (S-1,S) and (S-2,S-1).
        # v6: L per convention — a_ls (pos-pk), a_ls_same, a_ls_nog (no shared L).
        a_cd: list[tuple[float, float]] = []
        a_cd_same: list[tuple[float, float]] = []
        a_cd_nog: list[tuple[float, float]] = []
        a_ls: list[float] = []
        a_ls_same: list[float] = []
        a_ls_nog: list[float] = []
        for dom in domains:
            for si in range(n_samples):
                for p1, p2 in ((1, 0), (2, 1)):
                    J1, J2 = J[(dom, si, l, p1)], J[(dom, si, l, p2)]
                    H1, H2 = H[(dom, si, l, p1)], H[(dom, si, l, p2)]
                    d = (H1 - H2).norm().item()
                    J1r, J2r = JRAW[(dom, si, l, p1)], JRAW[(dom, si, l, p2)]
                    p1v = PK[(dom, si, l, p1)]  # earlier position's gauge on both
                    J1s = softmax_jacobian(J1r, p1v)
                    J2s = softmax_jacobian(J2r, p1v)
                    a_cd.append((d, _cos(J1, J2)))
                    a_ls.append((J1 - J2).norm().item() / max(d, 1e-12))
                    a_cd_same.append((d, _cos(J1s, J2s)))
                    a_ls_same.append((J1s - J2s).norm().item() / max(d, 1e-12))
                    a_cd_nog.append((d, _cos(J1r, J2r)))
                    a_ls_nog.append((J1r - J2r).norm().item() / max(d, 1e-12))
        res["intra_sample"] = _curve_from_pairs(a_cd, a_ls, 2 * n_samples * len(domains))
        res["intra_sample"]["C_raw"] = [c for _, c in a_cd]  # per-pair, pos-pk
        res["intra_sample"]["C_raw_samepk"] = [c for _, c in a_cd_same]
        res["intra_sample"]["C_raw_nogauge"] = [c for _, c in a_cd_nog]
        res["intra_sample"]["delta_raw"] = [d for d, _ in a_cd]
        res["intra_sample_samepk"] = _curve_from_pairs(a_cd_same, a_ls_same, 2 * n_samples * len(domains))
        res["intra_sample_nogauge"] = _curve_from_pairs(a_cd_nog, a_ls_nog, 2 * n_samples * len(domains))
        # v6: gauge share ||P(p)J||_F^2 / ||J||_F^2 per (sample, layer, pos),
        # and cos(p_t, p_{t+1}) on the token-set-normalized distributions.
        gs_vals, cp_vals = [], []
        for dom in domains:
            for si in range(n_samples):
                for pos in (0, 1, 2):
                    Jr = JRAW[(dom, si, l, pos)]
                    pv = PK[(dom, si, l, pos)]
                    Jp = softmax_jacobian(Jr, pv)
                    gs_vals.append(
                        float(Jp.pow(2).sum() / Jr.pow(2).sum().clamp_min(1e-12))
                    )
                for p1, p2 in ((1, 0), (2, 1)):
                    v1, v2 = PK[(dom, si, l, p1)], PK[(dom, si, l, p2)]
                    cp_vals.append(_cos(v1, v2))
        res["gauge_share_mean"] = float(np.mean(gs_vals))
        # gs_vals order: pos (0,1,2) per sample -> pos0 = gs_vals[0::3] etc.
        res["gauge_share_per_pos"] = {
            f"pos{q}": float(np.mean(gs_vals[q::3])) for q in (0, 1, 2)
        }
        res["cos_p_mean"] = float(np.mean(cp_vals))
        res["cos_p_per_pair"] = cp_vals
        # v6: PK reproducibility — hash of all PK on this layer + top-8 tokens
        # by mean pk (full PK not saved: 64x64x8x4x3 too big for the JSON).
        _ph = hashlib.sha256()
        _pk_sum = torch.zeros(K)
        for dom in domains:
            for si in range(n_samples):
                for pos in (0, 1, 2):
                    _ph.update(PK[(dom, si, l, pos)].numpy().tobytes())
                    _pk_sum += PK[(dom, si, l, pos)]
        res["pk_hash"] = _ph.hexdigest()[:16]
        res["pk_top_token_ids"] = [token_set[i] for i in _pk_sum.topk(8).indices.tolist()]
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
        res["intra_domain_C_raw"] = [c for _, c in b_cd_pool]  # per-pair, for histogram
        # sigma_d per spec: norm of per-dim std (unbiased=False) of full
        # samples' last-position h, mean over domains.
        stds = []
        for dom in domains:
            hs_d = torch.stack([H[(dom, si, l, 0)] for si in range(n_samples)])
            stds.append(hs_d.std(dim=0, unbiased=False).norm().item())
        res["sigma_d"] = float(np.mean(stds))

        def _c_at(cd: list[tuple[float, float]], x: float) -> float | None:
            centers, means = coh_curve(cd)
            if len(centers) < 2:
                return None
            return float(np.interp(x, centers, means))

        # v5: SPLIT gates, THREE gauge conventions. Canonical = between-sample
        # (b) at sigma_d; intra-sample (a) in all three conventions reported
        # alongside (not pooled — 1.1).
        res["C_at_sigma_d_b"] = _c_at(b_cd_pool, res["sigma_d"])
        res["C_at_sigma_d_a"] = _c_at(a_cd, res["sigma_d"])
        res["C_at_sigma_d_a_samepk"] = _c_at(a_cd_same, res["sigma_d"])
        res["C_at_sigma_d_a_nogauge"] = _c_at(a_cd_nog, res["sigma_d"])
        bc, bm = coh_curve(b_cd_pool)
        ac, am = coh_curve(a_cd)
        res["gate_curve_b"] = {"delta_centers": bc, "C": bm, "n_pairs": len(b_cd_pool)}
        res["gate_curve_a"] = {"delta_centers": ac, "C": am, "n_pairs": len(a_cd)}

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
        per_layer[str(l)] = res
        cb = res["C_at_sigma_d_b"]
        ca = res["C_at_sigma_d_a"]
        print(
            f"  L{l}: sigma_d={res['sigma_d']:.3f} "
            f"C_b@sigma_d={cb} C_a@sigma_d={ca}"
        )

    # Save JSON.
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
                    "min_bin_pairs": MIN_BIN_PAIRS,
            "n_bins": N_BINS,
            "token_set_size": len(token_set),
        },
    }
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "exp01_coherence_v6.json")
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
                blk_a = per_layer[str(l)]["intra_sample"]
                ax.plot(blk_a["delta_centers"], blk_a["C"], marker="o", ms=4,
                        label="a: intra-sample (positions)")
                for b in per_layer[str(l)]["intra_domain"]:
                    ax.plot(b["delta_centers"], b["C"], marker="s", ms=4, alpha=0.5)
                blk_c = per_layer[str(l)]["cross_domain"]
                ax.plot(blk_c["delta_centers"], blk_c["C"], marker="^", ms=4, alpha=0.7,
                        label="c: cross-domain")
                sd = per_layer[str(l)]["sigma_d"]
                ax.axvline(sd, color="gray", ls="--", lw=0.8)
                ax.set_title(f"L{l} (sigma_d={sd:.3f})")
                ax.set_xscale("log")
                ax.set_xlabel("|h-h'|")
                ax.set_ylabel("C")
                ax.set_ylim(-0.1, 1.05)
                ax.legend(fontsize=7)
            fig.suptitle("Exp 0.1 v4: Jacobian coherence — SPLIT curves (a) vs (b)")
            fig.tight_layout()
            png_path = os.path.join(results_dir, "exp01_coherence.png")
            fig.savefig(png_path, dpi=110)
            plt.close(fig)
            print(f"Saved to {png_path}")
        except Exception as e:
            print(f"PNG failed (non-fatal): {e}")

    # v5: three-convention summary (no field-character verdict — thresholds
    # below are v3-scale screening labels, recalibrate vs 1/sqrt(D) noise).
    # v6: verdict per layer per convention vs 1/sqrt(D) noise floor (D=hidden
    # dim, 1024 for 0.8B). ORTHOGONAL < 1/sqrt(D); WEAK < 0.15; COHERENT above.
    D = m.config.hidden_size
    NOISE_FLOOR = 1.0 / (D ** 0.5)

    def verdict(c: float | None) -> str:
        if c is None:
            return "n/a"
        if c < NOISE_FLOOR:
            return "ORTHOGONAL"
        if c < 0.15:
            return "WEAK"
        return "COHERENT"

    for l in layers:
        g = per_layer[str(l)]
        g["noise_floor"] = NOISE_FLOOR
        for key, tag in (
            ("C_at_sigma_d_a", "pospk"),
            ("C_at_sigma_d_a_samepk", "samepk"),
            ("C_at_sigma_d_a_nogauge", "raw"),
        ):
            g[f"verdict_{tag}"] = verdict(g[key])
        for key, tag in (("C_at_sigma_d_b", "b"),):
            g[f"verdict_{tag}"] = verdict(g[key])
        cb, ca = g["C_at_sigma_d_b"], g["C_at_sigma_d_a"]
        flag = (
            " [FLAG: C_b > 1.5*C_a]"
            if (cb is not None and ca is not None and cb > 1.5 * ca)
            else ""
        )
        r = lambda x: None if x is None else round(x, 3)
        print(
            f"L{l}: C_b={r(cb)} C_a(pospk)={r(ca)}{flag} "
            f"C_a(samepk)={r(g['C_at_sigma_d_a_samepk'])} "
            f"C_a(raw)={r(g['C_at_sigma_d_a_nogauge'])} "
            f"gauge_share={r(g['gauge_share_mean'])} cos_p={r(g['cos_p_mean'])}"
        )
        print(
            f"     verdict: pospk={g['verdict_pospk']} samepk={g['verdict_samepk']} "
            f"raw={g['verdict_raw']} b={g['verdict_b']} "
            f"(noise_floor=1/sqrt({D})={NOISE_FLOOR:.4f})"
        )


def _selfcheck() -> None:
    """Model-free check: smooth field C monotone + ell exists."""
    # 1) Smooth field: circle. C(delta) must decrease, ell found.
    D = 16
    n = 40
    ts = torch.linspace(0.0, 3.0, n)
    Js, Hs = [], []
    for t in ts:
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

    print(f"selfcheck OK: ell={ell:.3f}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description="Exp 0.1 v4: Jacobian coherence, fibered")
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
