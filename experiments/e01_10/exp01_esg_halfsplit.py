"""Exp 0.1 add-on (3.2): half-split reliability of the domain ESG gramian.

LOCAL, no GPU. Uses the exp16 J cache in results/targets/*.pt
(J_S_p: gauge-free sample-local J [K, D], 4 samples per layer).

Test: split the 4 samples 2+2, build G_A, G_B = mean J J^T / K per half,
compare top-r eigenspaces (principal angles). If mean_cos2 is high, the
domain gramian's main subspace is stable across samples (G^d makes sense
even pooled); if low, only sample-local gramians are meaningful.
Also winner-take-all probe: pooled G's top eigenvector vs each sample's.

Usage:
  python experiments/exp01_esg_halfsplit.py --selfcheck
  python experiments/exp01_esg_halfsplit.py --targets results/targets --r 16
"""

import argparse
import glob
import json
import os
import sys

import torch



TARGETS_DIR = "results/targets"
ESG_TOP_R = 16


def _top_r_basis(G: torch.Tensor, r: int) -> torch.Tensor:
    """Orthonormal basis of the top-r eigenspace of a symmetric PSD gramian."""
    w, V = torch.linalg.eigh(G)
    return V[:, -r:].flip(1).contiguous()


def _principal_alignment(
    A: torch.Tensor, B: torch.Tensor
) -> tuple[float, float]:
    """Subspace alignment via principal angles of A^T B.

    A, B: orthonormal bases [D, r]. Returns (mean cos^2, max principal angle deg).
    1.0/0.0 deg = same subspace; 0.0/90.0 deg = orthogonal.
    """
    sv = torch.linalg.svdvals(A.t() @ B)
    cs2 = sv.pow(2)
    angle = float(torch.rad2deg(torch.acos(cs2.min().clamp(0, 1).sqrt())).item())
    return float(cs2.mean().item()), angle


def _half_split_esg(
    js: list[torch.Tensor], r: int = ESG_TOP_R
) -> dict:
    """Half-split reliability of the domain ESG gramian.

    Splits n samples into halves (deterministic interleave), builds
    G_A = mean J J^T / K over half A, same for B; compares top-r subspaces
    (principal angles). Also winner-take-all probe: pooled G's top eigenvector
    vs each sample's top eigenvector (cos).
    """
    n = len(js)
    if n < 4:
        return {"n": n, "skip": "need >=4 samples"}
    idx_a = list(range(0, n, 2))
    idx_b = list(range(1, n, 2))

    def gram(idx):
        G = torch.zeros_like(js[0] @ js[0].t())
        for i in idx:
            Ji = js[i]
            # Frobenius-normalize each sample: J J^T / ||J||_F^2 (cos-based gramian)
            G += (Ji @ Ji.t()) / Ji.pow(2).sum().clamp_min(1e-12)
        return G / len(idx)

    G_a, G_b = gram(idx_a), gram(idx_b)
    base_a, base_b = _top_r_basis(G_a, r), _top_r_basis(G_b, r)
    mean_cs2, max_angle = _principal_alignment(base_a, base_b)

    G_pool = (G_a + G_b) / n
    top_pool = _top_r_basis(G_pool, 1)[:, 0]
    wta = []
    for i in range(n):
        Gi = js[i] @ js[i].t()
        _, Vi = torch.linalg.eigh(Gi)
        # abs: eigenvector sign gauge (v ~ -v is the same 1D subspace)
        wta.append(abs(float((top_pool * Vi[:, -1]).sum().item())))
    return {
        "n": n,
        "r": r,
        "mean_cos2": mean_cs2,
        "max_angle_deg": max_angle,
        "wta_top1_cos_per_sample": wta,
        "wta_top1_max": max(wta),
    }




def load_j_cache(targets_dir: str) -> dict[int, list[torch.Tensor]]:
    """l_S -> [J_S_p per sample] from exp16 targets."""
    per_layer: dict[int, list[torch.Tensor]] = {}
    for f in sorted(glob.glob(os.path.join(targets_dir, "*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        per_layer.setdefault(int(d["l_S"]), []).append(d["J_S_p"].float())
    return per_layer


def main(targets_dir: str = TARGETS_DIR, r: int = 16, results_dir: str = "results") -> None:
    per_layer = load_j_cache(targets_dir)
    out = {}
    for l in sorted(per_layer):
        js = per_layer[l]
        res = _half_split_esg(js, r)
        out[str(l)] = res
        fcs = res.get("mean_cos2")
        fa = res.get("max_angle_deg")
        print(
            f"  L{l}: n={res.get('n')} mean_cos2={fcs:.3f} max_angle={fa:.1f}deg "
            f"wta_max={res.get('wta_top1_max', float('nan')):.3f}"
        ) if fcs is not None else print(f"  L{l}: {res}")
    os.makedirs(results_dir, exist_ok=True)
    p = os.path.join(results_dir, "exp01_esg_halfsplit.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {p}")


def _selfcheck() -> None:
    g = torch.Generator().manual_seed(0)
    base = torch.randn(8, 4, generator=g)
    coherent = [base + 0.05 * torch.randn_like(base) for _ in range(4)]
    fibered = [torch.randn(8, 4, generator=g) for _ in range(4)]
    r = 2
    hs_c = _half_split_esg(coherent, r)
    hs_f = _half_split_esg(fibered, r)
    assert hs_c["mean_cos2"] > 0.8, f"coherent should align: {hs_c['mean_cos2']}"
    assert hs_f["mean_cos2"] < hs_c["mean_cos2"], "fibered must be less aligned"
    print(
        f"selfcheck OK: coherent={hs_c['mean_cos2']:.3f} fibered={hs_f['mean_cos2']:.3f} "
        f"wta_max={hs_c['wta_top1_max']:.3f}"
    )


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        raise SystemExit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=TARGETS_DIR)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--results-dir", default="results")
    a = ap.parse_args()
    main(a.targets, a.r, a.results_dir)
