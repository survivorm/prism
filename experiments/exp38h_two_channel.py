"""E-38h. Two-channel Goodhart decomposition of the readout-only ΔW.

Pre-registered (docs/E38.md §E-38h):
  - ΔW(s) from the E-16 readout-only snapshots; states — the 100-step grid
    (10..1000), subsample K_STEPS bracketing the knee (84 ≈ dose 1.0).
  - Fixed two-sided top-16 projector from the CACHED J23 (reference
    unperturbed model): U_t [128,16] (left/token-side, rows of J23),
    V_t [16,1024] (right/hidden-side, rows = right singular vectors).
    ΔW_top = U_t U_tᵀ ΔW V_tᵀ V_t, ΔW_tail = ΔW − ΔW_top.
  - Non-commutativity: R(s1,s2) = procrustes_align (rotation aligning the
    channel's top-16 frame s2 -> s1); associativity defect
    ‖R12 R23 − R13‖_F / r over triples.
  - Per-channel Jacobian response: J(s) = (√D/r) M_K(s)(I−ûûᵀ) is exact at
    L23 (E-35), so ΔJ(s) = ΔJ_top(s) + ΔJ_tail(s) via the same channel
    split applied to ΔW — computed from ΔW alone. NOTE (E38_fixes P1):
    "h_t invariant" is FALSE for tied Qwen3.5-0.8B (lm_head = embed_tokens),
    so ΔJ = ΔW @ J_norm(cached h) is a linear-response approximation, not
    an identity; no model forward.

BLOCKED (honest, documented): the per-channel `bridge(s)` = 1 −
jacobian_gap(J_L, J_S, E_L, E_S, p) needs J_L of the large target model,
which is not on this machine. Only the small-model-side per-channel
response (ΔJ_top/ΔJ_tail energy) is computed; the bridge attribution
stays blocked.

States: the snapshot keys are E-16 optimiser STEPS; knee_step=84 has
knee_dose≈100 (results/exp16_lm_head_knee_info.pt), so step ≈ dose*100.
K_STEPS = [10,20,40,80,90,100,200,400,800,1000] = dose 0.1..10.

Artifact: results/exp38h_two_channel_goodhart.json
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from prism.exp38_hi_common import (  # noqa: E402
    D, K_STEPS, load_snapshots, load_u128, top_r_subspaces,
    load_model_tensors,
)
from prism.holonomy import procrustes_align  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(REPO, "results", "exp38h_two_channel_goodhart.json")
R = 16
# pre-registered triples for the associativity defect (dose brackets)
TRIPLES = [(10, 100, 1000), (10, 80, 100), (100, 400, 1000)]


def main():
    t0 = time.time()
    U128, ck = load_u128()
    snaps_all, raw = load_snapshots()
    missing = [k for k in K_STEPS if k not in snaps_all]
    assert not missing, f"missing snapshot steps: {missing}"
    ks = list(K_STEPS)
    W_orig = raw["W_orig"].to(torch.float64)  # noqa: F841 (schema check only)
    # P0 fix (E38_fixes): snapshots[k] IS ΔW — do NOT subtract W_orig again
    dW = {k: snaps_all[k] for k in ks}

    # --- fixed reference projector: top-16 frame of ΔW at the smallest dose
    # (s=10). DEVIATION from TZ (documented): the TZ's two-sided projector
    # U_t ∈ R^{K×16} is taken from the cached J23, which lives on U128
    # (128 tokens); ΔW lives on the E-16 256-token set (13 shared tokens,
    # rank ≤ 13 — a 16-dim projector is undefined there). The leading
    # frame of the first edit is the honest in-space analog.
    A_ref = dW[ks[0]]
    Us, sU, Vt = torch.linalg.svd(A_ref, full_matrices=False)
    U_t = Us[:, :R]            # [256,16]
    V_t = Vt[:R].t()           # [16,1024]
    P_L = U_t @ U_t.t()        # [256,256]
    P_R = V_t @ V_t.t()        # [1024,1024]

    # --- per-state channel split + frames ---
    per = {}
    for k in ks:
        A = dW[k]
        A_top = P_L @ A @ P_R
        A_tail = A - A_top
        Qt, st = top_r_subspaces(A_top, R)          # right frame (top ch)
        Qta, sta = top_r_subspaces(A_tail, R)       # right frame (tail ch)
        Ut, _ = top_r_subspaces(A_top.t(), R)       # left frame  (top ch)
        Uta, _ = top_r_subspaces(A_tail.t(), R)     # left frame  (tail ch)
        e_top = float((A_top.pow(2).sum() / A.pow(2).sum()))
        s_full = torch.linalg.svdvals(A)
        per[k] = {
            "energy_top": e_top,
            "sigma_top16_top_channel": st.tolist(),
            "sigma_top16_tail_channel": sta.tolist(),
            "eigengap_16_17_dW": float(s_full[15] - s_full[16]),
            "_Q_top": Qt, "_Q_tail": Qta, "_U_top": Ut, "_U_tail": Uta,
        }

    # --- non-commutativity: Procrustes associativity defect ---
    def defect(framedict, a, b, c, side):
        R12, _ = procrustes_align(framedict[b], framedict[a])
        R23, _ = procrustes_align(framedict[c], framedict[b])
        R13, _ = procrustes_align(framedict[c], framedict[a])
        prod = R12 @ R23
        return float(((prod - R13).norm() / R))

    defects = {}
    for side in ("top", "tail"):
        framed = {k: per[k][f"_Q_{side}"] for k in ks}
        framedU = {k: per[k][f"_U_{side}"] for k in ks}
        d_right, d_left = {}, {}
        for (a, b, c) in TRIPLES:
            d_right[f"{a},{b},{c}"] = defect(framed, a, b, c, side)
            d_left[f"{a},{b},{c}"] = defect(framedU, a, b, c, side)
        defects[side] = {"right_hidden": d_right, "left_token": d_left}

    # --- per-channel J-response (exact: ΔJ = ΔW @ J_norm, J_norm invariant) ---
    # channel split of ΔJ: ΔJ_top = ΔW_top @ J_norm — measure energy share
    h0 = ck["H"][("wiki", 0, 60)].double()
    r0 = h0.norm()
    u0 = h0 / r0
    emb, gamma = load_model_tensors()
    Jnorm = (D ** 0.5 / r0) * (torch.eye(D, dtype=torch.float64)
                               - torch.outer(u0, u0)) * gamma[:, None]
    jresp = {}
    for k in ks:
        Jt = dW[k] @ Jnorm
        Jtt = (P_L @ dW[k] @ P_R) @ Jnorm
        Jtail = Jt - Jtt
        jresp[k] = {"energy_top": float((Jtt.pow(2).sum() / Jt.pow(2).sum()))}

    out = {
        "experiment": "E-38h",
        "tz": "docs/E38.md (E-38h) — Two-channel Goodhart decomposition",
        "data": {
            "snapshots": "results/exp16_deltas/exp16_dW_snapshots.pt",
            "grid": "E-16 steps 10..1000 (step≈dose*100; knee_step=84)",
            "subsample": ks,
            "schema": "snapshots[k] = ΔW = (W_curr − W_orig)[token_set]; "
                      "dW = snapshots[k] (E38_fixes P0: previous run "
                      "subtracted W_orig twice -> results of 2026-10-24 "
                      "invalid)",
            "tied_weights": "Qwen3.5-0.8B tie_word_embeddings=True: the "
                            "'readout-only' E-16 edit also changes "
                            "embed_tokens, so in the REAL model h_t changes "
                            "with the edit; ΔJ = ΔW @ J_norm(cached h) is a "
                            "linear-response approximation, not an identity "
                            "(E38_fixes P1)",
            "projector": "two-sided top-16 FIXED at ΔW(s=10) (smallest dose). "
                         "DEVIATION from TZ: J23 projector (U128, 128 tokens) "
                         "cannot act on ΔW (E-16 256-token set; 13 shared) — "
                         "rank ≤ 13 makes a 16-dim U128 projector undefined",
            "R": R,
        },
        "per_state": {k: {kk: vv for kk, vv in per[k].items()
                          if not kk.startswith("_")} for k in ks},
        "non_commutativity": {
            "definition": "‖R(s_a,s_b) R(s_b,s_c) − R(s_a,s_c)‖_F / r, "
                          "R = procrustes_align (top-16 frame)",
            "triples": [list(x) for x in TRIPLES],
            "values": defects,
            "verdict_rule": "pre-registered: defect_top >> defect_tail => "
                            "non-Abelian in top-16 (E-38d scenario 2); "
                            "equal => both channels Abelian",
        },
        "j_response_channel_split": jresp,
        "blocked": {
            "bridge_per_channel": "1 − jacobian_gap(J_L, J_S, E_L, E_S, p) "
                                  "needs J_L of the large target model "
                                  "(not on this machine) — honestly not run",
            "behavior_top_p_retention": "needs E-16 eval runs under the "
                                        "per-channel edits — not run",
        },
        "runtime_s": round(time.time() - t0, 1),
        "deterministic": True,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"OK -> {OUT}")
    for k in ks:
        print(f"  s={k:>5}: energy_top(ΔW)={per[k]['energy_top']:.4f} "
              f"energy_top(ΔJ)={jresp[k]['energy_top']:.4f}")
    for side in ("top", "tail"):
        print(f"  defect {side}:", {kk: round(vv, 5)
              for kk, vv in defects[side]["right_hidden"].items()})


if __name__ == "__main__":
    main()
