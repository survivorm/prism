#!/usr/bin/env python3
"""at0_1_tpu_dev_recompute.py — post-hoc детерминированный пересчёт для r6-cosmetic
аннотации AT0_1 (docs/AT0_1_review.md, пункты 3–6). НЕ экспериментальный прогон:
никакие числа прогона не пересчитываются; добавляются только
  (1) max_rel_dev реконструкции U·hn vs h (гейт G0a, review #5),
  (2) верификация gram-тождества sum_rho_full = (r−1)/2 для notes T_none_full (review #3),
  (3) self-audit: точное воспроизведение уже отчётных G0a evidence (own_w0) —
      если не совпадут bit-eq, пересчёт невалиден и правки не применять.
Входы зафиксированы sha256. Формулы зеркалят AT0_1_accounting.py:
  load_source(kind='state') строки 185–189 (h_raw → hn=‖h‖ → u=h/max(hn,1e-12)),
  cell-loop строки 275–276 (U=u/max(‖u‖,1e-12), h_rec=U·hn).
Запуск: .venv/bin/python experiments/at0_1_tpu_dev_recompute.py
Выход: один JSON в stdout.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWN = os.path.join(ROOT, "results/e64/tpu_v8/P1_own.npz")
COR = os.path.join(ROOT, "results/e64/tpu_v8/P2_corpus.npz")
STORE = os.path.join(ROOT, "results/AT0/store/23281c64cc418ff1.json")
CLAIM = os.path.join(ROOT, "results/AT0/T0_ACCOUNTING.json")
LAYERS = [1, 5, 11, 17, 23]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tpu(path, prefix):
    """Зеркало load_source(kind='state', AT0_1_accounting.py:180-197) + h_raw для dev."""
    z = np.load(path, allow_pickle=True)
    docs = {}
    for i in range(16):
        n = f"{prefix}{i}"
        d = {}
        for L in LAYERS:
            h = np.asarray(z[f"{n}_L{L}"], dtype=np.float64)
            hn = np.linalg.norm(h, axis=1)
            u = h / np.maximum(hn, 1e-12)[:, None]
            d[L] = (u, hn, h)
        docs[n] = d
    return docs


def main():
    out = {
        "sha": {
            "results/e64/tpu_v8/P1_own.npz": sha256_file(OWN),
            "results/e64/tpu_v8/P2_corpus.npz": sha256_file(COR),
            "results/AT0/store/23281c64cc418ff1.json": sha256_file(STORE),
        }
    }

    # ---- (1) self-audit: G0a evidence_own_w0 (из claim, отчётные значения) ----
    claim = json.load(open(CLAIM, encoding="utf-8"))
    g0a = claim["gates"]["G0a_tpu_hn_field"]
    z_own = np.load(OWN, allow_pickle=True)
    _hn0 = np.asarray(z_own["own_w0_hn"], dtype=np.float64)
    _col = np.linalg.norm(np.asarray(z_own["own_w0_L23"], dtype=np.float64)[:, :len(_hn0)], axis=0)
    own_w0_row_norms = np.linalg.norm(np.asarray(z_own["own_w0_L23"], dtype=np.float64), axis=1)
    recomputed = {
        "row_norm_median": float(np.median(own_w0_row_norms)),
        "hn_median": float(np.median(_hn0)),
        "hn_vs_column_norms_max_rel": float(np.max(np.abs(_hn0 / _col - 1.0))),
        "row_norm_vs_hn_corr": float(np.corrcoef(own_w0_row_norms, _hn0)[0, 1]),
    }
    out["self_audit_g0a_own_w0"] = {
        "recomputed": recomputed,
        "claimed": g0a["evidence_own_w0"],
        "bit_eq": all(recomputed[k] == g0a["evidence_own_w0"][k] for k in recomputed),
    }

    # ---- (2) tpu_dev: max_rel_dev U·hn vs h ----
    overall = (0.0, None, None)
    by_src = {}
    by_layer = {}
    for src, path, prefix in (("P1_own", OWN, "own_w"), ("P2_corpus", COR, "cor_w")):
        for n, d in sorted(load_tpu(path, prefix).items()):
            for L in LAYERS:
                u, hn, h = d[L]
                U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
                h_rec = U * hn[:, None]
                rel = np.linalg.norm(h_rec - h, axis=1) / np.maximum(np.linalg.norm(h, axis=1), 1e-300)
                i = int(np.argmax(rel))
                if float(rel[i]) > overall[0]:
                    overall = (float(rel[i]), n, L)
                by_src.setdefault(src, []).append(float(rel.max()))
                by_layer.setdefault(f"L{L}", []).append(float(rel.max()))
    out["tpu_dev"] = {
        "definition": ("max по TPU-докам (own_w0..15, cor_w0..15) × L ∈ {1,5,11,17,23} × row t "
                       "‖U·hn − h_raw‖₂/‖h_raw‖₂; hn=‖h_raw‖, u=h_raw/max(hn,1e-12), "
                       "U=u/max(‖u‖,1e-12) (load_source + cell-loop AT0_1_accounting.py:185-189, 275-276)"),
        "max_rel_dev": overall[0],
        "worst": {"doc": overall[1], "L": overall[2]},
        "max_by_source": {k: max(v) for k, v in sorted(by_src.items())},
        "max_by_layer": {k: max(v) for k, v in sorted(by_layer.items())},
    }

    # ---- (3) gram-тождество + T_none_full (v_tangent, L23, median-клетки, пул nondegen) ----
    st = json.load(open(STORE, encoding="utf-8"))
    recs = []
    max_gram_dev = 0.0
    n_checked = 0
    for nm in sorted(st["per_doc"]):
        d = st["per_doc"][nm]
        if d.get("degen_tz"):
            continue
        c = d["layers"]["23"]["v_tangent"]["median"]
        if c.get("sum_rho_full") is None or c.get("f_dc") is None or c.get("N") is None:
            continue
        n_checked += 1
        max_gram_dev = max(max_gram_dev, abs(c["sum_rho_full"] - (c["r"] - 1) / 2))
        recs.append(c["sum_rho_full"] - (c["N"] - 1) * c["f_dc"])
    claim_tnf = claim["aggregates"]["t0_headline"]["v_tangent"]["excess_median"]["T_none_full"]
    out["gram_identity_check"] = {
        "n_docs_nondegen_vtangent_L23": n_checked,
        "max_abs_sum_rho_full_minus_(r-1)/2": float(max_gram_dev),
        "T_none_full_median_recomputed": float(np.median(recs)),
        "T_none_full_median_claim": claim_tnf,
        "median_eq": float(np.median(recs)) == claim_tnf,
    }

    json.dump(out, sys.stdout, ensure_ascii=False, indent=1)
    print()


if __name__ == "__main__":
    main()
