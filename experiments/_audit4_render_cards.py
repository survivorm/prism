#!/usr/bin/env python
"""Render ref-exp markdown tables from rerun artifacts (audit v4)."""
import json
import statistics as st


def e11a(path="results/exp11a_defrag_fidelity.json"):
    d = json.load(open(path))
    print("params:", d["params"])
    print("\n### E-11a: KL_true (token_set-conditional) по (pert, scale), среднее по слоям/сэмплам")
    print("| pert | scale | raw | gf | dr4 | dr8 | dr16 | dr32 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    agg = {}
    for L, rows in d.items():
        if L == "params":
            continue
        for r in rows:
            key = (r["perturbation"], r["scale"])
            agg.setdefault(key, []).append(r)
    for (p, s), rs in agg.items():
        m = lambda k: st.median(e[k] for e in rs)  # noqa: E731
        print(f"| {p} | {s} | {m('KL_true_raw'):.4f} | {m('KL_true_gf'):.4f} | "
              f"{m('KL_true_dr4'):.4f} | {m('KL_true_dr8'):.4f} | {m('KL_true_dr16'):.4f} | "
              f"{m('KL_true_dr32'):.4f} |")
    # per-layer dr-convergence for sem/random: KL(dr_r) within 1% of KL(dr32)?
    print("\n### E-11a: r-сходимость (median KL_true_dr{r}/KL_true_dr32 по живым направлениям, sem+rand, s>=1)")
    for L, rows in d.items():
        if L == "params":
            continue
        rs = [r for r in rows if r["perturbation"].startswith(("sem", "rand")) and r["scale"] >= 1.0]
        if not rs:
            continue
        line = []
        for r_ in (4, 8, 16):
            ratios = [r[f"KL_true_dr{r_}"] / max(r["KL_true_dr32"], 1e-12) for r in rs
                      if r["KL_true_dr32"] > 1e-4]
            line.append(f"dr{r_}/{r_ and ''}ratio_med={st.median(ratios):.2f}")
        print(f"L{L} (n={len(rs)}): " + ", ".join(line))


def e12b(path="results/exp12b_quantization_map.json"):
    d = json.load(open(path))
    print("\nparams:", d["params"])
    print("\n### E-12b: по слоям (avg по сэмплам)")
    print("| L | KL(map) | KL(unif) | ppl_orig | ppl_map | ppl_unif | top1_map | top1_unif | dh_map | dh_unif |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for L in ("6", "12", "18"):
        a = d.get(L, {}).get("_avg", {})
        if not a:
            continue
        rows = [e for si, v in d.get(L, {}).items() if si != "_avg" for e in v]
        med = lambda k: st.median(e[k] for e in rows)  # noqa: E731
        print(f"| {L} | {a['kl_map']:.4f} | {a['kl_uniform']:.4f} | {med('ppl_orig'):.3g} | "
              f"{a['ppl_map']:.3g} | {a['ppl_uniform']:.3g} | "
              f"{sum(e['top1_map'] for e in rows)/len(rows):.2f} | "
              f"{sum(e['top1_uniform'] for e in rows)/len(rows):.2f} | "
              f"{med('delta_h_map'):.3f} | {med('delta_h_uniform'):.3f} |")
    print("\n### clamp-диагностика (доля направлений с |c|>max|x|; avg по всем записям)")
    tot = []
    for L in ("6", "12", "18"):
        for si, rows in d.get(L, {}).items():
            if si == "_avg":
                continue
            for r in rows:
                cf = r.get("clamp_diag", {}).get("clamp_frac", [])
                if cf:
                    tot.append(sum(cf) / len(cf))
    if tot:
        print(f"доля заклимпленных направлений: mean {st.mean(tot):.2f}, "
              f"max {max(tot):.2f} (n={len(tot)})")


def e12a(path="results/exp12a_perturbation_test.json"):
    d = json.load(open(path))
    print("\nparams:", d.get("params"))
    print("\n### E-12a: (avg по 8 сэмплиым; KL — полные V, якорь = истинный выход)")
    print("| eps | L | KL_g± | KL_d± | ratio_dg | ratio_lin | unif_g | sig_g/sig_d |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for ek in ("eps_0.01", "eps_0.1", "eps_0.5", "eps_1.0"):
        blk = d.get(ek, {})
        for L, rows in blk.items():
            g = [r["KL_g_plus"] for r in rows]
            dd = [r["KL_d_plus"] for r in rows]
            rl = [r["ratio_lin"] for r in rows]
            rt = st.median(dd) / max(st.median(g), 1e-12)
            unif = [r["uniformity_vg"] for r in rows]
            sg = st.median([r.get("sigma_g", 0) / max(r.get("sigma_d", 1), 1e-12) for r in rows])
            print(f"| {ek[4:]} | {L} | {st.mean(g):.4f}±{st.pstdev(g):.4f} | "
                  f"{st.mean(dd):.4f}±{st.pstdev(dd):.4f} | {rt:.2f}x | "
                  f"{st.geometric_mean(rl):.1f}x | {min(unif):.2f}–{max(unif):.2f} | {sg:.0f}x |")


if __name__ == "__main__":
    e11a()
    e12b()
    e12a()
