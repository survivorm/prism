"""E-39-дополнение, раунд 5: область применимости «усечённого p» (0 новых форвардов).

Отвечает на ревью `docs/E39+.md` §4 («sentinel 50.7 % — главное», «g_uni — uniform-доля под
усечённым `p`»). На кэше exp27b проверяется: (1) provenance `P128_RAW` — точная ли это
перенормировка `P128` того же файла; (2) распределение массы `U128` под полным softmax и по
доменам; (3) чувствительность `O1`, `O1/iso`, `Σp²`, `g_uni` к отказу от ренормировки `p`;
(4) то же для `frac_K` (E-37) и его референса `(I−P)`. Якоря `docs/E39_supplement.md` §2.1
проверяются fail-loud. Запуск из `experiments/`: `../.venv/bin/python exp39_supp_pscale_probe.py`.

Выход: `results/exp39_supp_sentinel_probe.json`, `results/exp39_supp_pscale_probe.json`.
"""
import json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) or ".")
from _e38_cache import DOMAINS, N_PER_DOMAIN, S, K, D, load_layer

RAW = "/data/prism_kernels/exp27b/results"                       # копия кэша с P128_RAW
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LAYERS = (6, 12, 18, 22, 23)
ANCH = {6: dict(o1=3.524e-4, g_uni=0.2150), 12: dict(o1=3.255e-4, g_uni=0.1882),
        18: dict(o1=2.057e-4, g_uni=0.1425), 22: dict(o1=1.719e-4, g_uni=0.1258),
        23: dict(o1=4.725e-4, g_uni=0.1403)}
E37_FRAC = {6: 0.996558, 12: 0.996758, 18: 0.998127, 22: 0.998288, 23: 0.993934}


def _stack(ck, field):
    return torch.stack([ck[field][(dm, si, t)].clone() for dm in DOMAINS
                        for si in range(N_PER_DOMAIN) for t in range(S)]).reshape(
        len(DOMAINS) * N_PER_DOMAIN, S, -1)


def sentinel_probe(L):
    """Масса U128, provenance P128_RAW, O1/O1-iso/Σp²/g_uni на ренормированном и сыром p."""
    J = load_layer(L, want=("J",))["J"]
    ck = torch.load(os.path.join(RAW, f"exp27b_jcache_L{L}.pt"), map_location="cpu", weights_only=False)
    Pr, P = _stack(ck, "P128_RAW").float(), _stack(ck, "P128").float()
    s = Pr.sum(-1).double()
    col = float(((Pr.double() - s.double().unsqueeze(-1) * P.double()).abs().max(-1).values
                 / Pr.double().max(-1).values).max())
    assert col <= 1e-4, f"L{L}: P128_RAW не является перенормировкой P128 (rel {col:.3g})"
    di = torch.tensor([DOMAINS.index(x) for x in np.repeat(DOMAINS, N_PER_DOMAIN)])[:, None].expand(-1, S)
    rec = {"collinearity_rel": col, "raw_mass_mean": float(s.mean()), "raw_mass_median": float(s.median()),
           "raw_mass_p10": float(s.quantile(.1)), "raw_mass_p90": float(s.quantile(.9)),
           "frac_below_half": float((s < .5).double().mean()),
           "raw_mass_by_domain": {dm: float(s[di == k].mean()) for k, dm in enumerate(DOMAINS)},
           "frac_below_half_by_domain": {dm: float((s[di == k] < .5).double().mean()) for k, dm in enumerate(DOMAINS)},
           "sump2_norm": float(P.double().pow(2).sum(-1).mean()),
           "sump2_raw": float(Pr.double().pow(2).sum(-1).mean())}
    okm = s >= 0.5
    rec["sump2_norm_alive"] = float(P.double().pow(2).sum(-1)[okm].mean())
    rec["sump2_raw_alive"] = float(Pr.double().pow(2).sum(-1)[okm].mean())
    Jd_all = None
    o1 = {"n": 0.0, "r": 0.0}; iso = {"n": 0.0, "r": 0.0}; gu = {"n": 0.0, "r": 0.0}
    den = pw = pr = uni_sum = 0.0
    for d in range(J.shape[0]):
        for c0 in range(0, S, 32):
            js = slice(c0, c0 + 32)
            Jc = J[d][js].double(); Pn, Pw = P[d][js].double(), Pr[d][js].double()
            raw = Jc.pow(2).sum(dim=(1, 2)); uni_sum += float(Jc.sum(1).pow(2).sum()); den += float(raw.sum())
            pw += float(Pn.pow(2).sum()); pr += float(Pw.pow(2).sum())
            for tag, P_ in (("n", Pn), ("r", Pw)):
                m = torch.einsum("sk,skd->sd", P_, Jc)
                o1[tag] += float((P_.unsqueeze(-1) * (Jc - m.unsqueeze(1))).pow(2).sum((1, 2)).sum())
                s2, s3 = P_.pow(2).sum(-1), P_.pow(3).sum(-1)
                iso[tag] += float(((s2 - 2 * s3 + s2.pow(2)) / K).sum())
                gu[tag] += float(torch.einsum("sk,skd->sd", P_, Jc).pow(2).sum())
            del Jc
    npos = J.shape[0] * S
    rec.update({"O1_norm": o1["n"] / den, "O1_raw": o1["r"] / den,
                "iso_norm": iso["n"] / npos, "iso_raw": iso["r"] / npos,
                "O1_rel_norm": (o1["n"] / den) / (iso["n"] / npos),
                "O1_rel_raw": (o1["r"] / den) / (iso["r"] / npos),
                "g_uni_corpus_p_independent": uni_sum / (K * den),
                "g_uni_w_norm": gu["n"] / (pw * den), "g_uni_w_raw": gu["r"] / (pr * den)})
    rec["g_uni_w_rel_change"] = abs(rec["g_uni_w_raw"] - rec["g_uni_w_norm"]) / rec["g_uni_w_norm"]
    assert abs(rec["O1_norm"] / ANCH[L]["o1"] - 1) < 0.01, f"L{L}: O1 поехал: {rec['O1_norm']}"
    assert abs(rec["g_uni_corpus_p_independent"] / ANCH[L]["g_uni"] - 1) < 0.01, f"L{L}: g_uni поехал"
    del J, ck, Pr, P
    return rec


def frac_probe(L):
    """frac_K (E-37) и его референс (I−P): то же, на ренормированном и на сыром p."""
    J = load_layer(L, want=("J",))["J"]
    ck = torch.load(os.path.join(RAW, f"exp27b_jcache_L{L}.pt"), map_location="cpu", weights_only=False)
    P, Pr = _stack(ck, "P128").float(), _stack(ck, "P128_RAW").float()
    al = (~((Pr.sum(-1) < 0.5))).reshape(-1)
    rec = {}
    for tag, Pt in (("norm", P), ("raw", Pr)):
        fr, ir = [], []
        for d in range(32):
            for c0 in range(0, S, 32):
                js = slice(c0, c0 + 32)
                Jc = J[d][js].double(); Pc = Pt[d][js].double()
                m = torch.einsum("sk,skd->sd", Pc, Jc)
                PJ = Pc.unsqueeze(-1) * (Jc - m.unsqueeze(1))
                fr.append(((Jc - PJ).pow(2).sum((1, 2)) / Jc.pow(2).sum((1, 2))))
                s2, s3 = Pc.pow(2).sum(-1), Pc.pow(3).sum(-1)
                ir.append((K - 2 * (1 - s2) + (s2 - 2 * s3 + s2.pow(2))) / K)
                del Jc
        fr, ir = torch.cat(fr), torch.cat(ir)
        rec[f"{tag}_frac_mean"] = float(fr.mean()); rec[f"{tag}_frac_median"] = float(fr.median())
        rec[f"{tag}_ipref_mean"] = float(ir.mean()); rec[f"{tag}_ipref_median"] = float(ir.median())
        rec[f"{tag}_ratio_mean"] = float(fr.mean() / ir.mean())
        rec[f"{tag}_ratio_median"] = float(fr.median() / ir.median())
        rec[f"{tag}_ratio_mean_alive"] = float(fr[al].mean() / ir[al].mean())
        del fr, ir
    assert abs(rec["norm_frac_median"] / E37_FRAC[L] - 1) < 2e-3, f"L{L}: frac_K поехал"
    del J, ck, P, Pr
    return rec


if __name__ == "__main__":
    t0 = time.time()
    a = {}
    for L in LAYERS:
        a[L] = sentinel_probe(L)
        print(L, json.dumps({k: (round(v, 8) if isinstance(v, float) else
                                 {x: round(y, 4) for x, y in v.items()} if isinstance(v, dict) else v)
                             for k, v in a[L].items()}, ensure_ascii=False), flush=True)
    json.dump(a, open(os.path.join(OUT, "exp39_supp_sentinel_probe.json"), "w"), indent=1, ensure_ascii=False)
    b = {}
    for L in LAYERS:
        b[L] = frac_probe(L)
        print("frac", L, json.dumps({k: round(v, 6) for k, v in b[L].items()}), flush=True)
    json.dump(b, open(os.path.join(OUT, "exp39_supp_pscale_probe.json"), "w"), indent=1)
    print("wall", round(time.time() - t0, 1))
