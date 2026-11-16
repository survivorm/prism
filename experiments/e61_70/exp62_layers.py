#!/usr/bin/env python
"""E-62 слой-батарея: факты (F1 двухвременка, osc/exp состав, clip, strict, фазы, Δ(1))
на каждом из L{1,5,11,17,23} ОДНИХ И ТЕХ ЖЕ 64 траекторий exp47b (траектория
побайтово воспроизведена, гейт sha+cos). Ответ: форма — свойство траектории или L23.
L23 здесь = контрольная точка (должна дать базлайн E-61: F1 0.913, osc 45/64)."""
import json, sys
import numpy as np

sys.path.insert(0, ".")
from exp62_batt_lib import run  # noqa: E402

NPZ = "/data/prism_kernels/exp62/gen_layers_qwen.npz"
LAYERS = [1, 5, 11, 17, 23]

z = np.load(NPZ, allow_pickle=True)
meta = json.load(open(NPZ.replace(".npz", "_meta.json")))
assert all(v["sha_ok"] for v in meta.values()), "есть несовпавшие траектории"
names = sorted(meta)
dom = {n: meta[n]["dom"] for n in names}
res = {}
for L in LAYERS:
    traj = {n: (z[f"{n}_L{L}_u"], z[f"{n}_L{L}_hn"]) for n in names}
    r = run(traj, dom, verbose=False)
    res[f"L{L}"] = {k: r[k] for k in ("F1", "n_osc", "n", "clip", "n_strict",
                                      "phase", "dip1")}
    print(f"L{L:>2}: F1={r['F1']} osc={r['n_osc']}/{r['n']} clip={r['clip']} "
          f"strict={r['n_strict']} lag1={r['dip1']}", flush=True)
    json.dump(res, open("../results/e62/layers_qwen.json", "w"), ensure_ascii=False)
print("фазы:")
for L in LAYERS:
    print(f"  L{L}: {res[f'L{L}']['phase']}")
