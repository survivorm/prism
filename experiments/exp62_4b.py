import json, sys, numpy as np
sys.path.insert(0,'.')
from exp62_batt_lib import run
z=np.load('../results/e52_4b/exp47b_gen_kinematics.npz',allow_pickle=True)
docs=json.load(open('../results/e52_4b/exp47b_gen_kinematics.json'))['docs']
traj={}; dom={}
for d in docs:
    nm=f"{d['dom']}_{d['si']}"
    it=z[nm].item() if z[nm].shape==() else None
    traj[nm]=(z[nm+"_u"], it["hnorm"]); dom[nm]=d['dom']
r=run(traj, dom)
print("E-62 4B (16 доков, n=4/домен):")
print(json.dumps({k:r[k] for k in ('F1','n_osc','n','clip','n_strict','phase')}, ensure_ascii=False))
