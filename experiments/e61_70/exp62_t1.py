import json, numpy as np, sys
sys.path.insert(0,'.')
from exp62_batt_lib import run
z=np.load('../results/exp47b_gen_kinematics.npz',allow_pickle=True)
docs=json.load(open('../results/exp47b_gen_kinematics.json'))['docs']
traj={}; dom={}
for d in docs[:8]:
    nm=f"{d['dom']}_{d['si']}"
    traj[nm]=(z[nm+"_u"], z[nm].item()["hnorm"]); dom[nm]=d['dom']
r=run(traj, dom)
print("t1 (первые 8 доков Qwen L23):")
print(json.dumps({k:r[k] for k in ('F1','n_osc','n','clip','n_strict','phase')}, ensure_ascii=False))
