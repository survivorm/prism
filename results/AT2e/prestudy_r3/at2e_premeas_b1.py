import sys; sys.modules["fla"] = None
import os
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["MKL_NUM_THREADS"] = "4"; os.environ["OPENBLAS_NUM_THREADS"] = "4"
import torch, numpy as np, time, json
from transformers import AutoModelForCausalLM
torch.set_num_threads(4)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained("/data/Qwen/Qwen3.5-0.8B", torch_dtype=torch.float32,
                                             attn_implementation="eager")
model.eval()
pool = torch.load("/data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt", map_location="cpu",
                  weights_only=False)
toks = np.asarray(pool["ids"]["wiki"][0], dtype=np.int64)
input_ids = torch.as_tensor(toks[None])
mask = torch.ones((1, 128), dtype=torch.long)
T_POS, H = 32, 8
SEED = 0
print(f"load {time.time()-t0:.1f}s", flush=True)

def patched(L_IDX, d):
    hh = model.model.layers[L_IDX].register_forward_hook(
        lambda m, i, o, d=d: (lambda oo: (oo.__setitem__((slice(None), T_POS, slice(None)),
            oo[:, T_POS, :] + d), oo)[1])(o.clone()))
    try:
        z = model(input_ids=input_ids, attention_mask=mask).logits[0]
    finally:
        hh.remove()
    return z

def q_of(dz, p):
    # dz [V] f64 (total change (z+ - z-)/2), p [V] f64 softmax baseline
    return float((p * dz * dz).sum() - (p * dz).sum() ** 2)

out = {}
for cell_id, L_IDX in ((0, 11), (1, 12)):
    with torch.no_grad():
        base = model(input_ids=input_ids, attention_mask=mask, output_hidden_states=True)
    z0 = base.logits[0]                       # [128, V]
    hL = base.hidden_states[L_IDX + 1][0]     # [128, 1024] (hidden_states[0]=embed)
    h_t = hL[T_POS].double()
    hnorm = float(h_t.norm())
    # B0: 8 top-PCA + 8 bottom-PCA (centered hidden, 128x1024, f64) + 16 gQR (per-cell seed)
    Hc = hL.double() - hL.double().mean(0, keepdim=True)
    vh = np.linalg.svd(Hc.numpy(), full_matrices=False)[2]   # [1024, 128]
    top8 = vh[:8]; bot8 = vh[119:127]
    # gQR (r3, N2 fix): A (16,1024) Gaussian; directions = COLUMNS of QR(A^T) (1024x16):
    # unit + orthonormal, Stiefel-Haar. (Rows of R from wide reduced QR — NOT unit/orthonormal,
    # verified numerically: norms ~30-33, cos ~0.07 — excluded.)
    A = np.random.RandomState(SEED + cell_id).standard_normal((16, 1024))
    q16 = np.linalg.qr(A.T)[0]                     # (1024, 16), columns = directions
    gqr = q16.T                                    # (16, 1024), rows = unit orthonormal directions
    gqr_check = float(np.abs(gqr @ gqr.T - np.eye(16)).max())
    gqr_norm_check = float(np.abs(np.linalg.norm(gqr, axis=1) - 1.0).max())
    b0 = np.vstack([top8, bot8, gqr]).astype(np.float32)
    b0 = b0 / np.linalg.norm(b0, axis=1, keepdims=True)
    pt = torch.softmax(z0[T_POS], dim=0).double().numpy()
    ps = {k: torch.softmax(z0[T_POS + k], dim=0).double().numpy() for k in range(1, H + 1)}
    pmax_t = float(pt.max())
    # a_small = 3e-4 * hnorm ; a_floor = 1e-9 * hnorm
    a_small = 3e-4 * hnorm
    a_floor = 1e-9 * hnorm
    qts, qfuts = [], []
    for j in range(32):
        v = torch.as_tensor(b0[j])
        with torch.no_grad():
            zp = patched(L_IDX, a_small * v)
            zm = patched(L_IDX, -a_small * v)
        rows_p = zp[T_POS:T_POS + H + 1].double().numpy()
        rows_m = zm[T_POS:T_POS + H + 1].double().numpy()
        q_t = q_of((rows_p[0] - rows_m[0]) / 2.0, pt)
        qf = [q_of((rows_p[k] - rows_m[k]) / 2.0, ps[k]) for k in range(1, H + 1)]
        qts.append(q_t); qfuts.append(float(np.mean(qf)))
    # floor
    v0 = torch.as_tensor(b0[0])
    with torch.no_grad():
        zf_p = patched(L_IDX, a_floor * v0)
        zf_m = patched(L_IDX, -a_floor * v0)
    q_floor_meas = q_of((zf_p[T_POS].double().numpy() - zf_m[T_POS].double().numpy()) / 2.0, pt)
    zmax_t = float(z0[T_POS].double().abs().max())
    q_quant = (1.1920929e-7 * zmax_t) ** 2 * pmax_t
    floor_q = max(q_floor_meas, q_quant)
    qts_s = sorted(qts)
    def pct(a, p):
        a = np.sort(np.asarray(a, dtype=float)); k = (len(a) - 1) * p / 100.0
        f, c = int(np.floor(k)), min(int(np.ceil(k)), len(a) - 1)
        return float(a[f] + (a[c] - a[f]) * (k - f))
    out[f"L{L_IDX}"] = {
        "hnorm_t32": hnorm, "a_small": a_small, "zmax_t": zmax_t, "pmax_t": pmax_t,
        "gqr_orth_max_dev": gqr_check, "gqr_norm_max_dev": gqr_norm_check,
        "q_t": {"min": min(qts), "p25": pct(qts, 25), "median": pct(qts, 50),
                "p75": pct(qts, 75), "max": max(qts), "sorted": qts_s},
        "qbar_fut": {"min": min(qfuts), "median": pct(qfuts, 50), "max": max(qfuts)},
        "floor": {"measured": q_floor_meas, "quant": q_quant, "used": floor_q},
        "median_qt_over_floor": pct(qts, 50) / floor_q,
        "frac_above_10floor": float(np.mean([q > 10 * floor_q for q in qts])),
    }
print(json.dumps(out, indent=1), flush=True)
print("DONE_PREMEAS_B1")
