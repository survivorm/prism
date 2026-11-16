#!/usr/bin/env python3
"""AT2b C2/C1 diagnostic — disk-only, БЕЗ форварда (всё из кэшей exp27b).

Контекст: docs/AT2b_review.md (2-й раунд, run-ревью) — C2 (обучающая цель LOO
не центрирована: yf = сырые з-логиты, intercept бьёт в global mean; в pc попадает
пер-док константный сдвиг c(d) = ȳf − ȳ(d)) и C1 (k=0-контроль сравнивает
вращающийся per-(l,d,t) J-базис с фиксированными через общий ridge).

Что делает (всё disk-only):
  1. Восстановление z_U128 = (H23 @ E.T)[:, :, U128] (те же операции, что в AT2b.py).
     V1: S3-ориентация совпадает с прогоном (hi 0.381/0.490, lo 0.014/0.017).
  2. Подпространства: точная копия стадии AT2b.py (rng_base(0), eigh, d_eff-
     rematching, VJ/VJdc/Vpca/HAAR/UNIF + _dc). V4: r_final == 7.
  3. Ridge LOO k=0..8, ДВА варианта (V2: RAW-вариант воспроизводит per_pos_cos
     артефакта max|diff| < 1e-6, V3: RAW-контрольный CI = CI артефакта):
       RAW — как в AT2b.py (yf = сырые логиты, pc = pred − ȳ(d));
       CD  — yf = per-doc центрированные (по пре-регу §2 y_c = y − ȳ),
             pc = pred (шкала centered-цели, DC = 0).
  4. Контроль (k=0) в обоих вариантах: CI top−weak per слой (та же bootstrap-
     конвенция: rngb(0), B=1000, stat = mean по (док,t)), control_ok (односторонний,
     lower > 0, AND over L).
  5. C1: (a) Eckart-Young: sum(lam_top) vs sum(lam_weak) per (l,d,t) (tautology-check:
     если top <= weak — баг split/eigh); (b) локальный per-(l,t) ridge (32 дока,
     центрирование по докаМ, без пулинга по t); (c) стационарный per-layer
     агрегированный J-базис (J_agg = mean_{d,t} J_{d,t}) + полный пул-ridge
     (CD-метрика) — кандидат контроля T2.1c.
  6. C3: per (L,arm) k=0: R2_mean и cos_median (RAW/CD) + weighted cos (веса
     ||yc_t||^2) по RAW-артефакту.

Выход: results/AT2b_diag/AT2b_diag.json + AT2b_diag_run.log; trackio
at2b_diag_c2c1_r1. Валидация V1-V4 — гейты: любой FAIL = STOP (интерпретация
запрещена).
"""
import atexit, importlib.util, json, os, sys, time
from datetime import datetime

# ТОЖЕ, что в AT2b.py: фиксация 4 тредов ДО импорта numpy/torch (иначе BLAS-
# порядок редукции GEMM другой -> f32-шум в z-маршруте и базисах)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
torch.set_num_threads(4)
from safetensors import safe_open

# ------------------------------------------------------------------- константы
# (идентичны AT2b.py)
CACHE_DIR = "/data/prism_kernels/exp27b/results"
MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.join(ROOT, "results", "AT2b_diag")
LOG_PATH = os.path.join(RESULTS, "AT2b_diag_run.log")
ART = os.path.join(RESULTS, "AT2b_diag.json")
ART_FULL = os.path.join(ROOT, "results", "AT2b", "AT2b.json")

LAYS = (6, 12, 18, 22)
KS = (0, 1, 2, 4, 8)
N_DOCS = 32
R = 32
ALPHA = 1.0
SEED = 0
BOOT_B = 1000
D = 1024
S_LEN = 128
N_PER_DOM = 8
DOMAINS = ("wiki", "math", "code", "chat")
ARM_MAIN = ("top", "weak", "pca", "haar", "uniform",
            "top_dc", "weak_dc", "pca_dc", "haar_dc", "uniform_dc")
REVISION = "r3"
# ожидаемые S3-значения из лога прогона (tol 0.005)
S3_EXPECT = {"hi_mean": 0.381, "hi_top1": 0.490, "lo_mean": 0.014, "lo_top1": 0.017}

# ------------------------------------------------------------------- логи
_LOG = []
def logline(s):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    _LOG.append(line)

_sp = importlib.util.spec_from_file_location("prism_field", os.path.join(ROOT, "prism", "field.py"))
_pf = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(_pf)
sha256_file = _pf.sha256_file

# ------------------------------------------------------------------- trackio
def trackio_start(name, cfg):
    try:
        import trackio
    except ImportError as e:
        raise SystemExit(f"trackio: модуль не установлен — прогон остановлен: {e}")
    os.environ.setdefault("TRACKIO_DIR", os.path.join(ROOT, ".trackio"))
    return trackio.init(project="prism", name=name, group="AT0", config=cfg)

def tlog(run, t, metrics):
    if run is not None:
        run.log(metrics, step=int(t))

def tstage(run, t, name):
    tlog(run, t, {"stage": name, "stage_s": round(float(t), 1)})

# ------------------------------------------------------------------- данные
def unpack(fld):
    """dict {(dom,si,t): vec} -> [32, 128, ...] (domain-major; как field_to_array)."""
    key0 = fld[("wiki", 0, 0)]
    arr = torch.empty((N_DOCS, S_LEN) + tuple(key0.shape), dtype=key0.dtype)
    for di, dom in enumerate(DOMAINS):
        for si in range(N_PER_DOM):
            for t in range(S_LEN):
                arr[di * N_PER_DOM + si, t] = fld[(dom, si, t)]
    del fld
    return arr

def _qr(A):
    return np.linalg.qr(A)[0]

def _lift_arr(lam, w, jrt, sp, idxs):
    """lift v = JR^T(P^{1/2} w)/sqrt(lam) + QR-реортонормализация (как в r1)."""
    V = np.zeros((D, len(idxs)), dtype=np.float32)
    for j, i in enumerate(idxs):
        if lam[i] > 1e-12:
            V[:, j] = jrt @ (sp * w[:, i]) / np.sqrt(lam[i])
    return _qr(V)

def dc_free(V, DC):
    """V_dc = QR(V − dc (dc^T V)); dc = ones(1024)/32 (как в AT2b.py)."""
    Vc = V - DC[:, None] * (V.T @ DC)[None, :]
    return _qr(Vc)

def d_eff_of(projs, Hdoc):
    X = Hdoc.double() - Hdoc.double().mean(0, keepdim=True)
    V = torch.as_tensor(projs).double()
    C = (V.T @ X.T @ X @ V) / max(len(X) - 1, 1)
    lam = torch.linalg.eigvalsh(C).clamp_min(0.0)
    lam = lam / lam.sum().clamp_min(1e-30)
    h = -(lam * torch.log(lam)).sum()
    return float(torch.exp(h).item())

# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    logline(f"AT2b C2/C1 diagnostic start {datetime.now().isoformat()} "
            f"py={sys.version.split()[0]} torch={torch.__version__}")

    tr_cfg = {
        "started_at": datetime.now().isoformat(),
        "revision": REVISION,
        "script_sha": sha256_file(os.path.abspath(__file__)),
        "seed": SEED, "device": "cpu",
        "model": "Qwen/Qwen3.5-0.8B (tied E, U128 readout)",
        "log": os.path.abspath(LOG_PATH),
        "layers": list(LAYS), "k": list(KS), "r": R, "alpha": ALPHA,
        "boot_b": BOOT_B, "n_docs": N_DOCS,
        "note": "disk-only diagnostic: C2 (centered target) / C1 (stationary basis) / C3",
    }
    tr = trackio_start(f"at2b_diag_c2c1_{REVISION}", tr_cfg)
    tlog(tr, 0, {"started": 1.0})
    logline(f"trackio: run at2b_diag_c2c1_{REVISION} init ok")

    def _trackio_on_exit():
        try:
            tlog(tr, time.time() - t0, {
                "finished_at": datetime.now().isoformat(), "status": "crash",
                "error": "unhandled exception / abort (fin-точка не записана)"})
            tr.finish()
        except Exception:
            pass
    atexit.register(_trackio_on_exit)
    tr_done = [False]

    DC = np.ones(D, dtype=np.float32) / 32.0
    validations = {}

    # ================= phase: load ==========================================
    tstage(tr, time.time() - t0, "load")
    ck23 = torch.load(os.path.join(CACHE_DIR, "exp27b_jcache_L23.pt"),
                      map_location="cpu", weights_only=False)
    H23 = unpack(ck23["H"]).float()                    # [32,128,1024]
    U128 = np.asarray(ck23["U128"], dtype=np.int64)
    del ck23
    assert len(U128) == 128 and len(set(U128.tolist())) == 128
    pool = torch.load(os.path.join(CACHE_DIR, "exp27b_pool_ids_L23.pt"),
                      map_location="cpu", weights_only=False)
    toks = {}
    for di, dom in enumerate(DOMAINS):
        toks[dom] = [np.asarray(pool["ids"][dom][si], dtype=np.int64)
                     for si in range(N_PER_DOM)]
        assert all(len(r) == S_LEN for r in toks[dom])
    del pool

    with safe_open(os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors"),
                   framework="pt", device="cpu") as f:
        E = f.get_tensor("model.language_model.embed_tokens.weight").float()
    logline(f"E loaded: {tuple(E.shape)}")

    # ТО же, что в AT2b.py: z_full = H23 @ E.T ; z_U128 = z_full[:, :, U128]
    z_full = (H23 @ E.T)                               # [32,128,V] f32
    z_U128 = z_full[:, :, torch.from_numpy(U128)].numpy()   # [32,128,128] f32
    logline(f"z_U128: {z_U128.shape} f32 (mean {z_U128.mean():.4f}, std {z_U128.std():.4f})")

    # V1: S3-ориентация (воспроизведение прогона)
    idx_next = torch.stack([torch.from_numpy(toks[dom][si][::-1][:-1].copy())
                            for di, dom in enumerate(DOMAINS)
                            for si in range(N_PER_DOM)])     # [32,127]
    p_soft = torch.softmax(z_full, dim=-1)
    hi = p_soft[:, 1:128, :].gather(-1, idx_next.unsqueeze(-1)).squeeze(-1)
    lo = p_soft[:, :127, :].flip(1).gather(-1, idx_next.unsqueeze(-1)).squeeze(-1)
    s3 = {"hi_mean": float(hi.mean()), "hi_top1": float((p_soft[:, 1:128, :].argmax(-1) == idx_next).float().mean()),
          "lo_mean": float(lo.mean()), "lo_top1": float((p_soft[:, :127, :].flip(1).argmax(-1) == idx_next).float().mean())}
    v1_ok = all(abs(s3[k] - S3_EXPECT[k]) < 0.005 for k in S3_EXPECT)
    validations["V1_s3_orient"] = {"got": s3, "expected": S3_EXPECT, "ok": bool(v1_ok)}
    logline(f"V1 S3: {s3} (ожид {S3_EXPECT}) -> {'OK' if v1_ok else 'FAIL'}")
    if not v1_ok:
        raise SystemExit("V1 FAIL: z-маршрут не воспроизводится — STOP")
    del p_soft, hi, lo, idx_next

    # ================= phase: subspaces (per layer) ========================
    tstage(tr, time.time() - t0, "subspaces")
    rng_base = np.random.default_rng(SEED)
    G32 = rng_base.standard_normal((D, R))
    UVEC = rng_base.standard_normal(D)
    UVEC /= np.linalg.norm(UVEC)
    HAAR32 = _qr(G32)
    UNIF = UVEC[:, None].astype(np.float32)

    H = {}
    VJ, VJdc = {}, {}
    Vpca, Vpcadc = {}, {}
    lam_store = {}    # (L,d,t) -> (lam, top_i, weak_i) для C1a
    H_np = {}
    ew_all = None

    # pass 1 + d_eff per layer (как в AT2b.py, r_final из rematching)
    deff = {}
    ew_cache = {}
    for L in LAYS:
        ck = torch.load(os.path.join(CACHE_DIR, f"exp27b_jcache_L{L}.pt"),
                        map_location="cpu", weights_only=False)
        Ht = unpack(ck["H"]).float()
        JRt = unpack(ck["JR"]).float()
        Pt = unpack(ck["P128"]).float()
        del ck
        H[L] = Ht
        H_np[L] = Ht.numpy().astype(np.float32)
        ew_cache[L] = []
        for d in range(N_DOCS):
            ew_cache[L].append([])
            for t in range(S_LEN):
                jr = JRt[d, t].double()
                P = Pt[d, t].double()
                sp = torch.sqrt(P)
                G = (sp.unsqueeze(1) * (jr @ jr.T)) * sp.unsqueeze(0)
                lam, w = torch.linalg.eigh(G)
                ew_cache[L][d].append((lam.cpu().numpy(), w.cpu().numpy(),
                                       jr.T.numpy().astype(np.float32),
                                       sp.cpu().numpy()))
        # d_eff (t=0, R=32) — для rematching
        for d in range(N_DOCS):
            lam0, w0, jrt0, sp0 = ew_cache[L][d][0]
            Vtop0 = _lift_arr(lam0, w0, jrt0, sp0, range(128 - R, 128))
            Vweak0 = _lift_arr(lam0, w0, jrt0, sp0, range(0, R))
            Hdoc = Ht[d]
            Hc = (Hdoc.double() - Hdoc.double().mean(0, keepdim=True))
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            Vpca32 = Vh.T[:, :R].numpy().astype(np.float32)
            deff[(L, d)] = {
                "top": d_eff_of(Vtop0, Hdoc), "weak": d_eff_of(Vweak0, Hdoc),
                "pca": d_eff_of(Vpca32, Hdoc),
                "haar": d_eff_of(HAAR32, Hdoc),
            }
        del JRt, Pt
        logline(f"subspaces L{L}: eigh done")
    ratios = [max(v.values()) / min(v.values()) for v in deff.values()]
    r_final = R
    rematched = False
    if max(ratios) > 1.2:
        r_final = int(round(min(min(v.values()) for v in deff.values())))
        rematched = True
        logline(f"d_eff rematching: max/min = {max(ratios):.2f} > 1.2 -> r = {r_final}")
    validations["V4_r_final"] = {"r_final": r_final, "expected": 7, "ok": bool(r_final == 7)}
    logline(f"V4 r_final = {r_final} (ожид 7) -> {'OK' if r_final == 7 else 'FAIL'}")
    if r_final != 7:
        raise SystemExit("V4 FAIL: r_final != 7 — базисы не воспроизведены — STOP")

    # pass 2: финальные базисы + проекции + C1a-энергии (per layer)
    HAAR = _qr(G32[:, :r_final])
    HAARdc = dc_free(HAAR, DC)
    UNIFdc = dc_free(UNIF, DC)
    c1a_ratios = []
    proj = {}
    for L in LAYS:
        Vpca[(L, 0)] = None
        for d in range(N_DOCS):
            Hdoc = H[L][d]
            Hc = (Hdoc.double() - Hdoc.double().mean(0, keepdim=True))
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            Vpca[(L, d)] = Vh.T[:, :r_final].numpy().astype(np.float32)
            Vpcadc[(L, d)] = dc_free(Vpca[(L, d)], DC)
            Hd = H_np[L][d]
            for t in range(S_LEN):
                lam, w, jrt, sp = ew_cache[L][d][t]
                top_i = np.argsort(lam)[::-1][:r_final]
                weak_i = np.argsort(lam)[:min(r_final, int((lam > 1e-12).sum()))]
                VJ[(L, d, t)] = {
                    "top": _lift_arr(lam, w, jrt, sp, top_i),
                    "weak": _lift_arr(lam, w, jrt, sp, weak_i),
                }
                VJdc[(L, d, t)] = {
                    "top_dc": dc_free(VJ[(L, d, t)]["top"], DC),
                    "weak_dc": dc_free(VJ[(L, d, t)]["weak"], DC),
                }
                lam_store[(L, d, t)] = (lam, top_i, weak_i)
            # проекции per (L, d, arm) (то же, что в AT2b.py)
            proj[(L, d, "top")] = np.stack([VJ[(L, d, t)]["top"].T @ Hd[t] for t in range(S_LEN)])
            proj[(L, d, "weak")] = np.stack([VJ[(L, d, t)]["weak"].T @ Hd[t] for t in range(S_LEN)])
            proj[(L, d, "top_dc")] = np.stack([VJdc[(L, d, t)]["top_dc"].T @ Hd[t] for t in range(S_LEN)])
            proj[(L, d, "weak_dc")] = np.stack([VJdc[(L, d, t)]["weak_dc"].T @ Hd[t] for t in range(S_LEN)])
            proj[(L, d, "pca")] = Hd @ Vpca[(L, d)]
            proj[(L, d, "pca_dc")] = Hd @ Vpcadc[(L, d)]
            proj[(L, d, "haar")] = Hd @ HAAR
            proj[(L, d, "haar_dc")] = Hd @ HAARdc
            proj[(L, d, "uniform")] = Hd @ UNIF
            proj[(L, d, "uniform_dc")] = Hd @ UNIFdc
            for arm in ARM_MAIN:
                proj[(L, d, arm)] -= proj[(L, d, arm)].mean(0, keepdims=True)
        # C1a: Eckart-Young энергии
        for d in range(N_DOCS):
            for t in range(S_LEN):
                lam, top_i, weak_i = lam_store[(L, d, t)]
                c1a_ratios.append(float(lam[top_i].sum() / max(lam[weak_i].sum(), 1e-30)))
        del ew_cache[L], H[L]
        logline(f"subspaces L{L}: bases+projections done")
    del ew_cache, G32, HAAR32, UNIF, UVEC
    logline(f"C1a Eckart-Young: min top/weak energy ratio = {min(c1a_ratios):.4f} "
            f"over {len(c1a_ratios)} (l,d,t)")
    c1a_ok = min(c1a_ratios) > 1.0
    validations["C1a_eckart_young"] = {
        "min_top_over_weak": min(c1a_ratios), "n_cells": len(c1a_ratios),
        "ok": bool(c1a_ok),
        "note": "ratio <= 1 => баг split/eigh; > 1 — алгебра верна (tautology по Eckart-Young)"}
    if not c1a_ok:
        raise SystemExit("C1a FAIL: top <= weak по энергиям — баг eigh/split — STOP")

    # ================= phase: ridge (RAW + CD) =============================
    tstage(tr, time.time() - t0, "ridge")
    cos_raw = {}   # (L,k,arm) -> [32, 128-k]
    cos_cd = {}
    r2_raw = {}
    r2_cd = {}
    for L in LAYS:
        for k in KS:
            n = S_LEN - k
            y_all = z_U128[:, :n, :]
            for arm in ARM_MAIN:
                r_arm = proj[(L, 0, arm)].shape[1]
                rows_raw, rows_cd = [], []
                r2rows_raw, r2rows_cd = [], []
                for d in range(N_DOCS):
                    X = proj[(L, d, arm)][k:]
                    Xf = np.vstack([proj[(L, d2, arm)][k:]
                                    for d2 in range(N_DOCS) if d2 != d])
                    # --- RAW (как в AT2b.py) ---
                    yf = np.vstack([y_all[d2] for d2 in range(N_DOCS) if d2 != d])
                    Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
                    A = Xb.T @ Xb + ALPHA * np.eye(r_arm + 1)
                    A[-1, -1] -= ALPHA
                    W = np.linalg.solve(A, Xb.T @ yf)
                    pred = np.hstack([X, np.ones((X.shape[0], 1))]) @ W
                    ym = y_all[d].mean(0)
                    pc = pred - ym
                    yc = y_all[d] - ym
                    cosv = (pc * yc).sum(1) / (
                        np.linalg.norm(pc, axis=1) * np.linalg.norm(yc, axis=1) + 1e-12)
                    rows_raw.append(cosv)
                    r2rows_raw.append(1.0 - float((pc ** 2).sum()) / float((yc ** 2).sum()))
                    # --- CD (per-doc центрированная цель) ---
                    yf_cd = np.vstack([y_all[d2] - y_all[d2].mean(0)
                                       for d2 in range(N_DOCS) if d2 != d])
                    W_cd = np.linalg.solve(A, Xb.T @ yf_cd)
                    pred_cd = np.hstack([X, np.ones((X.shape[0], 1))]) @ W_cd
                    pc_cd = pred_cd   # шкала centered-цели (DC = 0)
                    cosv_cd = (pc_cd * yc).sum(1) / (
                        np.linalg.norm(pc_cd, axis=1) * np.linalg.norm(yc, axis=1) + 1e-12)
                    rows_cd.append(cosv_cd)
                    r2rows_cd.append(1.0 - float((pc_cd ** 2).sum()) / float((yc ** 2).sum()))
                cos_raw[(L, k, arm)] = np.stack(rows_raw)
                cos_cd[(L, k, arm)] = np.stack(rows_cd)
                r2_raw[(L, k, arm)] = np.array(r2rows_raw)
                r2_cd[(L, k, arm)] = np.array(r2rows_cd)
        logline(f"ridge L{L} done (RAW+CD, all k)")
    del proj

    # V2: RAW-воспроизведение per_pos_cos артефакта
    logline("V2: сверка RAW с пер_pos_cos артефакта (860000)...")
    art = json.load(open(ART_FULL))
    pp = art["per_pos_cos"]
    maxdiff = 0.0
    ndiff = 0
    for (L, k, arm), rows in cos_raw.items():
        for d in range(N_DOCS):
            for t in range(rows.shape[1]):
                key = f"{L}|d{d}|t{t + k}|k{k}|{arm}"
                got = float(rows[d][t])
                ref = pp[key]
                diff = abs(got - ref)
                if diff > maxdiff:
                    maxdiff = diff
                if diff > 1e-6:
                    ndiff += 1
    validations["V2_raw_repro"] = {
        "max_abs_diff": float(maxdiff), "n_gt_1e-6": int(ndiff),
        "n_checked": int(sum(v.shape[0] * v.shape[1] for v in cos_raw.values())),
        "ok": bool(maxdiff < 1e-6)}
    logline(f"V2 RAW vs artifact: max|diff| = {maxdiff:.3e}, n>1e-6 = {ndiff} "
            f"-> {'OK' if maxdiff < 1e-6 else 'FAIL'}")
    if maxdiff >= 1e-6:
        raise SystemExit("V2 FAIL: RAW-вариант не воспроизводит артефакт — STOP")

    # ================= phase: control (k=0) + C3 ===========================
    tstage(tr, time.time() - t0, "control")
    def ci_of(dd, rngb):
        means = []
        for b in range(BOOT_B):
            sel = rngb.integers(0, N_DOCS, N_DOCS)
            means.append(float(dd[sel].mean()))
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

    def control_block(cos):
        ci, perL = {}, {}
        rngb = np.random.default_rng(SEED)
        for L in LAYS:
            da = np.stack([cos[(L, 0, "top")][d] - cos[(L, 0, "weak")][d]
                           for d in range(N_DOCS)])
            ci[L] = ci_of(da, rngb)
            perL[L] = float(da.mean())
        ok = all(ci[L][0] > 0.0 for L in LAYS)
        return {"CI": ci, "mean_top_minus_weak": perL, "control_ok": bool(ok)}

    ctrl_raw = control_block(cos_raw)
    ctrl_cd = control_block(cos_cd)
    # V3: RAW-контрольный CI = CI артефакта
    ref_ci = {int(kk): v for kk, v in art["bootstrap"]["control_k0_top_minus_weak_CI"].items()}
    v3 = {L: {"got": ctrl_raw["CI"][L], "ref": ref_ci[L],
              "max_diff": max(abs(ctrl_raw["CI"][L][0] - ref_ci[L][0]),
                             abs(ctrl_raw["CI"][L][1] - ref_ci[L][1]))}
          for L in LAYS}
    v3_ok = all(v3[L]["max_diff"] < 1e-5 for L in LAYS)
    validations["V3_control_ci"] = {**v3, "ok": bool(v3_ok),
                                    "ref_control_ok": art["verdict_detail"]["control_ok"]}
    logline(f"V3 RAW control CI vs artifact: {json.dumps({L: v3[L]['max_diff'] for L in LAYS})} "
            f"-> {'OK' if v3_ok else 'FAIL'}")
    if not v3_ok:
        raise SystemExit("V3 FAIL: RAW-контрольный CI не воспроизведён — STOP")

    # C3: per (L,arm) k=0 — R2/cos medians (RAW/CD) + weighted cos (артефакт, веса ||yc||^2)
    c3 = {}
    for L in LAYS:
        yc0 = z_U128[:, :128] - z_U128[:, :128].mean(1, keepdims=True)  # per (doc,t) centring
        w_t = (yc0 ** 2).sum(2)                        # [32,128] per (doc,t)
        for arm in ARM_MAIN:
            cr = cos_raw[(L, 0, arm)]
            cc = cos_cd[(L, 0, arm)]
            c3[f"{L}|{arm}"] = {
                "cos_median_raw": float(np.median(cr)), "R2_mean_raw": float(r2_raw[(L, 0, arm)].mean()),
                "cos_median_cd": float(np.median(cc)), "R2_mean_cd": float(r2_cd[(L, 0, arm)].mean()),
                "wcos_raw": float((w_t * cr).sum() / w_t.sum()),
                "wcos_cd": float((w_t * cc).sum() / w_t.sum()),
            }
    logline(f"control RAW ok={ctrl_raw['control_ok']} CD ok={ctrl_cd['control_ok']}")

    # ================= phase: C1 (local + stationary) ======================
    tstage(tr, time.time() - t0, "c1")
    # (b) локальный per-(l,t) ridge: 32 дока, центрирование по докаМ
    c1b = {}
    for L in LAYS:
        diffs = []
        for t in range(S_LEN):
            Xtop = np.stack([VJ[(L, d, t)]["top"].T @ H_np[L][d][t] for d in range(N_DOCS)])
            Xweak = np.stack([VJ[(L, d, t)]["weak"].T @ H_np[L][d][t] for d in range(N_DOCS)])
            yloc = z_U128[:, t, :].copy()
            for Xa, tag in ((Xtop, "top"), (Xweak, "weak")):
                Xa -= Xa.mean(0, keepdims=True)
                yc_loc = yloc - yloc.mean(0, keepdims=True)
                # per-doc LOO ridge (31 образцов)
                if tag == "top":
                    cos_d = np.empty(N_DOCS)
                    for d in range(N_DOCS):
                        Xf = np.vstack([Xa[d2] for d2 in range(N_DOCS) if d2 != d])
                        yf = np.vstack([yc_loc[d2] for d2 in range(N_DOCS) if d2 != d])
                        Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
                        A = Xb.T @ Xb + ALPHA * np.eye(Xa.shape[1] + 1)
                        A[-1, -1] -= ALPHA
                        W = np.linalg.solve(A, Xb.T @ yf)
                        pred = (np.hstack([Xa[d].reshape(1, -1), np.ones((1, 1))]) @ W)[0]
                        cos_d[d] = float(pred @ yc_loc[d] / (
                            np.linalg.norm(pred) * np.linalg.norm(yc_loc[d]) + 1e-12))
                    cos_top = cos_d
                else:
                    cos_d = np.empty(N_DOCS)
                    for d in range(N_DOCS):
                        Xf = np.vstack([Xa[d2] for d2 in range(N_DOCS) if d2 != d])
                        yf = np.vstack([yc_loc[d2] for d2 in range(N_DOCS) if d2 != d])
                        Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
                        A = Xb.T @ Xb + ALPHA * np.eye(Xa.shape[1] + 1)
                        A[-1, -1] -= ALPHA
                        W = np.linalg.solve(A, Xb.T @ yf)
                        pred = (np.hstack([Xa[d].reshape(1, -1), np.ones((1, 1))]) @ W)[0]
                        cos_d[d] = float(pred @ yc_loc[d] / (
                            np.linalg.norm(pred) * np.linalg.norm(yc_loc[d]) + 1e-12))
                    cos_weak = cos_d
            diffs.append(float(cos_top.mean() - cos_weak.mean()))
        diffs = np.array(diffs)
        c1b[f"L{L}"] = {
            "mean_diff_top_minus_weak": float(diffs.mean()),
            "median_diff_top_minus_weak": float(np.median(diffs)),
            "frac_t_top_gt_weak": float((diffs > 0).mean()),
        }
        logline(f"C1b L{L}: local per-(l,t) diff mean={diffs.mean():+.4f} "
                f"frac(t top>weak)={(diffs > 0).mean():.2f}")
    del VJ, VJdc, Vpca, Vpcadc

    # (c) стационарный per-layer агрегированный J-базис + полный пул-ridge (CD)
    c1c = {}
    for L in LAYS:
        ck = torch.load(os.path.join(CACHE_DIR, f"exp27b_jcache_L{L}.pt"),
                        map_location="cpu", weights_only=False)
        JRt = unpack(ck["JR"]).float()
        del ck
        Jagg = JRt.mean(axis=(0, 1)).double()          # [128, 1024]: mean_{d,t} J_{d,t}
        del JRt
        Gagg = Jagg.T @ Jagg                            # [1024,1024] rank<=128
        lam, w = torch.linalg.eigh(Gagg)
        order = torch.argsort(lam, descending=True)
        Vtop_a = w[:, order[:r_final]].cpu().numpy().astype(np.float32)
        Vweak_a = w[:, order[r_final:2 * r_final]].cpu().numpy().astype(np.float32)
        del Jagg, Gagg, lam, w, order
        outL = {}
        for arm, Va in (("top", Vtop_a), ("weak", Vweak_a)):
            Xs = np.stack([H_np[L][d] @ Va - (H_np[L][d] @ Va).mean(0, keepdims=True)
                           for d in range(N_DOCS)])      # [32,128,r]
            yf_cd = np.stack([z_U128[d] - z_U128[d].mean(0) for d in range(N_DOCS)])
            rows = []
            for d in range(N_DOCS):
                Xf = np.stack([Xs[d2] for d2 in range(N_DOCS) if d2 != d]).reshape(-1, r_final)
                yfc = np.vstack([yf_cd[d2] for d2 in range(N_DOCS) if d2 != d])
                Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
                A = Xb.T @ Xb + ALPHA * np.eye(r_final + 1)
                A[-1, -1] -= ALPHA
                W = np.linalg.solve(A, Xb.T @ yfc)
                pred = Xs[d].reshape(-1, r_final) @ W[:r_final] + W[-1]
                yc = yf_cd[d]
                rows.append((pred * yc).sum(1) / (
                    np.linalg.norm(pred, axis=1) * np.linalg.norm(yc, axis=1) + 1e-12))
            outL[arm] = np.stack(rows)
        da = outL["top"] - outL["weak"]
        rngb = np.random.default_rng(SEED)
        c1c[f"L{L}"] = {
            "cos_median_top": float(np.median(outL["top"])),
            "cos_median_weak": float(np.median(outL["weak"])),
            "mean_top_minus_weak": float(da.mean()),
            "CI_top_minus_weak": ci_of(da, rngb),
        }
        logline(f"C1c L{L}: stationary top/weak med = "
                f"{np.median(outL['top']):+.4f}/{np.median(outL['weak']):+.4f}, "
                f"CI diff = {c1c[f'L{L}']['CI_top_minus_weak']}")
        del outL

    # ================= phase: artifact =====================================
    tstage(tr, time.time() - t0, "artifact")
    out = {
        "meta": {
            "started_at": tr_cfg["started_at"],
            "script_sha": tr_cfg["script_sha"],
            "seed": SEED, "device": "cpu", "r_final": r_final,
            "wall_sec": round(time.time() - t0, 1),
            "note": "disk-only diagnostic (C2/C1/C3) для docs/AT2b_review.md 2-го раунда",
        },
        "validations": validations,
        "c2": {
            "control_raw": {str(k): v for k, v in ctrl_raw["CI"].items()} |
                           {"control_ok": ctrl_raw["control_ok"],
                            "mean_top_minus_weak": {str(k): v for k, v in ctrl_raw["mean_top_minus_weak"].items()}},
            "control_cd": {str(k): v for k, v in ctrl_cd["CI"].items()} |
                          {"control_ok": ctrl_cd["control_ok"],
                           "mean_top_minus_weak": {str(k): v for k, v in ctrl_cd["mean_top_minus_weak"].items()}},
            "offset_scale": {},
            "per_arm_k0": c3,
        },
        "c1": {
            "eckart_young": validations["C1a_eckart_young"],
            "local_per_lt": c1b,
            "stationary_pooled_cd": c1c,
        },
    }
    # offset scale (c(d,k) = ȳf − ȳ(d))
    for k in KS:
        n = S_LEN - k
        Y = z_U128[:, :n].astype(np.float64)
        cn = []
        for d in range(N_DOCS):
            yf = np.concatenate([Y[d2] for d2 in range(N_DOCS) if d2 != d])
            cn.append(float(np.linalg.norm(yf.mean(0) - Y[d].mean(0))))
        cn = np.array(cn)
        yc_med = float(np.median(np.linalg.norm(
            (Y - Y.mean(1, keepdims=True)), axis=2)))
        out["c2"]["offset_scale"][f"k{k}"] = {
            "c_max": float(cn.max()), "c_med": float(np.median(cn)),
            "yc_perpos_med": yc_med,
            "ratio_med": float(np.median(cn) / yc_med),
            "ratio_max": float(cn.max() / yc_med),
        }

    raw_json = json.dumps(out, ensure_ascii=False, indent=1)
    if "NaN" in raw_json or "Infinity" in raw_json:
        raise SystemExit("artifact: NaN/Infinity в JSON — STOP")
    with open(ART, "w") as f:
        f.write(raw_json)
    logline(f"artifact written: {ART} ({os.path.getsize(ART)} B)")
    tlog(tr, time.time() - t0, {
        "finished_at": datetime.now().isoformat(), "status": "ok",
        "control_ok_raw": ctrl_raw["control_ok"],
        "control_ok_cd": ctrl_cd["control_ok"],
        "v1": validations["V1_s3_orient"]["ok"],
        "v2": validations["V2_raw_repro"]["ok"],
        "v3": validations["V3_control_ci"]["ok"],
        "v4": validations["V4_r_final"]["ok"],
    })
    tr.finish()
    tr_done[0] = True
    logline(f"DONE wall={time.time() - t0:.1f}s "
            f"control RAW ok={ctrl_raw['control_ok']} CD ok={ctrl_cd['control_ok']}")
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_LOG) + "\n")

if __name__ == "__main__":
    main()
