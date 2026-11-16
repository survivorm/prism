#!/usr/bin/env python
"""AT2b (T2.1b) — центрированная state-readout клетка (disk-only).

Пре-рег: docs/AT2b_prereg.md (ревью: ACCEPT с фиксами C1-C3 + S9, docs/AT2_review_answers.md).
Клетка T2.1b (docs/AT2_results.md §5 + docs/AT2_review.md §10): та же клетка T2.1
(h_l^t -> readout будущего логита), но:
(1) центрированная метрика (DC-floor r1 = 0.9449-0.9456 выше всех readout; R^2 < 0);
(2) k=0 контроль SUBSPACE_CONTROL (односторонний CI-гейт, C1);
(3) настоящий Haar = QR(G), G ~ N(0,1) + uniform-направление (floor) + DC-free варианты;
(4) DELTA_MIN + ветка METRIC_UNINFORMATIVE;
(5) lk4: n_hits per k + U128-coverage c n и CI, one-hot vs y_c — non_informative;
(6) N8 report-only.
Фикс R2E-1: no-penalty на intercept — A[-1,-1] -= ALPHA (в r1 было A[0,0] — feature 0).
Ноль новых форвардов: кэш exp27b + E (tied embed). CPU, 4 треда.
"""
import atexit, importlib.util, json, os, sys, time
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
torch.set_num_threads(4)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root (скрипт в experiments/at2/)
CACHE_DIR = "/data/prism_kernels/exp27b/results"
MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
RESULTS = os.path.join(ROOT, "results")
SMOKE = os.environ.get("AT2b_SMOKE", "0") == "1"
ART = (os.path.join(RESULTS, "AT2b_smoke", "AT2b_smoke.json") if SMOKE
       else os.path.join(RESULTS, "AT2b", "AT2b.json"))
LOG_PATH = (os.path.join(RESULTS, "AT2b_smoke", "AT2b_smoke_run.log") if SMOKE
            else os.path.join(RESULTS, "AT2b", "AT2b_run.log"))
REVISION = "r1"

LAYS = (6, 12, 18, 22) if not SMOKE else (6,)     # J-слои
L_CTRL = 23                                        # L23-контроль (pca/haar/uniform)
KS = (0, 1, 2, 4, 8) if not SMOKE else (0, 1)
N_DOCS_RUN = 2 if SMOKE else 32
R = 32
ALPHA = 1.0
SEED = 0
BOOT_B = 100 if SMOKE else 1000
D = 1024
S_LEN = 128
DELTA_MIN = 0.01                                   # пре-рег §7 (1% абсолютного уровня)
DOMAINS = ["wiki", "math", "code", "chat"]
N_PER_DOM = 8
N_DOCS = 32
KER_SAMPLE = 32                                    # (l,t) на слой для ker-J контроля
ARM_MAIN = ("top", "weak", "pca", "haar", "uniform",
            "top_dc", "weak_dc", "pca_dc", "haar_dc", "uniform_dc")
ARM_L23 = ("pca", "haar", "uniform")
J_ARMS = ("top", "weak", "pca", "haar")            # в дерево вердикта + N8

SHA_FILES = {  # гейт S1 (те же 7, AT2-пре-рег §7)
    "exp27b_jcache_L6.pt":  "6d6b8ce784a2ad01a10112309aad7a74f13413d14636aad1a00652575f605dd2",
    "exp27b_jcache_L12.pt": "976079ce1d3c84c7016a236683f86f5fee1ce6571ff6edd53cf4f6f9ac228a24",
    "exp27b_jcache_L18.pt": "e2a03bc36b199695018171d0f0eab92ee139be83d0f0792a558cefa15f852d63",
    "exp27b_jcache_L19.pt": "0b31baff7dbc6e38c525d10d803ed9cc313774c6c8460cf1290a9ab9f4df82ae",
    "exp27b_jcache_L22.pt": "36ed09ebc49dd97bc7b8e0a8edd1c8d75567bb6431c4eea33c908ccfb7295a84",
    "exp27b_jcache_L23.pt": "af12505131be72b424515759c054646ea4526d94822287cbe7e8c50b7db46a43",
    "exp27b_pool_ids_L23.pt": "ded634f7d16cf33a9bb565de5cbc3faa288cfc90b15cdfa3e42365695fec4e03",
}
LK4_EXPECT = {"total": 90, "wiki": 8, "math": 41, "code": 36, "chat": 5}  # гейт S4 (канон as-is)

_LOG = []
def logline(s):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    _LOG.append(line)

# sha256_file — общий из prism/field.py (spec-лоад, минуя prism/__init__)
_sp = importlib.util.spec_from_file_location("prism_field", os.path.join(ROOT, "prism", "field.py"))
_pf = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(_pf)
sha256_file = _pf.sha256_file

# ------------------------------------------------------------------- trackio
def trackio_start(name, cfg):
    try:
        import trackio
    except ImportError as e:
        raise SystemExit(f"trackio: модуль не установлен — прогон остановлен "
                         f"(AGENTS: trackio обязателен для CPU-прогонов): {e}")
    os.environ.setdefault("TRACKIO_DIR", os.path.join(ROOT, ".trackio"))
    return trackio.init(project="prism", name=name, group="AT0", config=cfg)

def tlog(run, t, metrics):
    if run is not None:
        run.log(metrics, step=int(t))

def tstage(run, t, name):
    tlog(run, t, {"stage": name, "stage_s": round(float(t), 1)})

# ------------------------------------------------------------------- данные
def field_to_array(fld):
    """dict {(dom,si,t): vec} -> [32, 128, ...] (domain-major)."""
    key0 = fld[("wiki", 0, 0)]
    arr = torch.empty((N_DOCS, S_LEN) + tuple(key0.shape), dtype=key0.dtype)
    for di, dom in enumerate(DOMAINS):
        for si in range(N_PER_DOM):
            for t in range(S_LEN):
                arr[di * N_PER_DOM + si, t] = fld[(dom, si, t)]
    del fld
    return arr

def load_layer(L, want):
    ck = torch.load(os.path.join(CACHE_DIR, f"exp27b_jcache_L{L}.pt"),
                    map_location="cpu", weights_only=False)
    out = {k: field_to_array(ck[k]) for k in want}
    del ck
    return out

def lk4_hits(row):
    """Канон (experiments/exp64_at_battery.py:23-30): 4-грамма предсказывает СЛЕДУЮЩИЙ токен."""
    nxt = {}
    hit = np.zeros(len(row), dtype=bool)
    for i in range(3, len(row) - 1):
        k = (int(row[i - 3]), int(row[i - 2]), int(row[i - 1]), int(row[i]))
        pv = nxt.get(k)
        if pv is not None and pv == int(row[i + 1]):
            hit[i] = True
        nxt[k] = int(row[i + 1])
    return hit

def lk4_hits_kahead(row, k):
    """К-ahead (k >= 1): i + k <= 127, иначе промах (версионная нота пре-рега)."""
    nxt = {}
    hit = np.zeros(len(row), dtype=bool)
    for i in range(3, len(row) - k):
        key = (int(row[i - 3]), int(row[i - 2]), int(row[i - 1]), int(row[i]))
        pv = nxt.get(key)
        if pv is not None and pv == int(row[i + k]):
            hit[i] = True
        nxt[key] = int(row[i + k])
    return hit

def hit_to_cache_mask(hit):
    """forward i -> cache t = 127 - i."""
    m = np.zeros(S_LEN, dtype=bool)
    idx = S_LEN - 1 - np.where(hit)[0]
    m[idx] = True
    return m

def d_eff_of(projs, Hdoc):
    """d_eff(S) = exp(H(lam(Pr_S Cov(h) Pr_S))), float64, по t в доке. projs: [1024, r]."""
    X = Hdoc.double() - Hdoc.double().mean(0, keepdim=True)     # [128, 1024]
    V = torch.as_tensor(projs).double()                          # [1024, r]
    C = (V.T @ X.T @ X @ V) / max(len(X) - 1, 1)                # [r, r]
    lam = torch.linalg.eigvalsh(C).clamp_min(0.0)
    lam = lam / lam.sum().clamp_min(1e-30)
    h = -(lam * torch.log(lam)).sum()
    return float(torch.exp(h).item())

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
    """F1-фикс: V_dc = QR(V - dc (dc^T V)); dc = ones(1024)/32 (единичный). V: [1024, r]."""
    Vc = V - DC[:, None] * (V.T @ DC)[None, :]
    return _qr(Vc)

# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    logline(f"AT2b (T2.1b) start {datetime.now().isoformat()} "
            f"py={sys.version.split()[0]} torch={torch.__version__} smoke={SMOKE}")

    tr = None
    tr_done = [False]
    if not SMOKE:
        tr_cfg = {
            "started_at": datetime.now().isoformat(),
            "revision": REVISION,
            "script_sha": sha256_file(os.path.abspath(__file__)),
            "seed": SEED, "device": "cpu",
            "model": "Qwen/Qwen3.5-0.8B (tied E, U128 readout)",
            "log": os.path.abspath(LOG_PATH),
            "layers": list(LAYS), "k": list(KS), "r": R, "alpha": ALPHA,
            "boot_b": BOOT_B, "n_docs": N_DOCS_RUN, "delta_min": DELTA_MIN,
        }
        tr = trackio_start(f"at2b_t21b_{REVISION}", tr_cfg)
        tlog(tr, 0, {"started": 1.0})
        logline(f"trackio: run at2b_t21b_{REVISION} init ok")

        def _trackio_on_exit():
            if not tr_done[0]:
                try:
                    tlog(tr, time.time() - t0, {
                        "finished_at": datetime.now().isoformat(), "status": "crash",
                        "error": "unhandled exception / abort (fin-точка не записана)"})
                    tr.finish()
                except Exception:
                    pass
        atexit.register(_trackio_on_exit)

    gates = {}
    DC = np.ones(D, dtype=np.float32) / 32.0

    # --- S1: sha ------------------------------------------------------------
    tstage(tr, time.time() - t0, "load")
    for fn, expect in SHA_FILES.items():
        got = sha256_file(os.path.join(CACHE_DIR, fn))
        if got != expect:
            raise SystemExit(f"S1 FAIL: sha {fn} = {got} != {expect}")
    gates["S1_sha"] = True
    logline("S1 sha-гейт: 7/7 OK")

    # --- загрузка -------------------------------------------------------------
    H, JRs, P128s = {}, {}, {}
    for L in LAYS:
        f = load_layer(L, ("H", "JR", "P128"))
        H[L], JRs[L], P128s[L] = f["H"], f["JR"], f["P128"]
        logline(f"load L{L}: H {tuple(H[L].shape)} JR {tuple(JRs[L].shape)}")
    f23 = load_layer(L_CTRL, ("H", "P128_RAW"))
    H23, PR23 = f23["H"], f23["P128_RAW"]
    ck_meta = torch.load(os.path.join(CACHE_DIR, "exp27b_jcache_L23.pt"),
                         map_location="cpu", weights_only=False)
    U128 = np.asarray(ck_meta["U128"], dtype=np.int64)
    del ck_meta
    assert len(U128) == 128 and len(set(U128.tolist())) == 128
    pool = torch.load(os.path.join(CACHE_DIR, "exp27b_pool_ids_L23.pt"),
                      map_location="cpu", weights_only=False)
    toks = {}
    for di, dom in enumerate(DOMAINS):
        toks[dom] = [np.asarray(pool["ids"][dom][si], dtype=np.int64)
                     for si in range(N_PER_DOM)]
        assert all(len(r) == S_LEN for r in toks[dom])

    # --- E + z-маршрут ---------------------------------------------------------
    from safetensors import safe_open
    with safe_open(os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors"),
                   framework="pt", device="cpu") as f:
        E = f.get_tensor("model.language_model.embed_tokens.weight").float()
    logline(f"E loaded: {tuple(E.shape)}")
    z_full = (H23 @ E.T)                                # [32,128,V] f32
    p_soft = torch.softmax(z_full, dim=-1)
    U128_t = torch.from_numpy(U128)
    p_u128 = p_soft[:, :, U128_t]                       # [32,128,128]
    kl_doc = (p_u128 * (torch.log(p_u128.clamp_min(1e-30))
          - torch.log(PR23.clamp_min(1e-30)))).sum(-1)  # [32,128]
    kl2 = float(kl_doc[:2].mean())
    gates["S2_kl"] = bool(kl2 < 1e-5)
    kl_report = {"kl_gate_2docs": kl2, "kl_all32_mean": float(kl_doc.mean()),
                 "kl_all32_min": float(kl_doc.min()), "kl_all32_max": float(kl_doc.max())}
    logline(f"S2 z-маршрут: mean KL (2 дока) = {kl2:.3e} "
            f"all32 = {kl_report['kl_all32_mean']:.3e} "
            f"{'OK' if gates['S2_kl'] else 'FAIL'}")
    if not gates["S2_kl"]:
        raise SystemExit("S2 FAIL")
    # S3: ориентация (exploratory; пул 32 дока × 127 из r1 — не пререгистрирован).
    # z[cache t] = логиты forward 127-t => предсказывает row[128-t] (t>=1).
    idx_next = torch.stack([torch.from_numpy(toks[dom][si][::-1][:-1].copy())
                            for di, dom in enumerate(DOMAINS)
                            for si in range(N_PER_DOM)])     # [32,127]: row[128-t], t=1..127
    hi = p_soft[:, 1:128, :].gather(-1, idx_next.unsqueeze(-1)).squeeze(-1)
    lo = p_soft[:, :127, :].flip(1).gather(-1, idx_next.unsqueeze(-1)).squeeze(-1)
    mean_hi = float(hi.mean())
    top1_hi = float((p_soft[:, 1:128, :].argmax(-1) == idx_next).float().mean())
    mean_lo = float(lo.mean())
    top1_lo = float((p_soft[:, :127, :].flip(1).argmax(-1) == idx_next).float().mean())
    gates["S3_orient"] = bool(mean_hi > 0.1 and top1_hi > 0.1
                              and mean_lo < 0.05 and top1_lo < 0.05)
    logline(f"S3 ориентация (exploratory): hi {mean_hi:.3f}/{top1_hi:.3f} (ожид >0.1) "
            f"lo {mean_lo:.3f}/{top1_lo:.3f} (ожид <0.05) "
            f"{'OK' if gates['S3_orient'] else 'FAIL'}")
    if not gates["S3_orient"]:
        raise SystemExit("S3 FAIL")
    z_U128 = z_full[:, :, U128_t].numpy()               # [32,128,128] float32
    del z_full, p_soft, p_u128, hi, lo, kl_doc

    # --- lk4 + S4 ---------------------------------------------------------------
    lk4_tot = 0; lk4_dom = {d: 0 for d in DOMAINS}
    lk4_mask = {}                                        # k=0 канон (N8-стратум)
    lk4_k = {k: {} for k in KS}
    for d in range(N_DOCS):
        dom = DOMAINS[d // N_PER_DOM]
        row = toks[dom][d % N_PER_DOM]
        h = lk4_hits(row)
        lk4_tot += int(h.sum()); lk4_dom[dom] += int(h.sum())
        lk4_mask[d] = hit_to_cache_mask(h)
        for k in KS:
            m = h if k == 0 else lk4_hits_kahead(row, k)
            lk4_k[k][d] = hit_to_cache_mask(m)
    if SMOKE:
        gates["S4_lk4"] = True
        logline(f"S4 lk4 (smoke): total {lk4_tot}, per-домен "
                f"{{ {', '.join(f'{d} {lk4_dom[d]}' for d in DOMAINS)} }} — ожидаемые числа не гейтятся")
    else:
        gates["S4_lk4"] = (lk4_tot == LK4_EXPECT["total"] and
                           all(lk4_dom[d] == LK4_EXPECT[d] for d in DOMAINS))
        logline(f"S4 lk4: total {lk4_tot}/4096 per-домен "
                f"{{ {', '.join(f'{d} {lk4_dom[d]}' for d in DOMAINS)} }} "
                f"{'OK' if gates['S4_lk4'] else 'FAIL (ожид. %s)' % LK4_EXPECT}")
    if not gates["S4_lk4"]:
        raise SystemExit("S4 FAIL")
    # --- подпространства (F1, F7) -------------------------------------------------
    tstage(tr, time.time() - t0, "subspaces")
    # rng_base: ОДИН поток (SEED=0): сначала G (1024×32), затем uniform — пре-рег §3
    rng_base = np.random.default_rng(SEED)
    G32 = rng_base.standard_normal((D, R))              # Gaussian 1024×32
    UVEC = rng_base.standard_normal(D)
    UVEC /= np.linalg.norm(UVEC)                        # одна глобальная случайная прямая
    HAAR32 = _qr(G32)                                   # для d_eff (r=32)
    UNIF = UVEC[:, None].astype(np.float32)             # [1024, 1]
    # pass 1: eigh per (l,t) — хранить (lam, w) для rematching
    ew = {}
    for L in LAYS:
        for d in range(N_DOCS_RUN):
            ew[(L, d)] = []
            for t in range(S_LEN):
                jr = JRs[L][d, t].double()
                P = P128s[L][d, t].double()
                sp = torch.sqrt(P)
                G = (sp.unsqueeze(1) * (jr @ jr.T)) * sp.unsqueeze(0)
                lam, w = torch.linalg.eigh(G)
                ew[(L, d)].append((lam.cpu().numpy(), w.cpu().numpy(),
                                   jr.T.numpy().astype(np.float32),
                                   sp.cpu().numpy()))
    # d_eff per (l,doc,S) (J-базисы — t=0 как репрезентативный, как в r1)
    Vpca32 = {}
    deff = {}
    for L in LAYS:
        for d in range(N_DOCS_RUN):
            Hdoc = H[L][d]
            lam0, w0, jrt0, sp0 = ew[(L, d)][0]
            Vtop0 = _lift_arr(lam0, w0, jrt0, sp0, range(128 - R, 128))
            Vweak0 = _lift_arr(lam0, w0, jrt0, sp0, range(0, R))
            Hc = (Hdoc.double() - Hdoc.double().mean(0, keepdim=True))
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            Vpca32[(L, d)] = Vh.T[:, :R].numpy().astype(np.float32)
            deff[(L, d)] = {
                "top": d_eff_of(Vtop0, Hdoc), "weak": d_eff_of(Vweak0, Hdoc),
                "pca": d_eff_of(Vpca32[(L, d)], Hdoc),
                "haar": d_eff_of(HAAR32, Hdoc),
            }
    r_final = R
    rematched = False
    ratios = [max(v.values()) / min(v.values()) for v in deff.values()]
    if max(ratios) > 1.2:
        r_final = int(round(min(min(v.values()) for v in deff.values())))
        rematched = True
        logline(f"d_eff rematching: max/min = {max(ratios):.2f} > 1.2 -> r = {r_final}")
    # pass 2: финальные базисы с r_final + DC-free варианты
    HAAR = _qr(G32[:, :r_final])                        # true Haar: QR(G), глобальный
    HAARdc = dc_free(HAAR, DC)
    UNIFdc = dc_free(UNIF, DC)
    VJ = {}    # (L,d,t) -> {'top': [1024,r], 'weak': [1024,r]}
    VJdc = {}  # (L,d,t) -> {'top_dc': ..., 'weak_dc': ...}
    Vpca = {}
    Vpcadc = {}
    for L in LAYS:
        for d in range(N_DOCS_RUN):
            Vpca[(L, d)] = Vpca32[(L, d)][:, :r_final]
            Vpcadc[(L, d)] = dc_free(Vpca[(L, d)], DC)
            for t in range(S_LEN):
                lam, w, jrt, sp = ew[(L, d)][t]
                n_nz = int((lam > 1e-12).sum())
                top_i = np.argsort(lam)[::-1][:r_final]
                weak_i = np.argsort(lam)[:min(r_final, n_nz)]
                VJ[(L, d, t)] = {
                    "top": _lift_arr(lam, w, jrt, sp, top_i),
                    "weak": _lift_arr(lam, w, jrt, sp, weak_i),
                }
                VJdc[(L, d, t)] = {
                    "top_dc": dc_free(VJ[(L, d, t)]["top"], DC),
                    "weak_dc": dc_free(VJ[(L, d, t)]["weak"], DC),
                }
    # d_eff пересчёт с r_final (если rematching)
    if rematched:
        for (L, d) in deff:
            Hdoc = H[L][d]
            lam, w, jrt, sp = ew[(L, d)][0]
            deff[(L, d)] = {
                "top": d_eff_of(_lift_arr(lam, w, jrt, sp, np.argsort(lam)[::-1][:r_final]), Hdoc),
                "weak": d_eff_of(_lift_arr(lam, w, jrt, sp, np.argsort(lam)[:r_final]), Hdoc),
                "pca": d_eff_of(Vpca[(L, d)], Hdoc),
                "haar": d_eff_of(HAAR, Hdoc),
            }
    del Vpca32, HAAR32, G32
    # ker-J: выборка 32 (l,t) на слой (контроль, не в вердикт)
    rng = np.random.default_rng(SEED)
    kerV = {}
    for L in LAYS:
        for d in range(N_DOCS_RUN):
            tsel = rng.choice(S_LEN, KER_SAMPLE, replace=False)
            for t in tsel:
                jr = JRs[L][d, t].double()
                _, _, Vh = torch.linalg.svd(jr, full_matrices=True)
                kerV[(L, d, int(t))] = Vh[128:128 + r_final, :].T.numpy().astype(np.float32)
    del ew
    logline(f"подпространства: r_final={r_final} (rematched={rematched}); "
            f"top/weak per (l,t), pca per (l,doc), true-Haar/uniform глобальные, "
            f"+_dc варианты, ker-J выборка")

    # --- S5: eig-закрытость (как в r1) ------------------------------------------
    s5_ok = True
    for L in LAYS:
        for d in range(min(2, N_DOCS_RUN)):
            for t in (0, 64, 127):
                jr = JRs[L][d, t].double(); P = P128s[L][d, t].double()
                C = jr.T @ torch.diag(P) @ jr
                eC = torch.linalg.eigvalsh(C).sort(descending=True).values
                sp = torch.sqrt(P)
                G = (sp.unsqueeze(1) * (jr @ jr.T)) * sp.unsqueeze(0)
                eG = torch.linalg.eigvalsh(G).sort(descending=True).values
                if not torch.allclose(eC[:5], eG[:5], rtol=1e-9, atol=0):
                    s5_ok = False
                lam, w = torch.linalg.eigh(G)
                i = int(torch.argmax(lam))
                v = (jr.T @ (sp * w[:, i]) / torch.sqrt(lam[i]))
                if (C @ v - lam[i] * v).norm() / (C @ v).norm() > 1e-6:
                    s5_ok = False
    gates["S5_eig"] = bool(s5_ok)
    logline(f"S5 eig-закрытость: {s5_ok}")
    if not s5_ok:
        raise SystemExit("S5 FAIL")
    del JRs, P128s   # далее не нужны (J-базисы в VJ/VJdc, ker-V готов) — освобождаем до ridge
    # --- readout: проекции + ridge (LOO-doc), центрированная метрика -------------
    tstage(tr, time.time() - t0, "ridge")
    # Пайплайн X (один для ВСЕХ арм, включая *_dc — пре-рег §2):
    # проекция на V -> центрирование по t (t=0..127) per (l,doc,arm) -> срез [k:]
    proj = {}   # (L,d,arm) -> [128, r_arm]
    for L in LAYS:
        for d in range(N_DOCS_RUN):
            Hd = H[L][d].numpy().astype(np.float32)
            proj[(L, d, "top")] = np.stack([VJ[(L, d, t)]["top"].T @ Hd[t]
                                            for t in range(S_LEN)])
            proj[(L, d, "weak")] = np.stack([VJ[(L, d, t)]["weak"].T @ Hd[t]
                                             for t in range(S_LEN)])
            proj[(L, d, "top_dc")] = np.stack([VJdc[(L, d, t)]["top_dc"].T @ Hd[t]
                                               for t in range(S_LEN)])
            proj[(L, d, "weak_dc")] = np.stack([VJdc[(L, d, t)]["weak_dc"].T @ Hd[t]
                                                for t in range(S_LEN)])
            proj[(L, d, "pca")] = Hd @ Vpca[(L, d)]
            proj[(L, d, "pca_dc")] = Hd @ Vpcadc[(L, d)]
            proj[(L, d, "haar")] = Hd @ HAAR
            proj[(L, d, "haar_dc")] = Hd @ HAARdc
            proj[(L, d, "uniform")] = Hd @ UNIF
            proj[(L, d, "uniform_dc")] = Hd @ UNIFdc
            for arm in ARM_MAIN:
                proj[(L, d, arm)] -= proj[(L, d, arm)].mean(0, keepdims=True)
    # L23-контроль (без J-базисов и без _dc), тот же LOO-контур
    for d in range(N_DOCS_RUN):
        Hc23 = H23[d].double() - H23[d].double().mean(0, keepdim=True)
        _, _, Vh23 = torch.linalg.svd(Hc23, full_matrices=False)
        Vpca23 = Vh23.T[:, :r_final].numpy().astype(np.float32)
        Hd23 = H23[d].numpy().astype(np.float32)
        proj[(L_CTRL, d, "pca")] = Hd23 @ Vpca23
        proj[(L_CTRL, d, "haar")] = Hd23 @ HAAR
        proj[(L_CTRL, d, "uniform")] = Hd23 @ UNIF
        for arm in ARM_L23:
            proj[(L_CTRL, d, arm)] -= proj[(L_CTRL, d, arm)].mean(0, keepdims=True)
    del VJ, VJdc, Vpca, Vpcadc, HAAR, HAARdc, UNIF, UNIFdc

    SUBS_RUN = {L: (ARM_MAIN if L in LAYS else ARM_L23) for L in LAYS + (L_CTRL,)}
    cos_doc = {}   # (L,k,arm) -> [N_DOCS_RUN, 128-k]  (cos_c per позиция)
    r2doc = {}     # (L,k,arm) -> [N_DOCS_RUN] (R2_c per doc)
    for L in LAYS + (L_CTRL,):
        for k in KS:
            n = S_LEN - k
            y_all = z_U128[:, :n, :]                   # [32, n, 128] (будущий логит; k=0 — текущий)
            for arm in SUBS_RUN[L]:
                r_arm = proj[(L, 0, arm)].shape[1]
                rows = []
                r2rows = []
                for d in range(N_DOCS_RUN):
                    X = proj[(L, d, arm)][k:]
                    Xf = np.vstack([proj[(L, d2, arm)][k:]
                                    for d2 in range(N_DOCS_RUN) if d2 != d])
                    yf = np.vstack([y_all[d2] for d2 in range(N_DOCS_RUN) if d2 != d])
                    Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
                    A = Xb.T @ Xb + ALPHA * np.eye(r_arm + 1)
                    A[-1, -1] -= ALPHA                 # R2E-1: no-penalty на intercept (r1: A[0,0])
                    W = np.linalg.solve(A, Xb.T @ yf)
                    pred = np.hstack([X, np.ones((X.shape[0], 1))]) @ W
                    ym = y_all[d].mean(0)              # ȳ per (doc,k)
                    pc = pred - ym                     # предиктор и цель — одинаково центрированы
                    yc = y_all[d] - ym
                    cosv = (pc * yc).sum(1) / (
                        np.linalg.norm(pc, axis=1) * np.linalg.norm(yc, axis=1) + 1e-12)
                    rows.append(cosv)
                    r2rows.append(1.0 - float((pc ** 2).sum()) / float((yc ** 2).sum()))
                cos_doc[(L, k, arm)] = np.stack(rows)
                r2doc[(L, k, arm)] = np.array(r2rows)
        logline(f"readout L{L} done")
    agg = {}
    for (L, k, arm), rows in cos_doc.items():
        allc = np.concatenate(rows)
        agg[(L, k, arm)] = {"cos_median": float(np.median(allc)),
                            "cos_mean": float(allc.mean()),
                            "R2_mean": float(np.mean(r2doc[(L, k, arm)]))}
    # floor (C2): cos-метрика — uniform-арм (изотропия => cos_c ≈ 0);
    # медиана per (l,k) + doc-bootstrap CI (ниже, в bootstrap-блоке)
    # S6 (R2E-3, C2 — assert на ВЕСА, не на cos): ridge с нулевыми фичами,
    # LOO-doc (31 док в каждом фолде), per k: W_X = 0 (rel < 1e-10) и
    # intercept = mean(y_f) (rel < 1e-10)
    s6_ok = True
    s6_worst = {}
    for k in KS:
        n = S_LEN - k
        y_all = z_U128[:, :n, :]
        worst_w = 0.0; worst_i = 0.0
        for d in range(N_DOCS_RUN):
            yf = np.vstack([y_all[d2] for d2 in range(N_DOCS_RUN) if d2 != d])
            Xf = np.zeros(((N_DOCS_RUN - 1) * n, r_final), dtype=np.float64)
            Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
            A = Xb.T @ Xb + ALPHA * np.eye(r_final + 1)
            A[-1, -1] -= ALPHA
            W = np.linalg.solve(A, Xb.T @ yf)
            # W: [r_final+1, 128] — по 128 целевым измерениям; W[-1] — вектор интерцептов
            wr = float(np.abs(W[:r_final]).max()) / (float(np.abs(W).max()) + 1e-30)
            im_j = yf.astype(np.float64).mean(0)    # [128] mean(y_f) per dim, f64 (yf — f32)
            ir = float(np.abs(W[-1] - im_j).max()) / (float(np.abs(im_j).max()) + 1e-30)
            worst_w = max(worst_w, wr); worst_i = max(worst_i, ir)
        ok_k = (worst_w < 1e-10) and (worst_i < 1e-10)
        s6_worst[f"k{k}"] = {"W_X_rel_max": worst_w, "intercept_rel_max": worst_i, "ok": bool(ok_k)}
        s6_ok = s6_ok and ok_k
    gates["S6_pzero"] = bool(s6_ok)
    logline(f"S6 P-zero (веса): {s6_ok}")
    if not s6_ok:
        raise SystemExit("S6 FAIL")
    # S6b (C2): R^2-floor — константный предиктор in-sample => R2_c = 1 - TSS_c/TSS_c = 0;
    # проверка per (doc,k): TSS_c > 0 и |R2_c| < 1e-12
    s6b_ok = True
    for k in KS:
        n = S_LEN - k
        y_all = z_U128[:, :n, :]
        for d in range(N_DOCS_RUN):
            ym = y_all[d].mean(0)
            tss = float(((y_all[d] - ym) ** 2).sum())
            r2c = 1.0 - tss / tss
            if not (tss > 0.0 and abs(r2c) < 1e-12):
                s6b_ok = False
    gates["S6b_r2floor"] = bool(s6b_ok)
    logline(f"S6b R2-floor (TSS_c > 0, R2_c = 0): {s6b_ok}")
    if not s6b_ok:
        raise SystemExit("S6b FAIL")
    # --- lk4 (F5): n_hits per k, U128-coverage c n и CI, one-hot — non_informative ---
    tstage(tr, time.time() - t0, "lk4")
    u128_set = {int(v): j for j, v in enumerate(U128)}
    n_hits_per_k = {}
    cov_doc = {k: [] for k in KS}   # per (k, d) coverage или None (нет hit-позиций); append по d (d — по возрастанию)
    for k in KS:
        hcnt = 0
        for d in range(N_DOCS_RUN):
            row = toks[DOMAINS[d // N_PER_DOM]][d % N_PER_DOM]
            m = lk4_k[k][d]
            hcnt += int(m.sum())
            tt = [t for t in range(S_LEN) if m[t]]
            if tt:
                ahead = max(1, k)   # k=0 канон: следующий токен row[i+1]; k>=1: row[i+k]
                cov_doc[k].append((sum(1 for t in tt
                                       if int(row[S_LEN - 1 - t + ahead]) in u128_set)
                                   / len(tt)))
            else:
                cov_doc[k].append(None)
        n_hits_per_k[k] = hcnt
    # one-hot vs y_c cos (report-only, non_informative; честный y как в r1:
    # логиты, предсказывающие целевой токен; k=0: y = z[t] (текущие), k>=1: y = z[t+1-k])
    onehot_med = {}
    for k in KS:
        n = S_LEN - k
        vals = []
        for d in range(N_DOCS_RUN):
            row = toks[DOMAINS[d // N_PER_DOM]][d % N_PER_DOM]
            m = lk4_k[k][d]
            ym = z_U128[d, :n].mean(0)
            c = np.full(n, np.nan)
            for t in range(S_LEN):
                if not m[t]:
                    continue
                i = S_LEN - 1 - t
                ahead = max(1, k)
                if i + ahead > S_LEN - 1:
                    continue
                tok = int(row[i + ahead])
                if tok not in u128_set:
                    continue
                y_idx = t if k == 0 else t + 1 - k
                y = z_U128[d, y_idx] - ym
                one = np.zeros(128)
                one[u128_set[tok]] = 1.0
                c[t - k] = float(one @ y / (np.linalg.norm(y) + 1e-12))
            vals.extend(v for v in c if not np.isnan(v))
        onehot_med[f"k{k}"] = float(np.median(vals)) if vals else None
    # N8 (report-only, F4): cos_c per (l,k,S) на lk4-hit (k=0 канон) позициях
    n8 = {}
    for L in LAYS:
        for k in KS:
            for arm in J_ARMS:
                rows = []
                for d in range(N_DOCS_RUN):
                    tsel = np.arange(k, S_LEN)[lk4_mask[d][k:]]
                    if len(tsel):
                        rows.append(cos_doc[(L, k, arm)][d][tsel - k])
                n8[f"{L}|k{k}|{arm}"] = (float(np.median(np.concatenate(rows)))
                                         if rows else None)

    # --- bootstrap + вердикт (F6: DELTA_MIN; C1: односторонние CI) ---------------
    tstage(tr, time.time() - t0, "verdict")
    rngb = np.random.default_rng(SEED)
    def ci_of(dd):
        """per-doc bootstrap (B, SEED=0): ресэмплинг доков, stat = mean по (док,t) (конвенция r1)."""
        means = []
        for b in range(BOOT_B):
            sel = rngb.integers(0, N_DOCS_RUN, N_DOCS_RUN)
            means.append(float(dd[sel].mean()))
        return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
    def ci_of_scalars(vals):
        """doc-bootstrap пер-док скаляров: ресэмплинг доков, stat = mean."""
        stats = []
        for b in range(BOOT_B):
            sel = rngb.integers(0, len(vals), len(vals))
            stats.append(float(np.mean([vals[i] for i in sel])))
        return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]
    def ci_of_perdoc_medians(meds):
        """doc-bootstrap пер-док медиан: ресэмплинг доков, stat = медиана пер-док медиан (floor)."""
        stats = []
        for b in range(BOOT_B):
            sel = rngb.integers(0, len(meds), len(meds))
            stats.append(float(np.median([meds[i] for i in sel])))
        return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

    boot = {}
    # фаз 1: control CI (k=0): top - weak per слой
    ctrl_ci = {}
    for L in LAYS:
        da = np.stack([cos_doc[(L, 0, "top")][d] - cos_doc[(L, 0, "weak")][d]
                       for d in range(N_DOCS_RUN)])
        ctrl_ci[L] = ci_of(da)
    # фаз 2: per (L, k), k in KS, k > 0: weak-vs-max(pca,haar) + pairwise + (lookup — report)
    PAIRS = [("top", "weak"), ("top", "pca"), ("top", "haar"),
             ("weak", "pca"), ("weak", "haar"), ("pca", "haar")]
    for L in LAYS:
        for k in KS:
            if k == 0:
                continue
            wh = np.stack([cos_doc[(L, k, "weak")][d] - np.maximum(
                cos_doc[(L, k, "pca")][d], cos_doc[(L, k, "haar")][d])
                for d in range(N_DOCS_RUN)])
            boot[(L, k)] = {"weak_vs_maxPCAHaar_CI": ci_of(wh),
                            "weak_vs_maxPCAHaar_median": float(np.median(wh))}
            pcis = {}
            for a_, b_ in PAIRS:
                da = np.stack([cos_doc[(L, k, a_)][d] - cos_doc[(L, k, b_)][d]
                               for d in range(N_DOCS_RUN)])
                pcis[f"{a_}_vs_{b_}"] = ci_of(da)
            boot[(L, k)]["pairwise_CI"] = pcis
    # фаз 3: floor — uniform per (L, k): медиана + doc-bootstrap CI пер-док медиан
    floor_out = {}
    for L in LAYS:
        for k in KS:
            ucos = cos_doc[(L, k, "uniform")]
            perdoc_med = [float(np.median(ucos[d])) for d in range(N_DOCS_RUN)]
            floor_out[f"{L}|k{k}"] = {
                "uniform_cos_median": float(np.median(np.concatenate(ucos))),
                "uniform_cos_CI": ci_of_perdoc_medians(perdoc_med),
            }
    # фаз 4: lk4 coverage CI per k
    cov_out = {}
    for k in KS:
        vals = [v for v in cov_doc[k] if v is not None]
        cov_out[f"k{k}"] = {
            "value": float(np.mean(vals)) if vals else 0.0,
            "n_hits": int(n_hits_per_k[k]),
            "n_docs_with_hits": len(vals),
            "CI": (ci_of_scalars(vals) if vals else None),
        }

    # фаз 5 (V5 review): DELTA_MIN-калибровка — level vs центрированная полоса и floor CI
    # (безусловный отчёт; если level больше всей полосы — METRIC_UNINFORMATIVE съест всё)
    tree_ks = tuple(k for k in (1, 2, 4, 8) if k in KS)
    max_spread = 0.0
    for L in LAYS:
        for k in tree_ks:
            ms = {s: agg[(L, k, s)]["cos_median"] for s in J_ARMS}
            for a_, b_ in PAIRS:
                max_spread = max(max_spread, abs(ms[a_] - ms[b_]))
    floor_cw = max((v["uniform_cos_CI"][1] - v["uniform_cos_CI"][0])
                   for v in floor_out.values()) if floor_out else 0.0
    delta_calib = {
        "delta_min": DELTA_MIN,
        "max_pairwise_spread_Jarms": float(max_spread),
        "floor_CI_width_max": float(floor_cw),
        "delta_min_gt_spread": bool(DELTA_MIN > max_spread),
        "delta_min_gt_floor_width": bool(DELTA_MIN > floor_cw),
    }

    def win_ci(ci):
        return ci[0] > 0.0    # односторонний: ВЕСЬ CI выше 0 (C1)
    def contains_zero(ci):
        return ci[0] <= 0.0 <= ci[1]

    # SUBSPACE_CONTROL (C1: lower > 0 на ВСЕХ 4 слоях, AND-over-l)
    control_ok = all(ctrl_ci[L][0] > 0.0 for L in LAYS)
    detail = {
        "DELTA_MIN": DELTA_MIN,
        "control_k0_top_minus_weak_CI": {str(L): ctrl_ci[L] for L in LAYS},
        "control_ok": bool(control_ok),
    }
    if not control_ok:
        verdict = "SUBSPACE_CONTROL_FAIL"
        logline(f"VERDICT: {verdict} (k>=1 отчитаны, но не интерпретируются как ответ на T2)")
    else:
        # (a) SCRATCHPAD: >=2 из k in {2,4,8}: все 4 l — win_ci И median > DELTA_MIN
        a_per_k = {}
        for k in (2, 4, 8):
            if k not in KS:
                continue
            ci_ok = all(win_ci(boot[(L, k)]["weak_vs_maxPCAHaar_CI"]) for L in LAYS)
            meds = {str(L): float(boot[(L, k)]["weak_vs_maxPCAHaar_median"]) for L in LAYS}
            med_ok = all(m > DELTA_MIN for m in meds.values())
            a_per_k[str(k)] = {"ci_ok_all_l": bool(ci_ok), "med_ok_all_l": bool(med_ok),
                               "medians": meds}
        n_a = sum(1 for v in a_per_k.values() if v["ci_ok_all_l"] and v["med_ok_all_l"])
        detail["a_scratchpad"] = {"per_k": a_per_k, "n_pass": n_a}
        if n_a >= 2:
            verdict = "SCRATCHPAD"
        else:
            # (b) DEADWEIGHT: все (l,k) — |cos_median(top)| и |cos_median(weak)| <= DELTA_MIN
            ks_tree = tuple(k for k in (1, 2, 4, 8) if k in KS)
            mtop = max(abs(agg[(L, k, "top")]["cos_median"]) for L in LAYS for k in ks_tree)
            mweak = max(abs(agg[(L, k, "weak")]["cos_median"]) for L in LAYS for k in ks_tree)
            b_ok = (mtop <= DELTA_MIN) and (mweak <= DELTA_MIN)
            detail["b_deadweight"] = {"max_abs_top": float(mtop), "max_abs_weak": float(mweak),
                                      "b_ok": bool(b_ok)}
            if b_ok:
                verdict = "DEADWEIGHT"
            else:
                # (c) REDUNDANT_BASIS: все парные CI содержат 0 на всех (l,k) И
                # на k=1 top >= max(остальных) по медиане (без значимости)
                c_ok = True
                for L in LAYS:
                    for k in ks_tree:
                        for key, ci in boot[(L, k)]["pairwise_CI"].items():
                            if not contains_zero(ci):
                                c_ok = False
                k1ok = {}
                for L in LAYS:
                    if 1 in KS:
                        mt = agg[(L, 1, "top")]["cos_median"]
                        mx = max(agg[(L, 1, s)]["cos_median"] for s in ("weak", "pca", "haar"))
                        k1ok[str(L)] = bool(mt >= mx)
                    else:
                        k1ok[str(L)] = False
                c_ok = c_ok and all(k1ok.values())
                detail["c_redundant"] = {"c_ok": bool(c_ok), "k1_top_ge_max": k1ok}
                if c_ok:
                    verdict = "REDUNDANT_BASIS"
                else:
                    # (d) METRIC_UNINFORMATIVE: max парная |разность медиан| <= DELTA_MIN
                    maxdiff = 0.0
                    for L in LAYS:
                        for k in ks_tree:
                            ms = {s: agg[(L, k, s)]["cos_median"] for s in J_ARMS}
                            for a_, b_ in PAIRS:
                                maxdiff = max(maxdiff, abs(ms[a_] - ms[b_]))
                    d_ok = maxdiff <= DELTA_MIN
                    detail["d_metric_uninformative"] = {"max_pairwise_diff": float(maxdiff),
                                                        "d_ok": bool(d_ok)}
                    verdict = "METRIC_UNINFORMATIVE" if d_ok else "NOT_RESOLVED"
    logline(f"VERDICT: {verdict} (control_ok={control_ok})")
    # --- артефакт + S7 (C3: явная формула) -----------------------------------------
    tstage(tr, time.time() - t0, "artifact")
    per_pos = {}
    for (L, k, arm), rows in cos_doc.items():
        for d in range(N_DOCS_RUN):
            for t in range(S_LEN - k):
                per_pos[f"{L}|d{d}|t{t + k}|k{k}|{arm}"] = float(rows[d][t])
    pos_per_doc = sum(S_LEN - k for k in KS)          # full: 625 = 128+127+126+124+120
    S7_expected = (len(LAYS) * N_DOCS_RUN * pos_per_doc * len(ARM_MAIN)
                   + N_DOCS_RUN * pos_per_doc * len(ARM_L23))
    expected = set()
    for L in LAYS:
        for d in range(N_DOCS_RUN):
            for k in KS:
                for t in range(k, S_LEN):
                    for arm in ARM_MAIN:
                        expected.add(f"{L}|d{d}|t{t}|k{k}|{arm}")
    for d in range(N_DOCS_RUN):
        for k in KS:
            for t in range(k, S_LEN):
                for arm in ARM_L23:
                    expected.add(f"{L_CTRL}|d{d}|t{t}|k{k}|{arm}")
    set_eq = set(per_pos.keys()) == expected
    gates["S7_exact_set"] = bool(set_eq and len(per_pos) == S7_expected)
    logline(f"S7 exact-set: len(per_pos)={len(per_pos)} S7_expected={S7_expected} "
            f"set-eq={set_eq} -> {gates['S7_exact_set']}")
    if not gates["S7_exact_set"]:
        raise SystemExit(f"S7 FAIL: len={len(per_pos)} != {S7_expected} или set-разброс")
    # ker-J контроль (report, не в вердикт)
    ctrl = {"L23": {str(k): {arm: agg[(L_CTRL, k, arm)]["cos_median"] for arm in ARM_L23}
                    for k in KS}}
    for L in LAYS:
        for k in KS:
            vals = []
            for d in range(N_DOCS_RUN):
                for (L2, d2, t2), V in kerV.items():
                    if L2 != L or d2 != d or t2 < k:
                        continue
                    h = H[L2][d2, t2].numpy()
                    vals.append(float(np.linalg.norm(h @ V) / (np.linalg.norm(h) + 1e-12)))
            ctrl[f"kerJ_{L}_k{k}"] = float(np.median(vals)) if vals else None

    out = {
        "run": f"at2b_t21b_{REVISION}",
        "started_at": datetime.now().isoformat(),
        "config": {"layers": list(LAYS + (L_CTRL,)), "k": list(KS), "r": R,
                   "r_final": r_final, "rematched": rematched, "alpha": ALPHA,
                   "seed": SEED, "boot_b": BOOT_B, "delta_min": DELTA_MIN,
                   "n_docs": N_DOCS_RUN, "arms_main": list(ARM_MAIN),
                   "arms_L23": list(ARM_L23),
                   "smoke": SMOKE, "device": "cpu", "model": "Qwen/Qwen3.5-0.8B",
                   "script_sha": sha256_file(os.path.abspath(__file__))},
        "gates": gates,
        "metric": ("cos_median = центрированная метрика cos_c (пре-рег §2): "
                   "cos(pred_t - ybar, y_t - ybar), ybar = mean_t(y) per (doc,k); "
                   "R2_mean — по центрированному TSS; дерево вердиктов и все CI "
                   "используют ровно эту центрированную метрику (review V1)"),
        "s2_kl_report": kl_report,
        "s6_pzero": s6_worst,
        "s6b_r2floor": "pass: TSS_c > 0 и |R2_c| < 1e-12 per (doc,k)",
        "delta_min_calibration": delta_calib,
        "S7_expected": int(S7_expected),
        "kerJ_in_per_position": False,
        "kerJ_in_verdict": False,
        "d_eff": {f"{L}|d{d}": v for (L, d), v in deff.items()},
        "subspaces": {
            "definition": ("top/weak: eigh(G), G = P^{1/2}(JRJR^T)P^{1/2}, lift v = JR^T(P^{1/2}w)/sqrt(l), "
                           "QR-реортонормализация, per (l,t); weak = min-r lambda; "
                           "pca: per (l,doc) SVD центрированного H; "
                           "haar: true Haar = QR(G), G ~ N(0,1)^{1024 x r_final}, глобальный (rng_base SEED=0); "
                           "uniform: одна глобальная Gaussian-прямая (rng_base, после G); "
                           "ker-J: норма проекции h_l^t на SVD-null J_l (выборка 32 (l,t))"),
            "asymmetry": {
                "top": "per (l,t), из J-кэша", "weak": "per (l,t), из J-кэша",
                "pca": "per (l,doc), unsupervised (тот же док; утечек по y нет)",
                "haar": "глобальный", "uniform": "глобальный (floor-референс, не в вердикт)",
                "_dc": "V_dc = QR(V - dc (dc^T V)), dc = ones(1024)/32; X-пайплайн общий (только базис различается)",
            },
            "metric": ("cos_c(t) = cos(pred_t - ybar, y_t - ybar), ybar = mean_t(y) per (doc,k); "
                       "X: проекция на V -> центрирование по t (t=0..127) -> срез [k:] (однаков для всех арм); "
                       "y = z_U128[d, :128-k] (k=0 — текущий логит); "
                       "ridge alpha=1.0 LOO-doc (32 фолда), no-penalty на intercept: A[-1,-1] -= ALPHA (R2E-1); "
                       "R2_c = 1 - SSE_c/TSS_c per (doc,k), TSS_c = sum((y-ybar)^2)"),
            "floor": ("cos: uniform-арм (изотропия => cos_c ~ 0) per (l,k): медиана + doc-bootstrap CI; "
                      "R2: константный предиктор in-sample => R2_c = 0 (S6b)"),
        },
        "metrics": {f"{L}|k{k}|{arm}": m for (L, k, arm), m in agg.items()},
        "floor_uniform": floor_out,
        "lk4": {
            "hit_total": lk4_tot, "hit_per_dom": lk4_dom,
            "n_hits_per_k": {str(k): int(n_hits_per_k[k]) for k in KS},
            "u128_coverage": cov_out,
            "onehot_vs_yc_cos_median": onehot_med,
            "onehot_non_informative": True,
            "onehot_note": ("разные пространства (one-hot vs логиты), D3; r1: median 0.0 на всех k; "
                            "из условия (a) дерева lookup выведен (F5)"),
            "n8_strata_median": n8,   # report-only (F4/N8 — интерпретация отложена)
        },
        "bootstrap": {
            "control_k0_top_minus_weak_CI": {str(L): ctrl_ci[L] for L in LAYS},
            "pairwise": {f"{L}|k{k}": v for (L, k), v in boot.items()},
        },
        "control": ctrl,
        "verdict": verdict,
        "verdict_detail": detail,
        "per_pos_cos": per_pos,
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(ART, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logline(f"артефакт: {ART} ({os.path.getsize(ART) // 1024} KB)")
    if SMOKE:
        det = {"metrics": out["metrics"], "d_eff": out["d_eff"],
               "per_pos": per_pos, "bootstrap": out["bootstrap"],
               "verdict": verdict, "r_final": r_final}
        det_path = os.path.join(os.path.dirname(ART), "AT2b_smoke_det.json")
        with open(det_path, "w") as f:
            json.dump(det, f, sort_keys=True, ensure_ascii=False)
        logline(f"smoke det (S9): {det_path} ({os.path.getsize(det_path) // 1024} KB)")

    if not SMOKE:
        tr_done[0] = True
        tlog(tr, time.time() - t0, {"finished_at": datetime.now().isoformat(),
                                     "status": "ok", "verdict": verdict})
        tr.finish()
    logline(f"AT2b (T2.1b) DONE wall={time.time() - t0:.1f}s verdict={verdict}")
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_LOG) + "\n")

if __name__ == "__main__":
    main()
