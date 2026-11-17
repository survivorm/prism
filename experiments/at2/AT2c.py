#!/usr/bin/env python3
"""AT2c (T2.1c) — стационарная state-readout клетка (disk-only).

Пре-рег: docs/AT2c_prereg.md (ревью: r1 REVISE -> r2 REVISE -> r3 PASS ->
r4 PASS (pre-smoke фикс k=0) -> r5 PASS (2 строки) -> r6 PASS (S6c-дизайн),
docs/AT2c_review.md / _r2.._r6 + docs/AT2c_review_answers.md).

Революции (main-прогон): r1 (N8 4 арм) -> r2 (N8 6 арм, ТЗ §6) -> r3
(S6c dcos = max_d weighted-mean по формуле ТЗ + ybar строгий < 1e-5;
стоп: конфаунд max_d-pairwise = 1.03e-2 > DELTA_MIN — порог откалиброван
по unweighted-замеру) -> r4 (официальный: ТЗ r7 — статистика конфаунда =
per-arm weighted-wmean разность, verdict-количество, 7.6e-3 < DELTA_MIN;
max_d-pairwise — report-only). Наука (метрики/дерево/гейты) r1=r2=r4
бит-в-бит (S9-детерминизм; r3 не завершён).
Артефакт: results/AT2c/AT2c.json (r4); r1/r2 — /tmp/recovered/AT2c/.
Клетка T2.1c: та же клетка T2 (state h_l -> readout будущего логита), три
исправления конструкции T2.1b:
(1) СТАЦИОНАРНЫЙ per-layer J-базис: Jagg = mean_{d,t} JR_{d,t} ([128,1024], f64);
    Gagg = Jagg^T Jagg ([1024,1024], rank<=128); torch eigh; top = первые r_final;
    weak = слабейшие r_final НЕ-нулевых (weak-tail, конвенция T2.1/AT2b);
    weak_mid = позиции r..2r по убыванию (конвенция диагностики C1c; report-only
    + S8-якорь; НЕ в дерево);
(2) CENTERED-ОБУЧАЮЩАЯ ЦЕЛЬ (C2-фикс): y_tr = y_c = y - ybar_{d,k};
    intercept: k=0 = 0 ТОЧНО; k>0 = -(1^T Xf W)/n (малый, report-only, §3 ТЗ);
(3) ПЕРВИЧНАЯ МЕТРИКА — weighted cos (веса ||y_c(d,t)||^2, ОБЩИЕ для всех армов
    на позиции); вторичная — median (конвенция T2.1b). Все условия дерева и
    k=0-гейта — на первичной.
Дерево вердиктов, DELTA_MIN=0.01, lk4/N8/L23, S1-S7 — наследованы от T2.1b.
Новое: S5b (eig-закрытость Gagg), S8 (diagnostic anchor на C1c, max|diff|<=1e-6;
значения-референс из results/AT2b_diag/AT2b_diag.json c1.stationary_pooled_cd).
Report-only: C1b локальный per-(l,t) контроль (не гейт, §6 ТЗ), weak_vs_top /
weak_vs_weak_mid CI.
Thread-pinning OMP/OPENBLAS/MKL=4 + torch 4 тр. ДО импортов (S8 bit-identity;
поймано в диагностике r1/r2). Ноль новых форвардов: кэш exp27b + tied E. CPU.

Режимы:
  AT2c_SMOKE=1    : smoke (L=(6,), k=(0,1), 2 дока, B=100) — дет-под-артефакт S9;
  AT2c_PRERUN=1   : pre-run замер §11.0 (k=0, top vs weak-tail, полный пул,
                    weighted CI) -> results/AT2c/prerun_c0.json (НЕ гейт);
  AT2c_ARMSWAP=1  : arm-swap смок §11 (L6, k=0, 32 дока, r=7 фикс — конфигурация
                    main-прогона; swap обязан дать control FAIL, lower < 0);
  (без флагов)    : main-прогон at2c_t21c_r1 (trackio).
"""
import atexit, hashlib, importlib.util, json, os, sys, time
from datetime import datetime

# ТОЖЕ, что в AT2b.py/диагностике: фиксация 4 тредов ДО импорта numpy/torch
# (иначе BLAS-порядок редукции GEMM другой -> f32-шум в z-маршруте и базисах; S8)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
torch.set_num_threads(4)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # repo root
_sp = importlib.util.spec_from_file_location("prism_field", os.path.join(ROOT, "prism", "field.py"))
_pf = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(_pf)
sha256_file = _pf.sha256_file

CACHE_DIR = "/data/prism_kernels/exp27b/results"
MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
RESULTS = os.path.join(ROOT, "results", "AT2c")
SMOKE = os.environ.get("AT2c_SMOKE", "0") == "1"
PRERUN = os.environ.get("AT2c_PRERUN", "0") == "1"
ARMSWAP = os.environ.get("AT2c_ARMSWAP", "0") == "1"
if SMOKE:
    SMOKE_DIR = os.path.join(os.path.dirname(RESULTS), "AT2c_smoke")   # TZ: results/AT2c_smoke/
    ART = os.path.join(SMOKE_DIR, "AT2c_smoke.json")
    LOG_PATH = os.path.join(SMOKE_DIR, "AT2c_smoke_run.log")
elif PRERUN:
    ART = os.path.join(RESULTS, "prerun_c0.json")
    LOG_PATH = os.path.join(RESULTS, "prerun_c0.log")
elif ARMSWAP:
    ART = os.path.join(RESULTS, "armswap_l6.json")
    LOG_PATH = os.path.join(RESULTS, "armswap_l6.log")
else:
    ART = os.path.join(RESULTS, "AT2c.json")
    LOG_PATH = os.path.join(RESULTS, "AT2c_run.log")
REVISION = "r4"   # r1: N8 4 арм; r2: N8 6 арм; r3: S6c weighted (стоп: конфаунд max_d 1.03e-2);
                 # r4: S6c конфаунд = per-arm wmean (ТЗ r7) — официальный
MAIN_RUN = (not SMOKE) and (not PRERUN) and (not ARMSWAP)

LAYS = (6, 12, 18, 22) if (MAIN_RUN or PRERUN) else (6,)
L_CTRL = 23
if SMOKE:
    KS = (0, 1)
elif PRERUN or ARMSWAP:
    KS = (0,)
else:
    KS = (0, 1, 2, 4, 8)
N_DOCS_RUN = 2 if SMOKE else 32
R = 32
ALPHA = 1.0
SEED = 0
BOOT_B = 100 if SMOKE else 1000
D = 1024
S_LEN = 128
DELTA_MIN = 0.01                                  # пре-рег §5 (1% абсолютного уровня)
DOMAINS = ("wiki", "math", "code", "chat")
N_PER_DOM = 8
N_DOCS = 32
ARM_MAIN = ("top", "weak", "weak_mid", "pca", "haar", "uniform")
ARM_L23 = ("pca", "haar", "uniform")
J_ARMS = ("top", "weak", "pca", "haar")          # в дерево вердикта + N8
PAIRS = [("top", "weak"), ("top", "pca"), ("top", "haar"),
         ("weak", "pca"), ("weak", "haar"), ("pca", "haar")]
REPORT_PAIRS = [("weak", "top"), ("weak", "weak_mid")]   # report-only (ТЗ §5)

SHA_FILES = {  # гейт S1 (те же 7, AT2-пре-рег §7 / T2.1b)
    "exp27b_jcache_L6.pt":  "6d6b8ce784a2ad01a10112309aad7a74f13413d14636aad1a00652575f605dd2",
    "exp27b_jcache_L12.pt": "976079ce1d3c84c7016a236683f86f5fee1ce6571ff6edd53cf4f6f9ac228a24",
    "exp27b_jcache_L18.pt": "e2a03bc36b199695018171d0f0eab92ee139be83d0f0792a558cefa15f852d63",
    "exp27b_jcache_L19.pt": "0b31baff7dbc6e38c525d10d803ed9cc313774c6c8460cf1290a9ab9f4df82ae",
    "exp27b_jcache_L22.pt": "36ed09ebc49dd97bc7b8e0a8edd1c8d75567bb6431c4eea33c908ccfb7295a84",
    "exp27b_jcache_L23.pt": "af12505131be72b424515759c054646ea4526d94822287cbe7e8c50b7db46a43",
    "exp27b_pool_ids_L23.pt": "ded634f7d16cf33a9bb565de5cbc3faa288cfc90b15cdfa3e42365695fec4e03",
}
LK4_EXPECT = {"total": 90, "wiki": 8, "math": 41, "code": 36, "chat": 5}  # гейт S4
# S8-референс (results/AT2b_diag/AT2b_diag.json, c1.stationary_pooled_cd;
# sha артефакта 617f971e…; full float64 — ТЗ §7 S8)
S8_REF = {
    "L6":  {"mean": 0.2925798086059347,  "CI": [0.27130855793557396, 0.31440423884281526]},
    "L12": {"mean": 0.23564059495562337, "CI": [0.20970514751943972, 0.2651264985461837]},
    "L18": {"mean": 0.3448696221129545,  "CI": [0.3017648574350732, 0.3866173695844338]},
    "L22": {"mean": 0.3610377648577131,  "CI": [0.3398584447392122, 0.3841862938470581]},
}
S8_TOL = 1e-6
DIAG_SHA = "617f971e12bd25f6e71f5bca258fb05f792d2c0c43d78798c1fa4664a09a2326"

_LOG = []
def logline(s):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    _LOG.append(line)

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

def tstage(run, t, name, extra=None):
    m = {"stage": name, "stage_s": round(float(t), 1)}
    if extra:
        m.update(extra)
    tlog(run, t, m)

# ------------------------------------------------------------------- данные
def field_to_array(fld):
    """dict {(dom,si,t): vec} -> [32, 128, ...] (domain-major; как в AT2b.py/диагностике)."""
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
    """Канон (experiments/exp64_at_battery.py:23-30): 4-грамма предсказывает СЛЕДУЮЩИЙ токен.
    (дословно AT2b.py)"""
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
    """K-ahead (k >= 1): i + k <= 127, иначе промах (версионная нота пре-рега).
    (дословно AT2b.py)"""
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
    """forward i -> cache t = 127 - i. (дословно AT2b.py)"""
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
    """lift v = JR^T(P^{1/2} w)/sqrt(lam) + QR-реортонормализация (как в r1/диагностике)."""
    V = np.zeros((D, len(idxs)), dtype=np.float32)
    for j, i in enumerate(idxs):
        if lam[i] > 1e-12:
            V[:, j] = jrt @ (sp * w[:, i]) / np.sqrt(lam[i])
    return _qr(V)

# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    if SMOKE:
        os.makedirs(SMOKE_DIR, exist_ok=True)
    logline(f"AT2c (T2.1c) start {datetime.now().isoformat()} "
            f"py={sys.version.split()[0]} torch={torch.__version__} "
            f"smoke={SMOKE} prerun={PRERUN} armswap={ARMSWAP}")

    tr = None
    tr_done = [False]
    if MAIN_RUN:
        tr_cfg = {
            "started_at": datetime.now().isoformat(),
            "revision": REVISION,
            "script_sha": sha256_file(os.path.abspath(__file__)),
            "seed": SEED, "device": "cpu",
            "model": "Qwen/Qwen3.5-0.8B (tied E, U128 readout)",
            "log": os.path.abspath(LOG_PATH),
            "layers": list(LAYS), "k": list(KS), "r": R, "alpha": ALPHA,
            "boot_b": BOOT_B, "n_docs": N_DOCS_RUN, "delta_min": DELTA_MIN,
        }   # r_final неизвестен на t=0 — он в stage-точке subspaces (ТЗ S10)
        tr = trackio_start(f"at2c_t21c_{REVISION}", tr_cfg)
        tlog(tr, 0, {"started": 1.0})
        logline(f"trackio: run at2c_t21c_{REVISION} init ok")

        def _trackio_on_exit():
            if not tr_done[0]:
                try:
                    # лог пишем ПЕРВЫМ (до финальной точки, чтобы log_md5 имел смысл)
                    logline("CRASH: unhandled exception / abort (fin-точка не записана)")
                    with open(LOG_PATH, "w") as f:
                        f.write("\n".join(_LOG) + "\n")
                    tlog(tr, time.time() - t0, {
                        "finished_at": datetime.now().isoformat(), "status": "crash",
                        "error": "unhandled exception / abort (fin-точка не записана)",
                        "log_md5": hashlib.md5(open(LOG_PATH, "rb").read()).hexdigest()})
                    tr.finish()
                except Exception:
                    pass
        atexit.register(_trackio_on_exit)

    gates = {}

    # ================= phase: load ==========================================
    tstage(tr, time.time() - t0, "load")
    # --- S1: sha ------------------------------------------------------------
    for fn, expect in SHA_FILES.items():
        got = sha256_file(os.path.join(CACHE_DIR, fn))
        if got != expect:
            raise SystemExit(f"S1 FAIL: sha {fn} = {got} != {expect}")
    gates["S1_sha"] = True
    logline("S1 sha-гейт: 7/7 OK")

    # --- загрузка L23 + pool + E + z-маршрут --------------------------------
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
    # S3: ориентация (exploratory; пул 32 дока x 127 из r1 — не пререгистрирован)
    idx_next = torch.stack([torch.from_numpy(toks[dom][si][::-1][:-1].copy())
                            for di, dom in enumerate(DOMAINS)
                            for si in range(N_PER_DOM)])     # [32,127]
    hi = p_soft[:, 1:128, :].gather(-1, idx_next.unsqueeze(-1)).squeeze(-1)
    lo = p_soft[:, :127, :].flip(1).gather(-1, idx_next.unsqueeze(-1)).squeeze(-1)
    mean_hi = float(hi.mean())
    top1_hi = float((p_soft[:, 1:128, :].argmax(-1) == idx_next).float().mean())
    mean_lo = float(lo.mean())
    top1_lo = float((p_soft[:, :127, :].flip(1).argmax(-1) == idx_next).float().mean())
    gates["S3_orient"] = bool(mean_hi > 0.1 and top1_hi > 0.1
                              and mean_lo < 0.05 and top1_lo < 0.05)
    logline(f"S3 ориентация (exploratory): hi {mean_hi:.3f}/{top1_hi:.3f} "
            f"lo {mean_lo:.3f}/{top1_lo:.3f} {'OK' if gates['S3_orient'] else 'FAIL'}")
    if not gates["S3_orient"]:
        raise SystemExit("S3 FAIL")
    z_U128 = z_full[:, :, U128_t].numpy()               # [32,128,128] float32
    del z_full, p_soft, p_u128, hi, lo, kl_doc, idx_next, E

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
        logline(f"S4 lk4 (smoke): total {lk4_tot} — ожидаемые числа не гейтятся")
    else:
        gates["S4_lk4"] = (lk4_tot == LK4_EXPECT["total"] and
                           all(lk4_dom[d] == LK4_EXPECT[d] for d in DOMAINS))
        logline(f"S4 lk4: total {lk4_tot}/4096 per-домен "
                f"{{ {', '.join(f'{d} {lk4_dom[d]}' for d in DOMAINS)} }} "
                f"{'OK' if gates['S4_lk4'] else 'FAIL (ожид. %s)' % LK4_EXPECT}")
    if not gates["S4_lk4"]:
        raise SystemExit("S4 FAIL")

    # rng_base: ОДИН поток (SEED=0): сначала G (1024x32), затем uniform (ТЗ §2)
    rng_base = np.random.default_rng(SEED)
    G32 = rng_base.standard_normal((D, R))              # Gaussian 1024x32
    UVEC = rng_base.standard_normal(D)
    UVEC /= np.linalg.norm(UVEC)                        # одна глобальная случайная прямая
    HAAR32 = _qr(G32)                                   # для d_eff (r=32)
    UNIF = UVEC[:, None].astype(np.float32)             # [1024, 1]

    # ================= phase: subspaces pass 1 (d_eff, rematching) =========
    tstage(tr, time.time() - t0, "subspaces",
           {"phase": "pass1_deff", "note": "per (L,doc): t=0, R=32, P-взвешенный Грамиан"})
    ew_cache = {}
    deff = {}
    for L in LAYS:
        ck = load_layer(L, ("H", "JR", "P128"))
        Ht = ck["H"]; JRt = ck["JR"]; Pt = ck["P128"]
        ew_cache[L] = []
        for d in range(N_DOCS_RUN):
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
        # d_eff per (L,doc) (J-базисы — t=0 как репрезентативный, как в T2.1b)
        for d in range(N_DOCS_RUN):
            lam0, w0, jrt0, sp0 = ew_cache[L][d][0]
            Vtop0 = _lift_arr(lam0, w0, jrt0, sp0, range(128 - R, 128))
            Vweak0 = _lift_arr(lam0, w0, jrt0, sp0, range(0, R))
            Hdoc = Ht[d]
            Hc = (Hdoc.double() - Hdoc.double().mean(0, keepdim=True))
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            Vpca32 = Vh.T[:, :R].numpy().astype(np.float32)
            deff[(L, d)] = {
                "top": d_eff_of(Vtop0, Hdoc), "weak": d_eff_of(Vweak0, Hdoc),
                "pca": d_eff_of(Vpca32, Hdoc), "haar": d_eff_of(HAAR32, Hdoc)}
        del ck, JRt, Pt, Ht   # память: пер слой (как диагностика; pass2 перегружает H)
        logline(f"pass1 L{L}: eigh per (d,t) + d_eff done")
    ratios = [max(v.values()) / min(v.values()) for v in deff.values()]
    r_final = R
    rematched = False
    if ARMSWAP:
        r_final = 7      # конфигурация main-прогона (S5a): r фикс, rematching не нужен
    elif max(ratios) > 1.2:
        r_final = int(round(min(min(v.values()) for v in deff.values())))
        rematched = True
        logline(f"d_eff rematching: max/min = {max(ratios):.2f} > 1.2 -> r = {r_final}")
    # S5a: r_final == 7 (жёсткий гейт; smoke — только отчёт)
    s5a_ok = bool(r_final == 7)
    if SMOKE:
        logline(f"S5a r_final = {r_final} (smoke: отчёт, не гейт; ожидание main = 7)")
    else:
        gates["S5a_r_final"] = s5a_ok
        logline(f"S5a r_final = {r_final} (rematched={rematched}) -> "
                f"{'OK' if s5a_ok else 'FAIL (ожидание 7)'}")
        if not s5a_ok:
            raise SystemExit("S5a FAIL: r_final != 7 — базисы не воспроизведены — STOP")

    # ================= phase: subspaces pass 2 (stationary basis) ==========
    tstage(tr, time.time() - t0, "subspaces",
           {"phase": "pass2_stationary", "r_final": r_final, "rematched": rematched})
    HAAR = _qr(G32[:, :r_final])                        # true Haar: QR(G), глобальный
    del G32, HAAR32
    ARMS_RUN = ({L: ("top", "weak") for L in LAYS} if (PRERUN or ARMSWAP)
                else {L: ARM_MAIN for L in LAYS})
    proj = {}       # (L,d,arm) -> [128, r_arm] (t-центрированный)
    s5b = {}
    anchor = {}
    c1b = {}
    r_weak_eff = {}
    for L in LAYS:
        ck = load_layer(L, ("H", "JR"))
        Ht = ck["H"]; JRt = ck["JR"]
        # СТАЦИОНАРНЫЙ per-layer J-базис (ТОЧНО как диагностика C1c; ТЗ §2)
        Jagg = JRt.mean(axis=(0, 1)).double()           # [128, 1024]: mean_{d,t} J_{d,t}
        del JRt
        Gagg = Jagg.T @ Jagg                            # [1024,1024], rank <= 128
        lam, w = torch.linalg.eigh(Gagg)
        order = torch.argsort(lam, descending=True)
        lam_np = lam.cpu().numpy()                      # по возрастанию
        w_np = w.cpu().numpy()
        order_np = order.cpu().numpy()
        lammax = float(lam_np.max())
        nz = np.where(lam_np > 1e-12 * lammax)[0]      # ненулевые индексы (возр.)
        top_idx = order_np[:r_final]                   # сильнейшие (убыв.)
        weak_idx = nz[:min(r_final, len(nz))]          # слабейшие НЕ-нулевые (weak-tail)
        weakmid_idx = order_np[r_final:2 * r_final]    # позиции r..2r (C1c конвенция)
        Vtop_a = w_np[:, top_idx].astype(np.float32)
        Vweak_a = w_np[:, weak_idx].astype(np.float32)
        Vweakmid_a = w_np[:, weakmid_idx].astype(np.float32)
        r_weak_eff[f"L{L}"] = int(len(weak_idx))
        # S5b: eig-закрытость Gagg (21 вектор per слой, f64 — ТОЧНО по ТЗ §7:
        # проверка на F64-векторах eigh, до f32-каста для проекций)
        Gagg_np = Gagg.cpu().numpy()
        worst = 0.0
        for Va in (w_np[:, top_idx], w_np[:, weak_idx], w_np[:, weakmid_idx]):
            for j in range(Va.shape[1]):
                vv = Va[:, j].astype(np.float64)
                Gv = Gagg_np @ vv
                lam_j = float(vv @ Gv)                  # v нормирован (eigh)
                resid = float(np.linalg.norm(Gv - lam_j * vv)) / (abs(lam_j) + 1e-30)
                worst = max(worst, resid)
        s5b[f"L{L}"] = float(worst)
        del Gagg_np
        H_np = Ht.numpy().astype(np.float32)
        # S8-якорь (точная конфигурация диагностики C1c; только 32 дока)
        if N_DOCS_RUN == N_DOCS:
            anchor[f"L{L}"] = s8_anchor_block(Vtop_a, Vweakmid_a, H_np, z_U128, r_final)
        # C1b-report (локальные per-(l,d,t) базисы; только 32 дока; smoke — skip)
        if N_DOCS_RUN == N_DOCS:
            VJ_local = {}
            for d in range(N_DOCS_RUN):
                for t in range(S_LEN):
                    lam_l, w_l, jrt, sp = ew_cache[L][d][t]
                    top_i = np.argsort(lam_l)[::-1][:r_final]
                    n_nz = int((lam_l > 1e-12).sum())
                    weak_i = np.argsort(lam_l)[:min(r_final, n_nz)]
                    VJ_local[(d, t)] = {
                        "top": _lift_arr(lam_l, w_l, jrt, sp, top_i),
                        "weak": _lift_arr(lam_l, w_l, jrt, sp, weak_i)}
            c1b[f"L{L}"] = c1b_block(VJ_local, H_np, z_U128)
            del VJ_local
        del ew_cache[L]
        # Vpca per (l,doc) + проекции (один пайплайн для ВСЕХ арм: H @ V -> t-центр)
        for d in range(N_DOCS_RUN):
            Hdoc = Ht[d]
            Hc = (Hdoc.double() - Hdoc.double().mean(0, keepdim=True))
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            Vpca_d = Vh.T[:, :r_final].numpy().astype(np.float32)
            Hd = H_np[d]
            proj[(L, d, "top")] = Hd @ Vtop_a
            proj[(L, d, "weak")] = Hd @ Vweak_a
            proj[(L, d, "weak_mid")] = Hd @ Vweakmid_a
            proj[(L, d, "pca")] = Hd @ Vpca_d
            proj[(L, d, "haar")] = Hd @ HAAR
            proj[(L, d, "uniform")] = Hd @ UNIF
            for arm in ARMS_RUN[L]:
                proj[(L, d, arm)] -= proj[(L, d, arm)].mean(0, keepdims=True)
        del ck, Ht, H_np, Jagg, Gagg, lam, w, order
        logline(f"pass2 L{L}: stationary basis (top/weak-tail/weak_mid) + S5b + "
                f"{'anchor' if N_DOCS_RUN == N_DOCS else 'no-anchor'} + proj done")
    del UVEC   # UNIF сохранён — нужен L23-проекции
    # S5b гейт
    s5b_ok = all(v < 1e-9 for v in s5b.values())
    gates["S5b_eig"] = bool(s5b_ok)
    logline(f"S5b eig-закрытость Gagg: max per L = "
            f"{{ {', '.join(f'{k}: {v:.3e}' for k, v in s5b.items())} }} -> "
            f"{'OK' if s5b_ok else 'FAIL'}")
    if not s5b_ok:
        raise SystemExit("S5b FAIL")
    # S8: сверка с диагностикой (32 дока; main — гейт; prerun/armswap — отчёт)
    s8 = None
    if N_DOCS_RUN == N_DOCS:
        s8 = {}
        for L in LAYS:
            a = anchor[f"L{L}"]
            ref = S8_REF[f"L{L}"]
            s8[f"L{L}"] = {
                "got_mean": float(a["mean_top_minus_weak"]),
                "got_ci": a["CI_top_minus_weak"],
                "got_cos_median_top": float(a["cos_median_top"]),
                "got_cos_median_weak": float(a["cos_median_weak"]),
                "ref_mean": ref["mean"], "ref_ci": ref["CI"],
                "max_diff": float(max(abs(a["CI_top_minus_weak"][0] - ref["CI"][0]),
                                      abs(a["CI_top_minus_weak"][1] - ref["CI"][1]),
                                      abs(a["mean_top_minus_weak"] - ref["mean"])))}
        s8_ok = all(v["max_diff"] <= S8_TOL for v in s8.values())
        logline("S8 diagnostic anchor: " + " ".join(
            f"L{L} max|diff|={s8[f'L{L}']['max_diff']:.3e}" for L in LAYS)
            + f" (tol {S8_TOL}) -> {'OK' if s8_ok else 'FAIL'}")
        if N_DOCS_RUN == N_DOCS and not s8_ok:
            raise SystemExit("S8 FAIL: T2.1c не является корректным расширением "
                             "просмотренной диагностики — STOP")
        gates["S8_diag_anchor"] = bool(s8_ok)
    else:
        gates["S8_diag_anchor"] = "skipped (smoke: pool 2 != 32)"
    # L23-контроль (pca/haar/uniform; main + smoke)
    if MAIN_RUN or SMOKE:
        H23_np = H23.numpy().astype(np.float32)
        for d in range(N_DOCS_RUN):
            Hc23 = H23[d].double() - H23[d].double().mean(0, keepdim=True)
            _, _, Vh23 = torch.linalg.svd(Hc23, full_matrices=False)
            Vpca23 = Vh23.T[:, :r_final].numpy().astype(np.float32)
            Hd23 = H23_np[d]
            proj[(L_CTRL, d, "pca")] = Hd23 @ Vpca23
            proj[(L_CTRL, d, "haar")] = Hd23 @ HAAR
            proj[(L_CTRL, d, "uniform")] = Hd23 @ UNIF
            for arm in ARM_L23:
                proj[(L_CTRL, d, arm)] -= proj[(L_CTRL, d, arm)].mean(0, keepdims=True)
        del H23, H23_np

    # ================= phase: ridge (centered-цель, C2-фикс) ================
    tstage(tr, time.time() - t0, "ridge")
    SUBS_RUN = {L: ARMS_RUN[L] for L in LAYS}
    L_RUN = list(LAYS) + ([L_CTRL] if (MAIN_RUN or SMOKE) else [])
    if MAIN_RUN or SMOKE:
        SUBS_RUN[L_CTRL] = ARM_L23
    cos_doc = {}   # (L,k,arm) -> [N_DOCS_RUN, 128-k] (cos_c per позиция)
    r2doc = {}     # (L,k,arm) -> [N_DOCS_RUN] (R2_c per doc)
    binfo = {}     # (L,k,arm) -> [{b_over_W, dcos_mean} per doc] (S6c, TZ §11.2 r6)
    for L in L_RUN:
        for k in KS:
            n = S_LEN - k
            y_all = z_U128[:, :n, :]
            yc_all = y_all - y_all.mean(1, keepdims=True)   # centered-цель per (d,k)
            for arm in SUBS_RUN[L]:
                r_arm = proj[(L, 0, arm)].shape[1]
                rows = []
                r2rows = []
                for d in range(N_DOCS_RUN):
                    X = proj[(L, d, arm)][k:]
                    Xf = np.vstack([proj[(L, d2, arm)][k:]
                                    for d2 in range(N_DOCS_RUN) if d2 != d])
                    yf = np.vstack([yc_all[d2] for d2 in range(N_DOCS_RUN) if d2 != d])
                    Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
                    A = Xb.T @ Xb + ALPHA * np.eye(r_arm + 1)
                    A[-1, -1] -= ALPHA                 # no-penalty на intercept (R2E-1)
                    W = np.linalg.solve(A, Xb.T @ yf)
                    pred = np.hstack([X, np.ones((X.shape[0], 1))]) @ W
                    yc = yc_all[d]
                    cosv = (pred * yc).sum(1) / (
                        np.linalg.norm(pred, axis=1) * np.linalg.norm(yc, axis=1) + 1e-12)
                    rows.append(cosv)
                    r2rows.append(1.0 - float((pred ** 2).sum()) / float((yc ** 2).sum()))
                    # S6c: cos-вклад b̂ = |cos(ŷ+b̂·1, y_c) − cos(ŷ, y_c)| per позиция
                    pred_nob = X @ W[:r_arm]
                    den = np.linalg.norm(yc, axis=1) + 1e-12
                    cos_nob = (pred_nob * yc).sum(1) / (np.linalg.norm(pred_nob, axis=1) * den)
                    dcos_vec = np.abs(cosv - cos_nob)
                    wpos = (yc ** 2).sum(1)   # ‖y_c(d,t)‖² — те же веса, что метрика (C3)
                    binfo.setdefault((L, k, arm), []).append({
                        "b_over_W": float(np.abs(W[-1]).max() / (np.abs(W).max() + 1e-30)),
                        "dcos_wmean": float((dcos_vec * wpos).sum() / (wpos.sum() + 1e-30)),
                        "dcos_wdot": float((dcos_vec * wpos).sum()),
                        "dcos_wsum": float(wpos.sum())})
                cos_doc[(L, k, arm)] = np.stack(rows)
                r2doc[(L, k, arm)] = np.array(r2rows)
        logline(f"ridge L{L} done")

    # ================= S6/S6b gates =========================================
    s6_ok = True
    s6_worst = {}
    for k in KS:
        n = S_LEN - k
        yc_all = z_U128[:, :n, :] - z_U128[:, :n, :].mean(1, keepdims=True)
        worst_w = 0.0
        worst_i = 0.0
        for d in range(N_DOCS_RUN):
            yf = np.vstack([yc_all[d2] for d2 in range(N_DOCS_RUN) if d2 != d])
            Xf = np.zeros(((N_DOCS_RUN - 1) * n, r_final), dtype=np.float64)
            Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
            A = Xb.T @ Xb + ALPHA * np.eye(r_final + 1)
            A[-1, -1] -= ALPHA
            W = np.linalg.solve(A, Xb.T @ yf)
            wr = float(np.abs(W[:r_final]).max()) / (float(np.abs(W).max()) + 1e-30)
            im_j = yf.astype(np.float64).mean(0)
            ir = float(np.abs(W[-1] - im_j).max()) / (float(np.abs(im_j).max()) + 1e-30)
            worst_w = max(worst_w, wr)
            worst_i = max(worst_i, ir)
        ok_k = (worst_w < 1e-10) and (worst_i < 1e-10)
        s6_worst[f"k{k}"] = {"W_X_rel_max": float(worst_w),
                             "intercept_rel_max": float(worst_i), "ok": bool(ok_k)}
        s6_ok = s6_ok and ok_k
    gates["S6_pzero"] = bool(s6_ok)
    logline(f"S6 P-zero (weights, centered): {s6_ok}")
    if not s6_ok:
        raise SystemExit("S6 FAIL")
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
    # S6c (TZ §11.2 designed-сценарий, r6 + r7-уточнение статистики конфаунда):
    # centered-цель + cos-вклад intercept. Гейт (J-армы, L ∈ LAYS): per (L,k,arm)
    # max_d weighted-mean |dcos| ≤ 0.02 И попарный confounding (J_ARMS, per-arm
    # weighted-wmean — ТЗ r7) ≤ DELTA_MIN. Uniform/L23 — report-only
    # (uniform: 1-D фича, intercept механически велик, не в вердикт; L23 — контроль).
    # Веса-отношение |b̂|/max|W| — report-only (arm-зависимый proxy, не гейт).
    s6c_ok = True
    s6c = {}
    for k in KS:
        yc_k = z_U128[:, :S_LEN - k, :] - z_U128[:, :S_LEN - k, :].mean(1, keepdims=True)
        ybar_max = float(np.abs(yc_k.mean(1)).max())
        s6c[f"ybar_max_k{k}"] = ybar_max
        if ybar_max >= 1e-5:   # ТЗ: «< 1e-5» (строгий)
            s6c_ok = False
    for (L, k, arm), bi in sorted(binfo.items()):
        dcos_w = max(m["dcos_wmean"] for m in bi)   # max_d weighted-mean_t (ТЗ §11.2)
        b_over_W = max(m["b_over_W"] for m in bi)
        gated = (L in LAYS) and (arm != "uniform")   # J-армы кроме uniform (TZ: uniform — report-only)
        ok = (dcos_w <= 0.02) if gated else None
        if gated and not ok:
            s6c_ok = False
        s6c[f"{L}|k{k}|{arm}"] = {"dcos_max_d_wmean": float(dcos_w),
                                  "b_over_W_max": float(b_over_W), "ok": ok}
    # попарный confounding (J_ARMS) per (L,k) — ТЗ r7: разность per-arm
    # weighted-wmean dcos (количество, конфаундящее попарные wcos-контрасты
    # вердикта). Гейт ≤ DELTA_MIN. max_d-вариант — report-only (грубая
    # верхняя граница, не конфаунд контраста).
    s6c_conf = {}
    s6c_conf_maxd = {}
    for L in LAYS:
        for k in KS:
            avail = [a for a in J_ARMS if f"{L}|k{k}|{a}" in s6c]
            if len(avail) < 2:
                continue
            wmeans = {}
            for a in avail:
                bi = binfo[(L, k, a)]
                num = sum(m["dcos_wdot"] for m in bi)
                den = sum(m["dcos_wsum"] for m in bi)
                wmeans[a] = num / (den + 1e-30)
            s6c_conf[f"{L}|k{k}"] = float(max(
                abs(wmeans[a] - wmeans[b])
                for i, a in enumerate(avail) for b in avail[i + 1:]))
            ms = {a: s6c[f"{L}|k{k}|{a}"]["dcos_max_d_wmean"] for a in avail}
            s6c_conf_maxd[f"{L}|k{k}"] = float(max(
                abs(ms[a] - ms[b])
                for i, a in enumerate(avail) for b in avail[i + 1:]))
    s6c["confounding"] = {"per_lk": s6c_conf, "max": float(max(s6c_conf.values())),
                          "maxd_pairwise_report": s6c_conf_maxd,
                          "maxd_pairwise_max": float(max(s6c_conf_maxd.values()))}
    if s6c["confounding"]["max"] > DELTA_MIN:
        s6c_ok = False
    gates["S6c_centering"] = bool(s6c_ok)
    ybar_str = ", ".join(f"k{k}: {s6c[f'ybar_max_k{k}']:.2e}" for k in KS)
    gated_cells = [k_ for k_ in s6c
                   if isinstance(s6c[k_], dict) and "ok" in s6c[k_]
                   and s6c[k_]["ok"] is not None]
    dcos_worst = max(s6c[k_]["dcos_max_d_wmean"] for k_ in gated_cells)
    logline(f"S6c centering: ybar_max per k = {{ {ybar_str} }}; dcos worst (gated J) = "
            f"{dcos_worst:.2e} (≤ 2e-2); pairwise confounding max (wmean) = "
            f"{s6c['confounding']['max']:.2e} (≤ DELTA_MIN={DELTA_MIN}); "
            f"maxd-pairwise report = {s6c['confounding']['maxd_pairwise_max']:.2e} -> "
            f"{'OK' if s6c_ok else 'FAIL'}")
    if not s6c_ok:
        raise SystemExit("S6c FAIL: centering/intercept claim сломан — STOP")

    # ================= weights + agg (primary = weighted cos) ==============
    w_all = {}      # k -> [N_DOCS_RUN, 128-k] = ||y_c(d,t)||^2 (общиe для армов)
    for k in KS:
        yc_k = z_U128[:N_DOCS_RUN, :S_LEN - k, :] - z_U128[:N_DOCS_RUN, :S_LEN - k, :].mean(1, keepdims=True)
        w_all[k] = (yc_k ** 2).sum(2)
    agg = {}
    for (L, k, arm), rows in cos_doc.items():
        w = w_all[k]
        allc = np.concatenate(rows)
        agg[(L, k, arm)] = {
            "wcos": float((w * rows).sum() / w.sum()),        # первичная (ТЗ §3)
            "cos_median": float(np.median(allc)),             # вторичная
            "cos_mean": float(allc.mean()),
            "R2_mean": float(np.mean(r2doc[(L, k, arm)]))}

    # ================= PRE-RUN branch (TZ §11.0) ===========================
    if PRERUN:
        w0 = w_all[0]
        pre = {}
        for L in LAYS:
            da = np.stack([cos_doc[(L, 0, "top")][d] - cos_doc[(L, 0, "weak")][d]
                           for d in range(N_DOCS_RUN)])
            rngb = np.random.default_rng(SEED)
            ciw = ci_w(da, w0, rngb)
            ciu = ci_of(da, rngb)
            pre[f"L{L}"] = {
                "wcos_top": agg[(L, 0, "top")]["wcos"],
                "wcos_weak": agg[(L, 0, "weak")]["wcos"],
                "weighted_CI_top_minus_weak": ciw,
                "unweighted_CI_top_minus_weak": ciu,
                "pass": bool(ciw[0] > 0.0)}
        all_ok = all(v["pass"] for v in pre.values())
        out = {
            "what": "AT2c pre-run замер (TZ §11.0) — НЕ гейт; результат — в results-документ",
            "assembled_at": datetime.now().isoformat(),
            "config": {"layers": list(LAYS), "k": [0], "arms": ["top", "weak (weak-tail)"],
                       "n_docs": N_DOCS_RUN, "boot_b": BOOT_B, "seed": SEED,
                       "r_final": r_final, "rematched": rematched,
                       "metric": "weighted cos (primary)", "alpha": ALPHA,
                       "script_sha": sha256_file(os.path.abspath(__file__))},
            "s5a_r_final": {"r_final": r_final, "rematched": rematched, "ok": bool(s5a_ok)},
            "s8_anchor": ({"per_layer": s8, "ok": bool(s8_ok)} if s8 is not None else "skipped"),
            "control_top_vs_weaktail_k0": pre,
            "control_ok_all_l": bool(all_ok),
            "note": ("Pass/fail фиксируется до лонча main-прогона; fail на любом слое — "
                     "решение о запуске принимает супервизор (ТЗ §11.0)."),
            "wall_sec": round(time.time() - t0, 1)}
        raw = json.dumps(out, ensure_ascii=False, indent=1)
        if "NaN" in raw or "Infinity" in raw:
            raise SystemExit("prerun artifact: NaN/Infinity — STOP")
        with open(ART, "w") as f:
            f.write(raw)
        logline(f"PRE-RUN DONE: control_ok_all_l={all_ok} wall={time.time() - t0:.1f}s")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(_LOG) + "\n")
        return

    # ================= ARM-SWAP branch (TZ §11) =============================
    if ARMSWAP:
        L = 6   # TЗ: L6, k=0, 32 дока, r=7 фикс (конфиг main-прогона)
        # swapped control: CI(weak − top) — обязан дать lower < 0 (иначе логическая ошибка)
        da = np.stack([cos_doc[(L, 0, "weak")][d] - cos_doc[(L, 0, "top")][d]
                       for d in range(N_DOCS_RUN)])
        rngb = np.random.default_rng(SEED)
        ciw = ci_w(da, w_all[0], rngb)
        ciu = ci_of(da, rngb)
        ok_swap = bool(ciw[0] < 0.0)
        out = {
            "what": "AT2c arm-swap smoke (TZ §11): swapped control обязан FAIL (lower < 0)",
            "assembled_at": datetime.now().isoformat(),
            "config": {"layer": L, "k": 0, "n_docs": N_DOCS_RUN, "boot_b": BOOT_B,
                       "seed": SEED, "r_final": r_final, "metric": "weighted cos (primary)",
                       "script_sha": sha256_file(os.path.abspath(__file__))},
            "swapped_control_CI_top_minus_weak": ciw,
            "swapped_control_CI_unweighted": ciu,
            "s8_anchor_L6": s8.get('L6'),   # NIT-1 (код-ревью r2): якорь в артефакт
            "ok_swap": ok_swap,
            "wall_sec": round(time.time() - t0, 1)}
        raw = json.dumps(out, ensure_ascii=False, indent=1)
        if "NaN" in raw or "Infinity" in raw:
            raise SystemExit("armswap artifact: NaN/Infinity — STOP")
        with open(ART, "w") as f:
            f.write(raw)
        if not ok_swap:
            logline(f"ARMSWAP FAIL: swapped control lower={ciw[0]:.4f} >= 0 — "
                    f"ЛОГИЧЕСКАЯ ОШИБКА гейта (T2 не запускается)")
            with open(LOG_PATH, "w") as f:
                f.write("\n".join(_LOG) + "\n")
            raise SystemExit("ARMSWAP_FAIL")
        logline(f"ARMSWAP_DONE: swapped control lower={ciw[0]:.4f} < 0 — "
                f"логика гейта проверена (wall={time.time() - t0:.1f}s)")
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(_LOG) + "\n")
        return

    # ================= lk4-отчёты (наследованы от T2.1b, TZ §6) ============
    tstage(tr, time.time() - t0, "lk4")
    rngb = np.random.default_rng(SEED)   # ОДИН bootstrap-поток на все CI (порядок draws — как в коде)
    u128_set = {int(v): j for j, v in enumerate(U128)}
    n_hits_per_k = {}
    cov_doc = {k: [] for k in KS}   # per (k, d) coverage или None (нет hit-позиций)
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
    # one-hot vs y_c cos (report-only, non_informative; дословно T2.1b)
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
    # N8 (report-only, F4): cos_c per (l,k,arm) на lk4-hit (k=0 канон) позициях
    # (ТЗ §6: все 6 main-армов; r1-артефакт был с J_ARMS=4 — исправлено post-run)
    n8 = {}
    for L in LAYS:
        for k in KS:
            for arm in ARM_MAIN:
                rows = []
                for d in range(N_DOCS_RUN):
                    tsel = np.arange(k, S_LEN)[lk4_mask[d][k:]]
                    if len(tsel):
                        rows.append(cos_doc[(L, k, arm)][d][tsel - k])
                n8[f"{L}|k{k}|{arm}"] = (float(np.median(np.concatenate(rows)))
                                         if rows else None)
    cov_out = {}
    for k in KS:
        vals = [v for v in cov_doc[k] if v is not None]
        cov_out[f"k{k}"] = {
            "value": float(np.mean(vals)) if vals else 0.0,
            "n_hits": int(n_hits_per_k[k]),
            "n_docs_with_hits": len(vals),
            "CI": (ci_of_scalars(vals, rngb) if vals else None)}
    lk4_out = {
        "hit_total": int(lk4_tot),
        "per_dom": {d: int(lk4_dom[d]) for d in DOMAINS},
        "n_hits_per_k": {f"k{k}": int(n_hits_per_k[k]) for k in KS},
        "onehot_med": onehot_med,
        "onehot_non_informative": True,
        "coverage": cov_out}

    # ================= bootstrap + verdict (weighted, TZ §4-§5) ============
    tstage(tr, time.time() - t0, "verdict")
    # 1) control CI (k=0): top − weak per слой (weighted — первичный; unweighted — вторичный)
    ctrl_wci = {}
    ctrl_ci = {}
    for L in LAYS:
        da = np.stack([cos_doc[(L, 0, "top")][d] - cos_doc[(L, 0, "weak")][d]
                       for d in range(N_DOCS_RUN)])
        ctrl_wci[L] = ci_w(da, w_all[0], rngb)
        ctrl_ci[L] = ci_of(da, rngb)
    control_ok = all(ctrl_wci[L][0] > 0.0 for L in LAYS)
    # 2) k>=1: weak vs max(pca,haar) + pairwise (6 tree + 2 report)
    boot = {}
    for L in LAYS:
        for k in KS:
            if k == 0:
                continue
            w = w_all[k]
            wh = np.stack([cos_doc[(L, k, "weak")][d] - np.maximum(
                cos_doc[(L, k, "pca")][d], cos_doc[(L, k, "haar")][d])
                for d in range(N_DOCS_RUN)])
            pcis = {}
            for a_, b_ in PAIRS + REPORT_PAIRS:
                da = np.stack([cos_doc[(L, k, a_)][d] - cos_doc[(L, k, b_)][d]
                               for d in range(N_DOCS_RUN)])
                pcis[f"{a_}_vs_{b_}"] = ci_w(da, w, rngb)
            boot[(L, k)] = {
                "weak_vs_maxPCAHaar_wCI": ci_w(wh, w, rngb),
                "weak_vs_maxPCAHaar_wmean": float((w * wh).sum() / w.sum()),
                "pairwise_wCI": pcis}
    # 3) floor: uniform (T2.1b конвенция: per-doc median CI; + wcos первичный)
    floor_out = {}
    for L in LAYS:
        for k in KS:
            ucos = cos_doc[(L, k, "uniform")]
            perdoc_med = [float(np.median(ucos[d])) for d in range(N_DOCS_RUN)]
            floor_out[f"{L}|k{k}"] = {
                "uniform_wcos": float((w_all[k] * ucos).sum() / w_all[k].sum()),
                "uniform_cos_median": float(np.median(np.concatenate(ucos))),
                "uniform_cos_CI": ci_of_perdoc_medians(perdoc_med, rngb)}
    # 4) delta-min калибровка
    tree_ks = tuple(k for k in (1, 2, 4, 8) if k in KS)
    max_spread = 0.0
    for L in LAYS:
        for k in tree_ks:
            ms = {s: agg[(L, k, s)]["wcos"] for s in J_ARMS}
            for a_, b_ in PAIRS:
                max_spread = max(max_spread, abs(ms[a_] - ms[b_]))
    floor_cw = max((v["uniform_cos_CI"][1] - v["uniform_cos_CI"][0])
                   for v in floor_out.values()) if floor_out else 0.0
    delta_calib = {
        "delta_min": DELTA_MIN,
        "max_pairwise_spread_Jarms_wcos": float(max_spread),
        "floor_CI_width_max": float(floor_cw),
        "delta_min_gt_spread": bool(DELTA_MIN > max_spread),
        "delta_min_gt_floor_width": bool(DELTA_MIN > floor_cw),
        "metric": "weighted cos (primary)"}

    # L23-отчёт
    l23_out = {}
    if MAIN_RUN or SMOKE:
        for k in KS:
            for arm in ARM_L23:
                w = w_all[k]
                rows = cos_doc[(L_CTRL, k, arm)]
                l23_out[f"k{k}|{arm}"] = {
                    "wcos": float((w * rows).sum() / w.sum()),
                    "cos_median": float(np.median(np.concatenate(rows)))}

    # ================= дерево вердикта (TZ §5; первичная метрика) ==========
    def win_ci(ci):
        return ci[0] > 0.0

    def contains_zero(ci):
        return ci[0] <= 0.0 <= ci[1]

    detail = {
        "DELTA_MIN": DELTA_MIN,
        "control_k0_top_minus_weak_wCI": {str(L): ctrl_wci[L] for L in LAYS},
        "control_k0_top_minus_weak_CI": {str(L): ctrl_ci[L] for L in LAYS},
        "control_ok": bool(control_ok)}
    if not control_ok:
        verdict = "SUBSPACE_CONTROL_FAIL"
        logline(f"VERDICT: {verdict} (k>=1 в артефакте, но не интерпретируется как ответ T2)")
    else:
        # (a) scratchpad: >= 2 из k=2,4,8: CI_w(weak - max(pca,haar)) выше 0 И
        #     weighted-mean diff > DELTA_MIN (все L)
        a_per_k = {}
        for k in (2, 4, 8):
            if k not in KS:
                continue
            ci_ok = all(win_ci(boot[(L, k)]["weak_vs_maxPCAHaar_wCI"]) for L in LAYS)
            wmeans = {str(L): float(boot[(L, k)]["weak_vs_maxPCAHaar_wmean"]) for L in LAYS}
            wmean_ok = all(m > DELTA_MIN for m in wmeans.values())
            a_per_k[str(k)] = {"ci_ok_all_l": bool(ci_ok),
                               "wmean_ok_all_l": bool(wmean_ok), "wmeans": wmeans}
        n_a = sum(1 for v in a_per_k.values() if v["ci_ok_all_l"] and v["wmean_ok_all_l"])
        detail["a_scratchpad"] = {"per_k": a_per_k, "n_pass": n_a}
        if n_a >= 2:
            verdict = "SCRATCHPAD"
        else:
            # (b) deadweight: max |wcos(top)|, |wcos(weak)| <= DELTA_MIN
            mtop = max(abs(agg[(L, k, "top")]["wcos"]) for L in LAYS for k in tree_ks)
            mweak = max(abs(agg[(L, k, "weak")]["wcos"]) for L in LAYS for k in tree_ks)
            b_ok = (mtop <= DELTA_MIN) and (mweak <= DELTA_MIN)
            detail["b_deadweight"] = {"max_abs_top": float(mtop),
                                      "max_abs_weak": float(mweak), "b_ok": bool(b_ok)}
            if b_ok:
                verdict = "DEADWEIGHT"
            else:
                # (c) redundant: ВСЕ tree-pairwise CI (6) содержат 0 + top >= max(остальные) по wcos на k=1
                c_ok = True
                for L in LAYS:
                    for k in tree_ks:
                        for key, ci in boot[(L, k)]["pairwise_wCI"].items():
                            if key in (f"{a_}_vs_{b_}" for a_, b_ in REPORT_PAIRS):
                                continue     # report-only пары — вне дерева (ТЗ §5)
                            if not contains_zero(ci):
                                c_ok = False
                k1ok = {}
                if 1 in KS:
                    for L in LAYS:
                        mt = agg[(L, 1, "top")]["wcos"]
                        mx = max(agg[(L, 1, s)]["wcos"] for s in ("weak", "pca", "haar"))
                        k1ok[str(L)] = bool(mt >= mx)
                c_ok = (c_ok and all(k1ok.values())) if k1ok else c_ok
                detail["c_redundant"] = {"c_ok": bool(c_ok),
                                         "k1_top_ge_max_wcos": k1ok}
                if c_ok:
                    verdict = "REDUNDANT_BASIS"
                else:
                    # (d) metric-uninformative: max попарного расхождения J-армов <= DELTA_MIN
                    maxdiff = 0.0
                    for L in LAYS:
                        for k in tree_ks:
                            ms = {s: agg[(L, k, s)]["wcos"] for s in J_ARMS}
                            for a_, b_ in PAIRS:
                                maxdiff = max(maxdiff, abs(ms[a_] - ms[b_]))
                    d_ok = maxdiff <= DELTA_MIN
                    detail["d_metric_uninformative"] = {"max_pairwise_diff": float(maxdiff),
                                                        "d_ok": bool(d_ok)}
                    verdict = "METRIC_UNINFORMATIVE" if d_ok else "NOT_RESOLVED"
    logline(f"VERDICT: {verdict} (control_ok={control_ok}, metric=weighted cos)")

    # ================= артефакт ============================================
    tstage(tr, time.time() - t0, "artifact")
    # per_pos (ключ f"{L}|d{d}|t{t}|k{k}|{arm}", t = target-позиция = ctx_t + k)
    per_pos = {}
    for (L, k, arm), rows in cos_doc.items():
        for d in range(N_DOCS_RUN):
            for t in range(S_LEN - k):
                per_pos[f"{L}|d{d}|t{t + k}|k{k}|{arm}"] = float(rows[d][t])
    pos_per_doc = sum(S_LEN - k for k in KS)
    n_arms_l = 2 if (PRERUN or ARMSWAP) else len(ARM_MAIN)
    S7_expected = (len(LAYS) * N_DOCS_RUN * pos_per_doc * n_arms_l
                   + (N_DOCS_RUN * pos_per_doc * len(ARM_L23) if (MAIN_RUN or SMOKE) else 0))
    set_eq = (set(per_pos.keys()) ==
              {f"{L}|d{d}|t{t}|k{k}|{arm}"
               for L in LAYS for d in range(N_DOCS_RUN) for k in KS
               for t in range(k, S_LEN) for arm in ARMS_RUN[L]}
              | ({f"{L_CTRL}|d{d}|t{t}|k{k}|{arm}"
                 for d in range(N_DOCS_RUN) for k in KS for t in range(k, S_LEN)
                 for arm in ARM_L23} if (MAIN_RUN or SMOKE) else set()))
    gates["S7_exact_set"] = bool(set_eq and len(per_pos) == S7_expected)
    logline(f"S7 exact-set: expected {S7_expected}, got {len(per_pos)}, set_eq={set_eq}")
    if not gates["S7_exact_set"]:
        raise SystemExit("S7 FAIL: per_pos не равен ожидаемому точному множеству — STOP")

    out = {
        "run": f"at2c_t21c_{REVISION}",
        "assembled_at": datetime.now().isoformat(),   # НЕ started_at (T2.1b NIT-2)
        "config": {
            "layers": list(LAYS), "l23": L_CTRL, "k": list(KS),
            "arms_main": list(ARM_MAIN), "arms_L23": list(ARM_L23),
            "r_final": r_final, "rematched": rematched, "r": R, "alpha": ALPHA,
            "seed": SEED, "boot_b": BOOT_B, "n_docs": N_DOCS_RUN,
            "metric": "weighted_cos primary + cos_median secondary",
            "target": "per-doc centered y_c",
            "basis": "stationary per-layer Jagg",
            "delta_min": DELTA_MIN,
            "script_sha": sha256_file(os.path.abspath(__file__)),
            "diag_sha": DIAG_SHA,
            "model": "Qwen/Qwen3.5-0.8B (tied E, U128 readout)",
            "cache_dir": CACHE_DIR, "device": "cpu",
            "thread_pinning": "OMP/OPENBLAS/MKL=4 + torch 4 (S8 bit-identity)"},
        "gates": gates,
        "s2_kl_report": kl_report,
        "s5": {"r_final": r_final, "rematched": rematched,
               "r_weak_eff": r_weak_eff,
               "gagg_closure_max_rel_resid": s5b,
               "d_eff": {f"{L}|d{d}": v for (L, d), v in deff.items()}},
        "s6_pzero": s6_worst,
        "s6c_centering": s6c,
        "s7": {"S7_expected": int(S7_expected), "len_per_pos": int(len(per_pos)),
               "set_eq": bool(set_eq), "arms_main": list(ARM_MAIN),
               "arms_L23": list(ARM_L23), "k": list(KS),
               "weak_mid_in_verdict": False, "uniform_in_verdict": False},
        "s8_anchor": {
            "per_layer": (s8 if s8 is not None else "skipped (smoke: pool 2 != 32)"),
            "ok": (bool(s8_ok) if s8 is not None else "skipped"),
            "tol": S8_TOL,
            "ref_source": f"results/AT2b_diag/AT2b_diag.json (sha {DIAG_SHA[:16]}...)",
            "config": ("stationary top vs weak_mid [r..2r desc], centered target, "
                       "full-pool LOO-ridge, unweighted mean CI, B=1000, SEED=0 "
                       "(точная конфигурация диагностики C1c)")},
        "subspaces": {
            "definition": ("top/weak/weak_mid — стационарный per-layer J-базис "
                           "(Jagg=mean_{d,t} JR; Gagg=Jagg^T Jagg; eigh; r_final после rematching); "
                           "pca — per (l,doc) SVD центрированного H; haar — QR(G32[:, :r_final]), глобальный; "
                           "uniform — одна глобальная случайная прямая (floor-референс); "
                           "*dc-армов нет (§2: DC снята centered-целью + t-центрированным X)"),
            "asymmetry": {
                "top": "per-layer (stationary)", "weak": "per-layer (stationary)",
                "weak_mid": "per-layer (stationary)", "pca": "per (l,doc)",
                "haar": "глобальный", "uniform": "глобальный (floor-референс, не в вердикт)"},
            "d_eff_asymmetry_top_vs_weak": {f"L{L}": deff[(L, 0)]
                                            for L in LAYS}},
        "metrics": {f"{L}|k{k}|{arm}": m for (L, k, arm), m in agg.items()},
        "floor_uniform": floor_out,
        "lk4": lk4_out,
        "n8": n8,
        "l23": l23_out,
        "report_local_per_lt": c1b,
        "delta_min_calibration": delta_calib,
        "bootstrap": {
            "per_k": {f"{L}|k{k}": boot[(L, k)] for L in LAYS for k in KS if (L, k) in boot},
            "floor_CI": {k_: v["uniform_cos_CI"] for k_, v in floor_out.items()},
            "lk4_coverage": cov_out,
            "control_k0_top_minus_weak_wCI": {str(L): ctrl_wci[L] for L in LAYS},
            "control_k0_top_minus_weak_CI": {str(L): ctrl_ci[L] for L in LAYS},
            "control_ok": bool(control_ok)},
        "verdict": verdict,
        "verdict_detail": detail,
        "per_pos_cos": per_pos,
        "wall_sec": round(time.time() - t0, 1)}
    raw = json.dumps(out, ensure_ascii=False, indent=1)
    if "NaN" in raw or "Infinity" in raw:
        raise SystemExit("artifact: NaN/Infinity in the JSON — STOP")
    with open(ART, "w") as f:
        f.write(raw)

    if SMOKE:
        det = {"metrics": out["metrics"], "d_eff": out["s5"]["d_eff"],
               "per_pos": per_pos, "bootstrap": out["bootstrap"],
               "verdict": verdict, "r_final": r_final}
        det_path = os.path.join(SMOKE_DIR, "AT2c_smoke_det.json")
        with open(det_path, "w") as f:
            json.dump(det, f, sort_keys=True, ensure_ascii=False)
        logline(f"smoke det (S9): {det_path} ({os.path.getsize(det_path) // 1024} KB)")

    # ---- финиш: лог ПЕРВЫМ (log_md5 в финальной trackio-точке; ТЗ S10) ----
    logline(f"AT2c (T2.1c) DONE wall={time.time() - t0:.1f}s verdict={verdict}")
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_LOG) + "\n")
    if MAIN_RUN:
        tr_done[0] = True
        tlog(tr, time.time() - t0, {
            "finished_at": datetime.now().isoformat(), "status": "ok",
            "verdict": verdict,
            "log_md5": hashlib.md5(open(LOG_PATH, "rb").read()).hexdigest()})
        tr.finish()


# ------------------------------------------------------------------- bootstrap
def ci_of(dd, rngb):
    """per-doc bootstrap (B, SEED=0): ресэмплинг доков, stat = mean по (док,t)
    (конвенция T2.1b ci_of)."""
    means = []
    for b in range(BOOT_B):
        sel = rngb.integers(0, N_DOCS_RUN, N_DOCS_RUN)
        means.append(float(dd[sel].mean()))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def ci_w(dd, w, rngb):
    """per-doc bootstrap, weighted stat: weighted mean по позициям выбранных доков
    (ТЗ §4; веса w — общие для всех армов на позиции)."""
    means = []
    for b in range(BOOT_B):
        sel = rngb.integers(0, N_DOCS_RUN, N_DOCS_RUN)
        num = float((w[sel] * dd[sel]).sum())
        den = float(w[sel].sum())
        means.append(num / den)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def ci_of_scalars(vals, rngb):
    """doc-bootstrap пер-док скаляров: ресэмплинг доков, stat = mean."""
    stats = []
    for b in range(BOOT_B):
        sel = rngb.integers(0, len(vals), len(vals))
        stats.append(float(np.mean([vals[i] for i in sel])))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

def ci_of_perdoc_medians(meds, rngb):
    """doc-bootstrap пер-док медиан: stat = медиана пер-док медиан (floor)."""
    stats = []
    for b in range(BOOT_B):
        sel = rngb.integers(0, len(meds), len(meds))
        stats.append(float(np.median([meds[i] for i in sel])))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]

# ------------------------------------------------------------------- S8 anchor
def s8_anchor_block(Vtop_a, Vweakmid_a, H_np, z_U128, r):
    """Точная конфигурация диагностики C1c (ТЗ §7 S8): stationary top vs weak_mid
    [r..2r по убыванию], centered-цель, full-pool LOO-ridge, НЕ-взвешенный mean CI,
    B=1000, SEED=0. Код — дословно из experiments/at2/AT2b_diag_c2c1.py (C1c-блок).
    Возврат: per-слой {cos_median_top/weak, mean_top_minus_weak, CI_top_minus_weak}."""
    outL = {}
    for arm, Va in (("top", Vtop_a), ("weak", Vweakmid_a)):
        Xs = np.stack([H_np[d] @ Va - (H_np[d] @ Va).mean(0, keepdims=True)
                       for d in range(N_DOCS)])      # [32,128,r]
        yf_cd = np.stack([z_U128[d] - z_U128[d].mean(0) for d in range(N_DOCS)])
        rows = []
        for d in range(N_DOCS):
            Xf = np.stack([Xs[d2] for d2 in range(N_DOCS) if d2 != d]).reshape(-1, r)
            yfc = np.vstack([yf_cd[d2] for d2 in range(N_DOCS) if d2 != d])
            Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
            A = Xb.T @ Xb + ALPHA * np.eye(r + 1)
            A[-1, -1] -= ALPHA
            W = np.linalg.solve(A, Xb.T @ yfc)
            pred = Xs[d].reshape(-1, r) @ W[:r] + W[-1]
            yc = yf_cd[d]
            rows.append((pred * yc).sum(1) / (
                np.linalg.norm(pred, axis=1) * np.linalg.norm(yc, axis=1) + 1e-12))
        outL[arm] = np.stack(rows)
    da = outL["top"] - outL["weak"]
    rngb = np.random.default_rng(SEED)
    return {"cos_median_top": float(np.median(outL["top"])),
            "cos_median_weak": float(np.median(outL["weak"])),
            "mean_top_minus_weak": float(da.mean()),
            "CI_top_minus_weak": ci_of(da, rngb)}

# ------------------------------------------------------------------- C1b report
def c1b_block(VJ_local, H_np, z_U128):
    """Локальный per-(l,t) контроль (диагностика C1b; report-only, НЕ гейт, ТЗ §6):
    локальные per-(l,d,t) J-базисы, 32 дока, per-doc LOO ridge, центрирование по
    докаМ; per слой: mean/median diff top−weak per t + frac_t_top_gt_weak."""
    diffs = []
    for t in range(S_LEN):
        Xtop = np.stack([VJ_local[(d, t)]["top"].T @ H_np[d][t] for d in range(N_DOCS)])
        Xweak = np.stack([VJ_local[(d, t)]["weak"].T @ H_np[d][t] for d in range(N_DOCS)])
        yloc = z_U128[:, t, :].copy()
        cos_top = np.empty(N_DOCS)
        cos_weak = np.empty(N_DOCS)
        for Xa, cos_d in ((Xtop, cos_top), (Xweak, cos_weak)):
            Xa = Xa - Xa.mean(0, keepdims=True)
            yc_loc = yloc - yloc.mean(0, keepdims=True)
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
        diffs.append(float(cos_top.mean() - cos_weak.mean()))
    diffs = np.array(diffs)
    return {"mean_diff_top_minus_weak": float(diffs.mean()),
            "median_diff_top_minus_weak": float(np.median(diffs)),
            "frac_t_top_gt_weak": float((diffs > 0).mean())}

if __name__ == "__main__":
    main()
