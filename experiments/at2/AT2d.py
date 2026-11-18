#!/usr/bin/env python3
"""AT2d (T2.2) — причинная (интервенционная) state-клетка: zero-ablation J-компоненты
state h_l, замер ДЦE на БУДУЩИХ позициях (вариант A).

Пре-рег: docs/AT2d_prereg.md (r5; ревью r2/r3/r4 + ответки docs/AT2d_review_r{2,3,4}_answers.md).
Контент T2: H1-causal — weak-сторона mean-J спектра causally используется моделью
для предсказания будущего, даже если не линейно читаема (readout-часть отклонена AT2c).

Ключевые конструкции r4/r5:
(1) ВАРИАНТ A (ревью r3 B5): абляция ТОЛЬКО t<64 (позиции 0..63); метрика ТОЛЬКО
    logit-позиции 64..126 = цели 65..127 = 63 значения; бины (target-индексные)
    65-77/78-90/91-103/104-116/117-127. Конвенция: «позиция» = logit-позиция
    (предсказывает токен t+1), target = позиция+1. Позиция 63 в метрику НЕ входит.
(2) Первичные эстиманды (per-doc, оба ко-первичных; ĉ/energy-фит НЕТ, r3 B1/B2):
    excess_match(арм, док) = ΔCE(арм, док) − ΔCE(randG(α=1), док) — все не-sham
    армы α-matched к DOSE sham'а (post-norm ‖δn‖², НЕ raw E — r3 M1);
    excess_cal(арм, док) = ΔCE(арм, док) − [CE(τ*·z_B) − CE(z_B)] — температурный
    sham: τ* решает mean H(softmax(τ·z_B)) = mean H(p_arm) на окне 64..126;
    считается из B_plain-логитов самого прогона (в кэше полных логитов НЕТ —
    только pooled U128/P128/P128_RAW), 0 доп. форвардов.
(3) REFUTED = non-superiority (r3 B3, достижимость r4): на ВСЕХ 10 клетках
    CI_upper(97.5%) < δ_min = 0.020 (= 3.92·SE). Приоритет веток (r4 §3.1):
    SIGNAL > REFUTED > SOFTENING_DOMINATED > GBAR > NOT_RESOLVED.
(4) L23 = G-zero (архитектурный нуль: после layers[23] только per-position
    final-norm + head ⇒ ΔCE_P_future(L23) ≡ 0; детектор протечки окна, ≤1e-6)
    + L23-P-контроль (direction specificity, отчёт).
(5) gbar = ПОЛНАЯ конструкция G̸ = mean_{d,t} Jᵀ diag(P128) J на всех 4096 (d,t)
    (r3 M4: субсэмпл 8 поз был ~ортогонален — углы 64-87°, P0').
(6) α-лестница randG {0.5,1,2,4} — dose–response отчёт (заменяет S-energy гейт).
(7) doc-bootstrap B (SEED=0): per-doc эстиманды, ресемплинг доков, БЕЗ перефита;
    пары «арм + fрейм + z_B» ресемплируются вместе (та же выборка дока).

Thread-pinning OMP/OPENBLAS/MKL=4 + torch 4 тр. ДО импортов (бит-идентичность
форвардов; pre-study: max|Δlogits| = 0 между повторами). CPU, eager, без fla.

Режимы:
  AT2d_SMOKE=1 : smoke (2 дока (wiki,0),(wiki,1), L6, k=4, B=100) + S9-двойной
                 прогон (md5 дет-под-артефакта) + DS1-DS5 + G-zero smoke (1 док);
  (без флагов) : main-прогон at2d_t22_r1 (trackio; 32 дока, L=(6,12,18,22,19),
                 k=(4,8), B=1000; 3072 форварда ≈ 4.0-4.5 ч);
  --reanalysis : диск-реанализ frozen raw-артефакта БЕЗ форвардов (ревью r5 F1):
                 G4' (plumbing) + G5 (лестница-контроль) + ortho f64 (из кэша)
                 + dose-коррекция (F3, канон g = 1 + w) → AT2d_v2.json + manifest
                 + log c маркером AT2D_REANALYSIS_DONE; raw-артефакт НЕ меняется.

r6 (ревью r5, docs/AT2d_review.md): dose-формула канонизирована (g = 1 + w,
radial ПЕРВЫМ — F3); базисы f64, ortho-гейт действительно f64 (F4); G4' =
plumbing (строго > 0; магнитуда — лестница, новый G5 — F1/F6); α/FWER согласованы
(F2: 97.5-я перцентиль, per-cell односторонний α = 0.025, FWER ~2.5%); артефакт
двухпроходный + manifest, маркер — в лог-файл (m6/m7).
"""
import atexit, hashlib, importlib.util, json, os, sys, time
from datetime import datetime

# CPU-защита ДО импорта transformers: fla 0.5.x = triton/CUDA-only; без fla
# transformers использует чистый torch-путь (как в .venv pre-study).
sys.modules["fla"] = None

# ТОЖЕ, что в AT2b/AT2c/пре-стади: 4 треда ДО импорта numpy/torch (BLAS-порядок)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
torch.set_num_threads(4)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_sp = importlib.util.spec_from_file_location("prism_field", os.path.join(ROOT, "prism", "field.py"))
_pf = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(_pf)
sha256_file = _pf.sha256_file

CACHE_DIR = "/data/prism_kernels/exp27b/results"
MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
MODEL_SFT = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")
RESULTS = os.path.join(ROOT, "results", "AT2d")
SMOKE = os.environ.get("AT2d_SMOKE", "0") == "1"
if SMOKE:
    SMOKE_DIR = os.path.join(os.path.dirname(RESULTS), "AT2d_smoke")
    ART = os.path.join(SMOKE_DIR, "AT2d_smoke.json")
    LOG_PATH = os.path.join(SMOKE_DIR, "AT2d_smoke_run.log")
else:
    ART = os.path.join(RESULTS, "AT2d.json")
    LOG_PATH = os.path.join(RESULTS, "AT2d_run.log")

REVISION = "r6"
MAIN_RUN = (not SMOKE)

D = 1024
S_LEN = 128
N_DOCS = 32
N_DOCS_RUN = 2 if SMOKE else N_DOCS
DOMAINS = ("wiki", "math", "code", "chat")
N_PER_DOM = 8

LAYS = (6, 12, 18, 22, 19) if MAIN_RUN else (6,)
L_ZERO = 23
KS = (4, 8) if MAIN_RUN else (4,)
SEED = 0
BOOT_B = 100 if SMOKE else 1000
DELTA_MIN = 0.020            # r5 (ревью r4 §2): 3.92·SE, достижимость REFUTED
ALPHA_LO, ALPHA_HI = 0.05, 5.0
LADDER = (0.5, 1.0, 2.0, 4.0)
AB_W = 64                    # абляция: позиции 0..63 (вариант A)
MET_LO, MET_HI = 64, 127     # logit-позиции 64..126 = цели 65..127 (63 значения)
N_WIN = MET_HI - MET_LO      # 63
# бины (target-индексные) -> слайсы logit-позиций
BINS_TGT = ((65, 77), (78, 90), (91, 103), (104, 116), (117, 127))
BINS_POS = tuple((a - 1, b) for a, b in BINS_TGT)   # logit-позиции (target-1 .. target)
TAU_LO, TAU_HI, N_TAU = 0.01, 100.0, 200
TAU_GRID = np.logspace(np.log10(TAU_LO), np.log10(TAU_HI), N_TAU)

P0R_SHA = "beeadc935791e1c010efecbc9d6022db77bd0e0ce7eee5829a20fe48157c4813"  # /tmp/at2d_p0r.json (17.11)

SHA_FILES = {  # гейт S1 (те же 7, AT2c S1)
    "exp27b_jcache_L6.pt":  "6d6b8ce784a2ad01a10112309aad7a74f13413d14636aad1a00652575f605dd2",
    "exp27b_jcache_L12.pt": "976079ce1d3c84c7016a236683f86f5fee1ce6571ff6edd53cf4f6f9ac228a24",
    "exp27b_jcache_L18.pt": "e2a03bc36b199695018171d0f0eab92ee139be83d0f0792a558cefa15f852d63",
    "exp27b_jcache_L19.pt": "0b31baff7dbc6e38c525d10d803ed9cc313774c6c8460cf1290a9ab9f4df82ae",
    "exp27b_jcache_L22.pt": "36ed09ebc49dd97bc7b8e0a8edd1c8d75567bb6431c4eea33c908ccfb7295a84",
    "exp27b_jcache_L23.pt": "af12505131be72b424515759c054646ea4526d94822287cbe7e8c50b7db46a43",
    "exp27b_pool_ids_L23.pt": "ded634f7d16cf33a9bb565de5cbc3faa288cfc90b15cdfa3e42365695fec4e03",
}

_LOG = []

def _lk(a):
    return "d1" if a == 1.0 else f"d{a}"

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
def f2a_inv(fld, tail):
    """dict {(dom,si,t): vec} -> [N_DOCS, 128, ...] В FORWARD-ПОРЯДКЕ (ревью r3 M2).
    Кэш builder'а exp27b: H[(dom,si,t)] = позиция 127-t. Значит:
    arr[doc, pos] = fld[(dom, si, 127-pos)] — forward-позиция pos. ВСЕГДА."""
    arr = torch.empty((N_DOCS, S_LEN) + tuple(fld[("wiki", 0, 0)].shape),
                      dtype=fld[("wiki", 0, 0)].dtype)
    for di, dom in enumerate(DOMAINS):
        for si in range(N_PER_DOM):
            for pos in range(S_LEN):
                arr[di * N_PER_DOM + si, pos] = fld[(dom, si, S_LEN - 1 - pos)]
    del fld
    return arr

def eigh_basis(G, k, mode):
    """G [1024,1024] f64: top = первые k по убыванию λ; weak = первые k из
    ненулевых (λ > 1e-12·λ_max) по возрастанию. (дословно AT2c/P0')
    Возврат f64 (r6, F4 r5: ortho-гейт действительно f64; хук кастует в f32
    при использовании — те же биты, что f32-базис r5.1)."""
    lam, w = torch.linalg.eigh(G)
    lam_np = lam.cpu().numpy()
    nz = np.where(lam_np > 1e-12 * lam_np.max())[0]
    if mode == "desc":
        order = torch.argsort(lam, descending=True).cpu().numpy()
        idx = order[:k]
    else:
        idx = nz[:min(k, len(nz))]
    return w[:, torch.as_tensor(idx)].cpu().numpy()

def gbar_from(JRf, Pf, ts):
    """G̸ = mean_{d,t∈ts} Jᵀ diag(P128) J (f32-GEMM, f64-аккумуляция; P0')."""
    G = torch.zeros(D, D, dtype=torch.float64)
    n = 0
    for d in range(N_DOCS_RUN):
        Jd = JRf[d][list(ts), :, :]
        pd = Pf[d][list(ts), :]
        G += ((Jd.transpose(1, 2) * pd.unsqueeze(1)) @ Jd).to(torch.float64).sum(0)
        n += len(ts)
    return G / n

TS_SUB = (0, 16, 32, 48, 64, 80, 96, 112)   # P0' gbar-субсэмпл (дословно)

def e_halves(Hdd, V, w0, w1):
    """E_raw (mean ‖Vᵀh‖²/‖h‖²) по ДВУМ половинам окна [w0,w1) (m1 r5:
    E_early/E_late — обе половины, M2 r3). Возврат (E_early, E_late)."""
    Hw = torch.from_numpy(np.ascontiguousarray(Hdd[w0:w1])).double()
    Vd = torch.as_tensor(np.ascontiguousarray(V), dtype=torch.float64)
    vh = Hw @ Vd
    E = (vh ** 2).sum(1) / (Hw ** 2).sum(1)
    half = (w1 - w0) // 2
    return float(E[:half].mean()), float(E[half:].mean())

def principal_angles_deg(A, B):
    """Главные углы (градусы) между столбцами A [n,k1] и B [n,k2] (m1 r5:
    gbar full-vs-sub, M4 r3). SVD(AᵀB), arccos(σᵢ)."""
    _, s, _ = np.linalg.svd(A.T @ B)
    return [float(np.degrees(np.arccos(min(1.0, max(0.0, float(x))))))
            for x in s]

def frame_sha(V):
    """sha256 f64-байтов фрейма (m1 r5: randG-фреймы в артефакте, §2)."""
    return hashlib.sha256(np.ascontiguousarray(V.astype(np.float64)).tobytes()
                          ).hexdigest()[:16]

def win_stats(Hd, V, w0, w1, gain, JRd, Pdr):
    """Per-doc на окне [w0,w1): E_raw, dose (post-norm), KL_pred (disk-only; P0').
    Hd [128,1024] f32 np (forward); V [1024,k] np (f64 r6); gain = weight
    input_layernorm СЛЕДУЮЩЕГО слоя [1024] (w; канон g = 1 + w — F3 r5);
    JRd [128,128,1024] f32 (по forward-позициям);
    Pdr [128,128] f32 (P128_RAW по позициям).
    DOSE (r6, канон §2.5: ∂y/∂x = (1/r)·diag(1+w)·(I − ûûᵀ), diag — СЛЕВА):
    сначала радиальная проекция (I − ûûᵀ) на P_V(h/rms), ПОТОМ diag(1+w);
    r5.1-формула (g = w, gain первым) — баг F3, в reanalysis для сверки."""
    Hw = torch.from_numpy(np.ascontiguousarray(Hd[w0:w1])).double()
    Vd = torch.as_tensor(np.ascontiguousarray(V), dtype=torch.float64)
    vh = Hw @ Vd
    proj = vh @ Vd.T
    E = ((vh ** 2).sum(1) / (Hw ** 2).sum(1)).numpy()
    hn = (Hw ** 2).mean(1, keepdim=True).sqrt()
    u = Hw / (Hw.norm(dim=1, keepdim=True) + 1e-30)
    a = proj / hn
    a = a - (a * u).sum(1, keepdim=True) * u          # radial ПЕРВЫМ (I − ûûᵀ)
    r = a * (1.0 + gain[None, :])                     # ПОТОМ diag(g), g = 1 + w
    dose = (r ** 2).sum(1).numpy()
    dlog = torch.einsum("td,tvj->tv", proj.float(), JRd[w0:w1])   # [W,128] U128
    kl = (Pdr[w0:w1].double() * (dlog.double() ** 2)).sum(1) * 0.5
    return float(E.mean()), float(dose.mean()), float(kl.mean())

def dose_pair(Hdd, V, w, win0=0, win1=AB_W):
    """F3 r5 (reanalysis): per-doc dose (окно [win0,win1)) тремя способами:
    old — формула r5.1 (g = w, gain ПЕРВЫМ, радиальный ПОТОМ);
    new — канон (g = 1 + w, radial ПЕРВЫМ, diag ПОТОМ);
    g1  — g ≡ 1 (fingerprint ≈ E·D). Возврат (old, new, g1)."""
    Hw = torch.from_numpy(np.ascontiguousarray(Hdd[win0:win1])).double()
    Vd = torch.as_tensor(np.ascontiguousarray(V), dtype=torch.float64)
    vh = Hw @ Vd
    proj = vh @ Vd.T
    hn = (Hw ** 2).mean(1, keepdim=True).sqrt()
    u = Hw / (Hw.norm(dim=1, keepdim=True) + 1e-30)
    a = proj / hn
    P_a = a - (a * u).sum(1, keepdim=True) * u
    r0 = a * w[None, :]
    r_old = r0 - (r0 * u).sum(1, keepdim=True) * u
    r_new = P_a * (1.0 + w[None, :])
    return (float((r_old ** 2).sum(1).mean()),
            float((r_new ** 2).sum(1).mean()),
            float((P_a ** 2).sum(1).mean()))

def build_basis(L, JRf, Pf, Hf):
    """J/gbar-базисы + per-doc pca + randG + haar (дословно r5.1-конструкция:
    сиды/порядок те же). f64-векторы (r6, F4 r5). Возврат (VJ, Vpca, Vrg, Vhaar)."""
    Jagg = JRf.mean(dim=(0, 1)).double()
    Gagg = Jagg.T @ Jagg
    Gf = gbar_from(JRf, Pf, range(S_LEN))
    VJ = {}
    for k in KS:
        VJ[f"top{k}"] = eigh_basis(Gagg, k, "desc")
        VJ[f"weak{k}"] = eigh_basis(Gagg, k, "asc")
        VJ[f"gbar{k}"] = eigh_basis(Gf, k, "asc")
    del Gagg, Gf
    Vpca, Vrg = {}, {}
    for k in KS:
        arr = np.zeros((N_DOCS_RUN, D, k), np.float64)
        for d in range(N_DOCS_RUN):
            Hc = torch.from_numpy(np.ascontiguousarray(
                Hf[d] - Hf[d].mean(0, keepdims=True))).double()
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            arr[d] = Vh[:k].cpu().numpy().T
        Vpca[k] = arr
        arr = np.zeros((N_DOCS_RUN, D, k), np.float64)
        for d in range(N_DOCS_RUN):
            rng = np.random.default_rng(200_000 + L * 10 + k + 1000 * d)
            arr[d] = np.linalg.qr(rng.standard_normal((D, k)))[0]
        Vrg[k] = arr
    rng_base = np.random.default_rng(SEED)
    G32 = rng_base.standard_normal((D, 32))
    Vhaar = {k: np.linalg.qr(G32[:, :k])[0] for k in KS}
    return VJ, Vpca, Vrg, Vhaar

# ------------------------------------------------------------------- метрики
def ce_per_pos(z, ids1):
    """z [128,V] f32 (logits дока); ids1 = row [128] int. Возврат [127]: CE по
    logit-позициям 0..126 (позиция p предсказывает токен ids1[p+1])."""
    lg = torch.softmax(z[:S_LEN - 1].float(), dim=-1)
    tgt = torch.as_tensor(ids1[1:S_LEN], dtype=torch.long).unsqueeze(1)
    return -torch.log(lg.gather(1, tgt).clamp_min(1e-30)).squeeze(1)

def ent_per_pos(lg):
    return -(lg * torch.log(lg.clamp_min(1e-30))).sum(-1)

def h_curve(z_win):
    """H(softmax(τ·z_B)) на окне 64..126 по TAU_GRID (монотонно убывает по τ).
    z_win [63,V] f32. Пер-док — 1 раз (не per-слой: z_B не зависит от L)."""
    Hs = np.empty(N_TAU)
    for i, tau in enumerate(TAU_GRID):
        lg = torch.softmax(z_win * float(tau), dim=-1)
        Hs[i] = float(ent_per_pos(lg).mean())
    return Hs

def tau_star(Hs, h_target):
    """τ*: H(softmax(τ·z_B)) = h_target (окно). Линейная интерполяция в log-τ
    на сходящемся интервале сетки. None — вне сетки (флаг tsham_unbounded)."""
    if h_target >= Hs[0]:
        return None
    if h_target <= Hs[-1]:
        return None
    i = 0
    while i < N_TAU - 1 and Hs[i] > h_target:
        i += 1
    if Hs[i - 1] == Hs[i]:
        return float(TAU_GRID[i])
    frac = (Hs[i - 1] - h_target) / (Hs[i - 1] - Hs[i])
    lt = np.log(TAU_GRID[i - 1]) + frac * (np.log(TAU_GRID[i]) - np.log(TAU_GRID[i - 1]))
    return float(np.exp(lt))

def ce_at_tau(z_win, tau, tgt_win):
    lg = torch.softmax(z_win * float(tau), dim=-1)
    return float(-torch.log(lg.gather(1, tgt_win.unsqueeze(1)).clamp_min(1e-30)).mean())

def ci_doc(vals, rngb):
    """doc-bootstrap (B, SEED=0) per-doc скаляров: stat = mean; NaN-доки
    исключаются (n_matched). Возврат (CI95 [2.5,97.5] или None, n).
    CI = перцентили 2.5/97.5: односторонний upper-α = 0.025 на клетку;
    FWER по 10 клеткам (P(≥2)) ≈ 2.5% — TZ §5 (r5.3, F2 r5: старое
    «α = 0.0025 / 0.028%» было несогласовано с кодом и снято)."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return None, int(len(v))
    means = [float(v[rngb.integers(0, len(v), len(v))].mean()) for _ in range(BOOT_B)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))], int(len(v))

def _san(o):
    """Рекурсивный sanitize для json: NaN/Inf -> None, numpy-типы -> python."""
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, np.ndarray):
        return _san(o.tolist())
    if isinstance(o, np.floating):
        o = float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o

def _det_subset(o):
    return {k: v for k, v in o.items() if k not in _NONDET}

def _md5_det(o):
    return hashlib.md5(json.dumps(_san(_det_subset(o)), sort_keys=True,
                                  ensure_ascii=False).encode()).hexdigest()

_NONDET = ("wall_sec", "assembled_at", "started_at", "finished_at",
           "log_md5", "trackio_run")

# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    if SMOKE:
        os.makedirs(SMOKE_DIR, exist_ok=True)
    logline(f"AT2d (T2.2) start {datetime.now().isoformat()} "
            f"py={sys.version.split()[0]} torch={torch.__version__} smoke={SMOKE} rev={REVISION}")

    tr = None
    tr_done = [False]
    if MAIN_RUN:
        tr_cfg = {
            "started_at": datetime.now().isoformat(),
            "revision": f"TZ-{REVISION}",
            "script_sha": sha256_file(os.path.abspath(__file__)),
            "seed": SEED, "device": "cpu",
            "model": "Qwen/Qwen3.5-0.8B (float32, eager, без fla)",
            "log": os.path.abspath(LOG_PATH),
            "layers": list(LAYS) + [L_ZERO], "k": list(KS), "n_docs": N_DOCS_RUN,
            "boot_b": BOOT_B, "delta_min": DELTA_MIN,
            "alpha_bounds": [ALPHA_LO, ALPHA_HI], "ladder": list(LADDER),
            "window": {"ablate": "0..63", "metric_pos": "64..126",
                       "targets": "65..127", "n": N_WIN},
        }
        tr = trackio_start("at2d_t22_r1", tr_cfg)
        tlog(tr, 0, {"started": 1.0})
        logline("trackio: run at2d_t22_r1 init ok")

        def _trackio_on_exit():
            if not tr_done[0]:
                try:
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

    # ================= phase: load + sha-гейты ============================
    tstage(tr, time.time() - t0, "load")
    for fn, expect in SHA_FILES.items():
        got = sha256_file(os.path.join(CACHE_DIR, fn))
        if got != expect:
            raise SystemExit(f"S1 FAIL: sha {fn} = {got} != {expect}")
    gates["S1_sha"] = True
    logline("S1 sha-гейт: 7/7 OK")

    pool = torch.load(os.path.join(CACHE_DIR, "exp27b_pool_ids_L23.pt"),
                      map_location="cpu", weights_only=False)
    toks = {}
    for di, dom in enumerate(DOMAINS):
        toks[dom] = [np.asarray(pool["ids"][dom][si], dtype=np.int64)
                     for si in range(N_PER_DOM)]
    ok_g2a = (len(pool["ids"]) == 4 and
              all(len(toks[d]) == N_PER_DOM for d in DOMAINS) and
              all(len(r) == S_LEN for d in DOMAINS for r in toks[d]))
    gates["G2a_pool_ids"] = bool(ok_g2a)
    if not ok_g2a:
        raise SystemExit("G2a FAIL: структура pool_ids не 4 домена x 8 x 128 int")
    logline("G2a pool-структура: 4x8x128 OK")

    ck_l6_meta = torch.load(os.path.join(CACHE_DIR, "exp27b_jcache_L6.pt"),
                            map_location="cpu", weights_only=False)
    ok_g2b = (pool["meta"]["pool_fp"] == ck_l6_meta["pool_fp"])
    gates["G2b_pool_fp"] = bool(ok_g2b)
    if not ok_g2b:
        raise SystemExit(f"G2b FAIL: pool_fp {pool['meta']['pool_fp']} != "
                         f"{ck_l6_meta['pool_fp']}")
    logline(f"G2b pool_fp: 4/4 доменов совпадают ({dict(pool['meta']['pool_fp'])})")

    # токены-массивы (forward-порядок, как в pool)
    rows = [toks[DOMAINS[di // N_PER_DOM]][di % N_PER_DOM]
            for di in range(N_DOCS_RUN)]
    model = None
    def _load_model():
        nonlocal model
        if model is not None:
            return model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager")
        model.eval()
        logline(f"model loaded: {model.config.model_type}, "
                f"n_layers={len(model.model.layers)}")
        return model

    # layer_types (провенанс; после L23 — только per-position norm+head)
    cfg_j = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    ltypes = cfg_j["text_config"]["layer_types"] if "text_config" in cfg_j else cfg_j["layer_types"]
    logline(f"layer_types: full на {[i for i, t in enumerate(ltypes) if t == 'full_attention']}")

    # ================= phase: bases (disk-only, per слой) =================
    tstage(tr, time.time() - t0, "gbar_full",
           {"note": "G̸ = mean_{4096} J^T diag(P128) J (полная конструкция, M4 r3)"})
    from safetensors import safe_open
    GAIN = {}
    with safe_open(MODEL_SFT, framework="pt") as f:
        for L in list(LAYS) + [L_ZERO]:
            key = (f"model.language_model.layers.{L + 1}.input_layernorm.weight"
                   if L + 1 < len(ltypes)
                   else "model.language_model.norm.weight")
            GAIN[L] = f.get_tensor(key).double().numpy()
    model_weights_sha = sha256_file(MODEL_SFT)
    logline(f"model sha256[:16] = {model_weights_sha[:16]}")

    # P128_RAW — идентичен по слоям (spot-check P0': L12 vs L6 max|Δ| = 0.0);
    # L19-кэш старый, без P128_RAW → берём из L6 (зафиксировано в TZ §1.1)
    PFR_REF = None
    Hf_all, basis, dose_tab, alpha_tab = {}, {}, {}, {}
    angles_sub, frame_sha_tab = {}, {}
    ortho_max = 0.0
    for L in list(LAYS) + [L_ZERO]:
        tstage(tr, time.time() - t0, f"basis_L{L}")
        ck = torch.load(os.path.join(CACHE_DIR, f"exp27b_jcache_L{L}.pt"),
                        map_location="cpu", weights_only=False)
        if "P128_RAW" in ck:
            Pfr = f2a_inv(ck["P128_RAW"], (128,))
            if L == 12 and PFR_REF is not None:
                logline(f"P128_RAW L12 vs L6: max|Δ| = {(Pfr - PFR_REF).abs().max():.3e}")
            if PFR_REF is None:
                PFR_REF = Pfr.clone()
        else:
            if PFR_REF is None:  # m8 r5: PFR_REF None → явный стоп, не краш
                raise SystemExit(f"P128_RAW нет в кэше L{L} и референсного слоя нет "
                                 f"(PFR_REF is None) — стоп")
            Pfr = PFR_REF.clone()
        Hf = f2a_inv(ck["H"], (D,)).numpy().astype(np.float32)
        JRf = f2a_inv(ck["JR"], (128, D))
        Pf = f2a_inv(ck["P128"], (128,))
        Hf_all[L] = Hf

        # J-базис (Jagg/Gagg; gbar full — все 4096 (d,t)) — f64 (r6, F4 r5)
        VJ, Vpca, Vrg, Vhaar = build_basis(L, JRf, Pf, Hf)
        # m1 r5: gbar full-vs-sub углы (M4 r3, TS_SUB — дословно P0') +
        # randG-фреймы: sha per (L,k,d) (§2)
        Gs = gbar_from(JRf, Pf, TS_SUB)
        angles_sub[L] = {}
        frame_sha_tab[L] = {}
        for k in KS:
            Vs = eigh_basis(Gs, k, "asc")
            angles_sub[L][f"weak{k}"] = principal_angles_deg(VJ[f"weak{k}"], Vs)
            frame_sha_tab[L][k] = {f"d{d}": frame_sha(Vrg[k][d])
                                   for d in range(N_DOCS_RUN)}
        del Gs
        logline(f"L{L}: J/gbar-базисы готовы (gbar full: {S_LEN * N_DOCS_RUN} (d,t))")

        basis[L] = {"VJ": VJ, "Vpca": Vpca, "Vrg": Vrg, "Vhaar": Vhaar}

        # Ortho (f64) на всех направлениях
        for k in KS:
            for V in [VJ[f"top{k}"], VJ[f"weak{k}"], VJ[f"gbar{k}"], Vhaar[k]]:
                ortho_max = max(ortho_max,
                                float(np.abs(V.T @ V - np.eye(k)).max()))
            for d in range(N_DOCS_RUN):
                ortho_max = max(ortho_max,
                                float(np.abs(Vpca[k][d].T @ Vpca[k][d] - np.eye(k)).max()),
                                float(np.abs(Vrg[k][d].T @ Vrg[k][d] - np.eye(k)).max()))
        logline(f"L{L}: ortho f64 max|VᵀV−I| = {ortho_max:.3e}")

        # disk-only per-doc stats (окно 0..63) + α-матчинг
        for k in KS:
            dose_tab[(L, k)] = {}
            alpha_tab[(L, k)] = {}
            for d in range(N_DOCS_RUN):
                Hdd = Hf[d]
                JRd = JRf[d]; Pdr = Pfr[d]; gain = GAIN[L]
                st = {}
                for name, V in ((f"top{k}", VJ[f"top{k}"]), (f"weak{k}", VJ[f"weak{k}"]),
                                (f"gbar{k}", VJ[f"gbar{k}"]), (f"pca{k}", Vpca[k][d]),
                                (f"haar{k}", Vhaar[k]), (f"randG{k}", Vrg[k][d])):
                    E, Do, Kl = win_stats(Hdd, V, 0, AB_W, gain, JRd, Pdr)
                    eh = e_halves(Hdd, V, 0, AB_W)   # m1 r5: обе половины (M2 r3)
                    st[name] = {"E_raw": E, "E_early": eh[0], "E_late": eh[1],
                                "D": Do, "KL_pred": Kl}
                dose_tab[(L, k)][d] = st
                Drg = st[f"randG{k}"]["D"]
                for name in ("top", "weak", "gbar", "pca", "haar"):
                    Da = st[f"{name}{k}"]["D"]
                    a = float(np.sqrt(Drg / Da)) if Da > 0 else float("inf")
                    alpha_tab[(L, k)][d] = {
                        **alpha_tab.get((L, k), {}).get(d, {}),
                        name: a,
                        name + "_ok": bool(ALPHA_LO <= a <= ALPHA_HI),
                    }
        # L23: доза на полном окне (0..127) для P-контроля (report)
        if L == L_ZERO:
            dose_tab[(L_ZERO, "P")] = {}
            for d in range(N_DOCS_RUN):
                st = {}
                for name, V in (("top4", basis[L_ZERO]["VJ"]["top4"]),
                                ("randG4", basis[L_ZERO]["Vrg"][4][d])):
                    E, Do, Kl = win_stats(Hf[d], V, 0, S_LEN, GAIN[L],
                                          JRf[d], Pfr[d])
                    st[name] = {"E_raw": E, "D": Do, "KL_pred": Kl}
                dose_tab[(L_ZERO, "P")][d] = st
        del ck, JRf, Pfr
        logline(f"L{L}: базисы + дозы готовы ({time.time() - t0:.0f}s)")

    # ================= pipeline (S9: вызывается дважды в smoke) ==========
    def run_pipeline():
        # ================= phase: docs (baseline + arm-runs) ==================
        _load_model()
        tstage(tr, time.time() - t0, "baseline")
        rngb = np.random.default_rng(SEED)   # bootstrap-поток (одна на все CI)

        g1_max = 0.0
        g2c_cos = []
        rec = {}            # (L,k) -> {"arms": {arm: per-doc list}, "ladder": per-doc}
        for L in LAYS:
            for k in KS:
                rec[(L, k)] = {"arms": {}, "ladder": {}}
        l23_rec = {"gzero": {}, "P": {}}
        base = {}          # doc -> малые per-doc скаляры (z_win/lg_B_win живут в b)
        ds1_vals = {}
        smoke_extra = {}   # doc -> {P_top4, R_pf_top4}

        def run_arm(L, k, d, name, V, alpha, win, probe=None, is_reflect=False):
            """Один forward с хуком. Возврат: (logits, aux); aux — метрики хука
            (norm_ratio = mean_t ‖h'_t‖/‖h_t‖ на абляционном окне, m1 r5)."""
            layer = model.model.layers[L]
            W = AB_W if win == "pf" else S_LEN
            Vt = torch.as_tensor(np.ascontiguousarray(V), dtype=torch.float32)
            aux = {"norm_ratio": None}

            def h(mod, inp, out):
                ov = out[:, :W]
                proj = (ov @ Vt) @ Vt.T
                o2 = out.clone()
                o2[:, :W] = ov - (2.0 if is_reflect else alpha) * proj
                aux["norm_ratio"] = float(
                    (o2[:, :W].norm(dim=1) / (ov.norm(dim=1) + 1e-30)).mean())
                if probe is not None:
                    # DS1: max|proj| (α=1, калибровка pre-study 0.18) — независимо
                    # от α-арма (Δh = α·proj проверяется отдельно, DS5)
                    probe["maxdh"] = max(probe.get("maxdh", 0.0),
                                         float(proj.abs().max()))
                return o2

            hook = layer.register_forward_hook(h)
            try:
                with torch.no_grad():
                    out = model(input_ids=torch.as_tensor(rows[d][None], dtype=torch.long),
                                attention_mask=torch.ones((1, S_LEN), dtype=torch.long))
            finally:
                hook.remove()
            return out.logits[0], aux

        def metrics_from_logits(z, b, d, dce_rg1):
            """Все per-doc метрики из логитов run (z [128,V]) + baseline b.
            Окно: logit-позиции MET_LO..MET_HI (64..126, 63 значения; вариант A)."""
            ce = ce_per_pos(z, rows[d])
            dce = float((ce[MET_LO:MET_HI] - b["ce_B_win"]).mean())
            bins5 = [float((ce[s:e] - b["ce_B"][s:e]).mean()) for s, e in BINS_POS]
            lg_win = torch.softmax(z[MET_LO:MET_HI].float(), dim=-1)
            H_arm = float(ent_per_pos(lg_win).mean())
            KL = float((lg_win * (torch.log(lg_win.clamp_min(1e-30))
                      - torch.log(b["lg_B_win"].clamp_min(1e-30)))).sum(-1).mean())
            ddiff = (ce[MET_LO:MET_HI] - b["ce_B_win"])
            n_c = int(b["correct"].sum())
            dce_c = float(ddiff[b["correct"]].mean()) if n_c else float("nan")
            n_w = N_WIN - n_c
            dce_w = float(ddiff[~b["correct"]].mean()) if n_w else float("nan")
            # temp-sham (M3): τ* по H-кривой дока
            tau = tau_star(b["Hs"], H_arm)
            if tau is None:
                ec = float("nan")
                ce_cal = float("nan")
            else:
                ce_cal = ce_at_tau(b["z_win"], tau, b["tgt_win"])
                ec = dce - (ce_cal - float(b["ce_B_win"].mean()))
            em = dce - dce_rg1
            return {"dCE": dce, "bins5": bins5, "H_arm": H_arm, "KL": KL,
                    "dCE_correct": dce_c, "dCE_wrong": dce_w,
                    "n_correct": n_c, "n_wrong": n_w,
                    "tau_star": (float(tau) if tau is not None else None),
                    "ce_cal": ce_cal, "excess_match": em, "excess_cal": ec}

        doc_t0 = time.time()
        for d in range(N_DOCS_RUN):
            ids_row = torch.as_tensor(rows[d][None], dtype=torch.long)
            mask = torch.ones((1, S_LEN), dtype=torch.long)
            # --- B_plain (hidden states для G2c) --------------------------------
            with torch.no_grad():
                out_b = model(input_ids=ids_row, attention_mask=mask,
                              output_hidden_states=True)
            z_B = out_b.logits[0]
            z_win = z_B[MET_LO:MET_HI].contiguous()
            lg_B_win = torch.softmax(z_win.float(), dim=-1)
            ce_B = ce_per_pos(z_B, rows[d])
            tgt_win = torch.as_tensor(rows[d][MET_LO + 1:MET_HI + 1], dtype=torch.long)
            correct = (z_win.argmax(-1) == tgt_win)
            Hs = h_curve(z_win)
            b = {"ce_B": ce_B, "ce_B_win": ce_B[MET_LO:MET_HI], "z_win": z_win,
                 "lg_B_win": lg_B_win, "Hs": Hs, "correct": correct,
                 "tgt_win": tgt_win,
                 "CE_B_full": float(ce_B.mean()),
                 "H_B_win": float(ent_per_pos(lg_B_win).mean())}
            # --- G1: no-op hook bit-exact ---------------------------------------
            layer6 = model.model.layers[LAYS[0]]
            hook = layer6.register_forward_hook(lambda mod, inp, out: out)
            try:
                with torch.no_grad():
                    out_h = model(input_ids=ids_row, attention_mask=mask)
            finally:
                hook.remove()
            g1_max = max(g1_max, float((out_h.logits[0] - z_B).abs().max()))
            # --- G2c: state provenance (все 128 позиций, все L) -----------------
            for L in LAYS:
                hf = out_b.hidden_states[L + 1][0]
                Hfd = torch.from_numpy(np.ascontiguousarray(Hf_all[L][d]))
                cos = torch.nn.functional.cosine_similarity(hf, Hfd, dim=1)
                g2c_cos.extend([float(c) for c in cos])
            # --- arm-runs per (L, k) --------------------------------------------
            for L in LAYS:
                for k in KS:
                    st = dose_tab[(L, k)][d]
                    at = alpha_tab[(L, k)][d]
                    VJ = basis[L]["VJ"]
                    # base sham randG(α=1) — СНАЧАЛА (для dce_rg1)
                    z_rg, aux_rg = run_arm(L, k, d, "randG", basis[L]["Vrg"][k][d], 1.0, "pf")
                    m_rg = metrics_from_logits(z_rg, b, d, 0.0)
                    m_rg["norm_ratio"] = aux_rg["norm_ratio"]
                    dce_rg1 = m_rg["dCE"]
                    rec[(L, k)]["arms"].setdefault("randG", []).append(m_rg)
                    rec[(L, k)]["ladder"].setdefault("d1", []).append(m_rg["dCE"])
                    for name, V in (("top", VJ[f"top{k}"]), ("weak", VJ[f"weak{k}"]),
                                    ("gbar", VJ[f"gbar{k}"]),
                                    ("pca", basis[L]["Vpca"][k][d]),
                                    ("haar", basis[L]["Vhaar"][k])):
                        a = at[name]
                        ok = at[name + "_ok"]
                        probe = {"maxdh": 0.0} if (SMOKE and name == "top" and k == 4
                                                    and d < 2 and L == LAYS[0]) else None
                        z_a, aux_a = run_arm(L, k, d, name, V, a if ok else 1.0, "pf",
                                              probe=probe)
                        m_a = metrics_from_logits(z_a, b, d, dce_rg1)
                        m_a["norm_ratio"] = aux_a["norm_ratio"]
                        m_a["alpha"] = float(a)
                        m_a["unmatched"] = (not ok)
                        m_a["D"] = st[f"{name}{k}"]["D"]
                        m_a["D_matched"] = float((a if ok else 1.0) ** 2 * st[f"{name}{k}"]["D"])
                        m_a["E_raw"] = st[f"{name}{k}"]["E_raw"]
                        m_a["E_early"] = st[f"{name}{k}"]["E_early"]
                        m_a["E_late"] = st[f"{name}{k}"]["E_late"]
                        m_a["KL_pred"] = st[f"{name}{k}"]["KL_pred"]
                        m_a["dose_res"] = float(abs((a if ok else 1.0) ** 2 *
                                                     st[f"{name}{k}"]["D"] -
                                                     st[f"randG{k}"]["D"]) /
                                                 max(st[f"randG{k}"]["D"], 1e-30))
                        if probe is not None:
                            ds1_vals[(d, L, k, name)] = probe["maxdh"]
                        rec[(L, k)]["arms"].setdefault(name, []).append(m_a)
                    for a in LADDER:
                        if a == 1.0:
                            continue
                        z_l, _auxl = run_arm(L, k, d, "randG", basis[L]["Vrg"][k][d], a, "pf")
                        m_l = metrics_from_logits(z_l, b, d, dce_rg1)
                        rec[(L, k)]["ladder"].setdefault(f"d{a}", []).append(m_l["dCE"])
            # --- L23: G-zero (P_future) + P-контроль ----------------------------
            for name, V in (("top4", basis[L_ZERO]["VJ"]["top4"]),
                            ("randG4", basis[L_ZERO]["Vrg"][4][d])):
                z_g, _auxg = run_arm(L_ZERO, None, d, name, V, 1.0, "pf")
                diff = float((z_g[MET_LO:MET_HI] - z_B[MET_LO:MET_HI]).abs().max())
                ce_g = ce_per_pos(z_g, rows[d])
                dce_g = float((ce_g[MET_LO:MET_HI] - b["ce_B_win"]).mean())
                l23_rec["gzero"].setdefault(name, []).append(
                    {"max_logit_diff": diff, "dCE": dce_g})
            # L23 P-контроль (main-only; в смоке — только G-zero, ТЗ §6):
            # вариант P (все 128 позиций), метрика — 127 значений
            if MAIN_RUN:
                stp = dose_tab[(L_ZERO, "P")][d]
                VJz = basis[L_ZERO]["VJ"]
                a_top = float(np.sqrt(stp["randG4"]["D"] / stp["top4"]["D"]))
                a_top_ok = bool(ALPHA_LO <= a_top <= ALPHA_HI)
                a_top_eff = a_top if a_top_ok else 1.0
                def _p_metrics(z):
                    ce = ce_per_pos(z, rows[d])
                    dce_p = float((ce - b["ce_B"]).mean())
                    lg_f = torch.softmax(z[:S_LEN - 1].float(), dim=-1)
                    kl_f = float((lg_f * (torch.log(lg_f.clamp_min(1e-30))
                              - torch.log(b["lg_full"].clamp_min(1e-30)))).sum(-1).mean())
                    return dce_p, kl_f
                b["lg_full"] = torch.softmax(z_B[:S_LEN - 1].float(), dim=-1)
                z_p_top, _auxpt = run_arm(L_ZERO, None, d, "top4", VJz["top4"], a_top_eff, "P")
                z_p_rg, _auxpr = run_arm(L_ZERO, None, d, "randG4", basis[L_ZERO]["Vrg"][4][d],
                                         1.0, "P")
                dce_pt, kl_pt = _p_metrics(z_p_top)
                dce_pr, kl_pr = _p_metrics(z_p_rg)
                l23_rec["P"].setdefault("top4", []).append(
                    {"dCE": dce_pt, "KL": kl_pt, "alpha": float(a_top),
                     "unmatched": not a_top_ok,
                     "excess_match_P": dce_pt - dce_pr})
                l23_rec["P"].setdefault("randG4", []).append(
                    {"dCE": dce_pr, "KL": kl_pr})
                del b["lg_full"]
            # --- smoke-доп: DS2 ({top4, randG4}×P, ТЗ §11) + R (reflection) ----
            if SMOKE:
                z_p1, _auxp1 = run_arm(LAYS[0], None, d, "top4",
                                       basis[LAYS[0]]["VJ"]["top4"], 1.0, "P")
                ce_p1 = ce_per_pos(z_p1, rows[d])
                z_p2, _auxp2 = run_arm(LAYS[0], None, d, "randG4",
                                       basis[LAYS[0]]["Vrg"][4][d], 1.0, "P")
                ce_p2 = ce_per_pos(z_p2, rows[d])
                z_r, _auxr = run_arm(LAYS[0], None, d, "top4", basis[LAYS[0]]["VJ"]["top4"],
                                     1.0, "pf", is_reflect=True)
                ce_r = ce_per_pos(z_r, rows[d])
                smoke_extra[d] = {
                    "P_top4": float(ce_p1.mean() - b["ce_B"].mean()),
                    "P_randG4": float(ce_p2.mean() - b["ce_B"].mean()),
                    "R_pf_top4": float((ce_r[MET_LO:MET_HI] - b["ce_B_win"]).mean()),
                }
            base[d] = {"CE_B_full": b["CE_B_full"], "CE_B_win63": float(b["ce_B_win"].mean()),
                       "H_B_win": b["H_B_win"], "n_correct_win": int(correct.sum())}
            if MAIN_RUN:
                tstage(tr, time.time() - t0, f"doc_{d}",
                       {"dCE_B": b["CE_B_full"],
                        "dCE_top4_L6": rec[(LAYS[0], KS[0])]["arms"]["top"][d]["dCE"],
                        "dCE_weak4_L6": rec[(LAYS[0], KS[0])]["arms"]["weak"][d]["dCE"]})
            logline(f"doc {d + 1}/{N_DOCS_RUN} done ({time.time() - doc_t0:.0f}s, "
                    f"G1 max|Δlogits| = {g1_max:.3e})")

        # --- гейты по завершении doc-цикла ---------------------------------------
        gates["G1_hook_noop"] = {"max_logit_diff": float(g1_max),
                                 "ok": bool(g1_max == 0.0)}
        logline(f"G1 hook no-op: max|Δlogits| = {g1_max:.3e} "
                f"{'OK' if gates['G1_hook_noop']['ok'] else 'FAIL'}")
        if not gates["G1_hook_noop"]["ok"]:
            raise SystemExit("G1 FAIL: no-op хук не бит-идентичен — стоп")
        g2c_arr = np.asarray(g2c_cos)
        g2c_min = float(g2c_arr.min())
        g2c_med = float(np.median(g2c_arr))
        gates["G2c_state_prov"] = {"cos_min_all_pos": g2c_min,
                                   "cos_median": g2c_med,
                                   "ok": bool(g2c_min >= 0.999)}
        logline(f"G2c state provenance: cos_min = {g2c_min:.6f} "
                f"median = {g2c_med:.6f} "
                f"{'OK' if gates['G2c_state_prov']['ok'] else 'FAIL'}")
        if not gates["G2c_state_prov"]["ok"]:
            raise SystemExit("G2c FAIL: state provenance — стоп")
        gates["ortho_f64"] = {"max_norm_diff": float(ortho_max),
                              "ok": bool(ortho_max < 1e-6)}
        logline(f"Ortho f64: max|VᵀV−I| = {ortho_max:.3e} "
                f"{'OK' if gates['ortho_f64']['ok'] else 'FAIL'}")
        if not gates["ortho_f64"]["ok"]:
            raise SystemExit("Ortho FAIL")

        # ================= G-zero (L23) + G4 ===================================
        gz_ok = True
        gz_bad = []
        for name in ("top4", "randG4"):
            for r_ in l23_rec["gzero"][name]:
                if r_["max_logit_diff"] > 1e-6 or abs(r_["dCE"]) > 1e-6:
                    gz_ok = False
                    gz_bad.append((name, r_))
        gates["G_zero"] = {"ok": bool(gz_ok), "n_runs": 2 * N_DOCS_RUN,
                           "max_logit_diff": max(r["max_logit_diff"]
                                                 for n in l23_rec["gzero"]
                                                 for r in l23_rec["gzero"][n]),
                           "max_abs_dCE": max(abs(r["dCE"])
                                              for n in l23_rec["gzero"]
                                              for r in l23_rec["gzero"][n]),
                           "bad": [{"arm": n, "run": r} for n, r in gz_bad]}
        logline(f"G-zero L23: max|Δlogits[64..126]| = "
                f"{gates['G_zero']['max_logit_diff']:.3e}, max|ΔCE| = "
                f"{gates['G_zero']['max_abs_dCE']:.3e} "
                f"{'OK' if gz_ok else 'FAIL (протечка окна!)'}")
        if SMOKE:
            logline("G-zero (smoke): полный гейт — main (здесь 2 дока, отчёт)")
        elif not gz_ok:
            raise SystemExit("G-zero FAIL: L23 даёт ненулевой ΔCE — окно/метрика биты, стоп")

        # G4' (r6, F1 r5): sham responsiveness — PLUMBING (строго ≠ 0; мёртвый
        # хук даёт бит-ноль — логика G1). Магнитуда — на лестнице (G5).
        g4_cells = {}
        for L in LAYS:
            for k in KS:
                vals = [rec[(L, k)]["arms"]["randG"][d]["dCE"] for d in range(N_DOCS_RUN)]
                m = float(np.mean(vals))
                g4_cells[f"L{L}k{k}"] = {"abs_dce_randG4": abs(m), "mean": m,
                                         "ok": bool(abs(m) > 0.0)}
        n_resp = sum(1 for v in g4_cells.values() if v["ok"])
        g4_ok = n_resp >= 4
        gates["G4_sham"] = {"n_cells_responsive": n_resp, "of": len(g4_cells),
                            "threshold": 0.0,
                            "gate": ("r6 (F1 r5): plumbing — строго > 0 на ≥4/10 "
                                     "(мёртвый хук = бит-ноль); магнитуда — G5"),
                            "cells": g4_cells,
                            "ok": bool(g4_ok)}
        logline(f"G4' sham plumbing: {n_resp}/{len(g4_cells)} клеток > 0 "
                f"{'OK' if g4_ok else 'DEAD_CELL (флаг)'}")

        # G5 (r6, F6 r5): positive control ПЕРВИЧНОГО режима — верхняя ступень
        # лестницы (randG, α=4): окно 64..126 отвечает на абляцию 0..63.
        g5_cells = {}
        g5_ok = True
        for L in LAYS:
            for k in KS:
                m4 = float(np.mean(rec[(L, k)]["ladder"][_lk(4.0)]))
                g5_cells[f"L{L}k{k}"] = {"abs_dce_ladder_d4": abs(m4), "mean": m4,
                                         "ok": bool(abs(m4) >= 0.005)}
        n_g5 = sum(1 for v in g5_cells.values() if v["ok"])
        g5_ok = n_g5 >= 1
        gates["G5_ladder"] = {"n_cells_d4_gte_0.005": n_g5, "of": len(g5_cells),
                              "threshold": 0.005,
                              "gate": ("r6 (F6 r5): positive control первичного "
                                       "режима — лестница α=4 (16x dose)"),
                              "cells": g5_cells,
                              "ok": bool(g5_ok)}
        logline(f"G5 ladder positive control: {n_g5}/{len(g5_cells)} клеток |ΔCE(d4)| >= 0.005 "
                f"{'OK' if g5_ok else 'DEAD_CELL (флаг)'}")

        # ================= bootstrap + verdict ==================================
        tstage(tr, time.time() - t0, "bootstrap")
        def cell_vecs(L, k, arm_w, pairs):
            A = rec[(L, k)]["arms"]
            def em_of(arm, d):
                m = A[arm][d]
                return float("nan") if m["unmatched"] else m["excess_match"]
            def ec_of(arm, d):
                m = A[arm][d]
                return m["excess_cal"] if m["tau_star"] is not None else float("nan")
            em_w = np.array([em_of(arm_w, d) for d in range(N_DOCS_RUN)])
            ec_w = np.array([ec_of(arm_w, d) for d in range(N_DOCS_RUN)])
            em_p = {}
            for p in pairs:
                em_p[p] = np.array([em_w[d] - em_of(p, d)
                                    if not np.isnan(em_w[d]) and not np.isnan(em_of(p, d))
                                    else float("nan") for d in range(N_DOCS_RUN)])
            emc = np.array([(A[arm_w][d]["dCE_correct"] - A["randG"][d]["dCE_correct"])
                            if (not A[arm_w][d]["unmatched"]
                                and not np.isnan(A[arm_w][d]["dCE_correct"])
                                and not np.isnan(A["randG"][d]["dCE_correct"]))
                            else float("nan") for d in range(N_DOCS_RUN)])
            ek = np.array([(A[arm_w][d]["KL"] - A["randG"][d]["KL"])
                           if (not A[arm_w][d]["unmatched"]
                               and not np.isnan(A[arm_w][d]["KL"])
                               and not np.isnan(A["randG"][d]["KL"]))
                           else float("nan") for d in range(N_DOCS_RUN)])
            return {"em": em_w, "ec": ec_w, "em_pairs": em_p, "emc": emc, "ek": ek}

        cells = {}
        for L in LAYS:
            for k in KS:
                v = cell_vecs(L, k, "weak", ("pca", "haar"))
                vg = cell_vecs(L, k, "gbar", ("pca", "haar"))
                def _civs(v):
                    ci_em, n_em = ci_doc(v["em"], rngb)
                    ci_ec, n_ec = ci_doc(v["ec"], rngb)
                    cip = {}
                    for p in v["em_pairs"]:
                        cip[p] = ci_doc(v["em_pairs"][p], rngb)
                    ci_emc, n_emc = ci_doc(v["emc"], rngb)
                    ci_ek, n_ek = ci_doc(v["ek"], rngb)   # m1 r5: excess_KL
                    return ci_em, n_em, ci_ec, n_ec, cip, ci_emc, n_emc, ci_ek, n_ek
                ci_em, n_em, ci_ec, n_ec, cip, ci_emc, n_emc, ci_ek, n_ek = _civs(v)
                ci_gm, n_gm, ci_gec, n_gec, cgp, ci_gemc, n_gemc, ci_gk, n_gk = _civs(vg)
                cells[f"L{L}k{k}"] = {
                    "weak": {"em_mean": float(np.nanmean(v["em"])) if np.any(~np.isnan(v["em"])) else None,
                             "em_ci95": ci_em, "n_em": n_em,
                             "ec_mean": float(np.nanmean(v["ec"])) if np.any(~np.isnan(v["ec"])) else None,
                             "ec_ci95": ci_ec, "n_ec": n_ec,
                             "pair_ci": {p: cip[p][0] for p in cip},
                             "pair_n": {p: cip[p][1] for p in cip},
                             "emc_point": float(np.nanmean(v["emc"])) if not np.all(np.isnan(v["emc"])) else None,
                             "emc_ci95": ci_emc, "n_emc": n_emc,
                             "ek_mean": float(np.nanmean(v["ek"])) if np.any(~np.isnan(v["ek"])) else None,
                             "ek_ci95": ci_ek, "n_ek": n_ek,
                             "signal": bool(ci_em and ci_ec and ci_em[0] > 0 and ci_ec[0] > 0
                                            and cip["pca"][0] is not None and cip["pca"][1] >= 8
                                            and cip["haar"][0] is not None and cip["haar"][1] >= 8
                                            and cip["pca"][0][0] > 0 and cip["haar"][0][0] > 0
                                            and not np.all(np.isnan(v["emc"]))
                                            and float(np.nanmean(v["emc"])) > 0),
                             "refuted": bool(ci_em and ci_ec and n_em >= 8 and n_ec >= 8
                                             and ci_em[1] < DELTA_MIN
                                             and ci_ec[1] < DELTA_MIN),
                             "soft": bool(ci_em is not None and ci_em[0] <= 0.0 <= ci_em[1])},
                    "gbar": {"em_ci95": ci_gm, "n_em": n_gm,
                             "ec_ci95": ci_gec, "n_ec": n_gec,
                             "pair_ci": {p: cgp[p][0] for p in cgp},
                             "pair_n": {p: cgp[p][1] for p in cgp},
                             "emc_point": float(np.nanmean(vg["emc"])) if not np.all(np.isnan(vg["emc"])) else None,
                             "emc_ci95": ci_gemc, "n_emc": n_gemc,
                             "ek_mean": float(np.nanmean(vg["ek"])) if np.any(~np.isnan(vg["ek"])) else None,
                             "ek_ci95": ci_gk, "n_ek": n_gk,
                             "signal": bool(ci_gm and ci_gec and ci_gm[0] > 0 and ci_gec[0] > 0
                                            and cgp["pca"][0] is not None and cgp["pca"][1] >= 8
                                            and cgp["haar"][0] is not None and cgp["haar"][1] >= 8
                                            and cgp["pca"][0][0] > 0 and cgp["haar"][0][0] > 0
                                            and not np.all(np.isnan(vg["emc"]))
                                            and float(np.nanmean(vg["emc"])) > 0),
                             "refuted": bool(ci_gm and ci_gec and n_gm >= 8 and n_gec >= 8
                                             and ci_gm[1] < DELTA_MIN
                                             and ci_gec[1] < DELTA_MIN),
                             "soft": bool(ci_gm is not None and ci_gm[0] <= 0.0 <= ci_gm[1])}}
        n_sig = sum(1 for c in cells.values() if c["weak"]["signal"])
        n_ref = sum(1 for c in cells.values() if c["weak"]["refuted"])
        n_soft = sum(1 for c in cells.values() if c["weak"]["soft"])
        n_sig_g = sum(1 for c in cells.values() if c["gbar"]["signal"])
        n_ref_g = sum(1 for c in cells.values() if c["gbar"]["refuted"])
        n_cells = len(cells)
        if not g4_ok or not g5_ok:
            verdict = "NOT_RESOLVED"
            verdict_reason = (f"DEAD_CELL (G4'={g4_ok} / G5={g5_ok}): вердикт принуждён, "
                             f"фиксация в артефакте")
        elif n_sig >= 2:
            verdict, verdict_reason = "WEAK_CAUSAL_SIGNAL", f"{n_sig}/{n_cells} клеток signal"
        elif n_ref == n_cells:
            verdict, verdict_reason = "WEAK_CAUSAL_REFUTED_MEANJ", f"все {n_cells}/{n_cells} клеток refuted (CI_upper < δ_min)"
        elif n_soft >= 4:
            verdict, verdict_reason = "SOFTENING_DOMINATED", f"{n_soft}/{n_cells} клеток: CI95 через 0"
        elif n_sig_g >= 2:
            verdict, verdict_reason = "WEAK_CAUSAL_SIGNAL_GBAR", f"{n_sig_g}/{n_cells} клеток signal (canonical full)"
        elif n_ref_g == n_cells:
            verdict, verdict_reason = "WEAK_CAUSAL_REFUTED_GBAR", f"все {n_cells}/{n_cells} клеток refuted (canonical full)"
        else:
            verdict, verdict_reason = "NOT_RESOLVED", "смешанный случай (см. cells)"
        logline(f"VERDICT: {verdict} — {verdict_reason}")
        logline(f"  signal={n_sig} refuted={n_ref} softening={n_soft} (weak); "
                f"gbar: signal={n_sig_g} refuted={n_ref_g}; "
                f"g4_ok={g4_ok} g5_ok={g5_ok}")

        # ================= smoke-проверки DS1-DS5 (только SMOKE) =============
        smoke_ds = None
        if SMOKE:
            A0 = rec[(LAYS[0], KS[0])]["arms"]
            v1 = [ds1_vals.get((d, LAYS[0], KS[0], "top")) for d in range(N_DOCS_RUN)]
            smoke_ds = {"DS1": {"per_doc": v1,
                                "ok": bool(all(v is not None and v >= 0.1 for v in v1))}}
            p2 = [smoke_extra[d]["P_top4"] for d in range(N_DOCS_RUN)]
            ok_ds2 = bool(all(np.isfinite(v) and abs(v) >= 0.01 for v in p2)
                          and all(np.sign(v) == np.sign(p2[0]) for v in p2))
            smoke_ds["DS2"] = {"per_doc": p2, "mean_P_top4": float(np.mean(p2)),
                               "ok": ok_ds2,
                               "gate": ("r5.1: finite + однознак + |dCE_P| >= 0.01 "
                                        "(знак-ожидание r5 снято: off-by-one CE)")}
            r3 = [(smoke_extra[d]["R_pf_top4"] / A0["top"][d]["dCE"]
                   if abs(A0["top"][d]["dCE"]) > 1e-12 else float("nan"))
                  for d in range(N_DOCS_RUN)]
            smoke_ds["DS3"] = {"ratio_R_P_per_doc": [float(x) for x in r3],
                               "expect": "2..6 (pre-study 4.0)", "stop": False}
            t4 = [A0["top"][d]["tau_star"] for d in range(N_DOCS_RUN)]
            smoke_ds["DS4"] = {"tau_per_doc": t4,
                               "converged": bool(all(t is not None for t in t4)),
                               "in_0.1_10": bool(all(t is not None and 0.1 <= t <= 10
                                                      for t in t4)),
                               "ec_minus_em": [float(A0["top"][d]["excess_cal"]
                                                     - A0["top"][d]["excess_match"])
                                                for d in range(N_DOCS_RUN)],
                               "ok": bool(all(t is not None for t in t4))}
            a5 = {}
            for d in range(N_DOCS_RUN):
                for arm in ("weak", "gbar"):
                    a5[f"d{d}_{arm}"] = {"alpha": float(A0[arm][d]["alpha"]),
                                          "dose_res": float(A0[arm][d]["dose_res"])}
            smoke_ds["DS5"] = {"per_doc": a5,
                               "ok": bool(all(ALPHA_LO <= v["alpha"] <= ALPHA_HI
                                              and v["dose_res"] < 0.25
                                              for v in a5.values()))}
            gz0 = l23_rec["gzero"]["top4"][0]
            smoke_ds["Gzero_smoke"] = {"doc0_top4": dict(gz0),
                                       "ok": bool(abs(gz0["dCE"]) <= 1e-6
                                                  and gz0["max_logit_diff"] <= 1e-6)}
            logline(f"SMOKE DS1={smoke_ds['DS1']['ok']} DS2={smoke_ds['DS2']['ok']} "
                    f"DS3_r={ [round(x, 2) for x in r3] } DS4={smoke_ds['DS4']['ok']} "
                    f"DS5={smoke_ds['DS5']['ok']} Gzero={smoke_ds['Gzero_smoke']['ok']}")
            # стоп по DS-гейтам — в main ПОСЛЕ записи smoke-артефакта (диагностика)

        # ================= артефакт ==========================================
        tstage(tr, time.time() - t0, "artifact")
        deviations = []
        deviations.append(
            "r5.1: DS2-гейт переписан (plumbing: finite+однознак+|dCE_P|>=0.01); "
            "pre-study CE off-by-one (self-conditional) — знак-ожидание r5 снято; "
            "δ_min=0.020 перепроверен исправленной power (se(weak4-rand4)=0.0051, "
            "3.92*SE=0.0200) — значение не изменено")
        deviations.append(
            "r6 (ревью r5, docs/AT2d_review.md): dose-формула канонизирована "
            "(g = 1 + w, radial первым — F3); G4' = plumbing (строго > 0) + G5 = "
            "лестница-контроль первичного режима (F1/F6); базисы f64, ortho f64 (F4); "
            "CI 2.5/97.5 → односторонний per-cell α = 0.025, FWER ≈ 2.5% (F2); "
            "refuted требует n≥8 (m5); артефакт двухпроходный + manifest, маркер в "
            "лог-файл (m6/m7); PFR_REF guard (m8)")
        for (L, k) in rec:
            A = rec[(L, k)]["arms"]
            for arm in ("top", "weak", "gbar", "pca", "haar"):
                un = [d for d in range(N_DOCS_RUN) if A[arm][d].get("unmatched")]
                if un:
                    deviations.append(f"L{L}k{k} {arm}: unmatched (α вне границ) docs={un}")
            nt = [d for d in range(N_DOCS_RUN) if A["weak"][d]["tau_star"] is None]
            if nt:
                deviations.append(f"L{L}k{k} weak: tsham_unbounded (τ* вне сетки) docs={nt}")
        expectations = {}
        for (L, k) in rec:
            A = rec[(L, k)]["arms"]
            em_top = [m["excess_match"] for m in A["top"]]
            dce_rg = float(np.mean([m["dCE"] for m in A["randG"]]))
            expectations[f"L{L}k{k}"] = {
                "excess_match_top": {"mean": float(np.nanmean(em_top)),
                                     "ci95": ci_doc(em_top, rngb)[0]},
                "abs_dce_randG4": abs(dce_rg),
                "randG_responsive": bool(abs(dce_rg) >= 0.005),
                "ladder_means": {str(a): float(np.mean(rec[(L, k)]["ladder"][_lk(a)]))
                                 for a in LADDER}}
        if l23_rec.get("P"):
            dce_pt = float(np.mean([r["dCE"] for r in l23_rec["P"]["top4"]]))
            dce_pr = float(np.mean([r["dCE"] for r in l23_rec["P"]["randG4"]]))
            expectations["l23_P_specificity"] = {"top4": dce_pt, "randG4": dce_pr,
                                                  "specific": bool(abs(dce_pt) > abs(dce_pr))}
        if LAYS and 19 in LAYS:
            m19 = float(np.nanmean([rec[(19, k)]["arms"]["weak"][d]["excess_match"]
                                    for k in KS for d in range(N_DOCS_RUN)
                                    if not rec[(19, k)]["arms"]["weak"][d]["unmatched"]]))
            mgn = float(np.nanmean([rec[(L, k)]["arms"]["weak"][d]["excess_match"]
                                    for L in LAYS if L != 19 for k in KS
                                    for d in range(N_DOCS_RUN)
                                    if not rec[(L, k)]["arms"]["weak"][d]["unmatched"]]))
            expectations["l19_vs_GDN"] = {"L19_em_weak": m19, "GDN_mean_em_weak": mgn}

        out = {
            "run": ("at2d_t22_r1" if MAIN_RUN else "at2d_smoke"),
            "assembled_at": datetime.now().isoformat(),
            "config": {
                "revision": REVISION,
                "script_sha": sha256_file(os.path.abspath(__file__)),
                "seed": SEED, "k": list(KS), "layers": list(LAYS), "l_zero": L_ZERO,
                "layer_types": list(ltypes),
                "n_docs": N_DOCS_RUN, "boot_b": BOOT_B,
                "model": "Qwen/Qwen3.5-0.8B (float32, eager, без fla)",
                "device": "cpu",
                "thread_pinning": "OMP/OPENBLAS/MKL=4 + torch 4",
                "model_weights_sha": model_weights_sha,
                "delta_min": DELTA_MIN,
                "alpha_bounds": [ALPHA_LO, ALPHA_HI], "ladder": list(LADDER),
                "window": {"ablate": "0..63", "metric_pos": "64..126",
                           "targets": "65..127", "n": N_WIN,
                           "bins_targets": ["65-77", "78-90", "91-103",
                                            "104-116", "117-127"]},
                "gbar_full": {"n_dt": S_LEN * N_DOCS_RUN, "docs": N_DOCS_RUN,
                              "accum": "f64"},
                "p0r_sha": P0R_SHA,
                "at2c_script_sha": sha256_file(
                    os.path.join(os.path.dirname(__file__), "AT2c.py")),
                "started_at": datetime.fromtimestamp(t0).isoformat(),
            },
            "gates": gates,
            "baseline": {f"d{d}": base[d] for d in range(N_DOCS_RUN)},
            "arms": {f"L{L}k{k}": {
                "P_future": {arm: {
                    "per_doc": [m for m in rec[(L, k)]["arms"][arm]],
                    "n_matched": int(sum(1 for m in rec[(L, k)]["arms"][arm]
                                         if not m.get("unmatched"))),
                    "dCE_mean": float(np.nanmean([m["dCE"] for m in rec[(L, k)]["arms"][arm]])),
                    "dCE_ci95": ci_doc([m["dCE"] for m in rec[(L, k)]["arms"][arm]], rngb)[0],
                    "excess_match_mean": float(np.nanmean(
                        [m["excess_match"] for m in rec[(L, k)]["arms"][arm]])),
                    "excess_match_ci95": ci_doc(
                        [m["excess_match"] for m in rec[(L, k)]["arms"][arm]], rngb)[0],
                    "excess_cal_mean": float(np.nanmean(
                        [m["excess_cal"] for m in rec[(L, k)]["arms"][arm]])),
                    "excess_cal_ci95": ci_doc(
                        [m["excess_cal"] for m in rec[(L, k)]["arms"][arm]], rngb)[0],
                } for arm in ("top", "weak", "gbar", "pca", "haar", "randG")},
                "ladder": {f"d{a}": {"per_doc": [float(x) for x in rec[(L, k)]["ladder"][_lk(a)]],
                                     "mean": float(np.mean(rec[(L, k)]["ladder"][_lk(a)]))}
                           for a in LADDER},
            } for (L, k) in rec},
            "l23": {
                "gzero": {n: [r for r in l23_rec["gzero"][n]]
                          for n in l23_rec["gzero"]},
                "P": {n: [r for r in l23_rec["P"][n]] for n in l23_rec["P"]},
            },
            "expectations": expectations,
            # m1 r5: gbar full-vs-sub углы (M4 r3) + randG-фреймы sha per (L,k,d)
            "gbar_angles_full_vs_sub": angles_sub,
            "randG_frames": frame_sha_tab,
            "verdict": verdict,
            "verdict_reason": verdict_reason,
            "verdict_detail": {
                "DELTA_MIN": DELTA_MIN,
                "priority": "SIGNAL > REFUTED > SOFTENING_DOMINATED > GBAR > NOT_RESOLVED",
                "counts": {"signal": n_sig, "refuted": n_ref, "softening": n_soft,
                           "signal_gbar": n_sig_g, "refuted_gbar": n_ref_g,
                           "n_cells": n_cells, "g4_ok": bool(g4_ok),
                           "g5_ok": bool(g5_ok)},
                "cells": cells,
            },
            "smoke_ds": smoke_ds,
            "deviations": deviations,
            "wall_sec": round(time.time() - t0, 1),
        }
        return out

    # ================= S9 (smoke) / артефакт / финиш ======================
    if SMOKE:
        out1 = run_pipeline()
        md5_1 = _md5_det(out1)
        logline(f"S9 run 1: det md5 = {md5_1}")
        out2 = run_pipeline()
        md5_2 = _md5_det(out2)
        logline(f"S9 run 2: det md5 = {md5_2}")
        s9_ok = (md5_1 == md5_2)
        gates["s9_smoke"] = {"md5_pair": [md5_1, md5_2], "ok": bool(s9_ok)}
        out1["gates"]["s9_smoke"] = {"md5_pair": [md5_1, md5_2], "ok": bool(s9_ok)}
        if not s9_ok:
            raise SystemExit("S9 FAIL: два smoke-прогона не бит-идентичны")
        logline("S9: OK (md5 дет-под-артефакта бит-в-бит)")
        out = out1
    else:
        out = run_pipeline()

    raw = json.dumps(_san(out), ensure_ascii=False, indent=1)
    if "NaN" in raw or "Infinity" in raw:
        raise SystemExit("artifact: NaN/Infinity in the JSON — STOP")
    with open(ART, "w") as f:
        f.write(raw)
    logline(f"artifact (pass 1): {ART} ({os.path.getsize(ART) // 1024} KB)")
    if SMOKE:
        det_path = os.path.join(SMOKE_DIR, "AT2d_smoke_det.json")
        with open(det_path, "w") as f:
            f.write(json.dumps(_san(_det_subset(out)), sort_keys=True,
                               ensure_ascii=False))
        logline(f"smoke det (S9): {det_path} ({os.path.getsize(det_path) // 1024} KB)")
        # DS-стоп (TZ §11) — ПОСЛЕ записи артефакта (сбой = диагностируемый)
        for ds_name in ("DS1", "DS2", "DS4", "Gzero_smoke"):
            if not out["smoke_ds"][ds_name]["ok"]:
                raise SystemExit(f"SMOKE {ds_name} FAIL: {out['smoke_ds'][ds_name]}")

    # ---- финиш: маркер ДО записи лог-файла (m7 r5), лог ПЕРВЫМ
    #      (log_md5 в финальной trackio-точке), ПОТОМ pass-2 артефакта
    #      (finished_at/log_md5 внутри) + manifest (m6 r5) ----
    logline(f"AT2d (T2.2) DONE wall={time.time() - t0:.1f}s verdict={out['verdict']}")
    logline("AT2D_SMOKE_DONE" if SMOKE else "AT2D_MAIN_DONE")
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_LOG) + "\n")
    out["finished_at"] = datetime.now().isoformat()
    out["log_md5"] = hashlib.md5(open(LOG_PATH, "rb").read()).hexdigest()
    if MAIN_RUN:
        out["trackio_run"] = "at2d_t22_r1"
        tr_done[0] = True
        tlog(tr, time.time() - t0, {
            "finished_at": datetime.now().isoformat(), "status": "ok",
            "verdict": out["verdict"],
            "log_md5": out["log_md5"]})
        tr.finish()
    raw2 = json.dumps(_san(out), ensure_ascii=False, indent=1)
    with open(ART, "w") as f:
        f.write(raw2)
    man_path = os.path.splitext(ART)[0] + "_manifest.json"
    with open(man_path, "w") as f:
        json.dump({
            "artifact": os.path.abspath(ART),
            "artifact_sha256": sha256_file(ART),
            "log": os.path.abspath(LOG_PATH),
            "log_md5": out["log_md5"],
            "run": out["run"],
            "revision": REVISION,
            "verdict": out["verdict"],
            "finished_at": out["finished_at"],
            "script_sha": sha256_file(os.path.abspath(__file__)),
        }, f, indent=1, ensure_ascii=False)
    logline(f"artifact (pass 2) + manifest: {ART} / {man_path} "
            f"({os.path.getsize(ART) // 1024} KB)")


# ------------------------------------------------------------------- reanalysis
def reanalysis():
    """--reanalysis (F1 r5): переоценка вердикта из frozen raw-артефакта БЕЗ
    форвардов. G4' (plumbing: строго > 0) + G5 (лестница α=4, positive control
    первичного режима) + ortho f64 (пересчёт из кэша, r6-базисы) + dose-
    коррекция (F3: канон g = 1 + w, radial первым). Sanity: old-формула dose
    должна совпасть с артефактом (<1e-6). Raw-артефакт НЕ меняется; запись
    AT2d_v2.json + manifest + log c маркером AT2D_REANALYSIS_DONE."""
    t0 = time.time()
    os.makedirs(RESULTS, exist_ok=True)
    raw_path = ART
    raw_sha = sha256_file(raw_path)
    raw = json.load(open(raw_path))
    logline(f"reanalysis: raw = {raw_path} sha256[:16]={raw_sha[:16]}")
    logline(f"reanalysis: raw verdict = {raw['verdict']} ({raw['verdict_reason']})")
    if raw.get("config", {}).get("revision") not in ("r5.1", "r5.2"):
        logline(f"reanalysis NOTE: raw revision = {raw.get('config', {}).get('revision')}")

    # --- веса (F3): input_layernorm следующего слоя / final norm для L23 ---
    cfg_j = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    ltypes = cfg_j["text_config"]["layer_types"] if "text_config" in cfg_j else cfg_j["layer_types"]
    from safetensors import safe_open
    W = {}
    with safe_open(MODEL_SFT, framework="pt") as f:
        for L in list(LAYS) + [L_ZERO]:
            key = (f"model.language_model.layers.{L + 1}.input_layernorm.weight"
                   if L + 1 < len(ltypes) else "model.language_model.norm.weight")
            W[L] = f.get_tensor(key).double().numpy()

    # --- базисы (f64, r6) + dose old/new + ortho f64 ---
    ortho_max = 0.0
    dose_rep = {}
    l23_p_rep = {}
    worst_old = 0.0
    n_checked = 0
    for L in list(LAYS) + [L_ZERO]:
        ck = torch.load(os.path.join(CACHE_DIR, f"exp27b_jcache_L{L}.pt"),
                        map_location="cpu", weights_only=False)
        Hf = f2a_inv(ck["H"], (D,)).numpy().astype(np.float32)
        JRf = f2a_inv(ck["JR"], (128, D))
        Pf = f2a_inv(ck["P128"], (128,))
        VJ, Vpca, Vrg, Vhaar = build_basis(L, JRf, Pf, Hf)
        for k in KS:
            for V in [VJ[f"top{k}"], VJ[f"weak{k}"], VJ[f"gbar{k}"], Vhaar[k]]:
                ortho_max = max(ortho_max,
                                float(np.abs(V.T @ V - np.eye(k)).max()))
            for d in range(N_DOCS):
                ortho_max = max(ortho_max,
                                float(np.abs(Vpca[k][d].T @ Vpca[k][d] - np.eye(k)).max()),
                                float(np.abs(Vrg[k][d].T @ Vrg[k][d] - np.eye(k)).max()))
        if L == L_ZERO:
            # L23-P-контроль (окно 0..127): dose-коррекция для α-matчинга top4
            for d in range(N_DOCS):
                do_t, dn_t, _ = dose_pair(Hf[d], VJ["top4"], W[L], 0, S_LEN)
                do_r, dn_r, _ = dose_pair(Hf[d], Vrg[4][d], W[L], 0, S_LEN)
                a_used = raw["l23"]["P"]["top4"][d]["alpha"]   # list: индекс = док
                a_new = float(np.sqrt(dn_r / dn_t)) if dn_t > 0 else float("inf")
                l23_p_rep[f"d{d}"] = {"D_old_top": do_t, "D_new_top": dn_t,
                                      "D_old_rg": do_r, "D_new_rg": dn_r,
                                      "alpha_used": float(a_used),
                                      "alpha_new": a_new,
                                      "mismatch": float((a_used ** 2) * dn_t / dn_r)
                                      if dn_r > 0 else None}
        else:
            for k in KS:
                per_arm = {}
                for name in ("top", "weak", "gbar", "pca", "haar", "randG"):
                    rows_ = []
                    for d in range(N_DOCS):
                        V64 = (VJ[f"{name}{k}"] if name in ("top", "weak", "gbar")
                               else (Vpca[k][d] if name == "pca"
                                     else (Vhaar[k] if name == "haar" else Vrg[k][d])))
                        V32 = V64.astype(np.float32)   # r5.1-вектор (бит-в-бит)
                        do, dn, d1 = dose_pair(Hf[d], V32, W[L], 0, AB_W)
                        artD = raw["arms"][f"L{L}k{k}"]["P_future"][name]["per_doc"][d].get("D")
                        if artD is not None:  # randG без "D" в per_doc
                            worst_old = max(worst_old, abs(do - artD) / max(artD, 1e-30))
                            n_checked += 1
                        rows_.append((do, dn, d1))
                    per_arm[name] = rows_
                dose_rep[(L, k)] = per_arm
        del ck, JRf, Pf
        logline(f"reanalysis: L{L} done ({time.time() - t0:.0f}s)")
    if worst_old > 1e-6:
        raise SystemExit(f"reanalysis SANITY FAIL: old-dose vs артефакт "
                         f"max_rel_diff = {worst_old:.3e} > 1e-6")
    logline(f"reanalysis sanity: old-формула dose == артефакт "
            f"({n_checked} значений, max_rel_diff = {worst_old:.3e})")
    logline(f"reanalysis: ortho f64 max|VᵀV−I| = {ortho_max:.3e}")

    # --- сводка dose-коррекции (F3): α_old vs α_new, mismatch (1.0 = матч) ---
    # excess_KL post-hoc (m1 r5: в r5.1-артефакте KL per-doc есть, excess_KL — нет)
    ek_rep = {}
    for cell in [f"L{L}k{k}" for L in LAYS for k in KS]:
        A = raw["arms"][cell]["P_future"]
        vals = [(A["weak"]["per_doc"][d]["KL"] - A["randG"]["per_doc"][d]["KL"])
                for d in range(N_DOCS) if not A["weak"]["per_doc"][d].get("unmatched")]
        vals = [v for v in vals if v is not None and v == v]
        if vals:
            ek_rep[cell] = {"mean": float(np.mean(vals)),
                            "min": float(np.min(vals)), "max": float(np.max(vals)),
                            "n": len(vals)}
    def _cell_dose(L, k):
        rg_old = np.array([dose_rep[(L, k)]["randG"][d][0] for d in range(N_DOCS)])
        rg_new = np.array([dose_rep[(L, k)]["randG"][d][1] for d in range(N_DOCS)])
        s = {}
        for name in ("top", "weak", "gbar", "pca", "haar", "randG"):
            o = np.array([dose_rep[(L, k)][name][d][0] for d in range(N_DOCS)])
            nw = np.array([dose_rep[(L, k)][name][d][1] for d in range(N_DOCS)])
            a_old = np.sqrt(rg_old / o)
            a_new = np.sqrt(rg_new / nw)
            mismatch = (a_old ** 2) * nw / rg_new
            s[name] = {"D_old_mean": float(o.mean()), "D_new_mean": float(nw.mean()),
                       "ratio_new_old_mean": float((nw / o).mean()),
                       "alpha_old_mean": float(a_old.mean()),
                       "alpha_new_mean": float(a_new.mean()),
                       "mismatch_mean": float(mismatch.mean()),
                       "mismatch_range": [float(mismatch.min()), float(mismatch.max())]}
        return s
    dose_summary = {f"L{L}k{k}": _cell_dose(L, k) for L in LAYS for k in KS}

    # --- гейты из артефакта (G4', G5) ---
    g4_cells = {}
    for cell in [f"L{L}k{k}" for L in LAYS for k in KS]:
        vals = [raw["arms"][cell]["P_future"]["randG"]["per_doc"][d]["dCE"]
                for d in range(N_DOCS)]
        m = float(np.mean(vals))
        g4_cells[cell] = {"abs_dce_randG": abs(m), "mean": m,
                          "ok": bool(abs(m) > 0.0)}
    n_resp = sum(1 for v in g4_cells.values() if v["ok"])
    g4_ok = n_resp >= 4
    g5_cells = {}
    for cell in [f"L{L}k{k}" for L in LAYS for k in KS]:
        m4 = float(raw["arms"][cell]["ladder"][_lk(4.0)]["mean"])
        g5_cells[cell] = {"abs_dce_ladder_d4": abs(m4), "mean": m4,
                          "ok": bool(abs(m4) >= 0.005)}
    n_g5 = sum(1 for v in g5_cells.values() if v["ok"])
    g5_ok = n_g5 >= 1

    # --- дерево вердиктов (из raw cells; те же условия, что в main) ---
    cells = raw["verdict_detail"]["cells"]
    n_sig = sum(1 for c in cells.values() if c["weak"]["signal"])
    n_ref = sum(1 for c in cells.values() if c["weak"]["refuted"])
    n_soft = sum(1 for c in cells.values() if c["weak"]["soft"])
    n_sig_g = sum(1 for c in cells.values() if c["gbar"]["signal"])
    n_ref_g = sum(1 for c in cells.values() if c["gbar"]["refuted"])
    n_cells = len(cells)
    if not g4_ok or not g5_ok:
        verdict, reason = "NOT_RESOLVED", (f"DEAD_CELL (G4'={g4_ok} / G5={g5_ok}): "
                                            f"вердикт принуждён")
    elif n_sig >= 2:
        verdict, reason = "WEAK_CAUSAL_SIGNAL", f"{n_sig}/{n_cells} клеток signal"
    elif n_ref == n_cells:
        verdict, reason = "WEAK_CAUSAL_REFUTED_MEANJ", (
            f"все {n_cells}/{n_cells} клеток refuted (CI_upper < δ_min)")
    elif n_soft >= 4:
        verdict, reason = "SOFTENING_DOMINATED", f"{n_soft}/{n_cells} клеток: CI95 через 0"
    elif n_sig_g >= 2:
        verdict, reason = "WEAK_CAUSAL_SIGNAL_GBAR", f"{n_sig_g}/{n_cells} клеток signal (canonical full)"
    elif n_ref_g == n_cells:
        verdict, reason = "WEAK_CAUSAL_REFUTED_GBAR", (
            f"все {n_cells}/{n_cells} клеток refuted (canonical full)")
    else:
        verdict, reason = "NOT_RESOLVED", "смешанный случай (см. cells)"
    logline(f"reanalysis VERDICT: {verdict} — {reason}")
    logline(f"  signal={n_sig} refuted={n_ref} soft={n_soft} (weak); "
            f"gbar: sig={n_sig_g} ref={n_ref_g}; g4'={g4_ok} ({n_resp}/{n_cells}) "
            f"g5={g5_ok} ({n_g5}/{n_cells})")

    # --- v2-артефакт + manifest + log (маркер ДО записи файла) ---
    v2 = dict(raw)
    v2["reanalysis"] = {
        "mode": "reanalysis: frozen raw-артефакт → вердикт, без форвардов (F1 r5)",
        "assembled_at": datetime.now().isoformat(),
        "raw_sha256": raw_sha,
        "raw_verdict": raw["verdict"],
        "script_sha": sha256_file(os.path.abspath(__file__)),
        "revision": REVISION,
        "sanity": {"old_dose_vs_artifact": {"n_values": int(n_checked),
                                            "max_rel_diff": float(worst_old)}},
        "gates": {
            "G4_sham": {"n_cells_responsive": int(n_resp), "of": int(n_cells),
                        "threshold": 0.0, "cells": g4_cells, "ok": bool(g4_ok),
                        "gate": "r6 (F1 r5): plumbing — строго > 0 на ≥4/10"},
            "G5_ladder": {"n_cells_d4_gte_0.005": int(n_g5), "of": int(n_cells),
                          "threshold": 0.005, "cells": g5_cells, "ok": bool(g5_ok),
                          "gate": "r6 (F6 r5): positive control первичного режима (α=4)"},
            "ortho_f64": {"max_norm_diff": float(ortho_max),
                          "ok": bool(ortho_max < 1e-6),
                          "note": ("пересчёт из кэша в f64 (r6-базисы); в raw-прогоне "
                                   "считалось в f32 (F4 r5) — raw-значение 7.15e-7")},
        },
        "dose_correction_F3": {"per_cell": dose_summary, "l23_P": l23_p_rep},
        "excess_KL_posthoc_m1": ek_rep,
        "counts": {"signal": n_sig, "refuted": n_ref, "softening": n_soft,
                   "signal_gbar": n_sig_g, "refuted_gbar": n_ref_g,
                   "n_cells": n_cells, "g4_ok": bool(g4_ok), "g5_ok": bool(g5_ok)},
        "verdict": verdict,
        "verdict_reason": reason,
    }
    logline(f"AT2d reanalysis DONE wall={time.time() - t0:.1f}s verdict={verdict}")
    logline("AT2D_REANALYSIS_DONE")
    rean_log = os.path.join(RESULTS, "AT2d_v2_reanalysis.log")
    with open(rean_log, "w") as f:
        f.write("\n".join(_LOG) + "\n")
    v2["reanalysis"]["log_md5"] = hashlib.md5(open(rean_log, "rb").read()).hexdigest()
    rean_art = os.path.join(RESULTS, "AT2d_v2.json")
    rawv = json.dumps(_san(v2), ensure_ascii=False, indent=1)
    if "NaN" in rawv or "Infinity" in rawv:
        raise SystemExit("reanalysis artifact: NaN/Infinity — STOP")
    with open(rean_art, "w") as f:
        f.write(rawv)
    man_path = os.path.splitext(rean_art)[0] + "_manifest.json"
    with open(man_path, "w") as f:
        json.dump({
            "artifact": os.path.abspath(rean_art),
            "artifact_sha256": sha256_file(rean_art),
            "raw_artifact": os.path.abspath(raw_path),
            "raw_artifact_sha256": raw_sha,
            "log": os.path.abspath(rean_log),
            "log_md5": v2["reanalysis"]["log_md5"],
            "run": "at2d_t22_r1_reanalysis",
            "revision": REVISION,
            "verdict": verdict,
            "raw_verdict": raw["verdict"],
            "finished_at": v2["reanalysis"]["assembled_at"],
            "script_sha": sha256_file(os.path.abspath(__file__)),
        }, f, indent=1, ensure_ascii=False)
    logline(f"reanalysis artifact: {rean_art} "
            f"({os.path.getsize(rean_art) // 1024} KB) + manifest")


if __name__ == "__main__":
    if "--reanalysis" in sys.argv:
        reanalysis()
    else:
        main()
