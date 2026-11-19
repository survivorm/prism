# ==== T4 bootstrap (builder-injected, AT2e r3; TZ §6b) ====
import os as _t4os, sys as _t4sys, subprocess as _t4sp, time as _t4time, shutil as _t4sh
_T4_T0 = _t4time.time()
print(f"[t4-bootstrap] start py={_t4sys.version.split()[0]}", flush=True)
_t4sp.check_call([_t4sys.executable, "-m", "pip", "install", "-q",
                  "transformers==5.15.1", "tokenizers==0.22.2", "huggingface_hub"])
_t4os.makedirs("/data/Qwen", exist_ok=True)
_MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
if not _t4os.path.exists(_MODEL_DIR):
    from huggingface_hub import snapshot_download
    _t4os.symlink(snapshot_download("Qwen/Qwen3.5-0.8B"), _MODEL_DIR)
def _locate(fname):
    for _r, _d, _f in _t4os.walk("/kaggle/input"):
        if fname in _f:
            return _t4os.path.join(_r, fname)
    raise FileNotFoundError(fname)
_POOL_SRC = _locate("exp27b_pool_ids_L23.pt")
_t4os.makedirs("/data/prism_kernels/exp27b/results", exist_ok=True)
_t4sh.copy(_POOL_SRC, "/data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt")
# envVars metadata (kaggle_run.sh) — НЕ проверенный механизм; bootstrap — надёжный
# единственный путь (at16 паттерн: хардкод /kaggle/working)
_t4os.environ.setdefault("PRISM_KAGGLE_ROOT", "/kaggle/working")
_t4os.environ["AT2E_MODEL_DIR"] = _MODEL_DIR
_t4os.environ["AT2E_POOL"] = "/data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt"
_t4os.environ["AT2E_JCACHE"] = "/data/prism_kernels/exp27b/results/exp27b_jcache_L12.pt"
_t4os.environ["AT2E_DEVICE"] = "cuda"
_t4os.environ["AT2E_TRACKIO"] = "0"
_t4os.environ["AT2E_NO_JCACHE"] = "1"
_t4os.environ["AT2E_SCRIPT_SHA"] = "2137a7f912e719d0be17f7c5ccf47323012f9384e22cbce16c34c2db05c9efd2"
_t4sys.argv = ["at2e_t4", "--canary"]
print(f"[t4-bootstrap] ready in {_t4time.time()-_T4_T0:.1f}s; mode={'canary' if '--canary' in _t4sys.argv else 'main'}", flush=True)
# ==== конец T4 bootstrap ====

#!/usr/bin/env python3
# AT2e.py — стадия 1 (existence pilot): скретчпад-гипотеза T2, TZ r5.2 (docs/AT2e_prereg.md).
# Код r3 (ревью кода r2: C2/C3/C4, V1-V6, минорные — все приняты; C1 — не воспроизводится,
# evidence в docs/AT2e_response_r2.md). eager, fla guard'ом (инвариант 1), f32-модель /
# f64-аккумуляция. CPU — основной путь раунда (TZ §6b, r5.2); T4 — тонкая обёртка над
# этим же скриптом (AT2E_DEVICE=cuda ЯВНО, device-автоселекта нет — r3 C3).
#
# Режимы:
#   AT2e_SMOKE=1   smoke (S1-S13): 1 док wiki/0, L12, t=32, N_random=8, подъём 2 ит.
#   (без флагов)   main: at2e_s1_r3, 16 клеток {L11,L12}x{wiki,math,code,chat}x{t32,t96},
#                  условные стадии 1d/1b/1c + отчётные secondary/cext_ref.
#   --resume       main: продолжить с per-cell чекпойнта (at2e_s1_r3_partial.json);
#                  клетка считается по cell_id, стадии — по stage_done (B2).
#   --reanalysis   disk-реанализ frozen raw-артефакта БЕЗ форвардов (вердикты — чистая
#                  функция от сохранённых метрик).
#
# Переменные окружения (T4-обёртка, TZ §6b): AT2E_MODEL_DIR, AT2E_POOL, AT2E_JCACHE,
# AT2E_DEVICE (cpu|cuda), AT2E_TRACKIO=0 (в kernel trackio не пишется).
#
# Механика (pre-study r1-r4): патч выхода слоя L в позиции t (hook на model.model.layers[L]),
# центральные разности, q_s = dz^T P_s dz, P_s = diag(p_s) - p_s p_s^T (f64),
# VJP через vector-leaf (1 grad-fwd + 2 backward, retain_graph).
#
# Ключевые определения (TZ §3):
#   q_k(v)      = Var_{p_{t+k}}(dz_k), dz_k = (z+(t+k) - z-(t+k))/2, a = eps_a*||h_t||
#   lambda_K1   = 2*mean(q_1,q_2)/(q_0 + a^2*tau*eps0)          (объект подъёма, r3 N1)
#   lambda_K2   = 2*mean(q_4,q_8)/(q_0 + a^2*tau*eps0)          (валидация, out-of-sample)
#   lambda_full = 8*mean(q_1..q_8)/(q_0 + a^2*tau*eps0)         (эстиманд, отчёт)
#   eps0        = lam_max(G^{-1/2} C_t|_B G^{-1/2}), G = B0^T B0 (r3 N3)
#   rho         = mean(q_1..q_8)/q_0                            (нормировано на H, theory VII.6)
#   transfer-gap = lambda_K1(v*;t')/lambda_K1(v*_t';t') <= 0.3  (стадия 1d, r5 R1)
#
# Градиент подъёма: точный градиент зарегистрированного эстиманда
#   g = D*Ov_K1 - 2*qbar_K1*(Cv + eps_reg*v),  D = q_0 + a^2*tau*eps0, eps_reg = tau*eps0
# (r3: формула в r1-шапке была неверна — потеряны множитель 2 и ε-член; код всегда был
# верен, проверено cos(g, grad_analytic) = 1.0. TZ формула g ~ (v^T(C+epsI)v)Ov -
# lambda*(C+epsI)v — пропорциональна при v^TBv ~ 1; единицы согласованы с dose-free
# знаменателем a^2*tau*eps0, r1-правка).

import sys
sys.modules["fla"] = None  # инвариант 1: fla 0.5.x CUDA-only; guard ДО импортов

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import time
import json
import hashlib
import atexit
from datetime import datetime

import numpy as np
import torch
import transformers  # r3 V6: transformers_version в header
from transformers import AutoModelForCausalLM

torch.set_num_threads(4)

# r3 T4: в kernel'е PRISM_KAGGLE_ROOT=/kaggle/working (kaggle_run.sh); локально — репо
_KAGGLE_ROOT = os.environ.get("PRISM_KAGGLE_ROOT")
ROOT = (_KAGGLE_ROOT if _KAGGLE_ROOT
        else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# M1: пути — из окружения (T4-обёртка, TZ §6b r5.2); дефолт — локальный CPU-путь.
MODEL_DIR = os.environ.get("AT2E_MODEL_DIR", "/data/Qwen/Qwen3.5-0.8B")
POOL = os.environ.get("AT2E_POOL", "/data/prism_kernels/exp27b/results/exp27b_pool_ids_L23.pt")
JCACHE_L12 = os.environ.get("AT2E_JCACHE",
                            "/data/prism_kernels/exp27b/results/exp27b_jcache_L12.pt")
POOL_SHA = "ded634f7d16cf33a9bb565de5cbc3faa288cfc90b15cdfa3e42365695fec4e03"
JCACHE_SHA = "976079ce1d3c84c7016a236683f86f5fee1ce6571ff6edd53cf4f6f9ac228a24"
_DEVICE_ENV = os.environ.get("AT2E_DEVICE")
# r3 C3: дефолт cpu (TZ r5.2: CPU — основной путь раунда); cuda — только явное
# AT2E_DEVICE=cuda (T4-обёртка). Автоподхват GPU противоречит пре-регу и делал
# DEVICE-ловушкой любую машину с GPU.
DEVICE = _DEVICE_ENV if _DEVICE_ENV in ("cpu", "cuda") else "cpu"
assert DEVICE == "cpu" or torch.cuda.is_available(), \
    "AT2E_DEVICE=cuda задано, но CUDA недоступна"

RESULTS = os.path.join(ROOT, "results", "AT2e")
SMOKE = os.environ.get("AT2e_SMOKE", "0") == "1"
REVISION = "r3"
RUN_NAME = "at2e_smoke_r3" if SMOKE else "at2e_s1_r3"

D = 1024
S_LEN = 128
H = 8                      # окно будущего k=1..8
LAYERS = (11, 12)
T_POS = (32, 96)
DOCS = ("wiki", "math", "code", "chat")
SEED = 0
SEED_NULL = 1234
TAUS = (1e-3, 1e-2, 5e-2)
TAU_PRIMARY = 1e-2
# r5 R3: явные имена амплитуд (относительные: a = eps * ||h_t||)
A_LO_REL = 3e-4
A_HI_REL = 1e-3
A_FLOOR_REL = 1e-9
PROBE_AMPS_REL = (3e-3, 1e-2)   # 1c nonlinearity (report-only)
EPS_ASCENT_TAU = 1e-2            # eps_ascent = 1e-2 * eps0 (r2 B2)
N_RANDOM = 8 if SMOKE else 64
N_ASCENT = 2 if SMOKE else 8
GAP_MARGIN = 0.3                 # r5 R1 препрегистр. маржа (модель/якорь: TZ §3.11)
GAP_NEAR = 0.5                   # near-threshold полоса (0.3, 0.5]
FLOOR_FACTOR = 10                # все «10*floor» гейты
K1 = (1, 2)
K2 = (4, 8)
ULP32 = 1.1920929e-7            # 2^-23

_LOG = []
_TRACK_RUN = None
_T0 = None


def logline(s):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    _LOG.append(line)


def make_pwin(z0, t):
    """p_s = softmax(z⁰_s) в f64 (минор r5.2: одномасштабно с κ в probe —
    softmax(z⁺) f64; расхождение fp32-softmax/f64-softmax ~1e-7 снято)."""
    return {s: torch.softmax(z0[s].double(), dim=0).cpu().numpy()
            for s in range(t, t + H + 1)}  # r3 C3: .cpu() перед .numpy() (cuda-путь)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_tensor(t):
    """md5 битов tensor (f32) — дет-гейты."""
    return hashlib.md5(t.detach().cpu().numpy().tobytes()).hexdigest()


def sha_vec(v):
    """sha256 f64 вектора (1024,) — для траекторий v_i."""
    return hashlib.sha256(np.asarray(v, dtype=np.float64).tobytes()).hexdigest()


def _san(o):
    """sanitize для json.dump: numpy-типы/NaN/Inf/ndarray/torch -> python/None (AGENTS.md п.4)."""
    if isinstance(o, dict):
        return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_san(v) for v in o]
    if isinstance(o, np.ndarray):
        return _san(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, bytes):
        return o.hex()
    return o


def trackio_start(cfg):
    global _TRACK_RUN, _T0
    _T0 = time.time()
    if os.environ.get("AT2E_TRACKIO", "1") == "0":
        return  # T4-kernel: trackio не пишется (TZ §6b; JSON-лог, бэкилл migration)
    if "TRACKIO_DIR" not in os.environ:
        raise RuntimeError("TRACKIO_DIR не задан — запускать через ./run_one.sh "
                           "(он экспортирует TRACKIO_DIR) или export TRACKIO_DIR вручную")
    import trackio
    _TRACK_RUN = trackio.init(project="prism", name=RUN_NAME, group="AT2e", config=cfg)
    tlog(0, {"started_at": datetime.now().isoformat()})


def tlog(t, metrics):
    if _TRACK_RUN is not None:
        _TRACK_RUN.log(metrics, step=int(t))


def tstage(name, extra=None):
    m = {"stage": name, "stage_s": round(time.time() - _T0, 1)}
    if extra:
        m.update(extra)
    tlog(time.time() - _T0, m)


def trackio_finish(status):
    global _TRACK_RUN
    if _TRACK_RUN is not None:
        tlog(time.time() - _T0, {"finished_at": datetime.now().isoformat(), "status": status})
        _TRACK_RUN.finish()
        _TRACK_RUN = None  # atexit-trackio_crash: дубль finish не нужен


def trackio_crash():
    try:
        trackio_finish("crash")
    except Exception:
        pass

# ============================================================ модель и данные
def load_model():
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, torch_dtype=torch.float32, attn_implementation="eager").to(DEVICE)
    model.eval()
    model.requires_grad_(False)  # единственный градиентный лист — d (vjp_leaf)
    logline(f"model loaded: {model.config.model_type}, n_layers={len(model.model.layers)}, "
            f"vocab={model.config.vocab_size}, {time.time()-t0:.1f}s")
    return model


def load_pool():
    pool = torch.load(POOL, map_location="cpu", weights_only=False)
    ids = {}
    for dom in DOCS:
        ids[dom] = torch.as_tensor(np.asarray(pool["ids"][dom][0], dtype=np.int64))
    return pool, ids


def load_jcache_l12():
    return torch.load(JCACHE_L12, map_location="cpu", weights_only=False)


def init_null_weights(model, seed=SEED_NULL):
    """random weights x real text (стадия 1b): все параметры -> N(0, 1/sqrt(fan_in)),
    детерминированно одним RandomState(seed). Чанки по осям (минор r5.2: embedding
    248320x1024 — fp64-массив 2 GB + fp32-копия; чанк ~4M элементов -> пик < 100 MB)."""
    rng = np.random.RandomState(seed)
    n = 0
    for name, p in list(model.named_parameters()):
        shape = tuple(p.shape)
        if p.dim() >= 1:
            fan_in = int(p.shape[1]) if p.dim() >= 2 else 1
        else:
            fan_in = 1
        std = 1.0 / np.sqrt(fan_in) if fan_in > 1 else 1.0
        numel = int(np.prod(shape)) if len(shape) > 0 else 1
        if numel > 4_000_000 and len(shape) >= 1:
            # чанки вдоль оси 0 (r3 C2: последний чанк = остаток, не полная ось —
            # copy_ shape-mismatch на embedding 248320x1024)
            per = max(1, 4_000_000 // (numel // shape[0]))
            for st in range(0, shape[0], per):
                ch = min(per, shape[0] - st)
                with torch.no_grad():
                    p.data[st:st + ch].copy_(torch.as_tensor(
                        rng.normal(0.0, std, (ch,) + shape[1:]).astype(np.float32),
                        device=p.device))
        else:
            with torch.no_grad():
                p.data.copy_(torch.as_tensor(
                    rng.normal(0.0, std, shape).astype(np.float32), device=p.device))
        n += 1
    logline(f"null: reinit {n} параметр-тензоров (seed={seed})")


# ============================================================ forward-механика
class Fwd:
    """baseline + патч-форварды. input_ids/mask фиксированы per док."""

    def __init__(self, model, input_ids, mask):
        self.model = model
        self.input_ids = input_ids
        self.mask = mask

    def baseline(self, need_hidden=True):
        with torch.no_grad():
            out = self.model(input_ids=self.input_ids, attention_mask=self.mask,
                             output_hidden_states=need_hidden)
        z0 = out.logits[0]
        hs = {L: out.hidden_states[L + 1][0] for L in LAYERS} if need_hidden else {}
        return z0, hs

    def fwd(self, L_idx=None, t_pos=None, delta=None, grad=False):
        """Один форвард; если delta — патч выхода слоя L_idx в позиции t_pos."""
        def hook(m, i, o):
            oo = o.clone()
            oo[:, t_pos, :] = oo[:, t_pos, :] + delta
            return oo
        if delta is None:
            with torch.no_grad():
                return self.model(input_ids=self.input_ids, attention_mask=self.mask).logits[0]
        if grad:
            h = self.model.model.layers[L_idx].register_forward_hook(hook)
            try:
                z = self.model(input_ids=self.input_ids, attention_mask=self.mask).logits[0]
            finally:
                h.remove()
            return z
        with torch.no_grad():
            h = self.model.model.layers[L_idx].register_forward_hook(hook)
            try:
                z = self.model(input_ids=self.input_ids, attention_mask=self.mask).logits[0]
            finally:
                h.remove()
            return z

    def probe(self, L_idx, t_pos, v, a, z0, pwin):
        """Центральные разности в направлении v (np f32 unit), амплитуда a.
        Возвращает: dz_k (k=0..H) f64 numpy (V,), q_k, kappa_k, zplus (для det/отчётов)."""
        d = torch.as_tensor(np.asarray(a * v, dtype=np.float32), device=DEVICE)
        zp = self.fwd(L_idx, t_pos, d)
        zm = self.fwd(L_idx, t_pos, -d)
        out = {}
        for k in range(H + 1):
            s = t_pos + k
            dz = (zp[s].double().cpu().numpy() - zm[s].double().cpu().numpy()) / 2.0
            p = pwin[s]
            q = float(p @ (dz * dz) - (p @ dz) ** 2)
            # r3 C3: sp и lp — оба на CPU (f64): device-mismatch снят
            lp = torch.log(torch.as_tensor(p, dtype=torch.float64).clamp(min=1e-30))
            sp = torch.softmax(zp[s].double(), dim=0).cpu()
            kl = float((sp * (torch.log(sp.clamp(min=1e-30)) - lp)).sum())
            out[k] = {"dz": dz, "q": q, "kappa": kl}
        out["zplus"] = zp
        return out


def lambda_full(q0, qs, a, tau, eps0):
    return float(H * float(np.mean([qs[k] for k in range(1, H + 1)]))
                 / (q0 + a * a * tau * eps0))


# ============================================================ B0 и матрицы
def b0_construct(hL):
    """B0 (32 направления) из hidden-состояний (128 поз., 1024) f32:
    8 top-PCA + 8 sample_null_space (vh[119:127] — канон pre-study r3-r4: низко-энергетические
    правые сингулярные векторы; строки 119..126 из 128 — НЕ строго последние 8, канон не
    меняется: все pre-study-якоря и B0 16 клеток построены на нём) + 16 gQR (per-cell seed).
    Возвращает (b0 [32,1024] f32 unit, checks dict, labels)."""
    h = hL.double().cpu()  # r3 C3: B0 строится на CPU (f64 SVD)
    hc = h - h.mean(0, keepdims=True)
    _, _, vh = np.linalg.svd(hc.numpy(), full_matrices=False)  # (128, 1024): строки = правые с.в.
    top8 = vh[:8]
    bot8 = vh[119:127]
    return top8, bot8, vh


def gqr_directions(n, cell_id, seed=SEED):
    """n gQR-направлений: столбцы QR(A^T), A = RandomState(seed+cell_id).standard_normal(n, D)."""
    A = np.random.RandomState(seed + cell_id).standard_normal((n, D))
    q = np.linalg.qr(A.T)[0]  # (D, n) unit-ортонормальные столбцы
    return q.T  # (n, D)


def b0_full(cell_id, hL):
    top8, bot8, _ = b0_construct(hL)
    gqr = gqr_directions(16, cell_id)
    b0_f64 = np.vstack([top8, bot8, gqr])
    b0_f64 = b0_f64 / np.linalg.norm(b0_f64, axis=1, keepdims=True)
    b0 = b0_f64.astype(np.float32)  # пробы/матрицы — в f32-направлениях (registered)
    labels = (["top_pca"] * 8 + ["sample_null_space"] * 8 + ["gqr_cellseed"] * 16)
    g_top = float(np.abs(top8 @ top8.T - np.eye(8)).max())
    g_bot = float(np.abs(bot8 @ bot8.T - np.eye(8)).max())
    g_gqr = float(np.abs(gqr @ gqr.T - np.eye(16)).max())
    nrm = float(np.abs(np.linalg.norm(b0_f64, axis=1) - 1.0).max())
    # M7: норму меряем на f32-векторе b0 (f32->f64 для самого norm): отклонение ~ULP32
    nrm_f32 = float(np.abs(np.linalg.norm(b0.astype(np.float64), axis=1) - 1.0).max())
    checks = {"top_orth_max_dev": g_top, "bot_orth_max_dev": g_bot,
              "gqr_orth_max_dev": g_gqr, "norm_max_dev": nrm,
              "norm_f32_max_dev": nrm_f32}
    return b0, checks, labels


def matrices_from_Z(Z, pwin, t_pos, a):
    """Z[k] = (32, V) f64: dz_k по 32 направлениям (центральные, амплитуда a).
    Матрицы (J b)^T P (J b')/a^2 = dz_j^T P dz_j' / a^2 (f64):
    C_t (k=0), O_K1 (k in K1), O_K2 (k in K2)."""
    C = None
    O1 = np.zeros((32, 32))
    O2 = np.zeros((32, 32))
    for k in range(H + 1):
        p = pwin[t_pos + k]
        # C_k = Z P Z^T / a^2, P = diag(p) - p p^T. (Z@P)@Z^T — НЕ (Z@P)@(Z@P)^T
        # (последнее = Z P^2 Z^T; P не идемпотентен — проверено численно).
        A = Z[k] * p - (Z[k] @ p)[:, None] * p  # (32, V) = Z @ P
        M = (A @ Z[k].T) / (a * a)
        if k == 0:
            C = M
        elif k in K1:
            O1 += M
        elif k in K2:
            O2 += M
    return C, O1, O2


def _sym_gen_eig(C, G):
    """Собственные обобщённой задачи C x = lambda G x (G = L L^T, Cholesky):
    симметричная трансформация M = L^{-1} C L^{-T} (сходна обобщённым корням).
    Проверено численно vs eigh-based G^{-1/2} C G^{-1/2} (dev < 1e-8)."""
    Lc = np.linalg.cholesky(G + 1e-12 * np.eye(G.shape[0]))
    M = np.linalg.solve(Lc, C @ np.linalg.inv(Lc.T))
    return np.linalg.eigvalsh((M + M.T) / 2.0)


def eps0_generalized(C, G):
    """eps0 = lam_max(G^{-1/2} C G^{-1/2}) (r3 N3; Cholesky G, f64)."""
    w = _sym_gen_eig(C, G)
    return float(max(w.max(), 0.0))


def pencil_report(O1, C, G, eps):
    """нижняя граница: pencil (O_K1, C + eps*G) — крупнейший корень (отчёт, TZ §3.3)."""
    w = _sym_gen_eig(O1, C + eps * G)
    return float(w.max())

# ============================================================ подъём (ascent)
def vjp_leaf(fwd, L_idx, t_pos, dz, a, pwin, v, ks_obj):
    """VJP через vector-leaf (pre-study r1, at2e_vjp3/4): 1 grad-fwd + 2 backward.
    dz: dict k -> (V,) f64 (J_k v = dz/a). Возвращает (Ov, Cv) f64 (1024,).
    backward1: sum_{k in ks_obj} c_k . z_{t+k} -> Ov = sum J_k^T c_k
    backward2: c_0 . z_t -> Cv = J_0^T c_0
    c_k = P_k (J_k v) = p_k Jv - (p_k.Jv) p_k."""
    d = torch.zeros(D, requires_grad=True, device=DEVICE)
    Jv = {k: torch.as_tensor((dz[k] / a).astype(np.float32), device=DEVICE) for k in dz}
    c = {}
    for k in range(H + 1):
        p = torch.as_tensor(pwin[t_pos + k], dtype=torch.float32, device=DEVICE)
        s = float((p.double() @ Jv[k].double()).sum())
        c[k] = (p * Jv[k] - s * p)
    z = fwd.fwd(L_idx, t_pos, d, grad=True)
    loss1 = sum((c[k] * z[t_pos + k]).sum() for k in ks_obj)
    loss1.backward(retain_graph=True)
    Ov = d.grad.clone()
    d.grad = None
    loss2 = (c[0] * z[t_pos]).sum()
    loss2.backward()
    Cv = d.grad
    return Ov.double().cpu().numpy(), Cv.double().cpu().numpy()  # r3 C3


def ascent(fwd, cell, b0, eps0, n_iters):
    """Подъём на K1 (TZ §3.4). Итерация i (1..n): 2 FD + 1 grad-fwd + 2 bwd на v_{i-1}
    (6.5 ф.-э.); после цикла — финальный FD-only-проход на v_n (lambda(v_n)).
    Траектория (TZ §13, r5.2: i уникально, 1:1 с вектором): блочные поля
    v0_source/v0_idx/v0_sha256/v0_lambda_K1 (v0 из B0-батча; v0_recheck — дет-сверка
    повторного замера) + traj: n записей {i, v_sha256, lambda_K1, delta_v_norm, eta,
    gnorm, report} для v_1..v_n (запись v_j — на позиции j-1). vstar_argmax ∈ 0..n:
    0 -> v0, k -> traj[k-1].
    g = D*Ov_K1 - 2*qbar_K1*(Cv + eps_reg*v) (точный градиент registered-эстиманда,
    r3: формула исправлена — в r2 докстринге были потеряны множитель 2 и ε-член).
    Адаптивный нормированный шаг: eta0=0.2; lambda вниз -> eta/=2 (min 0.01),
    вверх -> eta=min(0.5, 1.5*eta); gnorm < 1e-8 -> ранний стоп."""
    L_idx, t = cell["L"], cell["t"]
    a = cell["a_hi"]
    pwin = cell["pwin"]
    b0q = cell["b0_q"]  # per j: dict k -> q (при a_hi)
    tau = EPS_ASCENT_TAU
    Dden = a * a * tau * eps0

    def lam_k1(qs):
        return float(2.0 * np.mean([qs[k] for k in K1]) / (qs[0] + Dden))

    def lam_rep(qs):
        return {"lambda_K2": float(2.0 * np.mean([qs[k] for k in K2]) / (qs[0] + Dden)),
                "lambda_full": lambda_full(qs[0], qs, a, TAU_PRIMARY, eps0)}

    v0_idx = int(np.argmax([lam_k1({k: b0q[j][k] for k in range(H + 1)}) for j in range(32)]))
    v_seq = [b0[v0_idx].astype(np.float64).copy()]
    lam_seq = [lam_k1({k: b0q[v0_idx][k] for k in range(H + 1)})]  # lambda(v0) из B0-батча
    records = []  # записи v_1..v_n (позиция j-1 — вектор v_j)
    v0_recheck = None
    eta = 0.2
    converged = False
    stop_iter = None
    for i in range(1, n_iters + 1):
        # проба состояния v_{i-1} (= v_seq[-1]): i=1 — повторный замер v0 (дет-сверка
        # с B0-батчом); i>=2 — первый замер v_{i-1}
        v_cur = v_seq[-1]
        probe = fwd.probe(L_idx, t, v_cur.astype(np.float32), a, cell["z0"], pwin)
        qs = {k: probe[k]["q"] for k in range(H + 1)}
        lam_cur = lam_k1(qs)
        if i == 1:
            v0_recheck = {"lambda_K1": lam_cur,
                          "rel_dev": float(abs(lam_cur - lam_seq[0])
                                           / max(abs(lam_seq[0]), 1e-30))}
            if v0_recheck["rel_dev"] > 1e-6:
                raise SystemExit(f"ascent: дет-сверка B0-батча FAIL "
                                 f"dev={v0_recheck['rel_dev']:.2e} — детерминизм нарушен, "
                                 f"артефакт не считается")
        else:
            records[i - 2]["lambda_K1"] = lam_cur  # запись v_{i-1} (позиция i-2)
            lam_seq.append(lam_cur)
        dz = {k: probe[k]["dz"] for k in range(H + 1)}
        Ov, Cv = vjp_leaf(fwd, L_idx, t, dz, a, pwin, v_cur, K1)
        qbar_k1 = float(np.mean([qs[k] for k in K1]))
        # точный градиент registered-эстиманда λ = 2*qbar/(q0 + a^2*tau*eps0):
        # dlambda ~ (v^T B v) A' v - (v^T A' v) B v, B = O0 + eps*I, A' = (O1+O2)/2
        # -> g = D*Ov - 2*qbar*(Cv + eps*v) (проверено: cos(g, grad_analytic) = 1.0;
        # TZ-формула с "∝" — аппроксимация, верная при v^TBv ~ 1)
        eps_reg = Dden / (a * a)  # = tau*eps0
        g = (qs[0] + Dden) * Ov - 2.0 * qbar_k1 * (Cv + eps_reg * v_cur)
        gn = float(np.linalg.norm(g))
        # отчёт о пробованном состоянии v_{i-1} — в его запись
        if i == 1:
            v0_recheck.update({"gnorm": gn, "report": lam_rep(qs)})
        else:
            records[i - 2].update({"gnorm": gn, "report": lam_rep(qs)})
        if gn < 1e-8:
            converged = True
            stop_iter = i
            if i == 1:
                v0_recheck["early_stop"] = True
            else:
                records[i - 2]["early_stop"] = True
            logline(f"ascent: ранний стоп (gnorm {gn:.2e} < 1e-8) на итерации {i}")
            break
        v_new = v_cur + eta * g / gn
        v_new = v_new / np.linalg.norm(v_new)
        records.append({"i": i, "v_sha256": sha_vec(v_new), "lambda_K1": None,
                        "eta": eta, "delta_v_norm": float(np.linalg.norm(v_new - v_cur))})
        v_seq.append(v_new)
        # адаптивный шаг (по изменению lambda между v_{i-2} -> v_{i-1})
        if len(lam_seq) >= 2:
            if lam_seq[-1] < lam_seq[-2]:
                eta = max(0.01, eta / 2.0)
            elif lam_seq[-1] > lam_seq[-2]:
                eta = min(0.5, 1.5 * eta)
    v_last = v_seq[-1]
    if not converged:
        # финальный FD-only-проход: lambda(v_n) (запись v_n — последняя)
        probe_f = fwd.probe(L_idx, t, v_last.astype(np.float32), a, cell["z0"], pwin)
        qs_f = {k: probe_f[k]["q"] for k in range(H + 1)}
        lam_f = lam_k1(qs_f)
        records[-1]["lambda_K1"] = lam_f
        records[-1]["report"] = lam_rep(qs_f)
        lam_seq.append(lam_f)  # lambda(v_n) — v_n не замерялся в цикле
    else:
        # ранний стоп: lambda(v_last) уже замерен на итерации стопа
        lam_f = lam_seq[-1]
    assert len(lam_seq) == len(v_seq), (len(lam_seq), len(v_seq))
    vstar_i = int(np.argmax(lam_seq))
    lam_prev = lam_seq[-2] if len(lam_seq) >= 2 else lam_seq[-1]
    not_converged = (False if converged
                     else bool(abs(lam_f - lam_prev) / max(abs(lam_prev), 1e-9) > 1e-3))
    return {"v": v_seq[vstar_i], "vstar_argmax": vstar_i,
            "lambda_vstar": float(lam_seq[vstar_i]),
            "eps_ascent": float(eps0 * EPS_ASCENT_TAU),  # r3 V6 (TZ §13): = tau*eps0
            "v0_source": "B0-батч (argmax K1, tau=1e-2)",
            "v0_idx": v0_idx,
            "v0_sha256": sha_vec(v_seq[0]),
            "v0_lambda_K1": float(lam_seq[0]),
            "v0_recheck": v0_recheck,
            "traj": records,
            "ascent_converged": converged, "early_stop_iter": stop_iter,
            "ascent_not_converged": not_converged}

# ============================================================ верификация (TZ §3.5)
def verify_vstar(fwd, cell, vstar, floor_q):
    """4 forward: +-a_lo*v*, +-a_hi*v*. Линейные гейты по режиму, denom/companion,
    kappa-флаги. a_lo/a_hi — явные имена (r5 R3)."""
    L_idx, t = cell["L"], cell["t"]
    hnorm = cell["hnorm"]
    a_lo = A_LO_REL * hnorm
    a_hi = A_HI_REL * hnorm
    pwin = cell["pwin"]
    out = {"a_lo": a_lo, "a_hi": a_hi}
    per_amp = {}
    md5s = {}
    for name, a in (("a_lo", a_lo), ("a_hi", a_hi)):
        probe = fwd.probe(L_idx, t, vstar.astype(np.float32), a, cell["z0"], pwin)
        md5s[f"verify_{name}"] = md5_tensor(probe["zplus"])
        qs = {k: probe[k]["q"] for k in range(H + 1)}
        kappa = {k: probe[k]["kappa"] for k in range(H + 1)}
        lams = {str(tau): lambda_full(qs[0], qs, a, tau, cell["eps0"]) for tau in TAUS}
        lams_k1 = {str(tau): float(2.0 * np.mean([qs[k] for k in K1])
                                    / (qs[0] + a * a * tau * cell["eps0"])) for tau in TAUS}
        lams_k2 = {str(tau): float(2.0 * np.mean([qs[k] for k in K2])
                                    / (qs[0] + a * a * tau * cell["eps0"])) for tau in TAUS}
        qbar_fut = float(np.mean([qs[k] for k in range(1, H + 1)]))
        qbar_k1 = float(np.mean([qs[k] for k in K1]))
        # r3 V6: доля ε в знаменателе a²·τ·ε₀/(q_t + a²·τ·ε₀) per τ (TZ §3.5):
        # режим чтения λ — q_t-доминирование vs ε-доминирование
        eps_share = {str(tau): (a * a * tau * cell["eps0"])
                     / (qs[0] + a * a * tau * cell["eps0"]) for tau in TAUS}
        per_amp[name] = {"q": {str(k): qs[k] for k in range(H + 1)},
                         "kappa": {str(k): kappa[k] for k in range(H + 1)},
                         "lambda_full_tau": lams, "lambda_K1_tau": lams_k1,
                         "lambda_K2_tau": lams_k2,
                         "qbar_fut": qbar_fut, "qbar_K1": qbar_k1,
                         "denom_eps_share_tau": eps_share}
    out["per_amp"] = per_amp
    lo, hi = per_amp["a_lo"], per_amp["a_hi"]
    # r3 U1: κ-флаги при a_hi (инвариант 4) — ГЕЙТ в обоих режимах (до ветки)
    kflags_hi = {}
    for k in range(H + 1):
        qk = hi["q"][str(k)]
        kflags_hi[str(k)] = abs(2.0 * hi["kappa"][str(k)] - qk) / max(qk, floor_q) > 0.5
    out["kappa_flags_hi"] = kflags_hi
    # режим: q_t(v*) > 10*floor (при a_lo — строже, r5 R3) -> normal, иначе denom_below_floor
    mode = "normal" if lo["q"]["0"] > FLOOR_FACTOR * floor_q else "denom_below_floor"
    out["mode"] = mode
    # linear-гейты (единое определение §3.5; провал -> NOT_TESTABLE)
    if mode == "normal":
        # r3 V5: L1 — на λ^{K1} (объект вердикта/ранг-теста, TZ r5.2 §3.5);
        # full-λ — отчёт (lambda_full_report), НЕ гейтится
        # r3 U1(b): + κ-флаги — симметрия с below_floor: K2-окно (k=4,8) не остаётся
        # непроверенным (0 доп. forward — κ уже посчитан)
        lam_lo, lam_hi = lo["lambda_K1_tau"][str(TAU_PRIMARY)], hi["lambda_K1_tau"][str(TAU_PRIMARY)]
        L1 = abs(lam_lo - lam_hi) / max(abs(lam_hi), 1e-30)
        L2 = np.log(hi["q"]["0"] / max(lo["q"]["0"], 1e-30)) / np.log(A_HI_REL / A_LO_REL)
        linear_ok = bool(L1 <= 0.10 and 1.9 <= L2 <= 2.1
                         and not any(kflags_hi.values()))
        out["linear"] = {"L1": float(L1), "L2": float(L2),
                         "kappa_flags_hi": kflags_hi,
                         "ok": linear_ok,
                         "mode": "L1+L2+kappa (K1-lambda tau=1e-2, TZ r5.2 V5/U1)",
                         "lambda_full_report": {"a_lo": lo["lambda_full_tau"][str(TAU_PRIMARY)],
                                                 "a_hi": hi["lambda_full_tau"][str(TAU_PRIMARY)]}}
    else:
        # r3 V5: L1_num/L2_num — на q̄_{K1} (числитель объекта вердикта); q̄_fut — отчёт
        r_lo = lo["qbar_K1"] / (a_lo * a_lo)
        r_hi = hi["qbar_K1"] / (a_hi * a_hi)
        L1n = abs(r_lo - r_hi) / max(abs(r_lo + r_hi) / 2.0, 1e-30)
        L2n = (np.log(max(hi["qbar_K1"], 1e-30) / max(lo["qbar_K1"], 1e-30))
               / np.log(A_HI_REL / A_LO_REL))
        # kappa-флаги (при a_hi): |2k - q|/max(q, floor) > 0.5 -> срабатывание
        kflags = {}
        for k in range(H + 1):
            qk = hi["q"][str(k)]
            kflags[str(k)] = abs(2.0 * hi["kappa"][str(k)] - qk) / max(qk, floor_q) > 0.5
        kfail = any(kflags.values())
        linear_ok = bool(L1n <= 0.10 and 1.9 <= L2n <= 2.1 and not kfail)
        out["linear"] = {"L1_num": float(L1n), "L2_num": float(L2n),
                         "qbar_fut_report": {"a_lo": lo["qbar_fut"], "a_hi": hi["qbar_fut"]},
                         "kappa_flags": kflags, "kappa_fail": bool(kfail), "ok": linear_ok,
                         "mode": "L1_num+L2_num+kappa (denom_below_floor, K1-объект, TZ r5.2 V5)"}
    # fallback-режим (сетка {1e-3, 3e-3}) — НЕ реализована (TZ r5.2 §3.5: правило ->
    # NOT_TESTABLE); flag — отчёт
    out["fallback_grid_triggered"] = False
    # denom-гейт v* (r2 B1, r5 R3: на a_lo)
    out["denom_gate_vstar"] = bool(lo["q"]["0"] > FLOOR_FACTOR * floor_q)
    # rho при a_hi (согласовано с random-референсом, проба которого — a_hi;
    # условие «осмыслен» — на a_lo, строже, r5 R3)
    out["rho"] = (float(hi["qbar_fut"] / hi["q"]["0"]) if out["denom_gate_vstar"] else None)
    # companion-гейт (инвариант 15, по K1, r4 m4; при a_hi)
    out["companion_ok"] = bool(hi["qbar_K1"] > max(FLOOR_FACTOR * floor_q, 1e-12))
    out["companion_qbar_K1"] = hi["qbar_K1"]
    # denom-гейт клетки: медиана по B0 q_t(a_lo) > 10*floor
    out["denom_gate_cell"] = bool(cell["b0_qt_alo_median"] > FLOOR_FACTOR * floor_q)
    out["denom_gate_cell_value"] = cell["b0_qt_alo_median"]
    out["det_md5"] = md5s
    return out


# ============================================================ random-референс (TZ §3.6)
def random_reference(fwd, cell, n_random, ref_seed, a_hi):
    """N случайных направлений (столбцы QR(A_ref^T), A_ref = RandomState(SEED+cell_id+1000));
    каждое — 2 forward (a_hi, центральные). per направление: q_k, lambda per tau (K1/K2/full),
    rho (при q_t > 10*floor)."""
    L_idx, t = cell["L"], cell["t"]
    A = np.random.RandomState(ref_seed).standard_normal((n_random, D))
    R = np.linalg.qr(A.T)[0]  # (D, n) unit-ортонормальные столбцы
    dirs = (R.T / np.linalg.norm(R.T, axis=1, keepdims=True)).astype(np.float32)
    out = []
    for j in range(n_random):
        probe = fwd.probe(L_idx, t, dirs[j], a_hi, cell["z0"], cell["pwin"])
        qs = {k: probe[k]["q"] for k in range(H + 1)}
        rec = {"q": {str(k): qs[k] for k in range(H + 1)},
               "lambda_K1_tau": {str(tau): float(2.0 * np.mean([qs[k] for k in K1])
                                            / (qs[0] + a_hi * a_hi * tau * cell["eps0"]))
                                for tau in TAUS},
               "lambda_K2_tau": {str(tau): float(2.0 * np.mean([qs[k] for k in K2])
                                            / (qs[0] + a_hi * a_hi * tau * cell["eps0"]))
                                for tau in TAUS},
               "lambda_full_tau": {str(tau): lambda_full(qs[0], qs, a_hi, tau, cell["eps0"])
                                   for tau in TAUS}}
        rec["rho"] = (float(np.mean([qs[k] for k in range(1, H + 1)]) / qs[0])
                      if qs[0] > FLOOR_FACTOR * cell["floor_q"] else None)
        rec["q_t"] = qs[0]
        out.append(rec)
    return dirs, out


def rank_test(vstar_lams, ref_lams, window):
    """per tau: rank v* среди {v*, r_1..r_N} по lambda (1 = лучший). window: K1|K2."""
    res = {}
    for tau in TAUS:
        vals = [vstar_lams[str(tau)]] + [r[f"lambda_{window}_tau"][str(tau)] for r in ref_lams]
        rank = 1 + sum(1 for v in vals[1:] if v > vals[0])
        res[str(tau)] = int(rank)
    return res


def pct(a, p):
    a = np.sort(np.asarray(a, dtype=float))
    k = (len(a) - 1) * p / 100.0
    f, c = int(np.floor(k)), min(int(np.ceil(k)), len(a) - 1)
    return float(a[f] + (a[c] - a[f]) * (k - f))


# ============================================================ pos-spec weak-скрин (TZ §3.10)
def pos_spec(fwd, cell, vstar, v0, ref_dirs16, ref_lams_t, sibling_refs):
    """Weak-скрин (r5 R1): v* и v0 в t' (2 forward each, центральные a_hi, фикс. eps из
    исходной клетки, r3 N4); референс: 16 парных замеров (для t32-клеток — проба в t'=96;
    для t96-клеток — референс sibling t32-клетки, 0 forward, r5).
    Скрин: ratio(v*) <= p10(ratio_ref) И ratio(v*) <= ratio(v0)."""
    L_idx, t = cell["L"], cell["t"]
    t_prime = 96 if t == 32 else 32
    a_p = A_HI_REL * cell["hnorm_prime"]
    eps_fixed = EPS_ASCENT_TAU * cell["eps0"]  # фикс. eps из ИСХОДНОЙ клетки (N4)
    Dden_p = a_p * a_p * TAU_PRIMARY * cell["eps0"]

    def lam_k1_p(qs):
        return float(2.0 * np.mean([qs[k] for k in K1]) / (qs[0] + Dden_p))

    res = {"t_prime": t_prime, "a_prime": a_p, "eps_fixed": eps_fixed}
    # v* в t'
    ps = fwd.probe(L_idx, t_prime, vstar.astype(np.float32), a_p, cell["z0"], cell["pwin_prime"])
    lam_v_p = lam_k1_p({k: ps[k]["q"] for k in range(H + 1)})
    lam_v_t = cell["verify"]["per_amp"]["a_hi"]["lambda_K1_tau"][str(TAU_PRIMARY)]
    res["lambda_K1_tprime_vstar"] = lam_v_p
    # M5: мёртвый отклик (lam_v_t = 0) — ratio не определён; скрин консервативно False
    res["ratio_vstar"] = (float(lam_v_p / lam_v_t) if lam_v_t > 0 else None)
    # kappa-флаги в t' (N4)
    kflags_p = {}
    for k in range(H + 1):
        qk = ps[k]["q"]
        kflags_p[str(k)] = abs(2.0 * ps[k]["kappa"] - qk) / max(qk, cell["floor_q"]) > 0.5
    res["kappa_flags_tprime"] = kflags_p
    # v0 в t'
    p0 = fwd.probe(L_idx, t_prime, v0.astype(np.float32), a_p, cell["z0"], cell["pwin_prime"])
    lam_v0_p = lam_k1_p({k: p0[k]["q"] for k in range(H + 1)})
    lam_v0_t = cell["ascent"]["v0_lambda_K1"]
    res["lambda_K1_tprime_v0"] = lam_v0_p
    res["ratio_v0"] = (float(lam_v0_p / lam_v0_t) if lam_v0_t > 0 else None)
    # референс (r4 m6: первые 16 в порядке столбцов QR(A_ref^T); смок N_RANDOM=8 -> 8)
    n_ref = min(16, len(ref_dirs16) if cell["t"] == 32 else len(sibling_refs))
    res["n_ref"] = n_ref
    if cell["t"] == 32:
        # проба реф. в t'=96 (2 forward each)
        ref16_lams_p = []
        for j in range(n_ref):
            pr = fwd.probe(L_idx, t_prime, ref_dirs16[j], a_p, cell["z0"], cell["pwin_prime"])
            ref16_lams_p.append(lam_k1_p({k: pr[k]["q"] for k in range(H + 1)}))
    else:
        # t96: t'=32 — референс = random-референс sibling (t32, тот же док/слой) при a_hi(t');
        # a_hi(t'=32) == a_hi sibling (тот же ||h_32||), eps — исходной (t96) клетки
        ref16_lams_p = []
        for j in range(n_ref):
            qs = sibling_refs[j]["q"]
            ref16_lams_p.append(float(2.0 * np.mean([qs[str(k)] for k in K1])
                                      / (qs["0"] + a_p * a_p * TAU_PRIMARY * cell["eps0"])))
    ref_lams_t16 = [ref_lams_t[j][f"lambda_K1_tau"][str(TAU_PRIMARY)] for j in range(n_ref)]
    # M5: мёртвое направление (lt = 0) — ratio не определён, исключается из
    # p10-множества (n_ref_finite отчитывается); все мёртвы -> p10 = None -> провал
    ratio_ref, n_finite = [], 0
    for lp, lt in zip(ref16_lams_p, ref_lams_t16):
        if lt > 0:
            ratio_ref.append(float(lp / lt))
            n_finite += 1
    res["ratio_ref"] = ratio_ref
    res["n_ref_finite"] = n_finite
    res["ratio_ref_p10"] = (float(pct(ratio_ref, 10)) if ratio_ref else None)
    # отчёты (0 forward): r* = argmax ref16 по lambda(r;t); f = инфляция знаменателя
    # r3 V2: r* — по КОНЕЧным направлениям (индексы совпадают с ratio_ref после
    # M5-фильтрации); все мёртвы -> rstar = None
    finite_idx = [j for j in range(n_ref) if ref_lams_t16[j] > 0]
    if finite_idx:
        rstar_j = finite_idx[int(np.argmax([ref_lams_t16[j] for j in finite_idx]))]
        res["rstar"] = {"idx": rstar_j, "ratio": ratio_ref[
            sum(1 for j in range(rstar_j) if ref_lams_t16[j] > 0)]}
    else:
        res["rstar"] = None
    # r3 гигиена: np.median([]) -> nan+warning; пустой список -> 0.0 (f_inflation = None)
    med_ref = (float(np.median(ref_lams_t16)) if ref_lams_t16 else 0.0)
    res["f_inflation"] = (float(lam_v_t / med_ref) if med_ref > 0 else None)
    # eps0(t')-диагностика (N4): ratio при плавающем eps0(t') — отчёт.
    # t96: t'=32, eps0(t') = eps0 sibling (доступен). t32: t'=96 — считается post-pass
    # в main (0 доп. forward, из сохранённых q пробы t').
    res["probe_q_tprime"] = {str(k): float(ps[k]["q"]) for k in range(H + 1)}
    if cell["t"] == 96:
        eps_tprime = cell["sibling_eps0"]
        # r3 V2: guard — lam_v_t = 0 (мёртвый отклик) -> diag не определён
        if eps_tprime and eps_tprime > 0 and lam_v_t > 0:
            Dden_p2 = a_p * a_p * TAU_PRIMARY * eps_tprime
            lam_v_p2 = float(2.0 * np.mean([ps[k]["q"] for k in K1]) / (ps[0]["q"] + Dden_p2))
            res["eps0_tprime_diag"] = float(lam_v_p2 / lam_v_t)
    # Скрин (r5): необходимое условие (None-ратии — провал, M5)
    res["screen_ok"] = bool(res["ratio_vstar"] is not None and res["ratio_v0"] is not None
                            and res["ratio_ref_p10"] is not None
                            and res["ratio_vstar"] <= res["ratio_ref_p10"]
                            and res["ratio_vstar"] <= res["ratio_v0"])
    return res

# ============================================================ стадия 1d: transfer-gap (TZ §3.11, r5 R1)
def stage_1d(fwd, cell, b0, vstar, pos_spec_res):
    """Независимый подъём в t' (B0(t')-проба 32x2 + 8 ит.) + гейт
    lambda(v*;t') <= max(10*floor_q(t'), 0.3*lambda(v*_t';t'))."""
    L_idx, t = cell["L"], cell["t"]
    t_prime = 96 if t == 32 else 32
    a_p = A_HI_REL * cell["hnorm_prime"]
    pwin_p = cell["pwin_prime"]
    # фикс. регуляризатор eps = 1e-2*eps0(t) (из pos-spec, N4); знаменатель = q0 + a'^2*eps
    eps_fixed = pos_spec_res["eps_fixed"]
    Dden_p = a_p * a_p * eps_fixed
    floor_p = cell["floor_q_prime"]

    def lam_k1_p(qs):
        return float(2.0 * np.mean([qs[k] for k in K1]) / (qs[0] + Dden_p))

    # 1. B0(t')-проба: 32 x 2 forward (центральные a_hi, фикс. eps)
    b0q_p = []
    for j in range(32):
        pr = fwd.probe(L_idx, t_prime, b0[j], a_p, cell["z0"], pwin_p)
        b0q_p.append({k: pr[k]["q"] for k in range(H + 1)})
    # 2. независимый подъём в t' (те же гиперпараметры, что в t).
    # ascent-объект: lam = 2*qbar/(q0 + a^2*tau*eps0_arg) — передать eps0_arg = eps0(t):
    # тогда tau*eps0_arg = eps_fixed = 1e-2*eps0(t) — объект ровно lambda^{K1,eps_fixed}(v; t'),
    # ТОТ ЖЕ eps, что у числителя gap (r5.2 B3: eps_fixed сюда передавать нельзя —
    # двойное применение tau давало знаменатель a^2*1e-4*eps0(t), gap до 100x занижен).
    cell_p = {"L": L_idx, "t": t_prime, "a_hi": a_p, "pwin": pwin_p,
              "z0": cell["z0"], "eps0": cell["eps0"], "b0_q": b0q_p,
              "floor_q": floor_p, "floor_q_prime": cell["floor_q"]}
    asc_p = ascent(fwd, cell_p, b0, cell["eps0"], N_ASCENT)
    lam_vstar_tp = asc_p["lambda_vstar"]  # lambda(v*_t'; t') при eps = eps_fixed
    # 3. lambda(v*; t') — из pos-spec блока (0 доп. forward)
    lam_vstar_at_tp = pos_spec_res["lambda_K1_tprime_vstar"]
    # 5. тест-страж: т' мёртвая -> NOT_TESTABLE
    test_guard = bool(lam_vstar_tp <= FLOOR_FACTOR * floor_p)
    # 4. гейт
    bound = max(FLOOR_FACTOR * floor_p, GAP_MARGIN * lam_vstar_tp)
    gap = float(lam_vstar_at_tp / lam_vstar_tp) if lam_vstar_tp > 0 else None
    gate_pass = bool(lam_vstar_at_tp <= bound) if not test_guard else False
    return {"t_prime": t_prime,
            "probe_b0_tprime": b0q_p,
            "ascent_tprime": {"vstar_argmax": asc_p["vstar_argmax"],
                              "vstar_tprime_sha": sha_vec(asc_p["v"]),
                              "lambda_vstar_tprime": lam_vstar_tp,
                              "traj": asc_p["traj"],
                              "ascent_not_converged": asc_p["ascent_not_converged"],
                              "ascent_converged": asc_p["ascent_converged"]},
            "lambda_vstar_at_tprime": lam_vstar_at_tp,
            "gap": gap,
            "gate": {"margin": GAP_MARGIN,
                     "floor_term": FLOOR_FACTOR * floor_p,
                     "best_term": GAP_MARGIN * lam_vstar_tp,
                     "bound": bound,
                     "pass": gate_pass},
            "near_threshold_flag": bool(gap is not None and GAP_MARGIN < gap <= GAP_NEAR),
            "test_guard": test_guard,
            "testable": bool(lam_vstar_tp > FLOOR_FACTOR * floor_p),
            "vstar_tprime": asc_p["v"].tolist()}


# ============================================================ floor (TZ §3.9), G-zero (TZ §3.8)
def measure_floor(fwd, cell, b0):
    """per (doc, L, t): a_floor = 1e-9*||h_t||, 2 forward (центральные, dir = b0[0]).
    floor_q = max(measured, q_quant), q_quant = (2^-23*max|z_t|)^2 * p_max."""
    L_idx, t = cell["L"], cell["t"]
    a_floor = A_FLOOR_REL * cell["hnorm"]
    pr = fwd.probe(L_idx, t, b0[0], a_floor, cell["z0"], cell["pwin"])
    q_meas = pr[0]["q"]
    zmax = float(np.max(np.abs(cell["z0"][t].double().cpu().numpy())))  # r3 C3
    pmax = float(np.max(cell["pwin"][t]))
    q_quant = (ULP32 * zmax) ** 2 * pmax
    return {"measured": q_meas, "quant": q_quant, "used": max(q_meas, q_quant),
            "zmax_t": zmax, "pmax_t": pmax, "a_floor": a_floor,
            "md5": md5_tensor(pr["zplus"])}


def g_zero(fwd, L23, t, v, a):
    """Патч выхода слоя 23 (последний) в позиции t: после него только per-position
    final-norm + head -> логиты t+k (k>=1) НЕ зависят от t (свойство графа).
    Гейт: max_k ||dz_{t+k}|| <= 1e-6."""
    d = torch.as_tensor((a * v).astype(np.float32), device=DEVICE)  # r3 C3
    zp = fwd.fwd(L23, t, d)
    zm = fwd.fwd(L23, t, -d)
    mx = 0.0
    per_k = {}
    for k in range(1, H + 1):
        dz = (zp[t + k].double().cpu().numpy() - zm[t + k].double().cpu().numpy()) / 2.0
        nrm = float(np.linalg.norm(dz))
        per_k[str(k)] = nrm
        mx = max(mx, nrm)
    return {"max_dz_fut": mx, "per_k": per_k, "gate_ok": bool(mx <= 1e-6)}


# ============================================================ secondary (TZ §3.7) и 1c, cext_ref
def secondary_k2(fwd, cell, b0, eps0):
    """Обратный split-half (отчёт): подъём симметрично на K2 (8 ит.), 4 клетки.
    v0_K2 = argmax B0 по lambda_K2(b; a_hi, tau=1e-2). Траектория — та же схема, что
    ascent (r5.2): v0 — блочные поля, traj = записи v_1..v_n (i уникально)."""
    L_idx, t = cell["L"], cell["t"]
    a = cell["a_hi"]
    pwin = cell["pwin"]
    b0q = cell["b0_q"]
    Dden = a * a * TAU_PRIMARY * eps0

    def lam_k2(qs):
        return float(2.0 * np.mean([qs[k] for k in K2]) / (qs[0] + Dden))

    v0_idx = int(np.argmax([lam_k2({k: b0q[j][k] for k in range(H + 1)}) for j in range(32)]))
    v_seq = [b0[v0_idx].astype(np.float64).copy()]
    lam_seq = [lam_k2({k: b0q[v0_idx][k] for k in range(H + 1)})]
    records = []  # v_1..v_n (позиция j-1 — вектор v_j)
    v0_recheck = None
    eta = 0.2
    converged = False
    stop_iter = None
    for i in range(1, N_ASCENT + 1):
        v_cur = v_seq[-1]  # v_{i-1}
        probe = fwd.probe(L_idx, t, v_cur.astype(np.float32), a, cell["z0"], pwin)
        qs = {k: probe[k]["q"] for k in range(H + 1)}
        lam_cur = lam_k2(qs)
        if i == 1:
            v0_recheck = lam_cur  # повторный замер v0 (отчёт)
        else:
            records[i - 2]["lambda_K2"] = lam_cur
            lam_seq.append(lam_cur)
        dz = {k: probe[k]["dz"] for k in range(H + 1)}
        Ov, Cv = vjp_leaf(fwd, L_idx, t, dz, a, pwin, v_cur, K2)
        qbar_k2 = float(np.mean([qs[k] for k in K2]))
        eps_reg = Dden / (a * a)  # = tau*eps0
        g = (qs[0] + Dden) * Ov - 2.0 * qbar_k2 * (Cv + eps_reg * v_cur)
        gn = float(np.linalg.norm(g))
        if gn < 1e-8:
            converged = True
            stop_iter = i
            logline("secondary K2: ранний стоп (gnorm < 1e-8)")
            break
        v_new = v_cur + eta * g / gn
        v_new = v_new / np.linalg.norm(v_new)
        records.append({"i": i, "v_sha256": sha_vec(v_new), "lambda_K2": None,
                        "eta": eta, "gnorm": gn,
                        "delta_v_norm": float(np.linalg.norm(v_new - v_cur))})
        v_seq.append(v_new)
        if len(lam_seq) >= 2:
            if lam_seq[-1] < lam_seq[-2]:
                eta = max(0.01, eta / 2.0)
            elif lam_seq[-1] > lam_seq[-2]:
                eta = min(0.5, 1.5 * eta)
    v_last = v_seq[-1]
    if not converged:
        probe_f = fwd.probe(L_idx, t, v_last.astype(np.float32), a, cell["z0"], pwin)
        lam_f = lam_k2({k: probe_f[k]["q"] for k in range(H + 1)})
        records[-1]["lambda_K2"] = lam_f
        lam_seq.append(lam_f)
    assert len(lam_seq) == len(v_seq), (len(lam_seq), len(v_seq))
    vstar_i = int(np.argmax(lam_seq))
    return {"v0_idx": v0_idx, "v0_sha256": sha_vec(v_seq[0]),
            "v0_lambda_K2": float(lam_seq[0]), "v0_recheck": v0_recheck,
            "vstar_argmax": vstar_i,
            "lambda_vstar_K2": float(lam_seq[vstar_i]),
            "vstar_sha": sha_vec(v_seq[vstar_i]), "traj": records,
            "early_stop_iter": stop_iter,
            "ascent_converged": converged}


def nonlin_probe_1c(fwd, cell, b0, vstar=None):
    """Отчёт: 32 направления B0 (+ v* — r3 U2: единственный вектор, входящий в вердикт,
    обязан иметь q(a)-кривую за пределами a_hi — «v* линеен до 3e-3?») x 2 доп. амплитуды
    (3e-3, 1e-2)*||h|| x 2 forward. q(a)-кривая (с a_lo/a_hi из батча) + асимметрия
    q(+a)/q(-a) (one-sided, из тех же fwd)."""
    L_idx, t = cell["L"], cell["t"]
    out = {}
    # r3 U2: v* — отдельный слот "vstar" (цена +4 ф.-э./клетку)
    dirs = [(j, b0[j]) for j in range(32)]
    if vstar is not None:
        dirs.append(("vstar", np.asarray(vstar, dtype=np.float64)))
    for rel in PROBE_AMPS_REL:
        a = rel * cell["hnorm"]
        for j, vec in dirs:
            d = torch.as_tensor((a * vec).astype(np.float32), device=DEVICE)  # r3 C3
            zp = fwd.fwd(L_idx, t, d)
            zm = fwd.fwd(L_idx, t, -d)
            key = f"{j}_{rel}"
            qs = {}
            for k in range(H + 1):
                dzc = (zp[t + k].double().cpu().numpy() - zm[t + k].double().cpu().numpy()) / 2.0
                dzp = zp[t + k].double().cpu().numpy() - cell["z0"][t + k].double().cpu().numpy()
                dzm = cell["z0"][t + k].double().cpu().numpy() - zm[t + k].double().cpu().numpy()
                p = cell["pwin"][t + k]
                qs[str(k)] = {"q_central": float(p @ (dzc * dzc) - (p @ dzc) ** 2),
                              "q_plus": float(p @ (dzp * dzp) - (p @ dzp) ** 2),
                              "q_minus": float(p @ (dzm * dzm) - (p @ dzm) ** 2)}
            out[key] = qs
    return out


def cext_ref(fwd, jcache, doc, z0, pwin, L12, t, a_hi, eps0):
    """C-extremes reference (report-only, TZ §3.12): 16 kэш-направлений per док
    (top-8+bottom-8 nonzero-C по C = J_R P128 J_R^T в U128-пространстве; направление в
    R^1024 = J_R^T u, нормировано) x 2 forward (a_hi, L12, t=32). Метка:
    cext_ref_L12_U128cache (direction-only bridge; P — полный вокабуляр AT2e —
    НЕ сравниваемо с AT2d U128-pooled gbar по числам)."""
    key = (doc, 0, 127 - t)  # jcache-builder exp27b: (dom, si, 127-pos) — AT2d f2a_inv
    JR = np.asarray(jcache["JR"][key].numpy(), dtype=np.float64)  # (128, 1024)
    P128 = np.asarray(jcache["P128"][key].numpy(), dtype=np.float64)  # (128,)
    C = JR @ np.diag(P128) @ JR.T  # (128, 128)
    w, u = np.linalg.eigh((C + C.T) / 2.0)
    wmax = float(np.abs(w).max())
    nz = list(np.where(np.abs(w) > 1e-10 * wmax)[0])
    top8 = nz[-8:] if len(nz) >= 8 else nz
    bot8 = nz[:8] if len(nz) >= 8 else nz
    idxs = list(dict.fromkeys(list(bot8) + list(top8)))[:16]
    dirs = []
    for j in idxs:
        dir1024 = JR.T @ u[:, j]
        n = float(np.linalg.norm(dir1024))
        if n < 1e-6:
            continue
        dirs.append((int(j), (dir1024 / n).astype(np.float32)))  # r3 U3: пара (idx, vec)
    out = []
    for j, v in dirs:
        probe = fwd.probe(L12, t, v, a_hi, z0, pwin)
        qs = {k: probe[k]["q"] for k in range(H + 1)}
        rec = {"dir_idx": j, "eig": float(w[j]),  # r3 U3: метка по СВОЕМУ индексу
               "q": {str(k): qs[k] for k in range(H + 1)},
               "lambda_K1_tau": {str(tau): float(2.0 * np.mean([qs[k] for k in K1])
                                    / (qs[0] + a_hi * a_hi * tau * eps0)) for tau in TAUS},
               "lambda_K2_tau": {str(tau): float(2.0 * np.mean([qs[k] for k in K2])
                                    / (qs[0] + a_hi * a_hi * tau * eps0)) for tau in TAUS},
               "lambda_full_tau": {str(tau): lambda_full(qs[0], qs, a_hi, tau, eps0)
                                   for tau in TAUS}}
        rec["rho"] = (float(np.mean([qs[k] for k in range(1, H + 1)]) / qs[0])
                      if qs[0] > 1e-30 else None)
        out.append(rec)
    return {"label": "cext_ref_L12_U128cache", "n_dirs": len(out), "per_dir": out}

# ============================================================ вердикты (TZ §5) — чистые функции
def cell_verdict(c):
    """Per-cell вердикт по сохранённым метрикам (чистая функция; reanalysis-friendly).
    Приоритет: NOT_TESTABLE -> FOUND -> SHO -> POSITION_GENERIC -> REDUNDANT -> NOT_FOUND."""
    v = c
    if (not v["verify"]["linear"]["ok"] or not v["verify"]["denom_gate_cell"]
            or not v["verify"]["companion_ok"]
            or v["verify"].get("fallback_grid_triggered")
            or (v.get("stage1d") and v["stage1d"]["test_guard"])):
        return "NOT_TESTABLE"
    rank1_k1 = sum(1 for tau in (str(t) for t in TAUS) if v["rank"]["K1"][tau] == 1)
    rank1_k2 = sum(1 for tau in (str(t) for t in TAUS) if v["rank"]["K2"][tau] == 1)
    rho = v["verify"]["rho"]
    p90_ref = v["rho_ref_p90"]
    signature = bool((rho is not None and p90_ref is not None and rho > p90_ref)
                     or (not v["verify"]["denom_gate_vstar"]))
    is_1d_eligible = bool(v.get("pos_spec") is not None
                         and rank1_k1 >= 2 and signature)
    if is_1d_eligible:
        ps = v["pos_spec"]
        if not ps["screen_ok"]:
            return "POSITION_GENERIC"  # early-exit, 1d не гоняется
        s1d = v["stage1d"]
        if s1d["gate"]["pass"]:
            return "FOUND" if rank1_k2 >= 2 else "SHORT_HORIZON_ONLY"
        return "POSITION_GENERIC"
    # REDUNDANT: companion OK, rank-1 K1 >= 2/3, q_t(a_lo) > 10*floor, rho-rank в [p10,p90]
    if (rank1_k1 >= 2 and v["verify"]["denom_gate_vstar"]
            and v["rho_rank"] is not None
            and v["rho_rank"] >= v["rho_ref_p10"] and v["rho_rank"] <= v["rho_ref_p90"]):
        return "REDUNDANT"
    return "NOT_FOUND"


def global_verdict(cells):
    """Global по 16 клеткам (ПОСЛЕ стадии 1d). Приоритет:
    SCRATCHPAD_CANDIDATE -> POSITION_GENERIC_DOMINANT -> REDUNDANT_DOMINANT ->
    DEAD_WEIGHT -> NOT_FOUND_IN_SUBSPACE -> NOT_RESOLVED."""
    vs = [c["verdict"] for c in cells]
    found = [c for c in cells if c["verdict"] == "FOUND"]
    sho = [c for c in cells if c["verdict"] == "SHORT_HORIZON_ONLY"]
    n_found_sho = len(found) + len(sho)
    docs_found = {c["doc"] for c in found + sho}
    n_pg = sum(1 for v in vs if v == "POSITION_GENERIC")
    n_red = sum(1 for v in vs if v == "REDUNDANT")
    n_ok8 = sum(1 for c in cells
                if c["verify"]["denom_gate_cell"] and c["verify"]["companion_ok"]
                and c["verdict"] != "NOT_TESTABLE")
    # DEAD_WEIGHT (r5.2: по тексту §5 — q_t И q̄_fut ниже пола для ВСЕХ v ∈ B0 ∪ {v*},
    # при a_hi — первичная амплитуда; данные: v* из verify, B0 — max-а по батчу)
    n_dw = sum(1 for c in cells
               if (c["verify"]["per_amp"]["a_hi"]["q"]["0"] <= FLOOR_FACTOR * c["floor"]["used"]
                   and c["verify"]["per_amp"]["a_hi"]["qbar_fut"] <= FLOOR_FACTOR * c["floor"]["used"]
                   and c["b0_qt_ahi_max"] <= FLOOR_FACTOR * c["floor"]["used"]
                   and c["b0_qbar_fut_max"] <= FLOOR_FACTOR * c["floor"]["used"]))
    if n_found_sho >= 4 and len(found) >= 2 and len(docs_found) >= 2:
        return "SCRATCHPAD_CANDIDATE"
    if n_found_sho == 0 and n_pg >= 8:
        return "POSITION_GENERIC_DOMINANT"
    if n_found_sho == 0 and n_red >= 8:
        return "REDUNDANT_DOMINANT"
    if n_found_sho == 0 and n_dw >= 8:  # r3 минор: FOUND/SHO первее (приоритет §5)
        return "DEAD_WEIGHT"
    if n_found_sho == 0 and n_ok8 >= 8 and n_pg < 8 and n_red < 8:
        return "NOT_FOUND_IN_SUBSPACE"
    return "NOT_RESOLVED"


def stage1b_verdict(null_cells, trained_global):
    """1b (условно): LEARNING / ARCHITECTURE / NULL_NOT_TESTABLE.
    r4 m9 + r5 U1: null-сравнение — по rank/lambda + 1d + weak-скрину ИДЕНТИЧНОГО
    поиска; ARCHITECTURE = null-поиск нашёл кандидата тем же усилием (включая 1d и скрин).
    NULL_NOT_TESTABLE — только если ВСЕ пред-замеры null в шумовом поле."""
    if trained_global != "SCRATCHPAD_CANDIDATE":
        return None
    pre_fail = [c for c in null_cells if c.get("null_not_testable")]
    if pre_fail and len(pre_fail) == len(null_cells):
        return "NULL_NOT_TESTABLE"  # отклик null в шумовом поле (все пред-замеры ниже пола)
    reproduced = [c for c in null_cells
                  if c["verdict"] in ("FOUND", "SHORT_HORIZON_ONLY")]
    if len(reproduced) >= 1:
        return "ARCHITECTURE"  # null воспроизвёл кандидата тем же усилием
    return "LEARNING"


# ============================================================ per-cell пайплайн
def run_cell(fwd, doc, L, t, cell_id, z0, hs, floor_map, floor_global,
             force_pos_spec=False, sibling=None, force_1d=False):
    """Полный per-cell пайплайн стадии 1 (TZ §3.2-3.11). Возвращает cell-record.
    floor_map: {(doc,L,t): floor-замер} (pre-pass, TZ §3.9, r5.2 M6: дедупликация —
    16 уникальных (doc,L,t), не 32); floor_global: max по 16 — верхняя граница для
    ВСЕХ гейтов (r3 M3). Стадия 1d: floor per (doc,L,t') (§3.11.5 — НЕ глобальный).
    sibling (для t96-клеток): {"refs": random-референс sibling (doc,L,32), "eps0": eps0 sibling}
    — pos-spec-референс в t'=32 берётся от него (0 forward, r5; r5.2 B4: shared-фрейм).
    force_1d: смок-режим (M2) — стадия 1d гоняется безусловно (скрин — научный исход,
    смок проверяет пайплайн)."""
    L_idx = L
    t_prime = 96 if t == 32 else 32
    hL = hs[L_idx]
    h_t = hL[t].double()
    hnorm = float(h_t.norm())
    hnorm_prime = float(hL[t_prime].double().norm())
    pwin = make_pwin(z0, t)
    pwin_prime = make_pwin(z0, t_prime)
    a_lo = A_LO_REL * hnorm
    a_hi = A_HI_REL * hnorm
    logline(f"cell {cell_id} ({doc} L{L} t{t}): hnorm={hnorm:.4f} hnorm'={hnorm_prime:.4f} "
            f"a_lo={a_lo:.3e} a_hi={a_hi:.3e}")

    # B0 + floor (pre-pass, TZ §3.9, r5.2 M6: 16 уникальных (doc,L,t)-замеров; гейты —
    # floor_global = max по 16 (r3 M3); стадия 1d — floor per (doc,L,t') (§3.11.5))
    b0, b0_checks, b0_labels = b0_full(cell_id, hL)
    floor_t = floor_map[(doc, L_idx, t)]
    floor_tp = floor_map[(doc, L_idx, t_prime)]
    if (not b0_checks["gqr_orth_max_dev"] < 1e-9
            or not b0_checks["top_orth_max_dev"] < 1e-9
            or not b0_checks["bot_orth_max_dev"] < 1e-9
            or not b0_checks["norm_max_dev"] < 1e-9
            or not b0_checks["norm_f32_max_dev"] < 1e-6):
        logline(f"cell {cell_id}: B0-проверки FAIL {b0_checks}")
    # B0-пробный батч: 32 x 2 амплитуды x 2 forward = 128 forward
    Z_hi = [np.zeros((32, z0.shape[1])) for _ in range(H + 1)]
    b0q_lo, b0q_hi = [], []
    md5_det = {}
    for j in range(32):
        pr_lo = fwd.probe(L_idx, t, b0[j], a_lo, z0, pwin)
        pr_hi = fwd.probe(L_idx, t, b0[j], a_hi, z0, pwin)
        b0q_lo.append({k: pr_lo[k]["q"] for k in range(H + 1)})
        b0q_hi.append({k: pr_hi[k]["q"] for k in range(H + 1)})
        for k in range(H + 1):
            Z_hi[k][j] = pr_hi[k]["dz"]
        if j < 2:
            md5_det[f"b0j{j}_hi"] = md5_tensor(pr_hi["zplus"])
    C_hi, O1_hi, O2_hi = matrices_from_Z(Z_hi, pwin, t, a_hi)
    G = (b0.astype(np.float64) @ b0.astype(np.float64).T)
    eps0 = eps0_generalized(C_hi, G)
    pencil = pencil_report(O1_hi, C_hi, G, TAU_PRIMARY * eps0)
    b0_qt_alo_median = float(np.median([b0q_lo[j][0] for j in range(32)]))
    b0_qt_alo_max = float(max(b0q_lo[j][0] for j in range(32)))
    # DEAD_WEIGHT-термины (r5.2: по тексту §5 — ВСЕ v ∈ B0 ∪ {v*}, при a_hi):
    b0_qt_ahi_max = float(max(b0q_hi[j][0] for j in range(32)))
    b0_qbar_fut_max = float(max(np.mean([b0q_hi[j][k] for k in range(1, H + 1)])
                                for j in range(32)))
    fallback_triggered = bool(b0_qt_alo_median / max(floor_global, 1e-30) < 100.0)
    if fallback_triggered:
        logline(f"cell {cell_id}: fallback-ПРАВИЛО срабатывает "
                f"(median q_t(a_lo)/floor = {b0_qt_alo_median / floor_global:.1f} < 100) — "
                f"клетка NOT_TESTABLE; сама сетка (1e-3, 3e-3) НЕ реализована — декларация "
                f"TZ §3.5 r5.2 (pre-meas r3: запас 3.3e3, активация не ожидается)")
    logline(f"cell {cell_id}: eps0={eps0:.4e} pencil={pencil:.4e} "
            f"b0_qt_alo_med={b0_qt_alo_median:.3e} floor_global={floor_global:.3e}")

    # M5: мёртвый отклик (C_t <= 0 -> eps0 = 0 -> q_t ≈ 0): знаменатели 0/0 — подъём
    # не определён; клетка NOT_TESTABLE early-exit (TZ §9: GDN-null путь; пред-замер
    # 1b ловит только q0(b0[0]), не q̄_K1 — гейт здесь обязателен)
    if not (eps0 > 0.0):
        logline(f"cell {cell_id}: eps0={eps0:.3e} <= 0 — мёртвый отклик, NOT_TESTABLE (M5)")
        rec_min = {
            "cell_id": cell_id, "doc": doc, "L": L_idx, "t": t, "t_prime": t_prime,
            "hnorm": hnorm, "hnorm_prime": hnorm_prime, "a_lo": a_lo, "a_hi": a_hi,
            "b0": {"n": 32, "labels": b0_labels, "gqr_seed": SEED + cell_id,
                   "sha256": sha_vec(b0.ravel()), "checks": b0_checks},
            "floor": {"at_t": floor_t, "at_t_prime": floor_tp, "used": floor_global},
            "eps0": eps0, "pencil_report": pencil,
            "b0_q_hi": b0q_hi, "b0_q_lo": b0q_lo,
            "G_conds": float(np.max(np.linalg.eigvalsh(G)) / np.min(np.linalg.eigvalsh(G))),
            "b0_qt_alo_median": b0_qt_alo_median, "b0_qt_alo_max": b0_qt_alo_max,
            "b0_qt_ahi_max": b0_qt_ahi_max, "b0_qbar_fut_max": b0_qbar_fut_max,
            "verify": {"mode": "denom_below_floor", "linear": {"ok": False,
                                                               "mode": "dead_response"},
                       "denom_gate_cell": False, "denom_gate_vstar": False,
                       "companion_ok": False, "companion_qbar_K1": 0.0,
                       "rho": None,
                       "per_amp": {"a_hi": {"q": {"0": 0.0}, "qbar_fut": 0.0,
                                             "qbar_K1": 0.0}}},
            "rank": {"K1": {str(tau): None for tau in TAUS},
                     "K2": {str(tau): None for tau in TAUS},
                     "rank1_K1": 0, "rank1_K2": 0},
            "verdict": "NOT_TESTABLE",
            "not_testable_reason": "eps0 <= 0 (мёртвый отклик, M5)",
        }
        return rec_min

    cellctx = {"L": L_idx, "t": t, "a_hi": a_hi, "a_lo": a_lo, "pwin": pwin,
               "pwin_prime": pwin_prime, "z0": z0, "eps0": eps0, "b0_q": b0q_hi,
               "floor_q": floor_global, "hnorm": hnorm, "hnorm_prime": hnorm_prime,
               "sibling_eps0": (sibling or {}).get("eps0"),
               "b0_qt_alo_median": b0_qt_alo_median,
               "floor_q_prime": floor_tp["used"]}

    # подъём
    tstage(f"cell{cell_id}_ascent")
    asc = ascent(fwd, cellctx, b0, eps0, N_ASCENT)
    cellctx["ascent"] = asc
    logline(f"cell {cell_id}: ascent v0_idx={asc['v0_idx']} vstar_i={asc['vstar_argmax']} "
            f"lambda_vstar={asc['lambda_vstar']:.4e} conv={asc['ascent_converged']} "
            f"not_conv={asc['ascent_not_converged']}")

    # верификация
    ver = verify_vstar(fwd, cellctx, asc["v"], floor_global)
    ver["fallback_grid_triggered"] = fallback_triggered
    cellctx["verify"] = ver
    logline(f"cell {cell_id}: verify mode={ver['mode']} linear_ok={ver['linear']['ok']} "
            f"denom_v*={ver['denom_gate_vstar']} companion={ver['companion_ok']} "
            f"rho={ver['rho']}")

    # random-референс + rank
    # B4 (r5.2): seed — shared per (doc, layer) пара (БЕЗ t-индекса): t32- и t96-клетки
    # одного (doc,L) делят ОДИН фрейм 64 направлений -> pos-spec-референс парный
    # (числитель/знаменатель — одно и то же направление в разных позициях), цена 0
    tstage(f"cell{cell_id}_random")
    ref_seed = SEED + (L_idx * 8 + DOCS.index(doc) * 2) + 1000
    ref_dirs, refs = random_reference(fwd, cellctx, N_RANDOM, ref_seed, a_hi)
    v_lams_k1 = ver["per_amp"]["a_hi"]["lambda_K1_tau"]
    v_lams_k2 = ver["per_amp"]["a_hi"]["lambda_K2_tau"]
    rank_K1 = rank_test(v_lams_k1, refs, "K1")
    rank_K2 = rank_test(v_lams_k2, refs, "K2")
    rank1_k1 = sum(1 for tau in rank_K1.values() if tau == 1)
    rank1_k2 = sum(1 for tau in rank_K2.values() if tau == 1)
    rhos = [r["rho"] for r in refs if r["rho"] is not None]
    rho_ref_dist = ([pct(rhos, p) for p in (10, 25, 50, 75, 90)] if rhos else [None] * 5)
    if rhos and ver["rho"] is not None:
        below = sum(1 for r in rhos if r < ver["rho"])
        equal = sum(1 for r in rhos if r == ver["rho"])
        rho_rank = 100.0 * (below + 0.5 * equal) / len(rhos)  # процентиль 0..100
    else:
        rho_rank = None
    logline(f"cell {cell_id}: rank K1={rank_K1} K2={rank_K2} (1s: {rank1_k1}/{rank1_k2}) "
            f"rho_ref_p90={rho_ref_dist[4]} rho_rank={rho_rank}")

    rec = {
        "cell_id": cell_id, "doc": doc, "L": L_idx, "t": t, "t_prime": t_prime,
        "hnorm": hnorm, "hnorm_prime": hnorm_prime, "a_lo": a_lo, "a_hi": a_hi,
        "b0": {"n": 32, "labels": b0_labels, "gqr_seed": SEED + cell_id,
               "sha256": sha_vec(b0.ravel()), "checks": b0_checks,
               # r3 V6 (TZ §13): per direction — λ per τ per окно (из b0_q_hi, 0 fwd)
               "per_dir": [{"j": j, "label": b0_labels[j],
                            "K1": {str(tau): float(2.0 * np.mean([b0q_hi[j][k] for k in K1])
                                   / (b0q_hi[j][0] + a_hi * a_hi * tau * eps0)) for tau in TAUS},
                            "K2": {str(tau): float(2.0 * np.mean([b0q_hi[j][k] for k in K2])
                                   / (b0q_hi[j][0] + a_hi * a_hi * tau * eps0)) for tau in TAUS},
                            "full": {str(tau): lambda_full(b0q_hi[j][0], b0q_hi[j],
                                                            a_hi, tau, eps0) for tau in TAUS}}
                           for j in range(32)]},
        "floor": {"at_t": floor_t, "at_t_prime": floor_tp, "used": floor_global},
        "eps0": eps0, "pencil_report": pencil,
        "b0_q_hi": b0q_hi, "b0_q_lo": b0q_lo,
        "G_conds": float(np.max(np.linalg.eigvalsh(G)) / np.min(np.linalg.eigvalsh(G))),
        "ascent": asc, "verify": ver,
        "rank": {"K1": rank_K1, "K2": rank_K2, "rank1_K1": rank1_k1, "rank1_K2": rank1_k2},
        "random_ref": {"N": N_RANDOM, "construction": "cols_qr(A_ref.T) (N2)",
                       "seed": ref_seed, "rho_distribution": rho_ref_dist,
                       "max_lambda_K1_tau": {str(tau): max(r["lambda_K1_tau"][str(tau)]
                                                          for r in refs) for tau in TAUS},
                       "per_dir": refs},
        "b0_qt_alo_median": b0_qt_alo_median, "b0_qt_alo_max": b0_qt_alo_max,
        "b0_qt_ahi_max": b0_qt_ahi_max, "b0_qbar_fut_max": b0_qbar_fut_max,
        "rho_ref_p10": rho_ref_dist[0], "rho_ref_p90": rho_ref_dist[4],
        "rho_rank": rho_rank,
        "p_cell": 1.0 / (N_RANDOM + 1),
        "det_md5": md5_det,
    }

    # pos-spec weak-скрин (TZ §3.10): клетки, прошедшие rank (K1) + companion
    eligible = bool(ver["companion_ok"] and rank1_k1 >= 2) or force_pos_spec
    if eligible:
        tstage(f"cell{cell_id}_pos_spec")
        sib_refs = (sibling or {}).get("refs") if t == 96 else None
        rec["pos_spec"] = pos_spec(fwd, cellctx, asc["v"], b0[asc["v0_idx"]],
                                   ref_dirs[:16], refs, sib_refs)
        rec["pos_spec"]["v0_sha"] = sha_vec(b0[asc["v0_idx"]].astype(np.float64))
        ps = rec["pos_spec"]
        # r3 C4: None-safe форматтер (M5-guard'ы дают None при мёртвом отклике)
        _f = lambda x: "None" if x is None else f"{x:.4f}"
        logline(f"cell {cell_id}: pos-spec ratio(v*)={_f(ps['ratio_vstar'])} "
                f"p10={_f(ps['ratio_ref_p10'])} ratio(v0)={_f(ps['ratio_v0'])} "
                f"screen_ok={ps['screen_ok']}")
    # 1d-кандидат: companion + rank + сигнатура + скрин
    if rec.get("pos_spec") is not None:
        ps = rec["pos_spec"]
        signature = bool((ver["rho"] is not None and rec["rho_ref_p90"] is not None
                          and ver["rho"] > rec["rho_ref_p90"])
                         or (not ver["denom_gate_vstar"]))
        rec["scratchpad_signature"] = signature
        if (ps["screen_ok"] and (signature or force_pos_spec)) or force_1d:
            tstage(f"cell{cell_id}_stage1d")
            rec["stage1d"] = stage_1d(fwd, cellctx, b0, asc["v"], ps)
            g = rec["stage1d"]
            logline(f"cell {cell_id}: 1d gap={g['gap']} bound={g['gate']['bound']:.4e} "
                    f"pass={g['gate']['pass']} test_guard={g['test_guard']} "
                    f"near={g['near_threshold_flag']}")
    # вердикт (чистая функция)
    rec["verdict"] = cell_verdict(rec)
    return rec

# ============================================================ стадия 1b (null, условная)
def run_stage1b(model, pool_ids):
    """Null (random weights x real text): 4 клетки (per док, L12, t=32), ИДЕНТИЧНЫЙ
    поиск — полный пайплайн (r4 m9 + r5 U1: включая pos-spec weak-скрин и 1d).
    r5.2: cell_id = 100+di (отдельный namespace, гигиена); floor pre-pass на NULL-
    модели per (doc,12,32/96) (TZ §3.9 — floor модель-специфичен: z0/P у null иные).
    Пред-замер: если отклик ниже квант-пола null-модели -> NULL_NOT_TESTABLE (early-exit)."""
    logline("stage 1b: reinit weights (null)")
    init_null_weights(model, SEED_NULL)
    # pass 1: baseline + null floor pre-pass (8 замеров) + пред-замер (лестница амплитуд)
    data = {}
    floor_map_null = {}
    for di, doc in enumerate(DOCS):
        ids = pool_ids[doc]
        ids_t = torch.as_tensor(ids[None]).to(DEVICE)  # r3 C3: input-тензоры на DEVICE
        fwd = Fwd(model, ids_t, torch.ones((1, S_LEN), dtype=torch.long).to(DEVICE))
        z0, hs = fwd.baseline()
        cid = 100 + di
        b0, _, _ = b0_full(cid, hs[12])
        for t in (32, 96):
            ctx = {"L": 12, "t": t, "hnorm": float(hs[12][t].double().norm()),
                   "z0": z0, "pwin": make_pwin(z0, t)}
            floor_map_null[(doc, 12, t)] = measure_floor(fwd, ctx, b0)
        # пред-замер (TZ: «пред-замер ||h_t||/||PДz|| ниже пола; масштаб eps_a_null
        # до 0.1 включительно»): лестница амплитуд 1e-3/1e-2/0.1 * ||h||
        t = 32
        hnorm = float(hs[12][t].double().norm())
        pwin = make_pwin(z0, t)
        zmax = float(np.max(np.abs(z0[t].double().cpu().numpy())))  # r3 C3
        pmax = float(np.max(pwin[t]))
        floor_quant = (ULP32 * zmax) ** 2 * pmax
        pre = {}
        above = False
        for rel in (1e-3, 1e-2, 0.1):
            pr = fwd.probe(12, t, b0[0], rel * hnorm, z0, pwin)
            pre[str(rel)] = pr[0]["q"]
            if pr[0]["q"] > FLOOR_FACTOR * floor_quant:
                above = True
                break
        if not above:
            logline(f"stage 1b {doc}: пред-замер на всех амплитудах ниже 10*floor_quant="
                    f"{floor_quant:.3e} (max q={max(pre.values()):.3e}) -> NULL_NOT_TESTABLE")
            data[doc] = {"not_testable": True,
                         "premeasure": {"q": pre, "floor_quant": floor_quant,
                                        "above_floor": False}}
            continue
        data[doc] = {"not_testable": False, "fwd": fwd, "z0": z0, "hs": hs,
                     "premeasure": {"q": pre, "floor_quant": floor_quant,
                                    "above_floor": True}}
    floor_global_null = max(f["used"] for f in floor_map_null.values())
    logline(f"stage 1b: null floor_global={floor_global_null:.3e} (8 замеров)")
    # pass 2: полный пайплайн для клеток, прошедших пред-замер
    out_cells = []
    for di, doc in enumerate(DOCS):
        d = data[doc]
        if d["not_testable"]:
            out_cells.append({"cell_id": 100 + di, "doc": doc, "L": 12, "t": 32,
                              "premeasure": d["premeasure"],
                              "verdict": "NOT_TESTABLE", "null_not_testable": True})
            continue
        # r3 V4: ИДЕНТИЧНЫЙ поиск (r5 U1) — БЕЗ force_pos_spec: weak-скрин и 1d
        # запускаются на null-клетках по тем же условиям, что в trained main
        # (companion + rank-1 K1 >= 2/3). force=True обходил бы rank/companion —
        # не та процедура, плюс расширял поверхность None-крашей на mёртвый отклик.
        cell_rec = run_cell(d["fwd"], doc, 12, 32, 100 + di, d["z0"], d["hs"],
                            floor_map_null, floor_global_null, force_pos_spec=False)
        cell_rec["null"] = True
        out_cells.append(cell_rec)
    logline(f"stage 1b: вердикты {[c['verdict'] for c in out_cells]}")
    return out_cells

def smoke_checks(S, ok, name, val=None):
    S[name] = {"ok": bool(ok), "value": val}
    if not ok:
        logline(f"SMOKE FAIL: {name} {val}")
    else:
        logline(f"SMOKE PASS: {name} {val if val is not None else ''}")

# ============================================================ main
def main():
    global _TRACK_RUN
    t_start = time.time()
    reanalysis = "--reanalysis" in sys.argv
    if reanalysis:
        reanalysis_mode()
        return
    canary = "--canary" in sys.argv
    if canary:
        canary_mode()
        return
    resume = "--resume" in sys.argv
    if SMOKE and resume:
        raise SystemExit("smoke не resume-ится (короткий прогон)")

    # r3 T4: в kernel'е __file__ — встроенный текст; провенанс — sha ЛОКАЛЬНОГО AT2e.py
    # (builder-инжект через env AT2E_SCRIPT_SHA; локально env не задан)
    script_sha = (sha256_file(os.path.abspath(__file__)) if "__file__" in globals()
                  else (os.environ.get("AT2E_SCRIPT_SHA") or "embedded-no-sha"))
    if "__file__" not in globals():
        logline(f"T4-режим: script_sha={script_sha[:12]}... из env (локальный AT2e.py)")
    started_iso = datetime.now().isoformat()
    cfg = {
        "device": DEVICE, "model": MODEL_DIR, "revision": REVISION,
        "tz": "r5.2", "script_sha": script_sha, "seed": SEED,
        "started_at": started_iso,
        "seed_null": SEED_NULL, "smoke": SMOKE, "n_layers": len(LAYERS),
        "tau_primary": TAU_PRIMARY, "gap_margin": GAP_MARGIN,
        "a_lo_rel": A_LO_REL, "a_hi_rel": A_HI_REL,
        "n_random": N_RANDOM, "n_ascent": N_ASCENT,
    }
    trackio_start(cfg)
    atexit.register(trackio_crash)
    logline(f"AT2e {RUN_NAME}: TZ r5.2, script_sha={script_sha[:12]}, "
            f"smoke={SMOKE}, resume={resume}, device={DEVICE}")

    # ---- S1: sha-гейты + guard (ДО любого форварда)
    pool_sha = sha256_file(POOL)
    assert pool_sha == POOL_SHA, f"S1 FAIL: pool sha {pool_sha} != {POOL_SHA}"
    logline(f"S1: pool sha OK ({pool_sha[:12]}...)")
    no_jcache = os.environ.get("AT2E_NO_JCACHE", "0") == "1"
    if no_jcache:
        jc_sha = None
        logline("S1: jcache L12 SKIP (T4: cext_ref — CPU-follow-up, TZ §6b; AT2E_NO_JCACHE=1)")
    else:
        jc_sha = sha256_file(JCACHE_L12)
        assert jc_sha == JCACHE_SHA, f"S1 FAIL: jcache sha {jc_sha}"
        logline(f"S1: jcache L12 sha OK ({jc_sha[:12]}...)")
    guard_ok = sys.modules.get("fla") is None
    assert guard_ok, "S1 FAIL: fla guard неактивен"
    logline("S1: fla guard активен")

    model = load_model()
    pool, pool_ids = load_pool()
    jcache = None if (SMOKE or no_jcache) else load_jcache_l12()
    tstage("model_and_data")

    # ---- M3 (r5.2): pool_fp cross-check — гейт S1b (TZ §12): кэш L12 обязан
    # соответствовать этому пулу (иначе cext_ref-направления — из другой выборки)
    pool_fp_ok = None
    if not SMOKE and jcache is not None:
        pool_fp_ok = (pool["meta"]["pool_fp"] == jcache["pool_fp"])
        logline(f"S1b: pool_fp pool vs jcache L12 = {pool_fp_ok} "
                f"({ {d: pool['meta']['pool_fp'][d][:8] for d in DOCS} })")
        assert pool_fp_ok, ("S1b FAIL: pool_fp mismatch pool vs jcache L12 — "
                            "cext-кэш не для этого пула")

    # ---- baseline per док
    fwds = {}
    baselines = {}
    smoke = {}
    docs_run = DOCS if not SMOKE else (DOCS[0],)
    for doc in docs_run:
        ids = torch.as_tensor(pool_ids[doc][None]).to(DEVICE)  # r3 C3
        mask = torch.ones((1, S_LEN), dtype=torch.long).to(DEVICE)
        fwd = Fwd(model, ids, mask)
        z0, hs = fwd.baseline()
        md5_1 = md5_tensor(z0)
        det_ok = None
        if SMOKE:
            # S2: baseline det — двойной forward (только в смоке; в main — финальный
            # дет-блок из 8 forward, TZ §4)
            z0b, _ = fwd.baseline(need_hidden=False)
            det_ok = md5_1 == md5_tensor(z0b)
        logline(f"baseline {doc}: md5={md5_1[:12]} det={det_ok}")
        if SMOKE and doc == "wiki":
            smoke_checks(smoke, det_ok, "S2_baseline_det", md5_1[:12])
        fwds[doc] = fwd
        baselines[doc] = {"z0": z0, "hs": hs, "md5": md5_1}

    # ---- клетки (порядок: doc -> L -> t32, t96; t96 получает sibling)
    # B2 (r5.2): чекпойнт cells после КАЖДОЙ клетки + stage_done-маркеры; --resume
    # по cell_id (образец resume_plan AT8_2). Новый прогон (без --resume) стартует
    # с чистого partial-файла.
    partial_path = os.path.join(RESULTS, f"{RUN_NAME}_partial.json")
    os.makedirs(RESULTS, exist_ok=True)
    partial = {"cells": [], "stage_done": {}, "secondary_k2": {},
               "nonlin_probe": {}, "cext_ref": {}, "stage1b": None}
    if resume:
        with open(partial_path) as f:
            partial = json.load(f)
        logline(f"resume: {len(partial['cells'])} клеток, "
                f"stage_done={sorted(partial['stage_done'])}")
    elif os.path.exists(partial_path):
        os.remove(partial_path)
        logline(f"новое начало: удалён старый {os.path.basename(partial_path)}")

    def save_partial():
        with open(partial_path, "w") as f:
            json.dump(_san(dict(partial, updated_at=datetime.now().isoformat())),
                      f, ensure_ascii=False)

    cells = partial["cells"]
    stage_done = partial["stage_done"]
    cells_done = {c["cell_id"] for c in cells}

    # ---- floor pre-pass (TZ §3.9, r5.2 M6): дедупликация — 16 уникальных (doc,L,t)
    # (каждый мерился бы дважды: как t и как t' соседа), 1 направление b0[0] x 2
    # forward = 32 форварда; floor_global = max по 16 — верхняя граница ВСЕХ гейтов
    # (r3 M3). Клетка получает floor_map; стадия 1d — floor per (doc,L,t') (§3.11.5).
    # r3 минор: результат кэшируется в partial — resume не пересчитывает (32 fwd)
    layers_run = LAYERS if not SMOKE else (12,)
    need_keys = {(d, L, t) for d in docs_run for L in layers_run for t in (32, 96)}
    floor_map, floor_global = {}, None
    if (not SMOKE and partial.get("floor_map") is not None
            and {(k.split("|")[0], int(k.split("|")[1]), int(k.split("|")[2]))
                 for k in partial["floor_map"].keys()} >= need_keys):
        floor_map = {((k.split("|")[0], int(k.split("|")[1]), int(k.split("|")[2]))): v
                     for k, v in partial["floor_map"].items()}
        floor_global = partial["floor_global"]
        logline(f"floor pre-pass: resume из partial ({len(floor_map)} замеров, "
                f"floor_global={floor_global:.3e})")
    else:
        tstage("floor_prepass")
        for doc in docs_run:
            fwd = fwds[doc]
            z0 = baselines[doc]["z0"]
            hs = baselines[doc]["hs"]
            for L in layers_run:
                b0_f, _, _ = b0_full(LAYERS.index(L) * 8 + DOCS.index(doc) * 2, hs[L])
                for t in (32, 96):
                    ctx = {"L": L, "t": t, "hnorm": float(hs[L][t].double().norm()),
                           "z0": z0, "pwin": make_pwin(z0, t)}
                    floor_map[(doc, L, t)] = measure_floor(fwd, ctx, b0_f)
        floor_global = max(f["used"] for f in floor_map.values())
        logline(f"floor pre-pass: {len(floor_map)} уникальных (doc,L,t), "
                f"floor_global={floor_global:.3e}")
        tstage("floor_prepass_done", {"n_unique": len(floor_map),
                                      "floor_global": floor_global})
        if not SMOKE:
            partial["floor_map"] = {f"{d}|{l}|{t}": v for (d, l, t), v in floor_map.items()}
            partial["floor_global"] = floor_global
            save_partial()

    for doc in docs_run:
        fwd = fwds[doc]
        z0 = baselines[doc]["z0"]
        hs = baselines[doc]["hs"]
        for L in layers_run:
            sib = None
            for ti, t in enumerate(T_POS if not SMOKE else (32,)):
                cell_id = LAYERS.index(L) * 8 + DOCS.index(doc) * 2 + ti
                if cell_id in cells_done:
                    logline(f"cell {cell_id} ({doc} L{L} t{t}): skip (resume)")
                    if t == 32:
                        c32 = next(c for c in cells if c["cell_id"] == cell_id)
                        rr = c32.get("random_ref") or {}
                        sib = {"refs": rr.get("per_dir", []), "eps0": c32.get("eps0", 0.0)}
                    continue
                tstage(f"cell{cell_id}_start", {"doc": doc, "L": L, "t": t})
                rec = run_cell(fwd, doc, L, t, cell_id, z0, hs,
                               floor_map, floor_global,
                               force_pos_spec=SMOKE, sibling=sib, force_1d=SMOKE)
                cells.append(rec)
                partial["cells"] = cells
                if not SMOKE:
                    save_partial()  # B2: чекпойнт после каждой клетки (main-only)
                if t == 32:
                    # r3 V1: мёртвая клетка (rec_min) — без random_ref: sibling-референс
                    # пустой (pos-spec т96-клетки: n_ref=0 -> screen_ok=False, провал)
                    rr = rec.get("random_ref") or {}
                    sib = {"refs": rr.get("per_dir", []), "eps0": rec.get("eps0", 0.0)}
                # смок-якоря (pre-study r4, канонический замер)
                if SMOKE and doc == "wiki" and L == 12 and t == 32:
                    S = smoke
                    c = rec
                    ok3 = (c["b0"]["checks"]["gqr_orth_max_dev"] < 1e-9
                           and c["b0"]["checks"]["top_orth_max_dev"] < 1e-9
                           and c["b0"]["checks"]["bot_orth_max_dev"] < 1e-9
                           and c["b0"]["checks"]["norm_max_dev"] < 1e-9
                           and c["b0"]["checks"]["norm_f32_max_dev"] < 1e-6)
                    smoke_checks(S, ok3, "S3_b0", c["b0"]["checks"])
                    smoke_checks(S, c["eps0"] > 0, "S4_eps0", c["eps0"])
                    smoke_checks(S, np.isfinite(c["pencil_report"]), "S4_pencil",
                                 c["pencil_report"])
                    hnorm_anchor = 2.1614  # pre-study r4: L12 t32 wiki (канон.)
                    smoke_checks(S, abs(c["hnorm"] - hnorm_anchor) / hnorm_anchor <= 0.02,
                                 "S3_hnorm_anchor", c["hnorm"])
                    # S5: VJP <-> FD при a_hi
                    v5 = np.random.RandomState(SEED).standard_normal(D)
                    v5 = (v5 / np.linalg.norm(v5)).astype(np.float32)
                    a5 = c["a_hi"]
                    t5, L5 = 32, 12
                    pwin5 = make_pwin(z0, t5)
                    pr5 = fwd.probe(L5, t5, v5, a5, z0, pwin5)
                    dz5 = {k: pr5[k]["dz"] for k in range(H + 1)}
                    Ov5, Cv5 = vjp_leaf(fwd, L5, t5, dz5, a5, pwin5, v5, K1)
                    Jv5 = {k: dz5[k] / a5 for k in range(H + 1)}
                    # каноническая сверка (at2e_vjp4): FD = sum_{k in K1} c_k . Jv_k
                    # vs vjp = Ov . v (Ov = sum J_k^T c_k; c_k = P_k Jv_k)
                    c5 = {}
                    for k in range(H + 1):
                        p = pwin5[t5 + k]
                        c5[k] = p * Jv5[k] - (p @ Jv5[k]) * p
                    fd_ov_v = float(sum(float(c5[k] @ Jv5[k]) for k in K1))
                    vjp_ov_v = float(Ov5 @ v5)
                    fd_cv_v = float(c5[0] @ Jv5[0])
                    vjp_cv_v = float(Cv5 @ v5)
                    rel_ov = abs(vjp_ov_v - fd_ov_v) / max(abs(fd_ov_v), 1e-9)
                    rel_cv = abs(vjp_cv_v - fd_cv_v) / max(abs(fd_cv_v), 1e-9)
                    smoke_checks(S, rel_ov <= 0.05 and rel_cv <= 0.05,
                                 "S5_vjp_fd", {"rel_ov": rel_ov, "rel_cv": rel_cv})
                    # eps0<=0 -> rec_min (нет ascent/rank/kappa) — чистый FAIL, не краш
                    if "ascent" in c:
                        a6 = c["ascent"]
                        # r5.2: ранний стоп — флаг в v0_recheck (ит. 1) или в записи v_{i-1}
                        ok6 = (all(np.isfinite(tr["lambda_K1"]) for tr in a6["traj"])
                               and all(tr.get("delta_v_norm") is None
                                       or tr["delta_v_norm"] >= 0 for tr in a6["traj"])
                               and (a6["v0_recheck"].get("early_stop")
                                    or any(tr.get("early_stop") for tr in a6["traj"])
                                    or any((tr.get("delta_v_norm") or 0.0) > 0
                                           for tr in a6["traj"])))
                        smoke_checks(S, ok6, "S6_ascent",
                                     {"vstar_i": a6["vstar_argmax"],
                                      "eta": [tr["eta"] for tr in a6["traj"]],
                                      "lambda": [tr["lambda_K1"] for tr in a6["traj"]]})
                        r8 = c["rank"]
                        ok8 = all(1 <= r8["K1"][tau] <= N_RANDOM + 1 for tau in r8["K1"]) \
                            and c["rho_ref_p90"] is not None
                        smoke_checks(S, ok8, "S8_random", r8)
                        smoke_checks(S, c["verify"]["kappa_flags_hi"] is not None,
                                     "S10_kappa", c["verify"]["kappa_flags_hi"])
                    else:
                        smoke_checks(S, False, "S6_ascent", "no ascent (eps0<=0, S4)")
                        smoke_checks(S, False, "S8_random", "no ascent (eps0<=0, S4)")
                        smoke_checks(S, False, "S10_kappa", "no ascent (eps0<=0, S4)")
                    v7 = c["verify"]
                    ok7 = (v7["linear"] is not None and v7["companion_qbar_K1"] is not None
                           and c.get("pos_spec") is not None)
                    smoke_checks(S, ok7, "S7_verify",
                                 {"mode": v7["mode"], "linear_ok": v7["linear"]["ok"],
                                  "denom_cell": v7.get("denom_gate_cell_value")})
    if SMOKE:
        # S11: дет-подмножество (B0-пробы j=0,1 a_hi + verify v* a_hi) — md5 идентичны
        c_sm = cells[0]
        fwd_sm = fwds["wiki"]
        z0_sm = baselines["wiki"]["z0"]
        b0_sm, _, _ = b0_full(c_sm["cell_id"], baselines["wiki"]["hs"][12])
        pwin_sm = make_pwin(z0_sm, 32)
        ok11 = "det_md5" in c_sm and "ascent" in c_sm
        if ok11:
            for j in (0, 1):
                pr = fwd_sm.probe(12, 32, b0_sm[j], c_sm["a_hi"], z0_sm, pwin_sm)
                ok11 = ok11 and md5_tensor(pr["zplus"]) == c_sm["det_md5"][f"b0j{j}_hi"]
        if "ascent" in c_sm:
            v0_arr = np.asarray(c_sm["ascent"]["v"], dtype=np.float64)
            prv = fwd_sm.probe(12, 32, v0_arr.astype(np.float32), c_sm["a_hi"],
                               z0_sm, pwin_sm)
            ok11 = ok11 and md5_tensor(prv["zplus"]) == c_sm["verify"]["det_md5"]["verify_a_hi"]
        else:
            ok11 = False
        smoke_checks(smoke, ok11, "S11_det_subset", "md5")
        # S13: стадия 1d на смок-клетке — в смоке force_1d: гоняется безусловно
        # (M2 r5.2: скрин — научный исход, гейт проверяет пайплайн, не данные)
        s1d = c_sm.get("stage1d")
        ok13 = (s1d is not None
                and s1d["testable"] is not None
                and (s1d["gap"] is None or np.isfinite(s1d["gap"]))
                and np.isfinite(s1d["ascent_tprime"]["lambda_vstar_tprime"]))
        smoke_checks(smoke, ok13, "S13_stage1d",
                     {"gap": s1d.get("gap") if s1d else None,
                      "test_guard": s1d.get("test_guard") if s1d else None})
    # ---- G-zero (wiki/0, t32, L23)
    fwd_g = fwds["wiki"]
    cell_g = next(c for c in cells if c["doc"] == "wiki" and c["L"] == 12 and c["t"] == 32)
    b0_g, _, _ = b0_full(cell_g["cell_id"], baselines["wiki"]["hs"][12])
    gz = g_zero(fwd_g, 23, 32, b0_g[0], cell_g["a_hi"])
    logline(f"G-zero L23: max_dz_fut={gz['max_dz_fut']:.3e} gate={gz['gate_ok']}")
    if SMOKE:
        smoke_checks(smoke, gz["gate_ok"], "S9_gzero", gz["max_dz_fut"])

    # ---- main-only: secondary (K2), 1c, cext_ref (resume: stage_done-маркеры)
    secondary_res, s1c_res, cext_res = (partial["secondary_k2"], partial["nonlin_probe"],
                                        partial["cext_ref"])
    if not SMOKE:
        # r3 self-audit: мёртвая клетка (rec_min, eps0=0) — lam-вычисления дают 0/0;
        # отчётные блоки (secondary/s1c/cext) пропускают её с пометкой
        sec_cells = [(c["doc"], c["L"]) for c in cells
                     if c["t"] == 32 and c["doc"] in ("wiki", "math") and "ascent" in c]
        for c in cells:
            if (c["t"] == 32 and c["doc"] in ("wiki", "math") and "ascent" not in c):
                logline(f"secondary/s1c {c['doc']} L{c['L']}: SKIP — мёртвая клетка "
                        f"(eps0<=0, rec_min)")
        if "secondary" not in stage_done:
            for doc, L in sec_cells:
                c = next(x for x in cells if x["doc"] == doc and x["L"] == L and x["t"] == 32)
                fwd_c = fwds[doc]
                pwin = make_pwin(baselines[doc]["z0"], 32)
                b0_c, _, _ = b0_full(c["cell_id"], baselines[doc]["hs"][L])
                logline(f"secondary {doc} L{L}: подъём на K2")
                sec_ctx = {"L": L, "t": 32, "a_hi": c["a_hi"], "pwin": pwin,
                           "z0": baselines[doc]["z0"], "eps0": c["eps0"],
                           "b0_q": _b0q_from_ref(c)}
                secondary_res[f"{doc}_L{L}_t32"] = secondary_k2(fwd_c, sec_ctx, b0_c, c["eps0"])
            stage_done["secondary"] = True
            save_partial()
        if "s1c" not in stage_done:
            for doc, L in sec_cells:
                c = next(x for x in cells if x["doc"] == doc and x["L"] == L and x["t"] == 32)
                fwd_c = fwds[doc]
                pwin = make_pwin(baselines[doc]["z0"], 32)
                ctx1c = {"L": L, "t": 32, "a_hi": c["a_hi"], "pwin": pwin,
                         "z0": baselines[doc]["z0"], "hnorm": c["hnorm"]}
                b0_c, _, _ = b0_full(c["cell_id"], baselines[doc]["hs"][L])
                # r3 U2: v* из ascent (resume: list из JSON -> ndarray)
                vstar_c = np.asarray(c["ascent"]["v"], dtype=np.float64)
                s1c_res[f"{doc}_L{L}_t32"] = nonlin_probe_1c(fwd_c, ctx1c, b0_c, vstar_c)
            stage_done["s1c"] = True
            save_partial()
        if jcache is None:
            logline("cext_ref: SKIP — jcache не загружен (T4, TZ §6b: CPU-follow-up)")
        elif "cext" not in stage_done:
            for doc in DOCS:
                c12 = next(x for x in cells if x["doc"] == doc and x["L"] == 12 and x["t"] == 32)
                if "ascent" not in c12:
                    logline(f"cext_ref {doc}: SKIP — мёртвая клетка (eps0<=0, rec_min)")
                    continue
                pwin = make_pwin(baselines[doc]["z0"], 32)
                cext_res[doc] = cext_ref(fwds[doc], jcache, doc, baselines[doc]["z0"], pwin,
                                         12, 32, c12["a_hi"], c12["eps0"])
                logline(f"cext_ref {doc}: {cext_res[doc]['n_dirs']} dir")
            stage_done["cext"] = True
            save_partial()

    # N4 post-pass: eps0(t'=96)-диагностика для t32 клеток (0 доп. forward:
    # q пробы t' сохранены в pos_spec["probe_q_tprime"]); идемпотентен на resume
    for c in cells:
        if (c["t"] == 32 and c.get("pos_spec") is not None
                and "eps0_tprime_diag" not in c["pos_spec"]):
            sib96 = next((x for x in cells
                          if x["doc"] == c["doc"] and x["L"] == c["L"] and x["t"] == 96),
                         None)
            if (sib96 is not None and sib96["eps0"] > 0
                    and c["verify"]["per_amp"]["a_hi"]["lambda_K1_tau"][str(TAU_PRIMARY)] > 0):
                q96 = c["pos_spec"]["probe_q_tprime"]
                a_p = c["pos_spec"]["a_prime"]
                lam_v_t = c["verify"]["per_amp"]["a_hi"]["lambda_K1_tau"][str(TAU_PRIMARY)]
                Dden2 = a_p * a_p * TAU_PRIMARY * sib96["eps0"]
                lam2 = float(2.0 * np.mean([q96[str(k)] for k in K1])
                             / (q96["0"] + Dden2))
                c["pos_spec"]["eps0_tprime_diag"] = float(lam2 / lam_v_t)

    # ---- агрегированный gates-блок (M3 r5.2, схема TZ §13)
    gates = {
        "S1_sha": {"pool": pool_sha == POOL_SHA, "jcache_l12": jc_sha == JCACHE_SHA,
                   "fla_guard": guard_ok},
        "S1b_pool_fp": pool_fp_ok,
        "G_floor": {"n_unique": len(floor_map), "floor_global": floor_global,
                    "per_cell": {f"c{c['cell_id']}": c["floor"]["used"] for c in cells}},
        "G_b0_ortho": {f"c{c['cell_id']}": c["b0"]["checks"] for c in cells},
        "G_denom": {f"c{c['cell_id']}": {"cell": c["verify"]["denom_gate_cell"],
                                         "vstar": c["verify"]["denom_gate_vstar"]}
                    for c in cells},
        "G_linear": {f"c{c['cell_id']}": c["verify"]["linear"]["ok"] for c in cells},
        "G_vjp_fd": None,  # main: однократная FD-сверка VJP (ниже, 5 ф.-э.)
        "G_pos_spec_ref": {f"c{c['cell_id']}":
                           {"n_ref": c["pos_spec"]["n_ref"],
                            "n_ref_finite": c["pos_spec"].get("n_ref_finite", 0),
                            "ratios_finite": (bool(np.all(np.isfinite(c["pos_spec"]["ratio_ref"])))
                                              if c["pos_spec"]["ratio_ref"] else False)}
                           for c in cells if c.get("pos_spec") is not None},
    }
    if not SMOKE:
        # G_vjp_fd (main): каноническая FD-сверка на клетке c0 (wiki L12 t32):
        # 1 probe (2 fwd) + 1 vjp (grad-fwd + 2 bwd) = 5 ф.-э. (смета TZ §4 r5.2)
        tstage("g_vjp_fd")
        c0v = next(c for c in cells if c["doc"] == "wiki" and c["L"] == 12 and c["t"] == 32)
        fwd0v = fwds["wiki"]
        z0v = baselines["wiki"]["z0"]
        v5 = np.random.RandomState(SEED).standard_normal(D)
        v5 = (v5 / np.linalg.norm(v5)).astype(np.float32)
        a5 = c0v["a_hi"]
        pwin5 = make_pwin(z0v, 32)
        pr5 = fwd0v.probe(12, 32, v5, a5, z0v, pwin5)
        dz5 = {k: pr5[k]["dz"] for k in range(H + 1)}
        Ov5, Cv5 = vjp_leaf(fwd0v, 12, 32, dz5, a5, pwin5, v5, K1)
        Jv5 = {k: dz5[k] / a5 for k in range(H + 1)}
        c5 = {k: pwin5[32 + k] * Jv5[k] - (pwin5[32 + k] @ Jv5[k]) * pwin5[32 + k]
              for k in range(H + 1)}
        fd_ov = float(sum(float(c5[k] @ Jv5[k]) for k in K1))
        vjp_ov = float(Ov5 @ v5)
        fd_cv = float(c5[0] @ Jv5[0])
        vjp_cv = float(Cv5 @ v5)
        rel_ov = abs(vjp_ov - fd_ov) / max(abs(fd_ov), 1e-9)
        rel_cv = abs(vjp_cv - fd_cv) / max(abs(fd_cv), 1e-9)
        gates["G_vjp_fd"] = {"rel_ov": rel_ov, "rel_cv": rel_cv,
                             "ok": bool(rel_ov <= 0.05 and rel_cv <= 0.05)}
        logline(f"G_vjp_fd: rel_ov={rel_ov:.3e} rel_cv={rel_cv:.3e} "
                f"ok={gates['G_vjp_fd']['ok']}")
        assert gates["G_vjp_fd"]["ok"], ("G_vjp_fd FAIL: VJP отклоняется от "
                                         "централь. разностей >5% — подъёмы невалидны")

    # ---- вердикты
    tstage("verdicts")
    gverdict = global_verdict(cells)
    logline(f"global verdict: {gverdict}")
    logline("per-cell: " + " ".join(
        f"c{c['cell_id']}({c['doc']}L{c['L']}t{c['t']})={c['verdict']}" for c in cells))

    # ---- B5 (r5.2): дет-подмножество main (8 forward) ДО стадии 1b — reinit
    # null-весов in-place разрушил бы trained-baseline и md5-сверку гарантированно
    # бы «провалилась» на самом интересном прогоне.
    # r3 V3: det_main хранится в partial (на resume не теряется);
    # r3 V1: c0 без ascent (мёртвая клетка) — дет-блок пропускается с пометкой
    det_main = partial.get("det_main")
    if not SMOKE and "det_main" not in stage_done:
        doc0 = "wiki"
        c0 = next(c for c in cells if c["doc"] == doc0 and c["L"] == 12 and c["t"] == 32)
        if "ascent" not in c0:
            det_main = {"skipped": "c0 (wiki L12 t32) без ascent (мёртвая клетка)",
                        "all_ok": None}
            logline("det_main: SKIP — c0 без ascent (мёртвая клетка)")
        else:
            tstage("det_main")
            # 8 forward финального чек-блока: baseline(1) + b0 j=0,1 a_hi(4) +
            # verify v* a_hi(2) + floor(1); md5-сверка с сохранёнными
            fwd0 = fwds[doc0]
            z0r, hs0 = fwd0.baseline()
            md5 = {"baseline": md5_tensor(z0r)}
            b0_c, _, _ = b0_full(c0["cell_id"], hs0[12])
            pwin0 = make_pwin(baselines[doc0]["z0"], 32)
            for j in (0, 1):
                pr = fwd0.probe(12, 32, b0_c[j], c0["a_hi"], baselines[doc0]["z0"], pwin0)
                md5[f"b0j{j}"] = md5_tensor(pr["zplus"])
            pwin0v = make_pwin(baselines[doc0]["z0"], 32)
            v0_arr = np.asarray(c0["ascent"]["v"], dtype=np.float64)  # resume: list из JSON
            prv = fwd0.probe(12, 32, v0_arr.astype(np.float32), c0["a_hi"],
                             baselines[doc0]["z0"], pwin0v)
            md5["verify_hi"] = md5_tensor(prv["zplus"])
            prf = fwd0.probe(12, 32, b0_c[0], A_FLOOR_REL * c0["hnorm"],
                             baselines[doc0]["z0"], pwin0v)
            md5["floor"] = md5_tensor(prf["zplus"])
            ok_all = (md5["baseline"] == baselines[doc0]["md5"]
                      and md5["b0j0"] == c0["det_md5"]["b0j0_hi"]
                      and md5["b0j1"] == c0["det_md5"]["b0j1_hi"]
                      and md5["verify_hi"] == c0["verify"]["det_md5"]["verify_a_hi"]
                      and md5["floor"] == c0["floor"]["at_t"]["md5"])
            det_main = {"md5": md5, "all_ok": bool(ok_all)}
            logline(f"det_main (8 forward): all_ok={ok_all}")
            if not ok_all:
                logline("det_main: ДЕТЕРМИНизм-сверка НЕ сошлась — артефакт помечен")
        partial["det_main"] = det_main
        stage_done["det_main"] = True
        save_partial()
    elif not SMOKE:
        logline(f"det_main: skip (resume, stage_done) — all_ok={det_main.get('all_ok') if det_main else None}")

    # ---- условная стадия 1b (только main, при SCRATCHPAD_CANDIDATE)
    stage1b_res, stage1b_verdict = None, None
    if not SMOKE and gverdict == "SCRATCHPAD_CANDIDATE" and "stage1b" not in stage_done:
        tstage("stage1b_start")
        stage1b_res = run_stage1b(model, pool_ids)
        stage1b_verdict = stage1b_verdict(stage1b_res, gverdict)
        logline(f"stage 1b verdict: {stage1b_verdict}")
        partial["stage1b"] = {"cells": stage1b_res, "verdict": stage1b_verdict}
        stage_done["stage1b"] = True
        save_partial()
    elif not SMOKE and "stage1b" in stage_done and partial.get("stage1b"):
        stage1b_res = partial["stage1b"]["cells"]
        stage1b_verdict = partial["stage1b"]["verdict"]
        logline(f"stage 1b: skip (resume) — verdict={stage1b_verdict}")

    # ---- артефакт
    tstage("artifact")
    finished = time.time()
    artifact = {
        "header": {"run": RUN_NAME, "run_revision": f"{REVISION} (TZ r5.2)",
                   "script_sha": script_sha, "seed": SEED, "device": DEVICE,
                   "model": MODEL_DIR,
                   "transformers_version": transformers.__version__,  # r3 V6
                   "canary": None,  # r3 V6: CPU-run (T4-canary — только в T4-обёртке, TZ §6b)
                   "forward_s": 4.33, "vjp_s": 8.41,  # r3 V6: провенанс (pre-study r1, TZ §6)
                   "log": os.path.join(RESULTS, f"{RUN_NAME}.log"),
                   "started_at": started_iso,
                   "finished_at": datetime.now().isoformat(),
                   "duration_s": round(finished - t_start, 1), "status": None},
        "config": cfg,
        "smoke": smoke if SMOKE else None,
        "gates": gates,
        "gzero_L23": gz,
        "cells": cells,
        "secondary_k2": secondary_res or None,
        "nonlin_probe": s1c_res or None,
        "cext_ref": cext_res or None,
        "det_main": det_main,
        "verdicts": {"per_cell": {f"c{c['cell_id']}": c["verdict"] for c in cells},
                     "global": gverdict,
                     "final": stage1b_verdict if stage1b_verdict else gverdict},
        "stage1b": ({"seed_null": SEED_NULL, "n_cells": len(stage1b_res),
                     "cells": stage1b_res,
                     "verdict": stage1b_verdict} if stage1b_res else None),
        "invariants": {
            "1_fla_guard": guard_ok,
            "2_num_denom_one_forward": True,  # q_t и q_fut — из одного forward-пара (механика probe)
            "4_q_and_kappa_reported": True,
            "12_relative_amplitude": True,
            "14_rank_screen": True,
            "15_companion_abs": True,
            "16_single_ensemble": True,
            "18_transfer_gap": True,  # r3 минор: 1d гоняется и в смоке (force_1d, S13)
        },
    }
    art_path = os.path.join(RESULTS, f"{RUN_NAME}.json")
    log_path = os.path.join(RESULTS, f"{RUN_NAME}.log")
    # предварительный dump (S12 проверяет его парсингом)
    with open(art_path, "w") as f:
        json.dump(_san(artifact), f, ensure_ascii=False, indent=1)
    # S12: артефакт валиден — парсинг + покрытие схемы (r5.2 M3: gates-блок на месте)
    if SMOKE:
        try:
            back = json.load(open(art_path))
            # r3 V6: S12 — schema-покрытие: gates + r3-поля (header/verify/ascent/b0)
            c0x = back["cells"][0]
            ok12 = (back.get("gates") is not None and back.get("cells") is not None
                    and back.get("verdicts") is not None and len(back["cells"]) == len(cells)
                    and back["header"].get("transformers_version") is not None
                    # canary: в CPU-ранге = None (легитимно, TZ §13) — проверяем КЛЮЧ
                    and "canary" in back["header"]
                    and back["header"].get("forward_s") is not None
                    and c0x["verify"]["per_amp"]["a_hi"].get("denom_eps_share_tau") is not None
                    and ("ascent" in c0x and c0x["ascent"].get("eps_ascent") is not None
                         or "ascent" not in c0x)
                    and len(c0x["b0"].get("per_dir", [])) == 32)
            smoke_checks(smoke, ok12, "S12_artifact",
                         {"gates_keys": sorted(back.get("gates", {}).keys())})
        except Exception as e:
            smoke_checks(smoke, False, "S12_artifact", str(e))
    smoke_ok = all(v["ok"] for v in smoke.values()) if SMOKE else True
    status = ("smoke" if smoke_ok else "smoke_fail") if SMOKE else "ok"
    # r3 минор: детерминизм-провал не гасится "ok"
    if (status == "ok" and det_main is not None
            and det_main.get("all_ok") is False):
        status = "ok_det_warn"
    # M4 (r5.2): маркер в _LOG ДО записи лог-файла (регресс AT2d m7: маркер
    # после write — вне файла)
    if SMOKE:
        if not smoke_ok:
            logline("SMOKE: есть провалы — артефакт не считается, стоп (пересмоук)")
        logline("AT2E_SMOKE_DONE" if smoke_ok else "AT2E_SMOKE_FAIL")
    else:
        logline("AT2E_MAIN_DONE")
    log_content = "\n".join(_LOG) + "\n"
    with open(log_path, "w") as f:
        f.write(log_content)
    artifact["header"]["status"] = status
    artifact["header"]["log_md5"] = hashlib.md5(log_content.encode()).hexdigest()
    if SMOKE:
        artifact["smoke"] = smoke
    with open(art_path, "w") as f:
        json.dump(_san(artifact), f, ensure_ascii=False, indent=1)
    # manifest — отдельный файл: sha256 финального артефакта
    manifest = {"script_sha": script_sha,
                "log_sha": artifact["header"]["log_md5"],
                "artifact_sha": sha256_file(art_path),
                "trackio_run": RUN_NAME,
                "started_at": artifact["header"]["started_at"],
                "finished_at": artifact["header"]["finished_at"]}
    man_path = os.path.join(RESULTS, f"{RUN_NAME}.manifest.json")
    with open(man_path, "w") as f:
        json.dump(_san(manifest), f, ensure_ascii=False, indent=1)
    if SMOKE:
        if not smoke_ok:
            trackio_finish("interrupted")
            raise SystemExit(1)
        trackio_finish("ok")
    else:
        trackio_finish("ok")

def _b0q_from_ref(c):
    """b0_q (per j: q per k при a_hi) из cell-rec (хранится явно, run_cell)."""
    return c["b0_q_hi"]


def canary_mode():
    """T4-canary (TZ §6b r3; review r3 §5.1): wiki/0 t=32, ОБЕ клетки L11+L12 (полный
    пайплайн — ascent с retain-двойным backward) + baseline всех 4 доков (VRAM-footprint:
    4 × 128 × 248320 × 4 B ≈ 0.5 GB, держатся в памяти на время клеток).
    Гейты: (1) VRAM_peak ≤ 12 GB; (2) GPU-дет — двойной forward md5; (3) wall-цена
    fwd/bwd → фиксация T4-бюджета. Провал → AT2E_T4_CANARY_FAIL + SystemExit(1)."""
    global _TRACK_RUN
    t_start = time.time()
    assert DEVICE == "cuda", "canary — T4-only (AT2E_DEVICE=cuda)"
    script_sha = (sha256_file(os.path.abspath(__file__)) if "__file__" in globals()
                  else (os.environ.get("AT2E_SCRIPT_SHA") or "embedded-no-sha"))
    started_iso = datetime.now().isoformat()
    cfg = {"device": DEVICE, "model": MODEL_DIR, "revision": REVISION,
           "tz": "r5.2 (canary)", "script_sha": script_sha, "seed": SEED,
           "started_at": started_iso, "canary": True}
    trackio_start(cfg)
    atexit.register(trackio_crash)
    logline(f"AT2e T4-canary: script_sha={script_sha[:12]}, device={DEVICE}")

    pool_sha = sha256_file(POOL)
    assert pool_sha == POOL_SHA, f"S1 FAIL: pool sha {pool_sha}"
    assert sys.modules.get("fla") is None, "S1 FAIL: fla guard"
    logline("S1: pool sha OK + fla guard (jcache — T4 skip, §6b)")

    model = load_model()
    pool, pool_ids = load_pool()
    # baseline всех 4 доков (footprint per review r3 §5.1)
    fwds, baselines = {}, {}
    for doc in DOCS:
        ids = torch.as_tensor(pool_ids[doc][None]).to(DEVICE)
        mask = torch.ones((1, S_LEN), dtype=torch.long).to(DEVICE)
        fwd = Fwd(model, ids, mask)
        z0, hs = fwd.baseline()
        fwds[doc] = fwd
        baselines[doc] = {"z0": z0, "hs": hs, "md5": md5_tensor(z0)}
    # (2) GPU-дет: двойной forward md5 (wiki)
    z0b, _ = fwds["wiki"].baseline(need_hidden=False)
    det_ok = bool(baselines["wiki"]["md5"] == md5_tensor(z0b))
    logline(f"GPU det (wiki baseline x2): {det_ok}")
    vram_after_baselines = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()

    # floor pre-pass wiki L11/L12 (4 замера, 8 fwd)
    floor_map = {}
    z0w, hsw = baselines["wiki"]["z0"], baselines["wiki"]["hs"]
    for li, L in enumerate(LAYERS):
        b0_f, _, _ = b0_full(li * 8, hsw[L])
        for t in (32, 96):
            ctx = {"L": L, "t": t, "hnorm": float(hsw[L][t].double().cpu().norm()),
                   "z0": z0w, "pwin": make_pwin(z0w, t)}
            floor_map[("wiki", L, t)] = measure_floor(fwds["wiki"], ctx, b0_f)
    floor_global = max(f["used"] for f in floor_map.values())
    logline(f"floor pre-pass (wiki, 4 замера): floor_global={floor_global:.3e}")

    # обе клетки L11+L12 t=32 — полный пайплайн (ascent: retain-двойной backward)
    cells = []
    for li, L in enumerate(LAYERS):
        tstage(f"canary_cell_L{L}_start")
        rec = run_cell(fwds["wiki"], "wiki", L, 32, li * 8, z0w, hsw,
                       floor_map, floor_global)
        cells.append(rec)
        logline(f"canary L{L}: eps0={rec.get('eps0')} verdict={rec.get('verdict')} "
                f"ascent={'yes' if 'ascent' in rec else 'NO (rec_min)'}")
    vram_peak = torch.cuda.max_memory_allocated() / 2**30

    # (3) wall-цена: fwd (baseline x2) и vjp (1 probe + 1 vjp_leaf)
    t0 = time.time()
    fwds["wiki"].baseline(need_hidden=False)
    fwds["wiki"].baseline(need_hidden=False)
    fwd_s = (time.time() - t0) / 2.0
    b0_c, _, _ = b0_full(8, hsw[12])
    a_c = 1e-3 * float(hsw[12][32].double().cpu().norm())
    t0 = time.time()
    pr_c = fwds["wiki"].probe(12, 32, b0_c[0], a_c, z0w, make_pwin(z0w, 32))
    t_probe = time.time() - t0
    dz_c = {k: pr_c[k]["dz"] for k in range(H + 1)}
    t0 = time.time()
    vjp_leaf(fwds["wiki"], 12, 32, dz_c, a_c, make_pwin(z0w, 32), b0_c[0], K1)
    vjp_s = time.time() - t0
    logline(f"wall: fwd={fwd_s:.2f}s vjp(1 grad-fwd + 2 bwd)={vjp_s:.2f}s "
            f"(probe={t_probe:.2f}s)")

    ok_vram = bool(vram_peak <= 12.0)
    ok_det = bool(det_ok)
    ok_cells = all("ascent" in c and c["eps0"] > 0 for c in cells)
    canary_ok = bool(ok_vram and ok_det and ok_cells)
    artifact = {
        "header": {"run": "at2e_t4_canary", "script_sha": script_sha,
                   "device": DEVICE, "model": MODEL_DIR,
                   "started_at": started_iso,
                   "finished_at": datetime.now().isoformat(),
                   "duration_s": round(time.time() - t_start, 1),
                   "vram_peak_gb": round(vram_peak, 3),
                   "vram_after_baselines_gb": round(vram_after_baselines, 3),
                   "fwd_s": round(fwd_s, 3), "vjp_s": round(vjp_s, 3),
                   "status": "ok" if canary_ok else "canary_fail"},
        "config": cfg,
        "gates": {"vram_peak_le_12gb": ok_vram, "gpu_det": ok_det,
                  "both_cells_ascent": ok_cells},
        "cells": [{"cell_id": c["cell_id"], "L": c["L"], "verdict": c["verdict"],
                   "eps0": c.get("eps0"),
                   "ascent": ({"vstar_argmax": c["ascent"]["vstar_argmax"],
                               "lambda_vstar": c["ascent"]["lambda_vstar"]}
                              if "ascent" in c else None)} for c in cells],
    }
    art_path = os.path.join(RESULTS, "at2e_t4_canary.json")
    os.makedirs(RESULTS, exist_ok=True)
    with open(art_path, "w") as f:
        json.dump(_san(artifact), f, ensure_ascii=False, indent=1)
    logline(f"canary: vram_peak={vram_peak:.2f}GB (<=12: {ok_vram}) det={ok_det} "
            f"cells_ok={ok_cells} -> {'PASS' if canary_ok else 'FAIL'}")
    log_content = "\n".join(_LOG) + "\n"
    with open(os.path.join(RESULTS, "at2e_t4_canary.log"), "w") as f:
        f.write(log_content)
    if canary_ok:
        logline("AT2E_T4_CANARY_DONE")
        trackio_finish("ok")
    else:
        logline("AT2E_T4_CANARY_FAIL")
        trackio_finish("interrupted")
        raise SystemExit(1)


def reanalysis_mode():
    """--reanalysis [path]: disk-реанализ frozen raw-артефакта БЕЗ форвардов —
    пересчёт вердиктов (чистые функции) по сохранённым метрикам + сверка со
    сохранёнными вердиктами."""
    i = sys.argv.index("--reanalysis")
    path = sys.argv[i + 1] if i + 1 < len(sys.argv) else os.path.join(RESULTS, "at2e_s1_r3.json")
    art = json.load(open(path))
    cells = art["cells"]
    out = []
    mismatch = 0
    for c in cells:
        v = cell_verdict(c)
        if v != c["verdict"]:
            mismatch += 1
            logline(f"reanalysis c{c['cell_id']}: {c['verdict']} -> {v} (МИСМАТЧ)")
        out.append(v)
    g = global_verdict(cells)
    logline(f"reanalysis global: {g} (сохранено: {art['verdicts']['global']}), "
            f"mismatch={mismatch}")
    out_path = path.replace(".json", "_reanalysis.json")
    with open(out_path, "w") as f:
        json.dump(_san({"source": path, "per_cell": out, "global": g,
                        "mismatch": mismatch}), f, ensure_ascii=False, indent=1)
    logline(f"AT2E_REANALYSIS_DONE {out_path}")


if __name__ == "__main__":
    main()
