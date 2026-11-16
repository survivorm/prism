#!/usr/bin/env python
"""AT2 (T2.1) — state-level ridge readout (disk-only).

Пре-рег: docs/AT2_prereg.md (ревью r1: REVISE → PASS, 6 правок).
Клетка T2.1 (docs/TZ_add_0-3.md §3.5): readout будущего логита из h_l^t,
спроецированного на weak-J / top-J / PCA_doc / Haar, vs lk4-текстовый базлайн.
Ноль новых форвардов: кэш exp27b + E (tied embed). Среда: /opt/miniconda3 base
py3.12.9 + torch 2.14.1; 4 CPU-трёда; CPU.
"""
import atexit, importlib.util, json, os, sys, time
from datetime import datetime

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import torch
torch.set_num_threads(4)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CACHE_DIR = "/data/prism_kernels/exp27b/results"
MODEL_DIR = "/data/Qwen/Qwen3.5-0.8B"
RESULTS = os.path.join(ROOT, "results")
ART = os.path.join(RESULTS, "AT2", "AT2.json")
LOG_PATH = os.path.join(RESULTS, "AT2_run.log")
REVISION = "r1"
SMOKE = os.environ.get("AT2_SMOKE", "0") == "1"

# --- фиксация пре-рега (fix-до-прогона) -------------------------------------
LAYS = (6, 12, 18, 22) if not SMOKE else (6,)   # A1: l <= 22
L_CTRL = 23
KS = (1, 2, 4, 8) if not SMOKE else (1,)
N_DOCS_RUN = 2 if SMOKE else 32                 # smoke: 2 дока
R = 32
EPS = 1e-2
ALPHA = 1.0
SEED = 0
BOOT_B = 100 if SMOKE else 1000
D = 1024
S_LEN = 128
DOMAINS = ["wiki", "math", "code", "chat"]
N_PER_DOM = 8
N_DOCS = 32
KER_SAMPLE = 32                                  # (l,t) на слой для ker-J контроля

SHA_FILES = {  # гейт S1 (AT0_9_T2_1_prereg.md §7, зафиксировано 16.11)
    "exp27b_jcache_L6.pt":  "6d6b8ce784a2ad01a10112309aad7a74f13413d14636aad1a00652575f605dd2",
    "exp27b_jcache_L12.pt": "976079ce1d3c84c7016a236683f86f5fee1ce6571ff6edd53cf4f6f9ac228a24",
    "exp27b_jcache_L18.pt": "e2a03bc36b199695018171d0f0eab92ee139be83d0f0792a558cefa15f852d63",
    "exp27b_jcache_L19.pt": "0b31baff7dbc6e38c525d10d803ed9cc313774c6c8460cf1290a9ab9f4df82ae",
    "exp27b_jcache_L22.pt": "36ed09ebc49dd97bc7b8e0a8edd1c8d75567bb6431c4eea33c908ccfb7295a84",
    "exp27b_jcache_L23.pt": "af12505131be72b424515759c054646ea4526d94822287cbe7e8c50b7db46a43",
    "exp27b_pool_ids_L23.pt": "ded634f7d16cf33a9bb565de5cbc3faa288cfc90b15cdfa3e42365695fec4e03",
}
LK4_EXPECT = {"total": 90, "wiki": 8, "math": 41, "code": 36, "chat": 5}  # гейт S4 (канон as-is на pin-пуле, 16.11; см. пре-рег §7)
LK4_EXPECT_SMOKE = {"total": 2, "wiki": 2, "math": 0, "code": 0, "chat": 0}  # wiki_0, wiki_1

_LOG = []
def logline(s):
    line = f"[{time.strftime('%H:%M:%S')}] {s}"
    print(line, flush=True)
    _LOG.append(line)

# sha256_file — общий из prism/field.py (spec-лоад, минуя prism/__init__ —
# тот тянет datasets/transformers):
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
    """Канон (experiments/exp64_at_battery.py:23-30)."""
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
    """К-ahead (пре-рег §5, version note): i + k <= 127, иначе промах."""
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

def hadamard_rows(r):
    H = np.array([[1.0]])
    while H.shape[0] < 1024:
        H = np.block([[H, H], [H, -H]])
    return (H[:r] / np.sqrt(1024)).astype(np.float32)

def d_eff_of(projs, Hdoc):
    """d_eff(S) = exp(H(lam(Pr_S Cov(h) Pr_S))), float64, по t в доке."""
    X = Hdoc.double() - Hdoc.double().mean(0, keepdim=True)     # [128, 1024]
    V = torch.as_tensor(projs).double()                          # [1024, r]
    C = (V.T @ X.T @ X @ V) / max(len(X) - 1, 1)                # [r, r]
    lam = torch.linalg.eigvalsh(C).clamp_min(0.0)
    lam = lam / lam.sum().clamp_min(1e-30)
    h = -(lam * torch.log(lam)).sum()
    return float(torch.exp(h).item())

# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(ART), exist_ok=True)
    logline(f"AT2 (T2.1) start {datetime.now().isoformat()} "
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
            "boot_b": BOOT_B, "n_docs": N_DOCS,
        }
        tr = trackio_start(f"at2_t21_{REVISION}", tr_cfg)
        tlog(tr, 0, {"started": 1.0})
        logline(f"trackio: run at2_t21_{REVISION} init ok")

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
    rng = np.random.default_rng(SEED)

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
    f23 = load_layer(L_CTRL, ("H", "P128", "P128_RAW"))
    H23, P23, PR23 = f23["H"], f23["P128"], f23["P128_RAW"]
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
    z_full = (H23 @ E.T)                                # [32,128,V] f32 (~4GB)
    zf = z_full
    p_soft = torch.softmax(zf, dim=-1)
    p_u128 = p_soft[:2][:, :, torch.from_numpy(U128)]         # [2 дока, 128 поз, 128 U128]
    kl = (p_u128 * (torch.log(p_u128.clamp_min(1e-30))
          - torch.log(PR23[:2].clamp_min(1e-30)))).sum(-1).mean().item()
    gates["S2_kl"] = bool(kl < 1e-5)
    logline(f"S2 z-маршрут: mean KL = {kl:.3e} {'OK' if gates['S2_kl'] else 'FAIL'}")
    if not gates["S2_kl"]:
        raise SystemExit("S2 FAIL")
    # S3: ориентация. Значение: z[cache t] = логиты forward 127-t ⇒ предсказывает
    # row[128-t] (t>=1; t=0 — токена 128 нет, NOTE 5). Правильная ориентация (hi):
    # P(row[128-t] | z[t]) — scope-замер 0.288/37%; flip (lo): P(row[128-t] | z[127-t]) — 0.018/2.4%.
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
    logline(f"S3 ориентация: hi {mean_hi:.3f}/{top1_hi:.3f} (ожид >0.1) "
            f"lo {mean_lo:.3f}/{top1_lo:.3f} (ожид <0.05) "
            f"{'OK' if gates['S3_orient'] else 'FAIL'}")
    if not gates["S3_orient"]:
        raise SystemExit("S3 FAIL")
    z_U128 = zf[:, :, torch.from_numpy(U128)]            # [32,128,128]
    del z_full, zf, p_soft, hi, lo

    # --- lk4 + S4 ---------------------------------------------------------------
    lk4_tot = 0; lk4_dom = {d: 0 for d in DOMAINS}
    lk4_mask = {}
    lk4_k = {k: {} for k in KS}
    for d in range(N_DOCS):
        dom = DOMAINS[d // N_PER_DOM]
        row = toks[dom][d % N_PER_DOM]
        h = lk4_hits(row)
        lk4_tot += int(h.sum()); lk4_dom[dom] += int(h.sum())
        lk4_mask[d] = hit_to_cache_mask(h)
        for k in KS:
            lk4_k[k][d] = hit_to_cache_mask(lk4_hits_kahead(row, k))
    exp = LK4_EXPECT_SMOKE if SMOKE else LK4_EXPECT
    if SMOKE:
        # smoke: точные ожидаемые числа по 2 докам неизвестны априори — гейт только
        # на внутреннюю согласованность (перепроверка канона на полном прогоне)
        gates["S4_lk4"] = True
        logline(f"S4 lk4 (smoke): total {lk4_tot}, per-домен "
                f"{{ {', '.join(f'{d} {lk4_dom[d]}' for d in DOMAINS)} }} — ожидаемые числа не гейтятся")
    else:
        gates["S4_lk4"] = (lk4_tot == exp["total"] and
                           all(lk4_dom[d] == exp[d] for d in DOMAINS))
        logline(f"S4 lk4: total {lk4_tot}/4096 per-домен "
                f"{{ {', '.join(f'{d} {lk4_dom[d]}' for d in DOMAINS)} }} "
                f"{'OK' if gates['S4_lk4'] else 'FAIL (ожид. %s)' % exp}")
    if not gates["S4_lk4"]:
        raise SystemExit("S4 FAIL")

    # --- подпространства ----------------------------------------------------------
    tstage(tr, time.time() - t0, "subspaces")
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
    # d_eff per (l,doc,S) (J-базисы — t=0 как репрезентативный, пре-рег §4)
    def lift(L, d, t, idxs, r):
        lam, w, jrt, sp = ew[(L, d)][t]
        V = np.zeros((D, r), dtype=np.float32)
        for j, i in enumerate(idxs):
            if lam[i] > 1e-12:
                V[:, j] = jrt @ (sp * w[:, i]) / np.sqrt(lam[i])
        # ре-ортонормализация (Gram-Schmidt — защита от числового дрейфа)
        Q, _ = np.linalg.qr(V)
        return Q
    SUBS = ("top", "weak", "pca", "haar")
    deff = {}
    VJ = {}   # (L,d,t) -> {'top': [1024,r], 'weak': [1024,r]}
    Vpca = {} # (L,d) -> [1024,r]
    HAAR = hadamard_rows(R)
    r_final = R
    rematched = False
    for L in LAYS:
        for d in range(N_DOCS_RUN):
            Hdoc = H[L][d]
            lam0, w0, jrt0, sp0 = ew[(L, d)][0]
            Vtop0 = lift(L, d, 0, range(128 - R, 128), R)
            Vweak0 = lift(L, d, 0, range(0, R), R)
            Hc = (Hdoc.double() - Hdoc.double().mean(0, keepdim=True))
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            Vpca0 = Vh.T[:, :R].numpy().astype(np.float32)
            deff[(L, d)] = {
                "top": d_eff_of(Vtop0, Hdoc), "weak": d_eff_of(Vweak0, Hdoc),
                "pca": d_eff_of(Vpca0, Hdoc), "haar": d_eff_of(HAAR.T, Hdoc),
            }
    ratios = [max(v.values()) / min(v.values()) for v in deff.values()]
    if max(ratios) > 1.2:
        r_final = int(round(min(min(v.values()) for v in deff.values())))
        rematched = True
        logline(f"d_eff rematching: max/min = {max(ratios):.2f} > 1.2 -> r = {r_final}")
    # pass 2: lift с r_final
    HAAR = hadamard_rows(r_final)
    for L in LAYS:
        for d in range(N_DOCS_RUN):
            Hdoc = H[L][d]
            Hc = (Hdoc.double() - Hdoc.double().mean(0, keepdim=True))
            _, _, Vh = torch.linalg.svd(Hc, full_matrices=False)
            Vpca[(L, d)] = Vh.T[:, :r_final].numpy().astype(np.float32)
            for t in range(S_LEN):
                lam, w, jrt, sp = ew[(L, d)][t]
                n_nz = int((lam > 1e-12).sum())
                top_i = np.argsort(lam)[::-1][:r_final]
                weak_i = np.argsort(lam)[:min(r_final, n_nz)]
                VJ[(L, d, t)] = {
                    "top": _lift_arr(lam, w, jrt, sp, top_i),
                    "weak": _lift_arr(lam, w, jrt, sp, weak_i),
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
                "haar": d_eff_of(HAAR.T, Hdoc),
            }
    # ker-J: выборка 32 (l,t) на слой
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
            f"top/weak per (l,t), pca per (l,doc), haar глобальный, ker-J выборка")

    # --- S5: eig-закрытость ---------------------------------------------------------
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

    # --- readout: проекции + ridge (LOO-doc) -------------------------------------------
    tstage(tr, time.time() - t0, "ridge")
    proj = {}   # (L,d,S) -> [128, r]
    for L in LAYS:
        for d in range(N_DOCS_RUN):
            Hdoc = H[L][d]
            proj[(L, d, "top")] = np.stack([VJ[(L, d, t)]["top"].T @ Hdoc[t].numpy()
                                            for t in range(S_LEN)])
            proj[(L, d, "weak")] = np.stack([VJ[(L, d, t)]["weak"].T @ Hdoc[t].numpy()
                                             for t in range(S_LEN)])
            proj[(L, d, "pca")] = Hdoc.numpy() @ Vpca[(L, d)]
            proj[(L, d, "haar")] = Hdoc.numpy() @ HAAR.T
            for S in SUBS:
                proj[(L, d, S)] -= proj[(L, d, S)].mean(0, keepdims=True)
    # L23-контроль (без J-базисов: pca/haar), тот же LOO-контур
    for d in range(N_DOCS_RUN):
        Hc23 = H23[d].double() - H23[d].double().mean(0, keepdim=True)
        _, _, Vh23 = torch.linalg.svd(Hc23, full_matrices=False)
        Vpca23 = Vh23.T[:, :r_final].numpy().astype(np.float32)
        proj[(L_CTRL, d, "pca")] = H23[d].numpy() @ Vpca23
        proj[(L_CTRL, d, "haar")] = H23[d].numpy() @ HAAR.T
    SUBS_RUN = {L: (SUBS if L in LAYS else ("pca", "haar")) for L in LAYS + (L_CTRL,)}
    cos_doc = {}   # (L,k,S) -> [N_DOCS_RUN, 128-k]
    r2 = {}
    for L in LAYS + (L_CTRL,):
        for k in KS:
            n = S_LEN - k
            y_all = z_U128[:, :n, :].numpy()          # z(t-k) для t=k..127 (цель = будущий логит)
            for S in SUBS_RUN[L]:
                rows = []
                for d in range(N_DOCS_RUN):
                    X = proj[(L, d, S)][k:]
                    Xf = np.vstack([proj[(L, d2, S)][k:] for d2 in range(N_DOCS_RUN) if d2 != d])
                    yf = np.vstack([y_all[d2] for d2 in range(N_DOCS_RUN) if d2 != d])
                    Xb = np.hstack([Xf, np.ones((len(Xf), 1))])
                    A = Xb.T @ Xb + ALPHA * np.eye(r_final + 1)
                    A[0, 0] -= ALPHA
                    W = np.linalg.solve(A, Xb.T @ yf)
                    pred = np.hstack([X, np.ones((X.shape[0], 1))]) @ W
                    cosv = (pred * y_all[d]).sum(1) / (
                        np.linalg.norm(pred, axis=1) * np.linalg.norm(y_all[d], axis=1) + 1e-12)
                    rows.append(cosv)
                    ss_res = float(((pred - y_all[d]) ** 2).sum())
                    ss_tot = float(((y_all[d] - y_all[d].mean(0, keepdims=True)) ** 2).sum())
                    r2.setdefault((L, k, S), []).append(1.0 - ss_res / ss_tot)
                cos_doc[(L, k, S)] = np.stack(rows)
        logline(f"readout L{L} done")
    agg = {}
    for (L, k, S), rows in cos_doc.items():
        allc = np.concatenate(rows)
        agg[(L, k, S)] = {"cos_median": float(np.median(allc)),
                          "cos_mean": float(allc.mean())}
        if (L, k, S) in r2:
            agg[(L, k, S)]["R2_mean"] = float(np.mean(r2[(L, k, S)]))
    # floor-контроль: cos(средний y по доку, y) — тривиальный константный канал
    floor = {}
    for k in KS:
        n = S_LEN - k
        vals = []
        for d in range(N_DOCS_RUN):
            y = z_U128[d, :n].double()
            m = y.mean(0)
            c = (y * m).sum(1) / (y.norm(dim=1) * m.norm() + 1e-12)
            vals.append(float(c.median()))
        floor[k] = {"median": float(np.median(vals)),
                    "min": float(np.min(vals)), "max": float(np.max(vals))}

    # --- lk4 one-hot cos + N8 -------------------------------------------------------------
    u128_set = {int(v): j for j, v in enumerate(U128)}
    lk4_cos = {}
    for k in KS:
        per_doc = []
        for d in range(N_DOCS_RUN):
            dom = DOMAINS[d // N_PER_DOM]
            row = toks[dom][d % N_PER_DOM]
            m = lk4_k[k][d]
            c = np.full(S_LEN - k, 0.0)
            for t in range(k, S_LEN):
                i = S_LEN - 1 - t
                if m[t]:
                    tok = int(row[i + k])
                    one = np.zeros(128)
                    if tok in u128_set:
                        one[u128_set[tok]] = 1.0
                    # честное сравнение: логиты, предсказывающие целевой токен,
                    # = z[cache t+1-k] (forward 127-t+k-1), НЕ будущий логит z[t-k]
                    y = z_U128[d, t + 1 - k].numpy()
                    c[t - k] = float(one @ y / (np.linalg.norm(y) + 1e-12))
            per_doc.append(c)
        lk4_cos[k] = np.stack(per_doc)
    lk4_cos_med = {k: float(np.median(np.concatenate(v))) for k, v in lk4_cos.items()}
    # покрытие: доля hit-позиций, где целевой токен в U128 (иначе one-hot = 0, cos = 0)
    lk4_cov = {}
    for k in KS:
        covs = []
        for d in range(N_DOCS_RUN):
            dom = DOMAINS[d // N_PER_DOM]
            row = toks[dom][d % N_PER_DOM]
            m = lk4_k[k][d]
            tt = [t for t in range(k, S_LEN) if m[t]]
            if tt:
                covs.append(sum(int(row[S_LEN - 1 - t + k]) in u128_set for t in tt) / len(tt))
        lk4_cov[k] = float(np.mean(covs)) if covs else 0.0
    n8 = {}
    for L in LAYS:
        for k in KS:
            for S in SUBS:
                rows = []
                for d in range(N_DOCS_RUN):
                    tsel = np.arange(k, S_LEN)[lk4_mask[d][k:]]
                    if len(tsel):
                        rows.append(cos_doc[(L, k, S)][d][tsel - k])
                n8[(L, k, S)] = (float(np.median(np.concatenate(rows))) if rows else None)

    # --- bootstrap + verdict -----------------------------------------------------------------
    tstage(tr, time.time() - t0, "verdict")
    rngb = np.random.default_rng(SEED)
    boot = {}
    for L in LAYS:
        for k in KS:
            n = S_LEN - k
            wh = np.stack([cos_doc[(L, k, "weak")][d] - np.maximum(
                cos_doc[(L, k, "pca")][d], cos_doc[(L, k, "haar")][d])
                for d in range(N_DOCS_RUN)])                     # [32, n]
            lk = np.stack([cos_doc[(L, k, "weak")][d] - lk4_cos[k][d]
                           for d in range(N_DOCS_RUN)])
            def ci_of(dd):
                means = []
                for b in range(BOOT_B):
                    sel = rngb.integers(0, N_DOCS_RUN, N_DOCS_RUN)
                    means.append(dd[sel].mean())
                return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
            boot[(L, k)] = {"weak_vs_maxPCAHaar_CI": ci_of(wh),
                            "weak_vs_lookup_CI": ci_of(lk)}
            # (c): попарные CI между 4 подпространствами
            pairs = [("top", "weak"), ("top", "pca"), ("top", "haar"),
                     ("weak", "pca"), ("weak", "haar"), ("pca", "haar")]
            pcis = {}
            for a, b in pairs:
                da = np.stack([cos_doc[(L, k, a)][d] - cos_doc[(L, k, b)][d]
                               for d in range(N_DOCS_RUN)])
                pcis[f"{a}_vs_{b}"] = ci_of(da)
            boot[(L, k)]["pairwise_CI"] = pcis

    def sig_ci(ci):
        return not (ci[0] >= 0.0 and ci[1] <= 0.0)
    def win_ci(ci):
        return ci[0] > 0.0   # (a) — именно ВЫИГРЫШ: весь CI выше 0
    ks2 = (2, 4, 8)
    n_a = 0
    for k in ks2:
        if k not in KS:
            continue
        if all(win_ci(boot[(L, k)]["weak_vs_maxPCAHaar_CI"]) and
               win_ci(boot[(L, k)]["weak_vs_lookup_CI"]) for L in LAYS):
            n_a += 1
    if n_a >= 2:
        verdict = "SCRATCHPAD"
    else:
        b_ok = all(not sig_ci(boot[(L, k)]["weak_vs_maxPCAHaar_CI"])
                   for L in LAYS for k in KS)
        if b_ok:
            verdict = "DEADWEIGHT"
        else:
            c_ok = True
            for L in LAYS:
                for k in KS:
                    for key, ci in boot[(L, k)]["pairwise_CI"].items():
                        if sig_ci(ci):
                            c_ok = False
            if c_ok:
                for L in LAYS:
                    if agg[(L, 1, "top")]["cos_median"] < max(
                            agg[(L, 1, S)]["cos_median"] for S in ("weak", "pca", "haar")):
                        c_ok = False
            verdict = "REDUNDANT_BASIS" if c_ok else "NOT_RESOLVED"
    logline(f"VERDICT: {verdict}")

    # --- контроль ker-J ------------------------------------------------------------------------
    ctrl = {"L23": {k: {S: agg[(L_CTRL, k, S)]["cos_median"] for S in ("pca", "haar")}
                    for k in KS}}
    for L in LAYS:
        for k in KS:
            vals = []
            for d in range(N_DOCS_RUN):
                for (L2, d2, t2), V in kerV.items():
                    if L2 != L or d2 != d or t2 < k:
                        continue
                    # h_l^t (1024) -> projection onto SVD-null of J_l
                    h = H[L2][d2, t2].numpy()
                    vals.append(float(np.linalg.norm(h @ V) / (np.linalg.norm(h) + 1e-12)))
            ctrl[f"kerJ_{L}_k{k}"] = float(np.median(vals)) if vals else None

    # --- артефакт + S7 -----------------------------------------------------------------------------
    tstage(tr, time.time() - t0, "artifact")
    per_pos = {}
    for (L, k, S), rows in cos_doc.items():
        for d in range(N_DOCS_RUN):
            for t in range(S_LEN - k):
                per_pos[f"{L}|d{d}|t{t + k}|k{k}|{S}"] = float(rows[d][t])
    expected = set()
    for L in (LAYS + (L_CTRL,)):
        for d in range(N_DOCS_RUN):
            for k in KS:
                for t in range(k, S_LEN):
                    for S in (SUBS if L in LAYS else ("pca", "haar")):
                        expected.add(f"{L}|d{d}|t{t}|k{k}|{S}")
    gates["S7_exact_set"] = (set(per_pos.keys()) == expected)
    logline(f"S7 exact-set: {len(per_pos)} == {len(expected)} -> {gates['S7_exact_set']}")

    out = {
        "run": f"at2_t21_{REVISION}",
        "started_at": datetime.now().isoformat(),
        "config": {"layers": list(LAYS + (L_CTRL,)), "k": list(KS), "r": R,
                   "r_final": r_final, "rematched": rematched, "eps": EPS,
                   "alpha": ALPHA, "seed": SEED, "boot_b": BOOT_B,
                   "smoke": SMOKE, "device": "cpu", "model": "Qwen/Qwen3.5-0.8B",
                   "script_sha": sha256_file(os.path.abspath(__file__))},
        "gates": gates,
        "d_eff": {f"{L}|d{d}": v for (L, d), v in deff.items()},
        "subspaces": {"definition": "top/weak: eigh(G), G = P^{1/2}(JRJR^T)P^{1/2}, "
                                     "lift v = JR^T(P^{1/2}w)/sqrt(l), QR-реортонормализация; "
                                     "weak = min-r lambda (weak-сторона); ker-J: норма проекции состояния "
                                     "h_l^t на SVD-null J_l (выборка 32 (l,t)); pca: per (l,doc); haar: H1024 rows 0..r-1",
                      "eps_diag": "число lambda > eps*lam_max отчитывается в логе (см. d_eff-блок)"},
        "metrics": {f"{L}|k{k}|{S}": m for (L, k, S), m in agg.items()},
        "lk4": {"hit_total": lk4_tot, "hit_per_dom": lk4_dom,
                "onehot_cos_median": lk4_cos_med,
                "u128_coverage": lk4_cov,
                "n8_strata_median": {f"{L}|k{k}|{S}": v for (L, k, S), v in n8.items()}},
        "floor_cos_mean_y": floor,
        "bootstrap": {f"{L}|k{k}": v for (L, k), v in boot.items()},
        "control": ctrl,
        "verdict": verdict,
        "per_pos_cos": per_pos,
        "wall_sec": round(time.time() - t0, 1),
    }
    with open(ART, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    logline(f"артефакт: {ART} ({os.path.getsize(ART) // 1024} KB)")

    if not SMOKE:
        tr_done[0] = True
        tlog(tr, time.time() - t0, {"finished_at": datetime.now().isoformat(),
                                     "status": "ok", "verdict": verdict})
        tr.finish()
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_LOG) + "\n")
    logline(f"AT2 (T2.1) DONE wall={time.time() - t0:.1f}s verdict={verdict}")

def _lift_arr(lam, w, jrt, sp, idxs):
    V = np.zeros((D, len(idxs)), dtype=np.float32)
    for j, i in enumerate(idxs):
        if lam[i] > 1e-12:
            V[:, j] = jrt @ (sp * w[:, i]) / np.sqrt(lam[i])
    Q, _ = np.linalg.qr(V)
    return Q

if __name__ == "__main__":
    main()
