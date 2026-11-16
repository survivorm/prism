#!/usr/bin/env python3
"""AT0_2 — T0.2 «Форма профиля при фиксированной сумме» (ТЗ v1.1 §1.5 T0.2, disk-only).

Σ инвариантна к перестановкам ТОЧНО (G7) ⇒ тест «Σ против полосы» ВЫРОЖДЕН (ТЗ).
Estimand — ФОРМА профиля при фиксированной сумме: доли площади по полосам
{1},{2},{3..8},{9..32},{33..96}, нормированные на (r−1)/2 = sum_rho_full (тождество G1,
полное суммирование k=1..N−1), + расстояние профилей real vs null.

Объекты/данные — те же, что AT0_1: h_raw (Δh), du_unit (Δu), v_tangent (v̂, frame u_t)
× 96 доков (64 gen + 16 P1_own + 16 P2_corpus) × L{1,5,11,17,23}. L1-TPU degraded —
absent: ровно own_w0, own_w4 (как AT0_1 excluded_L1; остальные TPU-доки L1 в норме).
Клеток: 1440 всего (1434 обычных + 6 absent = 2 дока × 3 объекта); G2 — set-равенство
всех 1440 ключей + число absent = 6.

Пре-рег (зафиксировано ДО прогона; ТЗ §0.4/§1.5-T0.2):
- Полосы — буква ТЗ: {1},{2},{3..8},{9..32},{33..96}. Хвост {97..N−1} НЕ входит в
  вектор профиля (объявленная K=96-обрезка, урок AT0_1); отчитан отдельным скаляром
  tail_share на каждом realization (real и null).
- Нормировка: (r−1)/2 = sum_rho_full — ПОЛНОЕ суммирование всех лагов (G1). Для
  permutation-null'ов (iid/blk/fourier) сумма инвариантна ТОЧНО (G7) ⇒ тот же
  знаменатель, что у real (гейт: |r_null − r_real| ≤ 1e-9·max(1,r_real)).
  Для AR-null знаменатель — собственная сумма суррогата (ТЗ: «Σ отчитывать отдельно»);
  r_AR, sum_rho_AR — в артефакте, с real НЕ СРАВНИВАЮТСЯ.
- Расстояние: tested_D_center_l1 = L1 по 5-компонентному вектору долей (основное,
  тестируется paired-вердиктом), tested_D_center_l2 — отчёт; descriptive_D_bands_L1 —
  reportable only (нет null-распределения, M2).
- Сиды (раунд 6, E3): B = 199 на каждый (iid/blk/AR) — p_floor = 1/(B+1) = 1/200 = 0.005
  (B=20 → 1/21 — saturated max-counter; стандарт T0.1 NNULL=199, p_perm = 0.005 per-test).
  Сиды 0..19 репродуктивны из round 1 (детерминистичные; гейт репродукции — сверка со
  store 5b05537017013a4f). Отдельный сид на (док, s): seed = 2026 + doc_idx·1009 + s.
  Fourier: B = 3 (positive control, не null). p_doc = (#{s: D'_s ≥ D_center}+1)/(n_valid+1)
  + явный exceedance_count (raw, E3).
- Null'ы (ТЗ):
  * iid-перестановка строк — null полной независимости (маргинал; Σ — автоматически).
  * blk-shuffle БЕЗ seam-exclusion, b ∈ {2,4,8,16,32} — DAMPED/SEAMED null (раунд 6, E2):
    within-block пары лага k сохраняются с весом w(k,b) = max(0, 1−k/b), k≥b уничтожены,
    швы добавлены — НЕ чистый null «локальности ≤ b» (локальность ≤ b у него затухает).
    Вердикт читается как «real отличается от unmasked block-shuffle null»; locality-
    прибор — P3-ladder (seam-excluded, AT0_1). Алгебра: within-block оценщик на blk-null
    ≡ within-block real (блоки нетронуты — identity-check в каждой клетке, fail-loud),
    поэтому seam-excluded within-block НЕ МОЖЕТ быть null'ом (D ≡ 0).
  * AR(p), p ∈ {1,2,4} — null «линейный процесс»: per-dim Yule-Walker на demeaned
    приращениях (согласовано по ρ̄(1) построением), симуляция, burn-in 256.
  * Fourier phase-randomization (per dim, zero-padded n_fft ≥ 2N−1, DC-бин фиксирован) —
    POSITIVE CONTROL: сохраняет спектр ⇒ обязан воспроизвести круговой ACF (спектральный
    объект) до 1e-6 (k=1..min(96,N−1)) и r по полному ряду точно (DC+Parseval).
    Линейный (обрезанный до N) ACF первых N сэмплов отличается на wrap-around хвостовой
    член — свойство ANY spectrum-preserving суррогата; отчитывается отдельно, не null.
    Не null; в вердикты не входит.
- Seam-excluded blk: ТОЛЬКО отдельный оценщик — within-block ACF (k < b), seed-средний
  профиль по доступным k, нормировка на собственную сумму; Σ с real НЕ СРАВНИВАТЬ (ТЗ).
  Identity-check (E2): c(seam_excl(blk-null)) ≡ c(within-block real) — max_dev ≤ 1e-12
  (fail-loud G-check).
- Вердикт (obj, null), раунд 6 (E4/M4/M5): гипотеза H = «форма профиля объясняется null».
  PAIRED (дизайн парный: real и null на одних доках): delta_d = D_real(doc) − D_null(doc)
  (per-doc mean по L, кластер = док); delta_mean, CI_delta — cluster-бутстрап по докам
  (B=1000, seed=117). DELTA_MIN = 0.05 — pre-registered floor (M4; порядок эффекта T0.1
  по полосам — dip ≈ 0.02–0.06; ниже — неразрешимо; half-CI отчитан рядом, не заменяет).
  |delta_mean| < DELTA_MIN → not_testable; delta_mean ≥ DELTA_MIN и CI_delta > 0 →
  refuted (real дальше от центра null, чем типичный null-сид); delta_mean ≤ −DELTA_MIN
  и CI_delta < 0 → confirmed_anomaly («более типичен, чем типично» — флаг, НЕ сигнал
  структуры; семантика: paired-разность, не separate-CI, M5); иначе → inconclusive.
  Separate-CI сравнение (D_real vs CI_null) — отчитано как вторичное (D_real_mean/_ci,
  D_null_mean/_ci), в вердикт не входит.
- Σ (E1): estimand T0.2 = ФОРМА при фиксированной сумме (ТЗ v1.2 G7: «Σ против полосы»
  ВЫРОЖДЕН — Σx и c0 инвариантны к перестановкам; численный чек: sum_invariance_max_dev
  ≤ 1e-9). AR-null — Σ своё (mismatch с real отчитан, с real НЕ СРАВНИВАТЬ — ТЗ).
- Пулы (раунд 6, E6/E7): PRIMARY = verified ∩ non-degen (37 доков); sensitivity:
  verified-only (80) и all_96 (96, включая corpus). corpus — provenance_limited (G16:
  нет токенов). degen-доки (loopy: периодическая лаговая структура — AT0_3) — вне
  primary; агрегат с loopy = «full-pool shape», не «clean text shape».
- Объекты (раунд 6, E8): h_raw/v_tangent gen — provenance_limited (hn не верифицирован —
  scale-пин только); du_unit — из U только (hn-независим) — primary на gen; state-объекты
  (h_raw, du_unit, v_tangent) — верифицированы конструкцией (hn = ‖raw h‖, load_source
  пересчитывает).
- Primary family (M3): pool=primary × objects {du_unit + state h_raw/v_tangent} × nulls
  {iid, blk32, ar4} = 9 тестов; FWER-коррекция не применяется — объявлено; остальное —
  sensitivity.
- G-гейты: G2 (1440 ключей, set-равенство, включая 6 absent; число absent = 6); G7 (hash-store + claim + manifest + run.log);
  G8 (gram_resid < 1e-10 на real-клетках); positive-control gate (1e-6/2e-4, ДО
  интерпретации null'ов); G7-invariance (permutation-null'ы: r — по ‖Σy‖²/c0);
  AR σ² ≤ 0 → clip + счётчик (σ² < −1e-8·r0 → fail-loud); |sum_rho_full| < 1e-6 →
  клетка absent-degenerate (не тише);
  field.py: sha + _selfcheck(); G0: sha 3 npz + gen_sha 64/64 + декод-проба 3 строк.
- AR diagnostics (раунд 6, E5): ar_max_root / n_nonstat (|root| > 1) / burn_mix_ratio
  (256·(1−max_root)) / sigma2_clip_frac / rho1..4 real vs null / sum-rho mismatch.
  Wording: AR1/AR2 refuted ≈ tautology при k3,4-excess real (по построению); AR4 —
  информативен только при ok diagnostics.
Артефакты: results/AT0/store/<sha16>.json (full), results/AT0/AT0_2_SHAPE.json (claim),
results/AT0/AT0_2_manifest.json, results/AT0/AT0_2_run.log.
Цена (бенч 14.11): ~35 с/клетку при B=199 (OMP=4) → ~11–14 ч CPU на 1434 клетки.
"""
import hashlib
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "4")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util  # noqa: E402

# G1′: тот же прибор, что AT0_1 — field.py напрямую из файла (без prism/__init__).
_fspec = importlib.util.spec_from_file_location(
    "prism_field_pinned", os.path.join(ROOT, "prism/field.py"))
pf = importlib.util.module_from_spec(_fspec)
_fspec.loader.exec_module(pf)
sys.modules["prism_field_pinned"] = pf
vpath, dcurve, degen, jclean, sha256_file = (
    pf.vpath, pf.dcurve, pf.degen, pf.jclean, pf.sha256_file)

import AT0_1_accounting as a1  # noqa: E402  (load_source, decode_probe, LAYERS, пути, L1_DEGRADE_TOL)

LAYERS = a1.LAYERS
GEN, OWN, COR = a1.GEN, a1.OWN, a1.COR
OUT_DIR = a1.OUT_DIR
STORE_DIR = a1.STORE_DIR
OUT_JSON = os.path.join(OUT_DIR, "AT0_2_SHAPE.json")
MAN_JSON = os.path.join(OUT_DIR, "AT0_2_manifest.json")
RUN_LOG = os.path.join(OUT_DIR, "AT0_2_run.log")

OBJECTS = ("h_raw", "du_unit", "v_tangent")
BANDS = ((1, 1), (2, 2), (3, 8), (9, 32), (33, 96))   # буква ТЗ; хвост 97..N−1 — скаляр
B_NNULL = 199            # раунд 6 E3: p_floor = 1/(B+1) = 0.005 (B=20 → 1/21 — saturated)
DELTA_MIN = 0.05         # раунд 6 M4: pre-registered floor paired-вердикта (T0.1-порядок
                         # эффекта по полосам 0.02–0.06); not_testable если |Δmean| < floor
B_FOURIER = 3           # positive control
SEED_BASE = 2026
SEED_STRIDE = 1009      # отдельный сид на (док, s)
B_BOOT = 1000
SEED_BOOT = 117
BLK_SIZES = (2, 4, 8, 16, 32)
AR_ORDERS = (1, 2, 4)
AR_BURN = 256
ALPHA = 0.05
GRAM_TOL = 1e-10
PC_RHO_TOL = 1e-6       # positive control: max|Δρ̂|
PC_R_TOL = 2e-4         # positive control: |Δr| (= ~2·96·1e-6)
INV_R_TOL = 1e-9        # G7-invariance r на permutation-null'ах (fp: reorder/FFT round-trip)
DEGEN_SUMRHO = 1e-6     # |sum_rho_full| < tol → absent-degenerate

LOG = []


def logline(s=""):
    LOG.append(str(s))
    print(s, flush=True)


def next_pow2(n):
    m = 1
    while m < n:
        m *= 2
    return m


def acf_full(x):
    """x (N,D) → c0, c (N−1,) — ЛИНЕЙНЫЙ ACF через zero-padded FFT (n_fft ≥ 2N−1 ⇒
    wrap-around для k ≤ N−1 отсутствует, точность fp64-FFTo ~1e-13)."""
    N = x.shape[0]
    n_fft = next_pow2(2 * N - 1)
    X = np.fft.rfft(x, n=n_fft, axis=0)          # (n_fft//2+1, D): FFT по ВРЕМЕНИ (axis=0)
    P = X.real * X.real + X.imag * X.imag
    ac = np.fft.irfft(P, n=n_fft, axis=0)        # (n_fft, D); ac[k, :] — circ ACF zero-padded
    c = ac[1:N, :].sum(axis=1)                   # (N−1,) — ровно линейный (k < N)
    c0 = float(np.einsum("ij,ij->i", x, x).sum())
    return c0, c, n_fft


def band_sums_direct(Y, bands):
    """Y (B, N, D) → (bandsums (B, len(bands)), c0 (B,)) — Σ_{k∈b} c(k) через cumsum.
    band (lo,hi): c(k) = Σ_{t<N−k} y_t·y_{t+k}; Σ_{k=lo}^{hi} c(k) = Σ_t y_t·(C[t+hi+1]−C[t+lo])
    (полная часть t ≤ N−1−hi) + частичная хвостовая часть (k ≤ N−1−t). ~30× быстрее
    полагам-эйнсума. Эквивалентность ground truth (полагам) — пин в main()."""
    B, N, D = Y.shape
    C = np.empty((B, N + 1, D))
    C[:, 0, :] = 0.0
    np.cumsum(Y, axis=1, out=C[:, 1:, :])
    Cend = C[:, N, :]
    out = np.zeros((B, len(bands)))
    for j, (lo, hi) in enumerate(bands):
        hi = min(hi, N - 1)
        if lo > N - 1:
            continue
        # инвариант: в BANDS lo ≤ hi всегда, и после continue выше lo ≤ N−1 ⇒ lo ≤ min(hi, N−1);
        # guard был бы мёртвым кодом (ревью 4, п. C) — защита от будущих изменений BANDS (ревью 5, п. E)
        assert lo <= hi, f"band ({lo},{hi}) нарушает lo≤hi (BANDS изменён — перепроверить инвариант)"
        t1 = N - hi
        S1 = C[:, hi + 1:N + 1, :] - C[:, lo:N - hi + lo, :]      # (B, N−hi, D)
        out[:, j] += np.einsum("btd,btd->b", Y[:, :t1, :], S1)
        if t1 <= N - 1 - lo:
            S2 = Cend[:, None, :] - C[:, N - hi + lo:N, :]        # (B, hi−lo+1, D)
            out[:, j] += np.einsum("btd,btd->b", Y[:, t1:N - lo, :], S2)
    c0 = (Y * Y).sum(axis=(1, 2))
    return out, c0


def acf_full_many(X):
    """X (B, N, D) → (c0 (B,), c (B, N−1,)) — batched zero-padded FFT ACF (axis=1 — время)."""
    B, N, D = X.shape
    n_fft = next_pow2(2 * N - 1)
    Xf = np.fft.rfft(X, n=n_fft, axis=1)
    P = Xf.real * Xf.real + Xf.imag * Xf.imag
    ac = np.fft.irfft(P, n=n_fft, axis=1)      # (B, n_fft, D)
    c = ac[:, 1:N, :].sum(axis=2)              # (B, N−1) — линейный (k < N); сумма по D
    c0 = (X * X).sum(axis=(1, 2))
    return c0, c, n_fft


def band_shares_vec(bandsums, c0, denom):
    """bandsums (M, 5), c0 (M,), denom (M,) → (shares (M, 5), tail (M,)).
    denom = (r−1)/2 (полная сумма всех лагов, G1) ⇒ tail = 1 − Σ долей точно"""
    out = bandsums / c0[:, None] / denom[:, None]
    tail = 1.0 - out.sum(axis=1)
    return out, tail


def _d_bands_l1(bandsums, c0n, ok, rho96):
    """L1: real band-структура (Σρ̂ по полосам) против mean null band-структуры.
    Без нормировки на полную сумму — прямое сравнение форм.
    REPORTABLE ONLY: null-распределения нет — p-value не вычисляется (ревью 4, п. H)."""
    K = len(rho96)
    real_b = np.array([sum(rho96[lo - 1:min(hi, K)]) for (lo, hi) in BANDS])
    null_b = (bandsums[ok] / c0n[ok, None]).mean(axis=0)
    return float(np.abs(null_b - real_b).sum())


def cell_real(x):
    """Real-клетка: c0, c, r (два способа), sum_rho, gram_resid, prof5, tail, rho96."""
    c0, c, _ = acf_full(x)
    N = x.shape[0]
    if c0 <= 0:
        raise ValueError("c(0) = 0: вырожденный объект")
    sum_rho = float(c.sum() / c0)
    r_sum = float(np.linalg.norm(x.sum(axis=0)) ** 2 / c0)
    r_id = 1.0 + 2.0 * sum_rho
    assert abs(r_sum - r_id) <= 1e-12 * max(1.0, r_sum), \
        f"G1: r(‖Σx‖²) {r_sum:.12f} != 1+2Σρ̂ {r_id:.12f} (N={N})"
    gram = abs(1.0 + 2.0 * c.sum() / c0 - r_sum) / max(1.0, r_sum)
    if abs(sum_rho) < DEGEN_SUMRHO:
        return {"N": int(N), "c0": c0, "r": r_sum, "sum_rho": sum_rho,
                "gram_resid": float(gram), "degenerate": True}
    bandsums = np.array([float(c[lo - 1:hi].sum()) for (lo, hi) in BANDS])
    prof, tail = band_shares_vec(bandsums[None, :], np.array([c0]), np.array([sum_rho]))
    K96 = min(96, N - 1)
    return {"N": int(N), "c0": c0, "r": r_sum, "sum_rho": sum_rho,
            "gram_resid": float(gram), "degenerate": False,
            "prof5": [float(v) for v in prof[0]], "tail": float(tail[0]),
            "rho96": [float(v) for v in c[:K96] / c0]}


def dists_to(p_real, P_nulls):
    """P_nulls (B,5) → C, D_center (L1,L2), D'_s (L1), D(real→null) (L1, отчёт)."""
    C = P_nulls.mean(axis=0)
    d_center = float(np.abs(p_real - C).sum())
    d_center2 = float(np.sqrt(((p_real - C) ** 2).sum()))
    d_nulls = np.abs(P_nulls - C).sum(axis=1)
    d_real_to_null = np.abs(P_nulls - p_real).sum(axis=1)
    return C, d_center, d_center2, d_nulls, d_real_to_null


def null_iid(x, rng):
    return x[rng.permutation(x.shape[0])]


def null_blk(x, b, rng):
    N = x.shape[0]
    nb = (N + b - 1) // b
    idx = np.arange(N)
    blocks = [idx[i * b:min((i + 1) * b, N)] for i in range(nb)]
    perm = rng.permutation(nb)
    return x[np.concatenate([blocks[i] for i in perm])]


def null_blk_seam_excl(x, b, rng):
    """Blk-permut + seam-excluded ACF: только within-block пары (k < b).
    ВАЖНО (раунд 6 E2, поймано identity-check'ом): границы блоков в y — по ПЕРМУТИРОВАНЫМ
    блокам (короткий последний блок N%b≠0 стоит на случайном месте — фиксированная сетка
    y[i*b:(i+1)*b] сдвигает все последующие блоки и «within-block» перестрачивает швы).
    Возвращает (y, c_seam, kmax): c_seam (kmax,) по k=1..min(b−1, 96)."""
    N = x.shape[0]
    nb = (N + b - 1) // b
    idx = np.arange(N)
    blocks = [idx[i * b:min((i + 1) * b, N)] for i in range(nb)]
    perm = rng.permutation(nb)
    y = x[np.concatenate([blocks[i] for i in perm])]
    kmax = min(b - 1, 96)
    c = np.zeros(kmax)
    pos = 0
    for i in range(nb):
        m = len(blocks[perm[i]])
        s = y[pos:pos + m]
        pos += m
        for k in range(1, min(kmax, m - 1) + 1):
            c[k - 1] += float(np.einsum("ij,ij->i", s[:-k], s[k:]).sum())
    assert pos == N, f"null_blk_seam_excl: границы блоков не закрывают y ({pos} != {N})"
    return y, c, kmax


def within_block_c(x, b, kmax):
    """within-block ACF real (те же блоки, что blk-shuffle), k=1..kmax.
    Для E2-identity: c(seam_excl(blk-null)) ≡ это (блоки перестановкой не трогаются)."""
    N = x.shape[0]
    c = np.zeros(kmax)
    for i in range((N + b - 1) // b):
        s = x[i * b:min((i + 1) * b, N)]
        m = s.shape[0]
        for k in range(1, min(kmax, m - 1) + 1):
            c[k - 1] += float(np.einsum("ij,ij->i", s[:-k], s[k:]).sum())
    return c


def ar_root_diag(phi):
    """AR-диагностика (раунд 6, E5): max |root| per-dim (companion matrix), число
    нестационарных dim (|root| > 1), burn/mix ratio = AR_BURN·(1−max_root)."""
    p, D = phi.shape
    if p == 1:
        roots = np.abs(phi[0, :])
    else:
        C = np.zeros((D, p, p))
        C[:, 0, :] = phi.T                     # первый ряд: phi_1..phi_p
        C[:, 1:p, :-1] = np.eye(p - 1)[None, :, :]
        roots = np.abs(np.linalg.eigvals(C)).max(axis=1)
    max_root = float(roots.max())
    n_nonstat = int((roots > 1.0 + 1e-9).sum())
    burn_mix = float(AR_BURN * (1.0 - max_root)) if max_root < 1.0 else 0.0
    return {"ar_max_root": max_root, "n_nonstationary": n_nonstat,
            "burn_mix_ratio": burn_mix}


def null_ar_fit(x, p):
    """Per-dim Yule-Walker на demeaned. Возвращает (a (p,D), sig (D,), mean (D,), info)."""
    N, D = x.shape
    m = x.mean(axis=0)
    xm = x - m
    r = np.empty((D, p + 1))
    r[:, 0] = (xm * xm).sum(axis=0) / N
    for j in range(1, p + 1):
        r[:, j] = (xm[:N - j] * xm[j:]).sum(axis=0) / N
    ij = np.abs(np.arange(p)[:, None] - np.arange(p)[None, :])      # (p,p)
    T = r[:, ij]                                                    # (D,p,p)
    rhs = r[:, 1:p + 1]                                             # (D,p)
    phi = np.linalg.solve(T, rhs[:, :, None]).squeeze(-1)   # (D,p)
    sig2 = r[:, 0] - np.einsum("dp,dp->d", phi, rhs)
    neg_fail = int((sig2 < -1e-8 * r[:, 0]).sum())
    if neg_fail:
        raise ValueError(f"AR({p}): σ² существенно < 0 на {neg_fail} dim — баг фита")
    clipped = int(((sig2 <= 0) & (sig2 > -1e-8 * r[:, 0])).sum())
    return phi.T, np.sqrt(np.maximum(sig2, 0.0)), m, {"sigma2_clipped": clipped}

def null_ar_batch(x, p, B, doc_idx):
    """B суррогатов AR(p) (batched-симуляция, один фит на всех сидов).
    Возвращает Y (B, N, D), info (+ AR-диагностика, E5)."""
    a, sig, m, info = null_ar_fit(x, p)
    info.update(ar_root_diag(a))
    info["sigma2_clip_frac"] = float(info["sigma2_clipped"] / x.shape[1])
    N, D = x.shape
    eps = np.stack([np.random.default_rng(SEED_BASE + doc_idx * SEED_STRIDE + s)
                    .standard_normal((AR_BURN + N, D)) for s in range(B)])   # (B, T, D)
    buf = np.zeros((B, p, D))
    out = np.empty((B, AR_BURN + N, D))
    for t in range(AR_BURN + N):
        v = np.einsum("bpd,pd->bd", buf, a) + sig * eps[:, t, :]
        buf = np.roll(buf, 1, axis=1)
        buf[:, 0, :] = v
        out[:, t, :] = v
    return out[:, AR_BURN:, :] + m[None, :], info


def null_fourier(x, rng, n_fft):
    """Phase randomization per dim на zero-padded ряду (n_fft ≥ 2N−1), DC-бин фиксирован.
    Сохраняет ТОЧНО: круговой ACF (весь n_fft) и полную сумму ряда (DC) + c0 (Parseval)
    ⇒ r по полному ряду — точно. ЛИНЕЙНЫЙ (обрезанный до N) ACF первых N сэмплов
    отличается на wrap-around хвостовой член — СВОЙСТВО any spectrum-preserving суррогата
    (не баг); отчитывается отдельно в positive control."""
    N, D = x.shape
    X = np.fft.rfft(x, n=n_fft, axis=0)            # (n_fft//2+1, D)
    M = X.shape[0]
    mag = np.abs(X)
    th = rng.uniform(0.0, 2.0 * np.pi, size=(M - 2, D))
    Xn = X.copy()
    # DC-бин (0) и Nyquist-бин (M−1) — для вещественного сигнала обязаны быть вещественными
    # (irfft молча берёт Re Nyquist-бина — фаза там не рандомизируется)
    Xn[1:M - 1, :] = mag[1:M - 1, :] * np.exp(1j * th)
    z = np.fft.irfft(Xn, n=n_fft, axis=0)          # (n_fft, D) — полный ряд
    return z


def run_cell(x, doc_idx, doc_name, L, obj, smoke=False):
    """Одна клетка (doc, L, obj): real + все null'ы + расстояния + p_doc."""
    real = cell_real(x)
    rec = {"doc": doc_name, "L": int(L), "obj": obj, "real": real}
    if real["degenerate"]:
        rec["status"] = "absent-degenerate"
        return rec
    B = 3 if smoke else B_NNULL
    p_real = np.array(real["prof5"])
    r_real = real["r"]
    nulls = {}
    r_inv_max = 0.0
    # --- permutation-null'ы: iid + blk{2,4,8,16,32} (без seam-exclusion) ---
    perm_nulls = [("iid", None)] + [(f"blk{b}", b) for b in BLK_SIZES]
    for nm, b in perm_nulls:
        def _mk(s, b=b):
            rng = np.random.default_rng(SEED_BASE + doc_idx * SEED_STRIDE + s)
            return null_iid(x, rng) if b is None else null_blk(x, b, rng)
        Y = np.stack([_mk(s) for s in range(B)])                # (B, N, D)
        bandsums, c0n = band_sums_direct(Y, BANDS)
        rnull = (Y.sum(axis=1) ** 2).sum(axis=1) / c0n
        r_inv_max = max(r_inv_max, float(np.max(np.abs(rnull - r_real) / max(1.0, r_real))))
        sgn = (rnull - 1.0) / 2.0            # (r−1)/2 = полная сумма всех лагов (G1)
        ok = np.abs(sgn) >= DEGEN_SUMRHO
        if not ok.any():
            nulls[nm] = {"n_valid": 0, "p_doc": None, "note": "все сиды degenerate"}
            continue
        Pn, _ = band_shares_vec(bandsums[ok], c0n[ok], sgn[ok])
        C, Dctr, Dctr2, Dns, DtoN2 = dists_to(p_real, Pn)
        m = int(ok.sum())
        p_doc = float((np.sum(Dns >= Dctr) + 1) / (m + 1))
        nulls[nm] = {"n_valid": m, "p_doc": p_doc,
                     "tested_D_center_l1": Dctr, "tested_D_center_l2": Dctr2,
                     "Dnull_l1_mean": float(np.mean(Dns)),
                     "Dnull_l1_p05": float(np.percentile(Dns, 5)),
                     "Dnull_l1_p95": float(np.percentile(Dns, 95)),
                     "DtoN_l1_mean": float(np.mean(DtoN2)),
                     "r_null_mean": float(np.mean(rnull[ok])),
                     "exceedance_count": int(round(p_doc * (m + 1) - 1)),   # raw (E3)
                     "prof_null_mean": [float(v) for v in C],
                     "per_seed": {"Dnull": [float(v) for v in Dns],
                                  "DtoN": [float(v) for v in DtoN2],
                                  "r_null": [float(v) for v in rnull[ok]],
                                  "sum_rho": [float((r - 1.0) / 2.0) for r in rnull[ok]]}}
    # --- seam-excluded (отдельный оценщик, Σ с real не сравнивать) ---
    # E2-identity: within-block(blk-null) ≡ within-block(real) (блоки нетронуты) —
    # алгебраически ТОЧНО; fail-loud check в каждой клетке.
    seam = {}
    seam_id = {}
    for b in BLK_SIZES:
        kmax = min(b - 1, 96)
        creal = within_block_c(x, b, kmax)
        profs = []
        dev = None
        for s in range(B):
            rng = np.random.default_rng(SEED_BASE + doc_idx * SEED_STRIDE + s)
            _, cs, kmax = null_blk_seam_excl(x, b, rng)
            if s == 0:
                dev = float(np.abs(cs - creal).max() / max(1.0, float(np.abs(creal).max())))
            tot = float(cs.sum())
            if abs(tot) < DEGEN_SUMRHO:
                continue
            pr = []
            for (lo, hi) in BANDS:
                hi2 = min(hi, kmax)
                if lo > kmax:
                    pr.append(0.0)
                else:
                    pr.append(float(cs[lo - 1:hi2].sum() / tot))
            profs.append(pr)
        assert dev <= 1e-12, (f"seam-identity ({doc_name} L{L} blk{b}): "
                              f"within-block(blk-null) != within-block(real), max_dev {dev:.2e} "
                              "(алгебраически обязаны совпадать — блоки нетронуты)")
        seam_id[f"blk{b}"] = dev
        if profs:
            seam[f"blk{b}"] = {"n_valid": len(profs),
                               "prof_mean": [float(v) for v in np.mean(profs, axis=0)],
                               "kmax": int(min(b - 1, 96)),
                               "note": "seam-excluded: within-block пары; Σ — собственный, с real НЕ сравнивать (ТЗ)"}
        else:
            seam[f"blk{b}"] = {"n_valid": 0}
    nulls["seam_excluded"] = seam
    nulls["seam_excluded_identity_max_dev"] = seam_id
    # --- AR (null «линейный процесс»; Σ своё — отчитывается отдельно) ---
    for p in AR_ORDERS:
        Y, info = null_ar_batch(x, p, B, doc_idx)        # (B, N, D), один фит на все сиды
        bandsums, c0n = band_sums_direct(Y, BANDS)
        rnull = (Y.sum(axis=1) ** 2).sum(axis=1) / c0n
        sgn = (rnull - 1.0) / 2.0            # (r−1)/2 (G1) — СВОЯ сумма суррогата (ТЗ)
        ok = (c0n > 0) & (np.abs(sgn) >= DEGEN_SUMRHO)
        if not ok.any():
            nulls[f"ar{p}"] = {"n_valid": 0, "p_doc": None,
                               "note": "все сиды degenerate", "sigma2_clipped": info["sigma2_clipped"]}
            continue
        Pn, _ = band_shares_vec(bandsums[ok], c0n[ok], sgn[ok])
        C, Dctr, Dctr2, Dns, _ = dists_to(p_real, Pn)
        m = int(ok.sum())
        p_doc = float((np.sum(Dns >= Dctr) + 1) / (m + 1))
        rho1 = bandsums[:, 0] / c0n
        # E5: rho1..4 real vs null (per-seed mean по сидам)
        Nn = Y.shape[1]
        rho4n = [float(((Y[:, :Nn - k, :] * Y[:, k:, :]).sum(axis=(1, 2)) / c0n).mean())
                 for k in range(1, 5)]
        nulls[f"ar{p}"] = {"n_valid": m, "p_doc": p_doc,
                           "tested_D_center_l1": Dctr, "tested_D_center_l2": Dctr2,
                           "Dnull_l1_mean": float(np.mean(Dns)),
                           "exceedance_count": int(round(p_doc * (m + 1) - 1)),   # raw (E3)
                           "r_null_mean": float(np.mean(rnull[ok])),
                           "sum_rho_null_mean": float(np.mean(sgn[ok])),
                           "rho1_null_mean": float(np.mean(rho1[ok])),
                           "rho1_real": float(real["rho96"][0]) if real["rho96"] else None,
                           "rho_real_1to4": [float(v) for v in real["rho96"][:4]],
                           "rho_null_1to4_mean": rho4n,
                           "sigma2_clipped": info["sigma2_clipped"],
                           "sigma2_clip_frac": info["sigma2_clip_frac"],
                           "ar_max_root": info["ar_max_root"],
                           "n_nonstationary": info["n_nonstationary"],
                           "burn_mix_ratio": info["burn_mix_ratio"],
                           "r_real": float(real["r"]),
                           "sum_mismatch": abs(float(real["r"]) - float(np.mean(rnull[ok]))),
                           "prof_null_mean": [float(v) for v in C],
                           "sum_rho_real": float(real["sum_rho"]),
                           "descriptive_D_bands_L1": _d_bands_l1(bandsums, c0n, ok, real["rho96"]),
                           "per_seed": {"Dnull": [float(v) for v in Dns],
                                        "r_null": [float(v) for v in rnull[ok]],
                                        "sum_rho": [float(v) for v in sgn[ok]],
                                        "rho1": [float(v) for v in rho1[ok]]},
                           "note": ("AR: Σ своё (ТЗ: «Σ отчитывать отдельно») — с real НЕ СРАВНИВАТЬ "
                                    "(sum_mismatch/r_real/r_null_mean — отчёт). Wording (ревью r6, E5): "
                                    "AR1/AR2 refuted ≈ тавтология при k3,4-excess real (по построению); "
                                    "AR4 информативен только при ok diagnostics (ar_max_root < 1, "
                                    "sigma2_clip_frac низкие, burn_mix_ratio достаточный). "
                                    "Интерпретация 'про band-форму' валидна при sign_check_ok "
                                    "(per-cell seed sign-mismatch ≤ 0.1). "
                                    "descriptive_D_bands_L1 — reportable only")}
    # --- fourier (positive control, не null) ---
    # Что сохранено ТОЧНО конструкцией: круговой ACF (весь n_fft) + полная сумма (DC) + c0
    # (Parseval) ⇒ r по полному ряду. Линейный (обрезанный до N) ACF первых N сэмплов —
    # с wrap-around хвостовым членом: отчитываем отдельно (свойство, не баг).
    pc_rows = []
    n_fft = next_pow2(2 * x.shape[0] - 1)
    rho96_arr = np.array(real["rho96"])
    K96 = len(rho96_arr)
    for s in range(B_FOURIER):
        rng = np.random.default_rng(SEED_BASE + doc_idx * SEED_STRIDE + 1000 + s)
        z = null_fourier(x, rng, n_fft)            # (n_fft, D) полный ряд
        c0s = float(np.einsum("ij,ij->i", z, z).sum())
        r_full_s = float(np.linalg.norm(z.sum(axis=0)) ** 2 / c0s)
        # круговой ACF суррогата = irfft(|Xn|²) (по строкам — по dim)
        Xn = np.fft.rfft(z, n=n_fft, axis=0)
        ac_s = np.fft.irfft(Xn.real * Xn.real + Xn.imag * Xn.imag, n=n_fft, axis=0)
        circ_err = float(np.max(np.abs(ac_s[1:K96 + 1, :].sum(axis=1) / c0s - rho96_arr)))
        # линейный ACF первых N сэмплов (объект нашей статистики) — wrap-отклонение
        y = z[:x.shape[0]]
        c0y, cy, _ = acf_full(y)
        r_lin_s = float(np.linalg.norm(y.sum(axis=0)) ** 2 / c0y)
        lin_err = float(np.max(np.abs(cy[:K96] / c0y - rho96_arr)))
        pc_rows.append({"r_full_null": r_full_s, "r_lin_null": r_lin_s,
                        "circ_max_abs_d_rho96": circ_err, "lin_max_abs_d_rho96": lin_err})
    nulls["fourier_positive_control"] = {
        "n": len(pc_rows),
        "max_abs_dr_full": max(abs(r_["r_full_null"] - r_real) for r_ in pc_rows),
        "max_abs_d_rho96_circular": max(r_["circ_max_abs_d_rho96"] for r_ in pc_rows),
        "max_abs_d_rho96_linear_wrap": max(r_["lin_max_abs_d_rho96"] for r_ in pc_rows),
        "r_lin_null_mean": float(np.mean([r_["r_lin_null"] for r_ in pc_rows])),
        "note": ("positive control: круговой ACF и полная сумма (r по полному ряду) сохранены "
                 "конструкцией ТОЧНО; линейный (обрезанный до N) ACF первых N сэмплов отличается "
                 "на wrap-around хвостовой член — свойство ANY spectrum-preserving суррогата, "
                 "не баг (ТЗ 1e-6 выполнимо для кругового/спектрального объекта)")}
    rec["nulls"] = nulls
    rec["r_inv_max"] = r_inv_max
    return rec


def g0a_hn_gate(docs):
    """G0a_hn_field, честная версия (ревью 4, п. A/F).
    state: load_source пересчитывает u, hn из raw h (npz-поле _hn — баг продюсера —
    НЕ читается; load_source:186-188, проверено чтением кода). Unit-пин здесь
    структурно истинен = регрессионный guard: если конвейер начнёт читать хранимые
    поля, u перестанет быть unit (deviation O(1) state / 1.63× gen) — пин упадёт.
    gen: в npz ТОЛЬКО _u и _hn (raw h отсутствует — recon-проверка U·hn vs raw h
    невозможна). Unit-пин — проверка поля _u (замер 2.4e-4 = fp16). Scale-пин —
    однородность row-скал U·hn (sdev ≈ CV²/2, замер 0.023 ⇒ CV ≈ 0.21): ловит
    сильную (CV ≳ 0.63) неоднородность скал; слабее — недетектируемо от файла,
    как и константный _hn (остаточный риск зафиксирован в claim; замеренная
    разбросность hn 0.93–1.87 согласуется с истинными row-нормами).
    state tol 1e-9: вырожденная строка с ‖h_row‖ < 1e-12 (u = h/1e-12 → ‖u‖≪1)
    уронит пин — desired fail-loud на degenerate h (ревью 5, п. A; в текущих
    данных таких строк нет). Пин не ловит ПЕРМУТИРОВАННЫЙ _hn (mean/rms/CV
    инвариантны к перестановке строк) — входит в residual_risk (ревью 5, п. D)."""
    # ревью 5, п. B: пустой класс docs → 0.0 и молчаливый PASS — ловим явно
    n_gen = sum(1 for d in docs.values() if d["source"] == "gen_layers")
    n_state = len(docs) - n_gen
    if n_gen == 0 or n_state == 0:
        raise ValueError(f"g0a_hn_gate: класс доков пуст (gen={n_gen}, state={n_state}) — "
                         "гейт обязан проверять оба класса (ревью 5, п. B)")
    meas = {"state": {"max_unit_dev": 0.0}, "gen_layers": {"max_unit_dev": 0.0, "max_scale_dev": 0.0}}
    for nm, d in docs.items():
        is_gen = d["source"] == "gen_layers"
        cls = "gen_layers" if is_gen else "state"
        tol = 5e-3 if is_gen else 1e-9
        for L in LAYERS:
            u, hn = d["layers"][L]
            udev = float(np.abs(np.linalg.norm(u, axis=1) - 1.0).max())
            meas[cls]["max_unit_dev"] = max(meas[cls]["max_unit_dev"], udev)
            if udev > tol:
                raise ValueError(f"u не unit ({nm}, L{L}: max dev {udev:.3e} > {tol:.0e}) — "
                                 f"{cls}: хранимые/используемые _u-поля биты (G0a_hn_field), "
                                 "h_raw недостоверен")
            if is_gen:
                h = u * hn[:, None]
                N_, D_ = h.shape
                rms = float(np.sqrt((h * h).sum() / (N_ * D_)))
                sdev = abs(hn.mean() - rms * np.sqrt(D_)) / (rms * np.sqrt(D_))
                meas[cls]["max_scale_dev"] = max(meas[cls]["max_scale_dev"], sdev)
                if sdev > 0.2:
                    raise ValueError(f"gen scale-пин ({nm}, L{L}): max dev {sdev:.3e} > 0.2 — "
                                     "порча row-скал U·hn (G0a_hn_field), h_raw недостоверен")
    return meas


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # раунд 6 M1: clamp [0,1] (float-артефакт hi > 1 при p→1)
    return [float(max(0.0, (c - w) / d)), float(min(1.0, (c + w) / d))]


def boot_ci(vals, seed=SEED_BOOT, b=B_BOOT):
    """Percentile-бутстрап (B=1000, seed=117): mean + [2.5, 97.5] по переданным
    значениям. Вызывать с кластерными значениями (в AT0_2: per-doc mean по L —
    кластер = док; внутри-док корреляция слоёв не учитывается cell-level бутстрапом)."""
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(vals)
    boots = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(b)])
    return float(vals.mean()), [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


NULL_NAMES = ["iid", "blk2", "blk4", "blk8", "blk16", "blk32", "ar1", "ar2", "ar4"]


def aggregate(cells_by_obj, exclude=()):
    """Пер (obj, null): paired-delta verdict (E4: real и null на одних доках — дизайн парный),
    separate-CI как вторичное, Σ-block (E1), exceedance-гистограмма (E3),
    frac (Wilson, clamp M1), домены (n≥12 гейт). exclude — доки вне пула (G16/degen)."""
    out = {}
    for obj in OBJECTS:
        out[obj] = {}
        for nm in NULL_NAMES:
            pdocs, per_doc_c, per_doc_n, docs_sig = [], {}, {}, set()
            sum_reals, sum_nulls, sum_devs = [], [], []
            for key, c in cells_by_obj[obj].items():
                if key[0] in exclude:
                    continue
                if c.get("status") in ("absent", "absent-degenerate") or "nulls" not in c:
                    continue
                nu = c["nulls"].get(nm)
                if not nu or nu.get("p_doc") is None:
                    continue
                doc = key[0]
                per_doc_c.setdefault(doc, []).append(nu["tested_D_center_l1"])
                per_doc_n.setdefault(doc, []).append(nu["Dnull_l1_mean"])
                pdocs.append(nu["p_doc"])
                if nu["p_doc"] < ALPHA:
                    docs_sig.add(doc)
                sum_reals.append(c["real"]["sum_rho"])
                if nm.startswith("ar"):
                    sum_nulls.extend(nu["per_seed"]["sum_rho"])
                else:
                    sum_devs.extend(abs(s - c["real"]["sum_rho"])
                                    for s in nu["per_seed"]["sum_rho"])
            if not pdocs:
                out[obj][nm] = {"n_docs": 0}
                continue
            Dc = [float(np.mean(v)) for v in per_doc_c.values()]    # per-doc mean по L (кластер)
            Dns = [float(np.mean(v)) for v in per_doc_n.values()]
            d_real, ci_real = boot_ci(Dc)
            d_null, ci_null = boot_ci(Dns)
            n_sig = int(np.sum(np.asarray(pdocs) < ALPHA))
            half_ci = (ci_real[1] - ci_real[0]) / 2
            # E4/M4/M5: paired difference (per-doc: D_real(doc) − D_null(doc)),
            # cluster-бутстрап по докам. DELTA_MIN — pre-registered floor (M4).
            delta_d = [float(np.mean(per_doc_c[dd]) - np.mean(per_doc_n[dd]))
                       for dd in sorted(set(per_doc_c) & set(per_doc_n))]
            delta_mean, ci_delta = boot_ci(delta_d)
            if abs(delta_mean) < DELTA_MIN:
                verdict = "not_testable"
                verdict_note = (f"|Δmean| {abs(delta_mean):.2e} < DELTA_MIN {DELTA_MIN} "
                                "(pre-registered floor, M4); half_ci_real "
                                f"{half_ci:.2e} (отчёт)")
            elif delta_mean > 0 and ci_delta[0] > 0:
                verdict = "refuted"
                verdict_note = (f"Δmean {delta_mean:.2e} ≥ DELTA_MIN и CI_Δ "
                                f"[{ci_delta[0]:.2e}, {ci_delta[1]:.2e}] > 0: real профиль дальше "
                                "от центра null, чем типичный null-сид (paired)")
            elif delta_mean < 0 and ci_delta[1] < 0:
                verdict = "confirmed_anomaly"
                verdict_note = (f"Δmean {delta_mean:.2e} ≤ −DELTA_MIN и CI_Δ < 0: real значимо "
                                "ближе к центру null, чем типичный null-сид — флаг «более "
                                "типичен, чем типично», НЕ сигнал структуры (M5)")
            else:
                verdict = "inconclusive"
                verdict_note = (f"Δmean {delta_mean:.2e} (≥ DELTA_MIN) но CI_Δ "
                                f"[{ci_delta[0]:.2e}, {ci_delta[1]:.2e}] пересекает 0")
            # E1: Σ — perm nulls: G7-вырождение (Σ_null ≡ Σ_real, численный чек);
            # AR: Σ своё — mismatch отчёт, с real не сравнивать (ТЗ)
            if nm.startswith("ar"):
                sum_block = {"sum_verdict": "not_compared (ТЗ: AR — Σ своё, с real НЕ сравнивать)",
                             "sum_real_mean": float(np.mean(sum_reals)),
                             "sum_null_mean": float(np.mean(sum_nulls)),
                             "sum_mismatch": abs(float(np.mean(sum_reals)) - float(np.mean(sum_nulls)))}
            else:
                sum_block = {"sum_verdict": ("degenerate (ТЗ v1.2 G7: Σ инвариантна к перестановкам — "
                                              "тест «Σ против полосы» вырожден; estimand = форма при "
                                              "фиксированной сумме); численный чек — sum_invariance_max_dev"),
                             "sum_real_mean": float(np.mean(sum_reals)),
                             "sum_null_equals_real": bool(sum_devs),
                             "sum_invariance_max_dev": float(max(sum_devs)) if sum_devs else None}
            # E3: raw exceedance counts (гистограмма по клеткам)
            hist = {}
            for key, c in cells_by_obj[obj].items():
                if key[0] in exclude or c.get("status") in ("absent", "absent-degenerate") \
                        or "nulls" not in c:
                    continue
                nu = c["nulls"].get(nm)
                if not nu or nu.get("p_doc") is None:
                    continue
                kk = str(int(nu["exceedance_count"]))
                hist[kk] = hist.get(kk, 0) + 1
            # домены (gen-доки; доменное утверждение только при n≥12 не-вырожденных — §0.3)
            dom = {}
            for key, c in cells_by_obj[obj].items():
                if c.get("status") in ("absent", "absent-degenerate") or "nulls" not in c:
                    continue
                domn = c.get("dom")
                if domn not in ("wiki", "math", "code", "chat"):
                    continue
                nu = c["nulls"].get(nm)
                if not nu or nu.get("p_doc") is None:
                    continue
                dom.setdefault(domn, {}).setdefault(key[0], []).append(
                    (c.get("degen", False), nu["tested_D_center_l1"]))
            dom_tab = {}
            for d, ddocs in sorted(dom.items()):
                rows = [v for dlist in ddocs.values() for v in dlist]
                nd = len(rows)
                nd_nondeg = sum(1 for (dg, _) in rows if not dg)
                n_docs_nondeg = sum(1 for dlist in ddocs.values()
                                    if any(not dg for (dg, _) in dlist))
                if n_docs_nondeg >= 12:   # ТЗ §0.3: «n≥12 невырожденных» = ДОКУМЕНТЫ (wiki = 1 док)
                    # cluster-бутстрап: per-doc mean по L (кластер = док, как D_real — N5)
                    doc_means = [float(np.mean([v for (dg, v) in dlist if not dg]))
                                 for dlist in ddocs.values()
                                 if any(not dg for (dg, v) in dlist)]
                    mv, civ = boot_ci(doc_means)
                    dom_tab[d] = {"n": nd, "n_nondeg": nd_nondeg, "n_docs": len(ddocs),
                                  "n_docs_nondeg": n_docs_nondeg,
                                  "D_real_mean": mv, "ci": civ, "claim": "ok"}
                else:
                    dom_tab[d] = {"n": nd, "n_nondeg": nd_nondeg, "n_docs": len(ddocs),
                                  "n_docs_nondeg": n_docs_nondeg,
                                  "claim": (f"n_docs_nondeg={n_docs_nondeg}<12 — без доменного "
                                            "утверждения (ТЗ §0.3: n — документы)")}
            out[obj][nm] = {
                "n_docs": len(Dc), "n_cells": len(pdocs),
                "D_real_mean": d_real, "D_real_ci": ci_real,
                "D_null_mean": d_null, "D_null_ci": ci_null,
                "delta_mean": delta_mean, "delta_ci": ci_delta, "delta_min": DELTA_MIN,
                "half_ci_real": half_ci,
                "verdict": verdict, "verdict_note": verdict_note,
                "frac_cells_p_lt_alpha": n_sig / len(pdocs),
                "frac_cells_p_lt_alpha_wilson": wilson_ci(n_sig, len(pdocs)),
                "exceedance_count_hist": hist,
                "n_docs_unique_p_lt_alpha": len(docs_sig),
                "p_docs_median": float(np.median(pdocs)),
                **sum_block,
                "domains": dom_tab,
            }
            if nm.startswith("ar") and pdocs:
                sel = [(k, cc) for k, cc in cells_by_obj[obj].items()
                       if k[0] not in exclude
                       and cc.get("status") not in ("absent", "absent-degenerate")
                       and cc["nulls"].get(nm, {}).get("p_doc") is not None]
                sr = np.array([cc["real"]["sum_rho"] for _, cc in sel])
                sn = np.array([cc["nulls"][nm]["sum_rho_null_mean"] for _, cc in sel])
                db = np.array([cc["nulls"][nm]["descriptive_D_bands_L1"] for _, cc in sel])
                # per-cell seed sign-mismatch (ревью 4, п. B/D): доля сидов с знаком,
                # отличным от знака real, усреднённая по клеткам
                mism = np.array([
                    float(np.mean(np.sign(np.array(cc["nulls"][nm]["per_seed"]["sum_rho"]))
                                 != np.sign(cc["real"]["sum_rho"])))
                    for _, cc in sel])
                out[obj][nm].update({
                    "sum_rho_real_mean": float(sr.mean()),
                    "sum_rho_null_mean": float(sn.mean()),
                    "sign_mismatch": bool(np.sign(sr.mean()) != np.sign(sn.mean())),
                    "frac_cells_seed_sign_mismatch": float(mism.mean()),
                    "sign_check_ok": bool(mism.mean() <= 0.1),
                    "descriptive_D_bands_L1_mean": float(db.mean()),
                    "descriptive_D_bands_L1_status": "reportable only (нет null-распределения — p-value не вычисляется)",
                    "note": ("AR: verdict — paired-delta по tested_D_center_l1 (профили sign-normalized "
                             "через band_shares_vec: деление на (r−1)/2). Интерпретация 'про band-форму' "
                             "валидна при sign_check_ok (per-cell seed sign-mismatch ≤ 0.1): при "
                             "расхождении знаков refuted был бы sign-driven. Порог 0.1 — эмпирический "
                             "(ревью 5, п. C): замеренный mean ≤ 0.01, при 0.5 знаки случайны и "
                             "refuted стал бы полностью sign-driven; это граница валидности "
                             "интерпретации, не тест. Механизм согласования "
                             "знаков: AR-фит (Yule-Walker) наследует знаки ACF real — в этих данных "
                             "frac ≥ 0.90 на клетку, mean 1.00. descriptive_D_bands_L1 — reportable only")})
    return out


def main():
    smoke = "--smoke" in sys.argv
    os.makedirs(STORE_DIR, exist_ok=True)
    env = {"python": sys.version.split()[0], "numpy": np.__version__,
           "threads": os.environ.get("OMP_NUM_THREADS"),
           "seed_base": SEED_BASE, "seed_stride": SEED_STRIDE,
           "b_null": B_NNULL, "b_fourier": B_FOURIER, "b_boot": B_BOOT, "seed_boot": SEED_BOOT}
    logline(f"AT0_2 T0.2 {'SMOKE' if smoke else 'FULL'}: py{env['python']} np{env['numpy']} thr={env['threads']}")

    # G1′: тот же прибор (field.py) — sha + selfcheck
    pf_sha = sha256_file(os.path.join(ROOT, "prism/field.py"))
    pf._selfcheck()
    # Пин acf_full: FFT-ACF == прямой einsum (ловит axis-конфузии и wrap-around)
    _rng_t = np.random.default_rng(123)
    _xt = _rng_t.standard_normal((37, 5))
    _c0t, _ct, _ = acf_full(_xt)
    _cd = np.array([float(np.einsum("ij,ij->i", _xt[:37 - k], _xt[k:]).sum()) for k in range(1, 37)])
    assert np.abs(_ct - _cd).max() < 1e-9 * max(1.0, float(np.abs(_cd).max())), \
        f"acf_full: FFT-ACF != direct (max Δ = {np.abs(_ct - _cd).max():.3e})"
    # Пин acf_full_many: batch == построчно (ловит axis-конфузии в batched-версии)
    _xb = _rng_t.standard_normal((2, 37, 5))
    _c0b, _cb, _ = acf_full_many(_xb)
    for _i in range(2):
        _c0s, _cs, _ = acf_full(_xb[_i])
        assert abs(_c0b[_i] - _c0s) < 1e-9 and np.abs(_cb[_i] - _cs).max() < 1e-9 * max(1.0, float(np.abs(_cs).max())), \
            f"acf_full_many != acf_full (row {_i})"
    # Пин band_sums_direct: cumsum-путь == полагам-einsum (ground truth)
    _xd = _rng_t.standard_normal((2, 200, 5))
    _bs, _c0d = band_sums_direct(_xd, BANDS)
    for j, (lo, hi) in enumerate(BANDS):
        hi = min(hi, 199)
        gt = sum(np.einsum("btd,btd->b", _xd[:, :200 - k, :], _xd[:, k:, :]) for k in range(lo, hi + 1))
        assert np.abs(_bs[:, j] - gt).max() < 1e-9 * max(1.0, float(np.abs(gt).max())), \
            f"band_sums_direct != per-lag (band {lo}..{hi})"
    # Пин r: ‖Σy‖²/c0 == 1+2Σρ̂ (FFT-full) и c0 по двум путям
    _c0f, _cf, _ = acf_full_many(_xd)
    _r_direct = (_xd.sum(axis=1) ** 2).sum(axis=1) / _c0d
    _r_fft = 1.0 + 2.0 * _cf.sum(axis=1) / _c0f
    assert np.abs(_r_direct - _r_fft).max() < 1e-9 * max(1.0, float(np.max(np.abs(_r_direct)))), \
        "r (‖Σy‖²/c0) != 1+2Σρ̂ (FFT-full)"
    assert np.abs(_c0d - _c0f).max() < 1e-9 * max(1.0, float(np.max(np.abs(_c0f)))), "c0: direct != FFT"
    logline(f"G1′: field.py sha {pf_sha[:16]} + _selfcheck ok + ACF-пины (cumsum/per-lag/FFT) ok")

    # G0: данные — sha + gen_sha + декод-проба
    data = {p: sha256_file(p) for p in (GEN, OWN, COR)}
    gen_meta = json.load(open(GEN.replace(".npz", "_meta.json")))
    gen_sha_ok = int(sum(1 for v in gen_meta.values() if v.get("sha_ok")))
    first_doc = sorted(gen_meta)[0]
    dec = a1.decode_probe(np.asarray(np.load(GEN, allow_pickle=True)[f"{first_doc}_tok"]), first_doc)
    # G0 per-source (G16): own/cor — статус по токенам (P2_corpus 0/16 ⇒ not_testable)
    own_z = np.load(OWN, allow_pickle=True)
    cor_z = np.load(COR, allow_pickle=True)
    own_toks = {f[:-4]: np.asarray(own_z[f]) for f in own_z.files if f.endswith("_tok")}
    n_own_toks = sum(1 for t in own_toks.values() if t.size > 0)
    n_cor_toks = sum(1 for f in cor_z.files if f.endswith("_tok") and cor_z[f].size > 0)
    own_first = sorted(n for n, t in own_toks.items() if t.size > 0)
    own_dec = a1.decode_probe(own_toks[own_first[0]], own_first[0]) if own_first else {"n_toks": 0, "lines": []}
    logline(f"G0: gen sha {data[GEN][:16]} gen_sha_ok {gen_sha_ok}/64; "
            f"decode_probe[{first_doc}]: {dec['lines'][0]['text'][:40]!r}...")
    logline(f"G0 sources: own токенов {n_own_toks}/16 (probe[{own_first[0] if own_first else '-'}]); "
            f"corpus токенов {n_cor_toks}/16 → {'ok' if n_cor_toks else 'not_testable (G16)'}")

    # Данные и объекты (та же машина, что AT0_1: load_source + vpath)
    docs = {}
    docs.update(a1.load_source(GEN, "gen"))
    docs.update(a1.load_source(OWN, "state"))
    docs.update(a1.load_source(COR, "state"))
    n_docs = len(docs)

    # G0a_hn_field (ревью 4, п. A/F): честный gate — см. g0a_hn_gate
    g0a_meas = g0a_hn_gate(docs)
    logline(f"G0a_hn_field: state unit-dev {g0a_meas['state']['max_unit_dev']:.2e} (регресс-гард; "
            f"_hn-поле не читается конвейером), gen unit-dev "
            f"{g0a_meas['gen_layers']['max_unit_dev']:.2e} (проверка поля _u) + scale-dev "
            f"{g0a_meas['gen_layers']['max_scale_dev']:.3f} (однородность U·hn; константный _hn "
            f"недетектируем — остаточный риск в claim) ok")

    # L1-TPU degraded (как AT0_1: |recomputed dip_V,med L1 − meta| > 1e-3 → absent)
    l1_degraded = {}
    for nm, d in docs.items():
        if not nm.startswith(("own_", "cor_")):
            continue
        u1, hn1 = d["layers"][1]
        U1 = u1 / np.maximum(np.linalg.norm(u1, axis=1, keepdims=True), 1e-12)
        V1, _ = vpath(U1, hn1)
        vc1 = dcurve(V1, np.arange(1, min(97, V1.shape[0])))
        meta_dip1 = d["curves_meta"].get(1, {}).get("dip")
        dev = None if (len(vc1) == 0 or meta_dip1 is None) else float(abs(float(vc1[0]) - meta_dip1))
        l1_degraded[nm] = bool(dev is not None and dev > a1.L1_DEGRADE_TOL)
    logline(f"L1-TPU degraded (dev > {a1.L1_DEGRADE_TOL}): "
            f"{[nm for nm, v in sorted(l1_degraded.items()) if v]}")

    # degen-флаги (§0.3, как AT0_1: L23 vcurve + token share; loopy = mcos>0.95 или wiki_0/4)
    degen_flags = {}
    for nm, d in docs.items():
        u23, hn23 = d["layers"][23]
        U23 = u23 / np.maximum(np.linalg.norm(u23, axis=1, keepdims=True), 1e-12)
        V23, _ = vpath(U23, hn23)
        vc23 = dcurve(V23, np.arange(1, min(97, V23.shape[0])))
        fl, mcos, _ = degen(vc23, d["toks"])
        degen_flags[nm] = {"degen": bool(fl or nm in ("wiki_0", "wiki_4")), "mcos": float(mcos)}

    # --- прогон клеток ---
    cells_by_obj = {o: {} for o in OBJECTS}
    n_cells = 0
    n_absent = 0
    expected = n_docs * len(LAYERS) * len(OBJECTS)   # ВСЕ ключи (включая absent-клетки)
    if smoke:
        smoke_names = sorted(nm for nm in docs if nm in ("wiki_0", "math_3"))
        doc_names = {nm: i for i, nm in enumerate(smoke_names)}
        expected = len(smoke_names) * len(LAYERS) * len(OBJECTS)
    else:
        doc_names = {nm: i for i, nm in enumerate(sorted(docs))}
    env["n_docs_run"] = len(doc_names)

    for nm, doc_idx in doc_names.items():
        d = docs[nm]
        l1_bad = bool(nm in l1_degraded and l1_degraded[nm]) and (not smoke)
        for L in LAYERS:
            if l1_bad and L == 1:
                for o in OBJECTS:
                    cells_by_obj[o][(nm, L, o)] = {
                        "doc": nm, "L": L, "obj": o, "dom": d["dom"],
                        "degen": degen_flags[nm]["degen"], "status": "absent",
                        "note": "L1-TPU: fp16-шаг ниже квантового шума (AT0_1 L1-degraded гейт)"}
                    n_absent += 1
                continue
            u, hn = d["layers"][L]
            U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
            h = U * hn[:, None]
            xs = {"h_raw": h[1:] - h[:-1], "du_unit": U[1:] - U[:-1]}
            V, _ = vpath(U, hn)
            xs["v_tangent"] = V
            for o in OBJECTS:
                x = xs[o]
                if x.shape[0] < 25:
                    raise ValueError(f"vacuous N={x.shape[0]} ({nm}, L{L}, {o})")
                rec = run_cell(x, doc_idx, nm, L, o, smoke=smoke)
                rec["dom"] = d["dom"]
                rec["degen"] = degen_flags[nm]["degen"]
                # E8: gen h_raw/v_tangent — conditional on не-верифицированном hn (scale-пин только)
                rec["provenance_limited"] = (not nm.startswith(("own_", "cor_"))) and o in ("h_raw", "v_tangent")
                cells_by_obj[o][(nm, L, o)] = rec
                n_cells += 1
        if not smoke and (doc_idx + 1) % 16 == 0:
            logline(f"  docs {doc_idx + 1}/{len(doc_names)}")

    # G2: полнота (set-равенство ВСЕХ ключей; absent-клетки тоже в наборе)
    got = sum(len(v) for v in cells_by_obj.values())
    n_l1_absent_exec = sum(1 for nm in doc_names if (not smoke) and nm in l1_degraded and l1_degraded[nm])
    g2_ok = (got == expected) and (n_absent == n_l1_absent_exec * len(OBJECTS))
    logline(f"G2: клеток {got}/{expected} (absent L1-TPU {n_absent} = {n_l1_absent_exec} доков × {len(OBJECTS)}) ok={g2_ok}")
    assert g2_ok, f"G2: клеток {got} != {expected} или absent {n_absent} != {n_l1_absent_exec * len(OBJECTS)}"

    # G8 + G7-invariance (по real-клеткам)
    gram_bad, inv_bad = [], []
    for o in OBJECTS:
        for key, c in cells_by_obj[o].items():
            if c.get("status") in ("absent", "absent-degenerate") or "real" not in c:
                continue
            if c["real"]["gram_resid"] >= GRAM_TOL:
                gram_bad.append(key)
            if c.get("r_inv_max", 0.0) > INV_R_TOL:
                inv_bad.append((key, c["r_inv_max"]))
    logline(f"G8: gram bad {len(gram_bad)}; G7-inv bad {len(inv_bad)}")
    assert not gram_bad, f"G8: {gram_bad[:5]}"
    assert not inv_bad, f"G7-inv: {inv_bad[:5]}"

    # G-seam (раунд 6 E2): within-block(blk-null) ≡ within-block(real) — алгебраически точно
    # (блоки перестановкой не трогаются) ⇒ seam-excluded within-block не может быть null'ом (D ≡ 0)
    seam_dev = 0.0
    for o in OBJECTS:
        for key, c in cells_by_obj[o].items():
            if "nulls" not in c or "seam_excluded_identity_max_dev" not in c["nulls"]:
                continue
            seam_dev = max(seam_dev, max(c["nulls"]["seam_excluded_identity_max_dev"].values()))
    logline(f"G-seam (E2): within-block(blk-null) ≡ within-block(real): max_dev {seam_dev:.2e} (≤ 1e-12) ok")
    assert seam_dev <= 1e-12, f"G-seam: max_dev {seam_dev:.2e} > 1e-12 — алгебраическое тождество сломано"

    # Positive control (ДО интерпретации null'ов): круговой ACF ≤ 1e-6, r (полный ряд) ≤ 2e-4
    pc_cells = [c for o in OBJECTS for c in cells_by_obj[o].values() if "nulls" in c]
    pc_max_circ = max(c["nulls"]["fourier_positive_control"]["max_abs_d_rho96_circular"] for c in pc_cells)
    pc_max_r = max(c["nulls"]["fourier_positive_control"]["max_abs_dr_full"] for c in pc_cells)
    pc_max_wrap = max(c["nulls"]["fourier_positive_control"]["max_abs_d_rho96_linear_wrap"] for c in pc_cells)
    pc_ok = pc_max_circ <= PC_RHO_TOL and pc_max_r <= PC_R_TOL
    logline(f"POSITIVE CONTROL: max|Δρ̂_circ(≤96)| = {pc_max_circ:.3e} (≤ {PC_RHO_TOL:.0e}), "
            f"max|Δr_full| = {pc_max_r:.3e} (≤ {PC_R_TOL:.0e}) ok={pc_ok}; "
            f"(отчёт) max|Δρ̂_lin| wrap = {pc_max_wrap:.3e}")
    assert pc_ok, "positive control FAILED — результат null'ов не интерпретируем"

    # Агрегаты + вердикты (раунд 6 E6/E7: primary = verified ∩ nondeg; sensitivity: 80/96)
    corpus_docs = {nm for nm, d in docs.items() if d["source"] == "P2_corpus"}
    degen_docs = {nm for nm, d in docs.items() if degen_flags[nm]["degen"]}
    primary_excl = corpus_docs | degen_docs
    agg = aggregate(cells_by_obj)                          # sensitivity: all_96
    agg_v = aggregate(cells_by_obj, exclude=corpus_docs)   # sensitivity: verified 80
    agg_p = aggregate(cells_by_obj, exclude=primary_excl)  # PRIMARY: verified ∩ nondeg
    logline(f"pools: all_96 (sensitivity) / verified_only {n_docs - len(corpus_docs)} (sensitivity) / "
            f"primary verified∩nondeg {n_docs - len(primary_excl)} "
            f"(исключено: corpus {len(corpus_docs)} + degen {len(degen_docs)})")

    # SMOKE: артефакты НЕ пишем (иначе затирает production claim/manifest/run.log — ревью N2)
    if smoke:
        for o in OBJECTS:
            for nm in NULL_NAMES:
                a = agg[o].get(nm, {})
                if a.get("n_docs"):
                    logline(f"[SMOKE] {o} {nm}: D_real {a['D_real_mean']:.4f} {a['D_real_ci']} "
                            f"D_null {a['D_null_mean']:.4f} {a['D_null_ci']} -> {a['verdict']}")
        logline("AT0_2 SMOKE OK (артефакты не пишутся)")
        return

    # --- store + claim + manifest + run.log ---
    store = {
        "experiment": "AT0_2 T0.2 (форма профиля при фиксированной сумме)",
        "tz": "docs/TZ_add_0-3.md §1.5 T0.2",
        "meta": {"env": env, "field_sha256": pf_sha, "data_sha256": data,
                 "gen_sha_ok": f"{gen_sha_ok}/64", "decode_probe": dec,
                 "script_sha256": sha256_file(os.path.join(ROOT, "experiments/AT0_2_shape.py")),
                 "n_docs": n_docs, "n_docs_verified": n_docs - len(corpus_docs),   # ревью 5, п. H
                 "confounds_note": ("единственный конфунд: P2_corpus (16/96) not_testable (G16) — "
                                    "provenance_limited, вне primary (E6); пулы: primary = verified ∩ "
                                    "nondeg (aggregates_primary), sensitivity: verified_only (80) и "
                                    "all_96; правило решения — gates.G0_sources.note"),
                 "layers": [int(l) for l in LAYERS], "objects": list(OBJECTS),
                 "smoke": bool(smoke)},
        "prereg": {
            "bands": [list(b) for b in BANDS],
            "tail": "97..N-1 — отдельный скаляр, в вектор профиля не входит (K=96-обрезка объявлена)",
            "normalization": "(r-1)/2 = sum_rho_full (полные лаги, G1); AR-null — своя сумма (ТЗ: Σ отдельно)",
            "estimand": ("T0.2 = ФОРМА профиля при фиксированной сумме (ТЗ v1.2 G7: тест «Σ против "
                         "полосы» ВЫРОЖДЕН — Σx, c0 инвариантны к перестановкам; численный чек — "
                         "sum_invariance_max_dev в агрегатах). AR-null — Σ своё (mismatch отчёт, "
                         "с real НЕ сравнивать — ТЗ)"),
            "distance": ("tested_D_center_l1 = L1 по 5-вектору долей (основное, тестируется), "
                         "tested_D_center_l2 — отчёт; descriptive_D_bands_L1 — reportable only; "
                         "p_doc = (#{D'_s ≥ D_center}+1)/(n+1) + raw exceedance_count"),
            "seeds": (f"B={B_NNULL} на каждый (iid/blk/AR) — p_floor = 1/(B+1) = {1/(B_NNULL+1):.3f} "
                      f"(раунд 6 E3; стандарт T0.1 NNULL=199); сиды 0..19 репродуктивны из round 1; "
                      f"fourier B={B_FOURIER}; seed = {SEED_BASE} + doc_idx*{SEED_STRIDE} + s"),
            "nulls": {"iid": "full independence",
                      "blk{2,4,8,16,32}": ("damped/seamed null (раунд 6 E2): within-block пары лага k — "
                                           "с весом w(k,b)=max(0,1-k/b), k≥b уничтожены, швы добавлены — "
                                           "НЕ чистый null локальности; вердикт = 'real differs from "
                                           "unmasked block-shuffle null'; locality-прибор — P3-ladder"),
                      "ar{1,2,4}": ("линейный процесс (per-dim Yule-Walker, demeaned, burn 256) + "
                                    "diagnostics (E5): ar_max_root, n_nonstationary, burn_mix_ratio, "
                                    "sigma2_clip_frac, rho1..4 real vs null, sum-mismatch; AR1/AR2 "
                                    "refuted ≈ тавтология при k3,4-excess real"),
                      "fourier": "POSITIVE CONTROL (не null): phase randomization, zero-pad >= 2N-1"},
            "seam_excluded": ("отдельный оценщик (within-block k<b), Σ с real НЕ СРАВНИВАТЬ; identity-check: "
                              "within-block(blk-null) ≡ within-block(real) (max_dev ≤ 1e-12, fail-loud) "
                              "⇒ seam-excluded within-block не может быть null'ом (D ≡ 0)"),
            "verdict": ("H='форма объясняется null'. PAIRED (раунд 6 E4/M4/M5): delta_d = D_real(doc) − "
                        "D_null(doc) (per-doc mean по L, кластер=док); delta_mean, CI_delta — cluster-"
                        "бутстрап по докам (B=1000, seed=117). DELTA_MIN = "
                        f"{DELTA_MIN} — pre-registered floor (M4; порядок эффекта T0.1 по полосам "
                        "0.02–0.06). |Δmean| < DELTA_MIN → not_testable; Δmean ≥ DELTA_MIN и CI_Δ > 0 → "
                        "refuted; Δmean ≤ −DELTA_MIN и CI_Δ < 0 → confirmed_anomaly (флаг «более типичен, "
                        "чем типично», НЕ сигнал структуры); иначе inconclusive. Separate-CI (D_real vs "
                        "CI_null) — вторичное, в вердикт не входит. AR: интерпретация 'про band-форму' "
                        "валидна при sign_check_ok (per-cell seed sign-mismatch ≤ 0.1)"),
            "pools": ("PRIMARY = verified ∩ non-degen (раунд 6 E6/E7); sensitivity: verified_only (80) "
                      "и all_96 (96, включая corpus provenance_limited + degen). corpus — G16 (нет "
                      "токенов). degen (loopy) — вне primary: агрегат с loopy = 'full-pool shape'"),
            "objects_provenance": ("h_raw/v_tangent gen — provenance_limited (hn не верифицирован — "
                                   "scale-пин); du_unit — из U только (hn-независим); state-объекты — "
                                   "верифицированы конструкцией (hn = ‖raw h‖)"),
            "primary_family": ("pool=primary × objects {du_unit + state h_raw/v_tangent} × nulls {iid, "
                               "blk32, ar4} = 9 тестов; FWER-коррекция не применяется — объявлено "
                               "(M3); остальное — sensitivity"),
            "expected_cells": int(expected),
        },
        "cells": {o: {f"{k[0]}|L{k[1]}": c for k, c in cells_by_obj[o].items()} for o in OBJECTS},
    }
    store_j = jclean(store)
    blob = json.dumps(store_j, ensure_ascii=False, sort_keys=True)
    store_sha = hashlib.sha256(blob.encode()).hexdigest()
    store_path = os.path.join(STORE_DIR, f"{store_sha[:16]}.json")
    with open(store_path, "w") as f:
        f.write(blob)

    claim = {
        "experiment": "AT0_2 T0.2",
        "round": "round 2 (ревью r6: E1–E8, M1–M5)",
        "date": "2026-11-14 (UTC+3)",
        "store": os.path.relpath(store_path, OUT_DIR),
        "store_sha256": store_sha,
        "meta": store["meta"],
        "prereg": store["prereg"],
        "gates": {
            "G0": {"data_sha256": data, "gen_sha_ok": f"{gen_sha_ok}/64",
                   "decode_probe_n_toks": dec["n_toks"]},
            "G0_sources": {
                "gen_layers": {"status": "ok", "gen_sha_ok": f"{gen_sha_ok}/64",
                               "decode_probe": dec},
                "P1_own": {"status": "ok" if n_own_toks else "not_testable",
                           "toks_present": f"{n_own_toks}/16", "decode_probe": own_dec},
                "P2_corpus": {"status": "not_testable" if n_cor_toks == 0 else "ok",
                              "toks_present": f"{n_cor_toks}/16",
                              "reason": (None if n_cor_toks else
                                         "нет токенов — декод-проба невыполнима в принципе (G16, ТЗ v1.2)")},
                "note": ("P2_corpus (16/96 доков) — источник not_testable (G16); corpus — "
                         "provenance_limited (E6): вне primary. ПУЛЫ (раунд 6 E6/E7): primary = "
                         "verified ∩ non-degen (aggregates_primary); sensitivity: verified_only (80, "
                         "aggregates_verified_only) и all_96 (aggregates). ЕСЛИ verdict'ы расходятся — "
                         "решает primary. Перепредставленность: 16/16 nondeg (100%) при 55% "
                         "общих (265/480)"),
            },
            "G0a_hn_field": {
                "state": {"check": ("unit-пин = регресс-гард: load_source пересчитывает u, hn из "
                                   "raw h, хранимое _hn-поле (баг продюсера) не читается"),
                         "verifies": ("только нормализацию (ревью 5, п. F): load_source — row-norm "
                                      "recompute, корректность hn доказана конструкцией (hn = ‖raw h‖)"),
                         "max_unit_dev": g0a_meas["state"]["max_unit_dev"], "tol": 1e-9,
                         "edge": "вырожденная строка ‖h_row‖ < 1e-12 → fail (desired fail-loud, ревью 5, п. A)"},
                "gen_layers": {"check": ("unit-пин — проверка поля _u (единственное representation: "
                                         "raw h в gen npz отсутствует, recon U·hn vs raw h "
                                         "невозможна); scale-пин — однородность row-скал U·hn"),
                               "verifies": ("поле _u (нормализация) + однородность row-скал U·hn; "
                                            "перестановку/константный _hn не ловит (ревью 5, п. D)"),
                               "max_unit_dev": g0a_meas["gen_layers"]["max_unit_dev"],
                               "tol_unit": 5e-3,
                               "max_scale_dev": g0a_meas["gen_layers"]["max_scale_dev"], "tol_scale": 0.2,
                               "residual_risk": ("scale-пин sdev≈CV²/2: ловит CV ≳ 0.63; слабее — "
                                                  "недетектируемо от файла, как константный и "
                                                  "пермутированный _hn (mean/rms/CV инвариантны к "
                                                  "перестановке); замеренная разбросность hn 0.93–1.87 "
                                                  "согласуется с истинными row-нормами")},
                "ok": True,
            },
            "G1_field": {"sha256": pf_sha, "selfcheck": "ok"},
            "G2": {"cells": int(got), "expected": int(expected), "ok": bool(g2_ok)},
            "G7_store": {"store_sha256": store_sha},
            "G8_gram": {"tol": GRAM_TOL, "bad": len(gram_bad), "ok": not gram_bad},
            "G7_invariance": {"tol": INV_R_TOL, "bad": len(inv_bad), "ok": not inv_bad},
            "positive_control": {"max_abs_d_rho96_circular": pc_max_circ,
                                 "max_abs_dr_full": pc_max_r,
                                 "max_abs_d_rho96_linear_wrap": pc_max_wrap,
                                 "tol_rho": PC_RHO_TOL, "tol_r": PC_R_TOL, "ok": bool(pc_ok),
                                 "note": "1e-6 — для кругового/спектрального ACF (сохранён конструкцией); "
                                         "линейный (обрезанный) ACF — wrap-around член, свойство суррогата"},
        },
        "aggregates_primary": agg_p,
        "aggregates_verified_only": agg_v,
        "aggregates": agg,
        "primary_pool": "verified_nondeg",
        "pool_roles": {
            "primary": f"aggregates_primary (verified ∩ non-degen, {n_docs - len(primary_excl)} доков)",
            "sensitivity_verified_only": f"aggregates_verified_only ({n_docs - len(corpus_docs)} доков, включая degen)",
            "sensitivity_all_96": "aggregates (96 доков, включая corpus + degen)"},
        "corpus_status": "provenance_limited (G16: нет токенов — декод-проба невыполнима в принципе)",
        "degen_docs_n": len(degen_docs),
        "degen_note": ("AT0_3: loopy-доки несут периодическую лаговую структуру — агрегат, включающий "
                       "loopy, = 'full-pool shape', не 'clean text shape'; primary исключает degen-доки; "
                       "strata (loopy/clean/remainder) — n в degen_docs_n + per-doc degen-флаги в store"),
        "object_provenance": {
            "h_raw": ("state: верифицирован конструкцией (hn = ‖raw h‖, load_source пересчитывает); "
                      "gen: provenance_limited (hn не верифицирован — scale-пин только; E8)"),
            "du_unit": "из U только (hn-независим) — primary на gen (E8)",
            "v_tangent": ("U·hn: gen — provenance_limited (E8); state — верифицирован конструкцией")},
        "primary_objects": ("du_unit (все пулы, hn-независим) + state-объекты (h_raw, v_tangent: "
                            "верифицированы конструкцией); gen h_raw/v_tangent — sensitivity "
                            "(provenance_limited, per-cell флаг в store)"),
        "aggregate_levels_note": ("frac_cells_p_lt_alpha — КЛЕТКИ (doc×layer×obj); "
                                  "n_docs_unique_p_lt_alpha — уникальные ДОКУМЕНТЫ; уровни разные — "
                                  "не сравнивать напрямую; exceedance_count_hist — raw-счётчики "
                                  "(E3); p_floor = 1/(B+1) = 0.005 при B=199"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(jclean(claim), f, ensure_ascii=False, indent=1)

    man = {
        "experiment": "AT0_2 T0.2",
        "round": "round 2 (ревью r6: E1–E8, M1–M5)",
        "date": "2026-11-14 (UTC+3)",
        "claim_sha256": hashlib.sha256(open(OUT_JSON, "rb").read()).hexdigest(),
        "store_sha256": store_sha,
        "script_sha256": sha256_file(os.path.join(ROOT, "experiments/AT0_2_shape.py")),
        "note": "claim AT0_2_SHAPE.json (aggregates+gates+prereg); store = full per-doc null-детали; "
                "run_log = stdout-лог (tracked).",
    }
    man_blob = json.dumps(man, ensure_ascii=False, indent=1)
    man["manifest_sha256"] = hashlib.sha256(man_blob.encode()).hexdigest()
    with open(MAN_JSON, "w") as f:
        f.write(json.dumps(man, ensure_ascii=False, indent=1) + "\n")

    for o in OBJECTS:
        for nm in NULL_NAMES:
            a = agg[o].get(nm, {})
            if not a.get("n_docs"):
                continue
            logline(f"{o} {nm}: Δmean {a['delta_mean']:.4f} {a['delta_ci']} "
                    f"(DELTA_MIN={a['delta_min']}) D_real {a['D_real_mean']:.4f} "
                    f"frac(p<α)={a['frac_cells_p_lt_alpha']:.2f} -> {a['verdict']}")
    logline(f"store: {store_path} (sha {store_sha[:16]})")
    logline(f"written: {OUT_JSON}, {MAN_JSON}, {RUN_LOG}")
    logline("AT0_2 gates: ALL PASS")
    with open(RUN_LOG, "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
