#!/usr/bin/env python3
"""
AT0_5 — T0.5 «Генеративная проверка (disk-only)» — пре-рег (текст = хедер скрипта,
docs/AT0_5_prereg.md)

Дата: 13.11.2026. ТЗ: docs/TZ_add_0-3.md §1.5 T0.5 (sha см. в артефакте).
Контекст (T0.1): DC+MA(1) refuted на пуле (p_perm = 1/199), но объясняет ≈93% уровня
плато; остаток 6.2–6.6% статистически надёжен ⇒ estimand T0.5 — **per-doc** (G10):
воспроизводит ли минимальная линейная модель пути (DC+MA(1), параметры fit-нуты на
доке) dip+плато **у каждого дока**. Вердикт — IN-SAMPLE GOF (ревью r4 C3): a из ρ̂(1)
⇒ дип циркулярен по построению; held-out/generative-claim НЕ делается (OOS — только
plat_hi_oos, reportable). Оба исхода публикуемы.

Клетки: 96 доков (gen 64 + P1_own 16 + P2_corpus 16) × L ∈ {1,5,11,17,23} = 480 − skips
(k < 16; T0.1 K=0: own_w0/own_w4 L1, fp16_l1_degraded).
Объект: joint (v̂_t, log‖Δh_t‖), h = u·hn (load_source AT0_1; TPU/corpus — raw fp16,
‖row‖ пересчитан; построение U/h — ТОЧНО машина T0.1: U = u/max(‖u‖,1e-12), h = U·hn),
Δh_t = h_{t+1} − h_t, t = 0..382 (N = 383). v̂_t = Δh_t/‖Δh_t‖ (guard ‖Δh_t‖ > 1e-12,
вырожденный шаг — fail-loud), n_t = log‖Δh_t‖.

Часть 1 (РЕШАЮЩАЯ) — DC+MA(1) per-doc. Модель (G8): x_t = m + (ε_t − aε_{t−1}),
ε ~ N(0,1) iid, m = const (DC-компонент; f_DC-плато безцентр. ACF возникает из m²-
вклада в c(k)). ACF в безцентровом T-нормировочном согласии (поле.acf_moments):
ρ̄(1) = (1−1/N)(f − (1−f)a/(1+a²)), ρ̄(k≥2) = (1−k/N)·f — DC-компонент несёт
наклон (1−k/N), а не плоское f (ревью р2). m² = f(1+a²)/(1−f). Фит per-doc (pooled
запрещён — E-56): (f, a) — LS к ACF Δh (T-нормировка, k=1..96): f = mean_k ρ̂(k)/(1−k/N)
(k=2..96); a из ρ̂(1)/(1−1/N) = f − (1−f)a/(1+a²) closed-form (корень с |a| < 1 —
стационарность MA); 1 шаг
Gauss-Newton с backtracking (≤8 halving); выход из области (0≤f<1, |a|<1) исключён
(backtracking принимает только доменные шаги, начальные доменные — raise недостижим);
out_of_model (disc ≤ 0, d ≥ (1−f)/2 — unit-корень/вне области) — reportable flag;
acf_r2 — R² ACF-фита (reportable). f ≤ 0 ⇒ f заморожен = 0 (GN по f
не выполняется, gn_skipped), a — из closed-form с (1−f) = 1 (чистый MA(1), если ρ̂(1) < 0; a = 0, если
ρ̂(1) ≥ 0) — flag f_le_0 = «DC нет». OOS-вариант (reportable, разделение GOF/out-of-sample):
fit k=2..48 (нормировка (1−k/n) — PER-CELL n) → симуляция → тест plat_hi_oos (k=49..96)
— ОДНА метрика, слабая проверка;
дип НЕ тестируется OOS (a из ρ̂(1) ⇒ циркулярно); основной вердикт — по TZ (fit
k=2..96). Симуляция — D-МЕРНЫЙ null из FIT-НУТЫХ параметров (ревью р3 + р4 C2):
x_{t,j} = m′_j + s′_j·(ε_{t,j} − aε_{t−1,j})/√(1+a²); масштаб: total power сохраняем
(Σ(m′²+s′²) = Σ(m²+s²) = E), DC-доля = fit-нутый f (Σm′² = f·E, Σs′² = (1−f)·E),
анизотропия сохраняется (направление m_j, форма s_j); f = 0 ⇒ m′ = 0. Старая версия
(real m/s as-is, f не передавался) давала f_DC_sim = f_DC_real BY CONSTRUCTION
(f_dc = ‖mean‖²/mean‖x‖²) ⇒ покрытие плато было циркулярным. B = 200 путей длины n_i,
seed = 100000 + 5·doc_idx + layer_idx (layer_idx = 0..4 по (1,5,11,17,23)). Метрики (все scale-invariant): dip = ρ̂(1), plat_lo = mean
ρ̂(k), k=9..32, plat_hi = mean ρ̂(k), k=33..96 — для real и каждого sim-пути.
Per-doc covered = real (dip, plat_lo, plat_hi) все ∈ [q_0.025, q_0.975] sim-
распределения (короткая клетка — по доступным полосам, sensitivity). Агрегат:
per-layer frac_covered по PRIMARY-пулу (non-degen ∩ K96; degen_tz — флаг T0.1 store;
degen-пул и short-клетки (16 ≤ k < 96) — отдельные отчёты, в вердикт не входят;
per-source стратификация gen/own/corpus — reportable; corpus: h-данные есть, токены
0/16 — G16 not_testable = только provenance-оговорка). КАЛИБРОВКА (ревью r4 C4,
reportable, не гейт): d_eff per-cell = participation ratio per-dim дисперсий
(анизотропия: d_eff ≪ 1024 ⇒ effective dof мал); null-on-null: 2 независимые партии
B=200 из fit-нутых (f,a) — партия A → квантили [q2.5,q97.5], партия B → marginal/joint
покрытие (первые K96 non-degen доки per layer); quantile MC error при B=200 ≈ 0.19σ
(meta). ВЕРДИКТ (IN-SAMPLE GOF, ревью r4 C3 — generative-claim не делается): branch1
«три моды мертвы как in-sample generative объяснение» (T3 остаётся только горб) ⇔
frac_covered ≥ 0.5 на КАЖДОМ слое по eligible клеткам; пустой primary-пул на слое ⇒
branch2; иначе branch2: структура сверх DC+MA(1) есть, T3 получает работу.
Reportable: r_sim/f_DC_sim/Σ96_sim/bump_sim по sim-пути + gaps real−sim; per-док
(f, a, m, acf_r2, f_fit_vs_real, d_eff).

Часть 2 (reportable) — лестница VAR per-doc. PCA(v̂) per (doc,L) (SVD 383×1024).
Сетка r ∈ {8,16,32} × p ∈ {1,2,4}: VAR(p) на r-проекции v̂, LS-фит (intercept;
регрессия x_t = c + Σ_i A_i x_{t−i}; n = 383−p — число LS-выборок; SSR = Σ_t‖e_t‖²).
BIC (PRIMARY, ревью r4 C6) = n·ln det Σ + (p·r²+r)·ln n — гауссов likelihood,
Σ = EᵀE/n (full df: A_i — r×r); bic_tz = n·ln(SSR/n) + (p·r+r)·ln n (TZ —
SENSITIVITY, reportable: trace-SSR теряет cross-output covariance, TZ-df
недоучитывает A_i; ДОП. смещение: BIC по СУММАРНОМУ SSR — низкодисперсионные
компоненты почти ничего не платят за свои df ⇒ selected смещён к r=32 сильнее, чем
df-объяснение). selected (r,p) = argmin BIC (primary); selected_tz — по TZ
(reportable). Норма —
AR(1): n_t = α + βn_{t−1} + ε (LS; reportable β, R²; в BIC-выбор не входит).
Генеративный путь с selected (r,p): B = 100 (seed = 200000 + 5·doc_idx + layer_idx):
v̂̃ ∈ R^r — ЕДИНИЧНОЕ направление сгенерированного пути (X/‖X‖ по оси r; VAR,
нач. усл. нули ⇒ первые p шагов ‖X_t‖ = 0 — артефакт нач. условий, ≤ p/383 ≈ 1%
ACF-массы, задокументировано),
ñ_t (AR(1), нач. ñ_0 = mean(n_real), e ~ N(0, sd_res)), Δh̃_t =
exp(ñ_t)·v̂̃_t (r-мер) → метрики (r, f_DC, dip, plat_lo, plat_hi, Σ96) + gaps real−sim.
|β| ≥ 1 ⇒ ar1_unstable flag, симуляция пропускается. Примечание: real D = 1024,
sim D = r — gap = «доля структуры в r главных проекциях», НЕ точное воспроизведение.
DC+MA(1) в лестнице — L0, ВНЕ BIC-выбора (другое семейство: MA-ошибка vs AR-регрессия)
— отчитывается ACF-fit R² (reportable).

Часть 3 (гейт G8) — negative control. Blk-shuffle Δh (b = 8 по t; 383 = 47×8 + 7,
хвост 7 остаётся на месте; 5 сидов 1001..1005; перестановка блоков = перестановка
строк v̂/scores — проекция коммутирует с перестановкой строк, SVD один per (doc,L)).
Свойство блок-нуля: внутри-блок структура (k < b) сохраняётся — BIC-лестница на null
закономерно может выбирать p > 1 (это сохранённая локальная структура, не баг).
«Плоская кривая» операционализуется по ДАЛЕКОМУ хвосту: при k > b пары всегда в
разных блоках ⇒ корреляции нет. Gate (G8-v3, 15.11): (а) median |tail_excess|
(plat_hi − (1−k̄/N)·f_DC, k=33..96, k̄ = 64.5) по (doc,L,seed) ≤ 3·sd_EMP, где
sd_EMP = std эмпирического null-распределения (пул по всем клеткам). Параметрическая
D-мерная DC+MA(1)-симуляция занижает флуктуацию null в 1.6–5.5× (реальные строки Δh —
не гауссовы по колонкам: концентрация row-power ⇒ дисперсия произведений пар больше,
чем в sim с iid-колонками) — sd_null (sim) отчитывается как калибровочная
диагностика, НЕ как масштаб гейта; (б) sim-среднее (TAIL-FIT: f на k=33..96, a из
лаг-1; внутри-блок структура k<8 сохранена по построению и из fit исключена) RAW —
согласование с null-статистиками на эмпирическом масштабе: frac клеток с |null −
sim_mean| ≤ 3·sd_pool_stat ≥ 90% (dip, plat_lo, plat_hi — RAW; без base-вычитания:
sample f_dc sim перегружен шумом среднего MA(1) ≈ V/n ≫ f — баг G8-v3).
Reportable: распределение selected (r,p) на null + BIC-сводка + raw-покрытие sim.
Нарушение (а)/(б) ⇒ баг пайплайна.

Нормировка (TZ): все отчитываемые метрики scale-invariant (ρ̂, r = 1+2Σρ̂ (G1), f_DC
(G3), dip, plateau, bump). Σ/S(0) — ТОЛЬКО как r; абсолютные Σ и ‖Δh‖² не сравниваются
real-vs-sim (разные D).

Гейты (fail-loud, до вердикта):
G0: sha256 3 npz + gen_sha 64/64 + decode-probe 3 строки/gen + 1 строка/own;
    P2_corpus — not_testable (G16, 0/16 tok).
G1′: field.py sha-префикс bcc607c561f95e3e + field._selfcheck().
G2: все клетки (480 − skips; skips: k < 16, T0.1 K=0), per-cell ключи полные.
S0: rho_metrics ≡ field.acf_moments (exec на 2 матрицах, точность 1e-14).
S1: телескоп (TZ §1.7) per (doc,L): ‖ΣΔu − (u_T−u_0)‖ < 1e-6.
S2: iid-нуль (ТЗ §1.7, D5): DC+MA(1) (f=0,a=0), B=2000: mean ρ̂(1) = 0 ± 3·SE.
    UNCENTERED iid: E[ρ̂(k)] = 0 ТОЧНО (симметрия по знаку одной координаты);
    −1/(N−1) — target DEMEANED ACF, не наше соглашение (ревью r4 C1). D13: старый
    «срыв на 2.81 SE» был артефактом wrong target — тот же seed 42: mean = +0.000593
    = 0.52 SE от 0; tol 3·SE сохранён как консервативный запас.
S3: AR(1) φ=0.5, B=2000: mean ρ̂(1) ∈ [0.3, 0.6] (ориентир +0.414).
S4: period-2: x_t = (−1)^t: чётный N=384 ⇒ r = 0; нечётный N=383 ⇒ r = 1/N
    (краевой член, свойство uncentered T-норм. ACF; старый докстринг «r=0» для odd N
    неверен — ревью r4 C7.3).
S5: Fourier-суррогат (positive control): 1 док L23, seed=7; фазовая рандомизация с
    сопряжённой симметрией ⇒ КРУГОВОЙ ACF сохраняется с точностью ≤ 1e-6
    (max |c_circ(сурр.) − c_circ(ориг.)|; круговой r самих данных — O(1) из-за
    DC-плато, так что «r ≤ 1e-6» на данных невозможно — (r) в TZ = расхождение);
    ЛИНЕЙНЫЙ ACF отличается wrap-around хвостом — отчитывается отдельно, «свойство,
    не баг» (урок AT0_2: линейный 1e-6 для spectrum-preserving суррогата
    конструктивно недостижим).
S6: gram_resid < 1e-10 на всех 480 клетках (G8-аудит field.acf_moments).
S7: r = N·f_DC (G3) ≤ 1e-12 на всех 480 клетках.
S8: Σρ̂_full перм-инвариантность: 1 док L23, 5 сидов, spread < 1e-12.
S9 (ЯКОРЬ): ACF Δh per (doc,L) == T0.1 store (results/AT0/store/23281c64cc418ff1.
    json, per_doc[doc].layers[L].h_raw.mean.rho) ≤ 1e-12; якоря 478 (own_w0/own_w4
    L1 — T0.1 K=0, L1-TPU деградация — skip, в клетках не участвуют); + K/N-сверка
    с T0.1 (ревью r4 C5.1): N_store == n; K-алфавит {0,24,32,96}: K=96 ⇔ k=96,
    K=32 ⇔ 32≤k≤95, K=24 ⇔ 24≤k≤31, K=0 ⇒ клетки нет. Пин — на STORE: sha256
    23281c64cc418ff1fedad2a5013c4bf7c8b7dd7680ac8d4fa7cf746c62fd1732 (бит-в-бит к TZ);
    claim-файл T0_ACCOUNTING.json (TZ-шина d4a4b101…) был перезаписан AT0_1 r5
    (коммит 6c1cdf3) — его sha только отчитывается.

Числа-ссылки (не пересчитывать без причины): T0.1 плато ≈ f_DC ≈ 0.057–0.065
(МЕДИАНА ПУЛА, не per-doc значение: f_DC per-doc варьируется, например до ~0 —
сверять per-doc с пулом нельзя);
избыток +0.00194 (v_tangent); горб +0.0121/−0.0134/−0.0137; E_null blk-2 = −0.002228;
аналит. E_null = −(1−f)/(N−1).

Формат: claim results/AT0/AT0_5_GENCHECK.json; store results/AT0/store/<sha>.json;
run.log; manifest. RUN_RC=0 + gates ALL PASS — обязательны; гейт с отсутствующим/
False ok ⇒ RUN_RC=1, claim/manifest НЕ пишутся (ревью r4 C7).
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "prism"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import field  # noqa: E402  (G1′: sha-пин)
import AT0_1_accounting as a1  # noqa: E402  (load_source, decode_probe, GEN/OWN/COR)
from field import sha256_file  # noqa: E402

GEN, OWN, COR = a1.GEN, a1.OWN, a1.COR
LAYERS = (1, 5, 11, 17, 23)
K = 96
N = 383          # T-1 приращений
T = 384
PLAT_LO = (9, 32)
PLAT_HI = (33, 96)
R_GRID = (8, 16, 32)
P_GRID = (1, 2, 4)
B1, B2, BNULL = 200, 100, 2000
BLK, NSEED = 8, 5
SEED1_OFF, SEED2_OFF, SEEDN_OFF = 100000, 200000, 300000
FIELD_PIN = "bcc607c561f95e3e"
T01_FILE = os.path.join(ROOT, "results/AT0/T0_ACCOUNTING.json")
T01_STORE = os.path.join(ROOT, "results/AT0/store/23281c64cc418ff1.json")
T01_STORE_SHA = "23281c64cc418ff1fedad2a5013c4bf7c8b7dd7680ac8d4fa7cf746c62fd1732"
OUT_DIR = os.path.join(ROOT, "results/AT0")
STORE_DIR = os.path.join(OUT_DIR, "store")
RUN_LOG = os.path.join(OUT_DIR, "AT0_5_run.log")
MANIFEST = os.path.join(OUT_DIR, "AT0_5_manifest.json")
CLAIM = os.path.join(OUT_DIR, "AT0_5_GENCHECK.json")
FRAC_MIN = 0.5  # порог пре-рега (branch1)
G8_TAIL_SIG, G8_COV_MIN = 3.0, 0.90   # G8-v3: (а) median|tail_excess| ≤ 3·sd_EMP (эмпирич. пул); (б) sim-среднее: frac |null−sim| ≤ 3·sd_pool_stat ≥ 0.90
SEED_OOS_OFF = 400000   # OOS-вариант (fit k=2..48, тест dip+plat_hi) — reportable
SEED_CAL_A, SEED_CAL_B = 500000, 510000   # null-on-null калибровка (ревью r4 C4;
# r5 C3: seed = OFF + 5·idx + li — уникален по (idx, li), нет коллизий)
SEED_FCOV_OFF = 600000   # full-cov null (ревью r5 C2, опция 2)
SEED_PERM_BASE = 1000000  # block_perm (ревью r5 C3.2: 25·idx + 5·li + seed_i)
REVISION = "r1"   # trackio run name: at05_gencheck_r1 (AGENTS: trackio обязателен)

_LOG = []


def logline(s=""):
    print(s, flush=True)
    _LOG.append(s)


# ---------------------------------------------------------------- trackio
# AGENTS: trackio обязателен для ВСЕХ новых локальных (CPU) экспериментов.
# run name = at05_gencheck_r1 (ревизия в имени); step = wall-секунды от старта;
# старт t=0 (config со started_at), финиш t=duration (finished_at + status);
# крупным этапам — {stage, stage_s}; краш — status=crash в finally + finish().

def trackio_start(name, cfg):
    try:
        import trackio
    except ImportError as e:
        raise SystemExit(f"trackio: модуль не установлен — прогон остановлен "
                         f"(AGENTS: trackio обязателен для CPU-прогонов): {e}")
    run = trackio.init(project="prism", name=name, group="AT0", config=cfg)
    return run


def tlog(run, t, metrics):
    if run is not None:
        run.log(metrics, step=int(t))


def tstage(run, t, name, extra=None):
    m = {"stage": name, "stage_s": round(float(t), 1)}
    if extra:
        m.update(extra)
    tlog(run, t, m)


# ---------------------------------------------------------------- данные

def load_docs():
    docs = {}
    docs.update(a1.load_source(GEN, "gen"))
    docs.update(a1.load_source(OWN, "state"))
    docs.update(a1.load_source(COR, "state"))
    return docs


def build_paths(docs, layers=LAYERS, t01_per=None):
    """Точная машина T0.1: U = u/max(‖u‖,1e-12); h = U·hn; dh = h[1:]−h[:-1].
    Per-cell N/K (протокол T0.1, ревью r4 C5.1): n = min(T_i, 384) − 1 = строк Δh
    (= N в T0.1 store, 0 расхождений по 478 клеткам), k = min(96, n − 1) — макс. лаг
    ряда длины n; НЕ старая min(96, n//4) (82/480 клеток расходились с T0.1; S9
    rho-якорь это не ловил: rho не зависит от K, зависят полосы). K-алфавит T0.1
    {0,24,32,96} (K = max {24,32,96} ≤ n−1): K=96 ⇔ n ≥ 97 (PRIMARY-пул, все 3
    полосы), 16 ≤ k < 96 — short-клетки (sensitivity), k < 16 или T0.1 K=0
    (fp16_l1_degraded: own_w0/own_w4 L1) — skip (reportable).
    Возврат: (paths, skipped)."""
    paths, skipped = {}, []
    for nm, d in sorted(docs.items()):
        for L in layers:
            u, hn = d["layers"][L]
            T = u.shape[0]
            n_i = min(T, N + 1)
            k_i = min(K, n_i - 2)   # строк Δh = n_i − 1; макс. лаг = n_i − 2
            t01K = (t01_per[nm]["layers"][str(L)]["h_raw"]["mean"].get("K")
                    if t01_per is not None else None)
            if t01K == 0:
                skipped.append({"doc": nm, "L": int(L), "T": int(T), "k": int(k_i),
                                "reason": "T0.1 K=0 (fp16_l1_degraded provenance)"})
                continue
            if k_i < 16:
                skipped.append({"doc": nm, "L": int(L), "T": int(T), "k": int(k_i)})
                continue
            u, hn = u[:n_i], hn[:n_i]
            U = u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
            h = U * hn[:, None]
            dh = h[1:] - h[:-1]
            du = U[1:] - U[:-1]
            assert float(np.linalg.norm(du.sum(axis=0) - (U[-1] - U[0]))) < 1e-6, \
                f"S1 телескоп (TZ §1.7): {nm} L{L}"
            norms = np.linalg.norm(dh, axis=1)
            if float(norms.min()) <= 1e-12:
                raise ValueError(f"вырожденный шаг ‖Δh‖ ≤ 1e-12: {nm} L{L} (fail-loud)")
            paths[(nm, L)] = {"dh": dh, "vhat": dh / norms[:, None],
                              "nlog": np.log(norms), "n": int(n_i - 1), "k": int(k_i)}
    if skipped:
        logline(f"build_paths: skip {len(skipped)} (k < 16 / T0.1 K=0): "
                + ", ".join(f"{s['doc']}L{s['L']}(T={s['T']},K={s['k']}{'; ' + s['reason'] if s.get('reason') else ''})" for s in skipped))
    return paths, skipped


# ---------------------------------------------------------------- метрики

def rho_metrics(x, k=None):
    """T-нормированный ACF k=1..k + r, f_DC, dip, plateau, sumK, bump. x (n, D).
    k по умолчанию = min(K, n−1); короткие доки — полосы ограничиваются доступными
    лагами (None, если лагов нет). Численно идентичен field.acf_moments(x, k) на
    первых k лагах (гейт S0)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:  # скалярный sim-путь (n,) → (n, 1)
        x = x[:, None]
    n = x.shape[0]
    k = min(K, n - 1) if k is None else k
    c0 = float(np.einsum("ij,ij->i", x, x).sum())
    if c0 <= 0:
        raise ValueError("rho_metrics: c(0) = 0")
    rho = np.empty(k)
    for kk in range(1, k + 1):
        rho[kk - 1] = float(np.einsum("ij,ij->i", x[:n - kk], x[kk:]).sum()) / c0
    fdc = field.f_dc(x)

    def band(lo, hi):
        hi = min(hi, k)
        return float(np.mean(rho[lo - 1:hi])) if hi >= lo else None

    return {"rho": rho, "k": k,
            "r": float(np.linalg.norm(x.sum(axis=0)) ** 2 / c0),
            "f_dc": fdc, "dip": float(rho[0]),
            "plat_lo": band(PLAT_LO[0], PLAT_LO[1]),
            "plat_hi": band(PLAT_HI[0], K),
            "sumK": float(rho.sum()),
            "bump": (float(0.5 * (rho[2] + rho[3]) - fdc) if k >= 4 else None)}


# ---------------------------------------------------------------- DC+MA(1)

def dcma1_model_rho(f, a, n, k):
    """Модельный ACF в безцентровом T-нормировочном согласии (поле.acf_moments):
    x_t = m + (ε_t − aε_{t−1}) ⇒ c(kk) ≈ (n−kk)(m² + γ_u(kk)), c0 ≈ n(m² + 1 + a²)
    ⇒ ρ̄(1) = (1−1/n)(f − (1−f)a/(1+a²)), ρ̄(kk≥2) = (1−kk/n)·f, f = m²/(m²+1+a²)."""
    kk = np.arange(1, k + 1)
    r = (1.0 - kk / n) * f
    r[0] = (1.0 - 1.0 / n) * (f - (1.0 - f) * a / (1.0 + a * a))
    return r


def _fit_step(f, a, rho, n, k):
    """1 шаг Gauss-Newton с backtracking. Возвращает (f, a, rss, сдвиг_ok)."""
    def resid(ff, aa):
        return dcma1_model_rho(ff, aa, n, k) - rho

    def rss_of(ff, aa):
        r = resid(ff, aa)
        return float((r * r).sum())

    r = resid(f, a)
    kk = np.arange(1, k + 1)
    J = np.zeros((k, 2))
    J[0, 0] = (1.0 - 1.0 / n) * (1.0 + a / (1.0 + a * a))
    J[0, 1] = (1.0 - 1.0 / n) * (-(1.0 - f) * (1.0 - a * a) / (1.0 + a * a) ** 2)
    J[1:, 0] = 1.0 - kk[1:] / n
    try:
        delta = np.linalg.solve(J.T @ J, -J.T @ r)
    except np.linalg.LinAlgError:
        return f, a, rss_of(f, a), False
    cur = rss_of(f, a)
    lam = 1.0
    for _ in range(8):
        fn, an = f + lam * delta[0], a + lam * delta[1]
        if 0.0 <= fn < 1.0 and abs(an) < 1.0 and rss_of(fn, an) <= cur:
            return fn, an, rss_of(fn, an), True
        lam *= 0.5
    return f, a, cur, False


def dcma1_fit(rho, n, k):
    """LS-фит (f, a) к ACF (k=1..k) в безцентровом T-согласии; n — длина ряда.
    f = mean_k ρ̂(k)/(1−k/n) (k=2..k); d ≡ f − ρ̂(1)/(1−1/n) = (1−f)a/(1+a²)
    ⇒ d·a² − (1−f)·a + d = 0 ⇒ a = ((1−f) − √((1−f)²−4d²))/(2d) (корень |a|<1).
    f ≤ 0: f заморожен = 0, a из того же closed-form с (1−f) = 1 (чистый MA(1))
    или a = 0 (ρ̂(1) ≥ 0); GN по f не выполняется (f_le_0)."""
    kk = np.arange(2, k + 1)
    f0 = float(np.mean(rho[1:] / (1.0 - kk / n)))
    f_le_0 = bool(f0 <= 0.0)
    dip_c = float(rho[0]) / (1.0 - 1.0 / n)   # ρ̂(1) в согласии модели
    if f_le_0:
        f = 0.0
        if dip_c >= 0.0:
            a = 0.0            # ρ̂(1) ≥ 0: iid (пре-рег)
            out_of_model = False
        else:
            d = -dip_c         # > 0: чистый MA(1), (1−f) = 1
            disc = 1.0 - 4.0 * d * d
            out_of_model = bool(disc <= 0.0)   # ≤ 0: корень — unit-корень/вне области
            a = (1.0 - float(np.sqrt(max(disc, 0.0)))) / (2.0 * d)
            if not np.isfinite(a) or abs(a) >= 1.0:
                a = 0.0
        m2 = 0.0
        rss = float(((dcma1_model_rho(f, a, n, k) - rho) ** 2).sum())
        r2 = float(1.0 - rss / max(float(((rho - rho.mean()) ** 2).sum()), 1e-30))
        return {"f": 0.0, "a": float(a), "m2": 0.0, "m": 0.0,
                "rss": rss, "f_le_0": True, "polish_ok": True,
                "gn_skipped": True, "out_of_model": out_of_model, "acf_r2": r2}
    f = min(max(f0, 0.0), 1.0 - 1e-9)
    g = 1.0 - f
    d = f - dip_c
    if abs(d) < 1e-12:
        a = 0.0
        out_of_model = False
    else:
        disc = g * g - 4.0 * d * d
        out_of_model = bool(disc <= 0.0)
        a = (g - float(np.sqrt(max(disc, 0.0)))) / (2.0 * d)
        if not np.isfinite(a) or abs(a) >= 1.0:
            a = 0.0  # за пределами области модели — fail-safe
    f, a, rss, ok = _fit_step(f, a, rho, n, k)
    # выход за (0≤f<1, |a|<1) исключён: backtracking принимает только доменные шаги,
    # начальные (f, a) доменные — fail-loud raise здесь недостижим (ревью р3)
    m2 = float(f * (1.0 + a * a) / (1.0 - f)) if f > 0.0 else 0.0
    r2 = float(1.0 - rss / max(float(((rho - rho.mean()) ** 2).sum()), 1e-30))
    return {"f": float(f), "a": float(a), "m2": m2, "m": float(np.sqrt(m2)),
            "rss": float(rss), "f_le_0": False, "polish_ok": bool(ok),
            "out_of_model": out_of_model, "acf_r2": r2}


def dcma1_fit_tail(rho, n, k, lo=33):
    """G8-v2 (15.11): fit ТОЛЬКО на хвостовой области k=lo..k. Модель DC+MA(1) при
    k ≥ 2 линейна по f (ρ(k) = f(1−k/n); член a — только лаг 1) ⇒ ЛС-решение по f
    closed-form, GN не нужен; a — closed-form из лаг-1 (реалистичность sim, на хвост
    не влияет). Обоснование: blk-null сохраняет внутри-блок структуру (k < b = 8) по
    построению — full-fit (k=2..k) вынужден её «впитывать» → неверный (f, a) → ложный
    sd_null (сбой v1: frac_cov 0.091, ratio 4.29). Хвост (k ≥ b): пары всегда в разных
    блоках ⇒ iid-внесение + DC-базовая ⇒ tail-fit корректен."""
    kk = np.arange(lo, k + 1)
    f0 = float(np.mean(rho[lo - 1:] / (1.0 - kk / n)))
    f_le_0 = bool(f0 <= 0.0)
    dip_c = float(rho[0]) / (1.0 - 1.0 / n)
    if f_le_0:
        f = 0.0
        if dip_c >= 0.0:
            a = 0.0
            out_of_model = False
        else:
            d = -dip_c
            disc = 1.0 - 4.0 * d * d
            out_of_model = bool(disc <= 0.0)
            a = (1.0 - float(np.sqrt(max(disc, 0.0)))) / (2.0 * d)
            if not np.isfinite(a) or abs(a) >= 1.0:
                a = 0.0
    else:
        f = min(max(f0, 0.0), 1.0 - 1e-9)
        g = 1.0 - f
        d = f - dip_c
        if abs(d) < 1e-12:
            a = 0.0
            out_of_model = False
        else:
            disc = g * g - 4.0 * d * d
            out_of_model = bool(disc <= 0.0)
            a = (g - float(np.sqrt(max(disc, 0.0)))) / (2.0 * d)
            if not np.isfinite(a) or abs(a) >= 1.0:
                a = 0.0
    tail = rho[lo - 1:]
    model_tail = dcma1_model_rho(f, a, n, k)[lo - 1:]
    rss = float(((model_tail - tail) ** 2).sum())
    r2 = float(1.0 - rss / max(float(((tail - tail.mean()) ** 2).sum()), 1e-30))
    return {"f": float(f), "a": float(a), "m2": 0.0, "m": 0.0, "rss": rss,
            "f_le_0": f_le_0, "polish_ok": True, "gn_skipped": True,
            "out_of_model": out_of_model, "acf_r2": r2, "lo": int(lo),
            "tail_fit": True}


def fit_scale(mvec, svec, f):
    """Масштаб D-null (ревью r4 C2, вариант A) + M1 edge (ревью r5): total power
    сохраняем (Σ(m′²+s′²) = Σ(m²+s²) = E), DC-доля = fit-нутый f (Σm′² = f·E),
    анизотропия сохраняется. M = 0 и f > 0: направление = svec/‖svec‖ (ось
    наибольшей дисперсии) — flag dc_dir (контракт «DC-доля = fit-нутый f» не
    нарушается); E = 0 — degenerate (flag). Возврат (m2, s2, dc_dir)."""
    M = float((mvec * mvec).sum())
    V = float((svec * svec).sum())
    E = M + V
    f = min(max(f, 0.0), 1.0 - 1e-9)
    Mp, Vp = f * E, (1.0 - f) * E
    if Mp <= 0.0:
        m2 = np.zeros_like(mvec)
        dc_dir = "none (f=0)"
    elif M > 0:
        m2 = mvec * np.sqrt(Mp / M)
        dc_dir = "mvec"
    elif V > 0:
        m2 = (svec / np.sqrt(V)) * np.sqrt(Mp)
        dc_dir = "svec (M=0: направление осн. дисперсии)"
    else:
        m2 = np.zeros_like(mvec)
        dc_dir = "degenerate (E=0)"
    s2 = svec * np.sqrt(Vp / V) if V > 0 else np.zeros_like(svec)
    return m2, s2, dc_dir


def dcma1_sim_d_fit(m2, s2, a, B, seed, n=N):
    """D-мерный null из FIT-НУТЫХ параметров (ревью р3 + р4 C2):
    x_{t,j} = m2_j + s2_j·u_{t,j}, u ~ MA(1) единичной дисперсии.
    m2/s2 — fit_scale (total power сохранён, DC-доля = fit-нутый f; M1 edge —
    flag dc_dir в вызывающем месте). Старая версия (real m/s as-is, f не
    передавался) давала f_DC_sim = f_DC_real BY CONSTRUCTION ⇒ покрытие плато
    было циркулярным; теперь plateau-покрытие тестирует генерацию DC fit-нутым f."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((B, n + 1, m2.size))
    u = (eps[:, 1:] - a * eps[:, :-1]) / np.sqrt(1.0 + a * a)
    return m2[None, None, :] + s2[None, None, :] * u


def fullcov_sim(dh, f, a, B, seed, n):
    """Гaussian null с ПОЛНОЙ ковариацией (ревью r5 C2, опция 2): y_t = m′ + L·u_t,
    u_t = (ε_t − aε_{t−1})/√(1+a²) — независимый по координатам MA(1), един. дисперсия;
    L Lᵀ = Σ_c — ковариация centred Δh (cross-dim ковариация сохранена; чего не
    хватает против real — не-гауссовость: её измеряет G8-v3 эмпирический пул).
    m′ — fit_scale (DC-доля = f). Возврат (y, how): how = 'chol'|'svd' (PSD-fallback)."""
    D = dh.shape[1]
    dhc = dh - dh.mean(axis=0)
    Sigma = (dhc.T @ dhc) / (n - 1)
    try:
        L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(D))
        how = "chol"
    except np.linalg.LinAlgError:
        U, sv, _ = np.linalg.svd(Sigma, full_matrices=False)
        L = U * np.sqrt(np.maximum(sv, 0.0))
        how = "svd"
    m2, s2, _ = fit_scale(dh.mean(axis=0), dh.std(axis=0), f)
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((B, n + 1, D))
    u = (eps[:, 1:] - a * eps[:, :-1]) / np.sqrt(1.0 + a * a)
    y = m2[None, None, :] + u @ L.T
    return y, how


def wilson_ci(c, n, z=1.959963985):
    """Wilson 95% CI для доли c/n (ревью r5 C1). n = 0 → None."""
    if n <= 0:
        return None
    p = c / n
    den = 1.0 + z * z / n
    ctr = (p + z * z / (2.0 * n)) / den
    half = z * float(np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / den
    return [float(ctr - half), float(ctr + half)]


def coverage(real_m, simmats):
    """real_m: dict метрик; simmats: (B,3) [dip, plat_lo, plat_hi].
    covered = все НЕ-None real ∈ [q_0.025, q_0.975] соответствующего sim-распределения."""
    out = {"covered": True}
    for j, key in enumerate(("dip", "plat_lo", "plat_hi")):
        if real_m[key] is None:
            continue   # короткая клетка: полосы нет — покрываем доступные
        lo, hi = np.quantile(simmats[:, j], (0.025, 0.975))
        out[f"{key}_q"] = [float(lo), float(hi)]
        if not (lo <= real_m[key] <= hi):
            out["covered"] = False
    return out


def _bn(v):
    """None → NaN (метрики с отсутствующей полосой в sim-матрицах)."""
    return float(v) if v is not None else float("nan")


def sim_mats(X, k=K):
    """X (B, N[, D]) — sim-пути; возвращает (mats (B,4) [dip, plat_lo, plat_hi,
    plat_hi_oos (k=49..min(96,k))], agg метрик). k — per-cell лаги.
    (G8-v3.1: fdcs удалён — нигде не используется; sample f_dc sim перегружен
    шумом среднего MA(1) ≈ V/n ≫ fit f и вычисление 2400×B мертво.)"""
    B = X.shape[0]
    mats = np.empty((B, 4))
    agg = {"r": [], "f_dc": [], "sumK": [], "bump": []}
    for b in range(B):
        m = rho_metrics(X[b], k)
        oos = float(np.mean(m["rho"][48:min(96, k)])) if k > 48 else float("nan")
        mats[b] = (_bn(m["dip"]), _bn(m["plat_lo"]), _bn(m["plat_hi"]), oos)
        for key in agg:
            agg[key].append(m[key])
    return mats, {key: float(np.mean(v)) for key, v in agg.items()}


# ---------------------------------------------------------------- VAR-лестница

def svd_scores(vhat, rgrid=R_GRID):
    """SVD per (doc,L); scores[r] = U[:, :r] · s[:r] (N, r)."""
    U, s, _ = np.linalg.svd(vhat, full_matrices=False)
    return {r: U[:, :r] * s[:r] for r in rgrid}


def var_fit(X, p):
    """LS-фит VAR(p): x_t = c + Σ_i A_i x_{t−i}. X (N, r).
    BIC (PRIMARY, ревью r4 C6) = n·ln det Σ + (p·r²+r)·ln n — гауссов likelihood,
    Σ = EᵀE/n, full df (A_i — r×r); bic_tz = n·ln(SSR/n) + (p·r+r)·ln n (TZ —
    sensitivity: trace-SSR теряет cross-output covariance, TZ-df недоучитывает A_i).
    n = N−p (число LS-выборок)."""
    n, r = X.shape
    nn = n - p
    reg = np.empty((nn, 1 + p * r))
    reg[:, 0] = 1.0
    for i in range(1, p + 1):
        # строка j ↔ t = p+j: регрессор x_{t−i} = X[p+j−i] ⇒ срез X[p−i : n−i]
        reg[:, 1 + (i - 1) * r:1 + i * r] = X[p - i:n - i]
    Y = X[p:]
    sol, *_ = np.linalg.lstsq(reg, Y, rcond=None)
    E = Y - reg @ sol
    ssr = float((E * E).sum())
    sign, logdet = np.linalg.slogdet((E.T @ E) / nn)
    bic = float(nn * logdet + (p * r * r + r) * float(np.log(nn))) if sign > 0 else None
    bic_tz = float(nn * float(np.log(ssr / nn)) + (p * r + r) * float(np.log(nn)))
    sst = float(((Y - Y.mean(axis=0)) ** 2).sum())
    return {"ssr": ssr, "nn": nn, "bic": bic, "bic_tz": bic_tz,
            "r2": float(1.0 - ssr / sst) if sst > 0 else None, "coef": sol}


def var_ladder(scores, rgrid=R_GRID, pgrid=P_GRID):
    """Таблица BIC {(r, p): fit} + selected (argmin primary BIC) + selected_tz.
    primary = logdet + full df (ревью r4 C6); TZ BIC — sensitivity (reportable)."""
    out = {}
    for r in rgrid:
        for p in pgrid:
            f = var_fit(scores[r], p)
            out[(r, p)] = f
    sel = min((k_ for k_ in out if out[k_]["bic"] is not None), key=lambda k_: out[k_]["bic"])
    sel_tz = min(out, key=lambda k_: out[k_]["bic_tz"])
    return out, sel, sel_tz


def ar1_fit(nlog):
    """AR(1) на n_t = log‖Δh‖: n_t = α + β n_{t−1} + ε (LS)."""
    y = nlog[1:]
    x = nlog[:-1]
    A = np.column_stack([np.ones_like(x), x])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ sol
    ssr = float((e * e).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {"beta": float(sol[1]), "alpha": float(sol[0]),
            "R2": float(1.0 - ssr / sst) if sst > 0 else None,
            "sd": float(np.sqrt(ssr / (len(y) - 2)))}


def gen_sim(r, p, sol, ar1, g0, B, seed, n=N):
    """Генеративные пути с selected (r,p): B × (n, r). VAR — нач. нули; AR(1) —
    ñ_0 = g0 (mean n_real), e ~ N(0, sd_res). Δh̃ = exp(ñ)·v̂̃."""
    rng = np.random.default_rng(seed)
    # sol-раскладка: строка 0 = intercept c; строки 1+(i−1)r..1+ir = коэффициенты x_{t−i}
    c = sol[0]
    As = [sol[1 + (i - 1) * r:1 + i * r, :] for i in range(1, p + 1)]
    X = np.zeros((B, n, r))
    Ev = rng.standard_normal((B, n - p, r))
    for t in range(p, n):
        X[:, t] = c + Ev[:, t - p]
        for i in range(1, p + 1):
            X[:, t] = X[:, t] + X[:, t - i] @ As[i - 1]
    g = np.empty((B, n))
    g[:, 0] = g0
    Eg = rng.standard_normal((B, n - 1)) * ar1["sd"]
    for t in range(1, n):
        g[:, t] = ar1["alpha"] + ar1["beta"] * g[:, t - 1] + Eg[:, t - 1]
    vhat = X / np.maximum(np.linalg.norm(X, axis=2, keepdims=True), 1e-12)  # направление (unit по r)
    return np.exp(g)[:, :, None] * vhat


def sim_metrics(dhs, k=K):
    """dhs (B, n, r) → (B,3) [dip, plat_lo, plat_hi] + агрегаты (r, f_dc, sumK, bump)."""
    return sim_mats(dhs, k)


# ---------------------------------------------------------------- negative control

def block_perm(n, b, seed):
    """Перестановка блоков длины b (хвост n−(n//b)·b остаётся на месте)."""
    rng = np.random.default_rng(seed)
    nb = n // b
    full = nb * b
    perm = np.arange(n)
    seg = perm[:full].reshape(nb, b)
    seg = seg[rng.permutation(nb)]
    out = np.empty(n, dtype=int)
    out[:full] = seg.reshape(-1)
    out[full:] = perm[full:]   # хвост на месте
    return out


def null_cell(idx, li, pth, scores, seed_i):
    """Часть 3: blk-shuffle (seed SEED_PERM_BASE + 25·idx + 5·li + seed_i — ревью r5 C3.2:
    уникален по (doc, layer, seed_i), НЕ paired design) → лестница + DC+MA(1) + покрытие.
    Лестница на null: scores[r][perm] — РЯДОКИ РЕАЛЬНЫХ PCA-проекций, переставленные
    тем же perm — корректный null: порядок времени разорван ⇒ временные корреляции
    уничтожены; базис r-проекций фиксирован real-данными (изоляция ordering от
    выбора базиса; пересчитывать SVD на переставленных рядах не нужно). АЦХ
    безцентровый: перестановка сохраняет mean ⇒ хвост null несёт DC-базовую
    (1−k/N)·f_DC (для Δh: f_dc ≈ 1e-5 — Δh почти чистый шум; «плато 0.05» —
    объект h_raw T0.1, не наш). Gate (а) — на ИЗБЫТОК сверх базовой («плоская кривая»
    = нет структуры сверх DC-уровня). Per-cell n/k; короткая клетка (k<33) — хвост None.
    G8-v3.1 (15.11): (а) масштаб = ЭМПИРИЧЕСКОЕ null-распределение (пул по клеткам):
    параметрическая DC+MA(1)-sim занижает флуктуацию null в 1.6–5.5× (реальные
    строки Δh не гауссовы по колонкам). (б) sim-СРЕДНЕЕ (TAIL-FIT: f на k=33..96,
    a из лаг-1) согласовано с null-статистиками RAW на эмпирическом масштабе.
    ВАЖНО (баг v3): sample f_dc sim (fdcs) перегружен шумом СРЕДНЕГО MA(1) по t
    (≈V/n ≈ 0.001 >> fit f ≈ 0.0004) ⇒ «excess» sim = plat_hi − fdcs(1−k̄/n)
    систематически отрицателен; base для sim-шума — КОНСТРУКЦИОННЫЙ fit f
    (x = m2 + s2·u: E[rho(k≥2)] = f(1−k/n) точно). Возврат: raw-статистики null
    (dip, plat_lo, plat_hi) + tail_excess (null-избыток, гейт а) + raw sim-средние
    (sim_dip, sim_plat_lo, sim_plat_hi) + sd_null (sim-шум, construction-based)."""
    n, k = pth["n"], pth["k"]
    perm = block_perm(n, BLK, SEED_PERM_BASE + 25 * idx + 5 * li + seed_i)
    dh_s = pth["dh"][perm]
    m = rho_metrics(dh_s, k)
    fdc_s = field.f_dc(dh_s)
    if k >= PLAT_HI[0]:
        kbar_hi = float(np.mean(np.arange(PLAT_HI[0], min(PLAT_HI[1], k) + 1)))
        tail_excess = m["plat_hi"] - fdc_s * (1.0 - kbar_hi / n)
    else:
        kbar_hi, tail_excess = None, None
    # G8-v2/v3: sim от TAIL-FIT (f на k=33..min(96,k) — область части 1; a из
    # лаг-1, на хвост не влияет); full-fit — reportable (внутри-блок k<8 сохранена).
    fit = dcma1_fit_tail(m["rho"], n, k, lo=PLAT_HI[0])
    fit_full = dcma1_fit(m["rho"], n, k)
    # D-мерный null из FIT-НУТЫХ (f, a) (ревью р3 + р4 C2): m/s per-dim из dh_s,
    # total power сохранён, DC-доля = fit-нутый f ⇒ sim в масштабе реального D.
    m2, s2, dc_dir = fit_scale(dh_s.mean(axis=0), dh_s.std(axis=0), fit["f"])
    X = dcma1_sim_d_fit(m2, s2, fit["a"], B1,
                        SEEDN_OFF + NSEED * (5 * idx + li) + seed_i, n=n)   # r5 C3.2: +layer
    mats, _ = sim_mats(X, k)
    cov = coverage(m, mats)
    # sim-шум хвоста (G8-v3.1: base = КОНСТРУКЦИОННЫЙ f, не sample fdcs):
    if kbar_hi is not None and np.isfinite(mats[:, 2]).all():
        sd_null = float(np.std(mats[:, 2] - fit["f"] * (1.0 - kbar_hi / n)))
    else:
        sd_null = None
    # raw sim-средние (G8-v3.1 b): сравнение с raw null-статистиками
    sim_dip = float(np.mean(mats[:, 0])) if np.isfinite(mats[:, 0]).all() else None
    sim_plat_lo = float(np.mean(mats[:, 1])) if np.isfinite(mats[:, 1]).all() else None
    sim_plat_hi = float(np.mean(mats[:, 2])) if np.isfinite(mats[:, 2]).all() else None
    ladder, sel, _ = var_ladder({r: scores[r][perm] for r in R_GRID})
    return {"sel": list(sel), "sel_p": int(sel[1]),
            "cov": cov["covered"], "fit_f": fit["f"], "fit_a": fit["a"],
            "fit_f_full": fit_full["f"], "fit_a_full": fit_full["a"],
            "dc_dir": dc_dir,
            "dip": m["dip"], "plat_lo": m["plat_lo"], "plat_hi": m["plat_hi"],
            "tail_excess": float(tail_excess) if tail_excess is not None else None,
            "sim_dip": sim_dip, "sim_plat_lo": sim_plat_lo, "sim_plat_hi": sim_plat_hi,
            "fdc": float(fdc_s),
            "sd_null": sd_null}


def calibrate_null_on_null(cal_cells, paths, cells, idx_of, layers):
    """Null-on-null калибровка (ревью r4 C4, reportable — не гейт): для клетки 2
    независимые партии B=200 sim-путей из fit-нутых (f, a): партия A → квантили
    [q2.5, q97.5], партия B → marginal/joint покрытие. Marginal ≈ 0.95 ⇒ CI
    калиброван; joint — эмпирический (метрики коррелированы, НЕ 0.95³). Если
    marginal ≪ 0.95 — CI слишком узок (B1 / d_eff) — интерпретация вердикта с
    оговоркой. Subset: первые K96 non-degen доки per layer (детерминированно)."""
    out = {}
    for (nm, L) in cal_cells:
        pth = paths[(nm, L)]
        fit = cells[(nm, L)]["part1"]["fit"]
        n, k = pth["n"], pth["k"]
        m2, s2, _ = fit_scale(pth["dh"].mean(axis=0), pth["dh"].std(axis=0), fit["f"])
        li = layers.index(L)
        # r5 C3.1: seed = OFF + 5·idx + li — уникален (старый idx+li давал коллизии:
        # (idx=1,li=0) == (idx=0,li=1))
        Xa = dcma1_sim_d_fit(m2, s2, fit["a"], B1,
                             SEED_CAL_A + 5 * idx_of[nm] + li, n=n)
        Xb = dcma1_sim_d_fit(m2, s2, fit["a"], B1,
                             SEED_CAL_B + 5 * idx_of[nm] + li, n=n)
        ma = sim_mats(Xa, k)[0]
        mb = sim_mats(Xb, k)[0]
        marg, inband = [], []
        for j in range(3):
            lo, hi = np.quantile(ma[:, j], (0.025, 0.975))
            marg.append(float(np.mean((mb[:, j] >= lo) & (mb[:, j] <= hi))))
            inband.append((mb[:, j] >= lo) & (mb[:, j] <= hi))
        joint = float(np.mean(np.prod(np.stack(inband, axis=1), axis=1)))
        out[f"{nm}|L{L}"] = {"marginal": marg, "joint": joint,
                             "f_fit": float(fit["f"]), "a_fit": float(fit["a"]),
                             "d_eff_diag": float(cells[(nm, L)]["real"]["d_eff"]),
                             "d_eff_full": float(cells[(nm, L)]["real"]["d_eff_full"])}
    return out


# ---------------------------------------------------------------- гейты

def gate_s0():
    """S0: rho_metrics ≡ field.acf_moments (exec на 2 матрицах, точность 1e-14)."""
    rng = np.random.default_rng(0)
    maxdev = 0.0
    for shape in ((N, 32), (N, 1)):
        x = rng.standard_normal(shape)
        a = field.acf_moments(x, K)
        b = rho_metrics(x)
        dev = max(float(np.max(np.abs(a["rho"] - b["rho"]))),
                  abs(a["r"] - b["r"]), abs(field.f_dc(x) - b["f_dc"]))
        assert dev <= 1e-14, f"S0: rho_metrics vs field dev={dev} shape={shape}"
        maxdev = max(maxdev, dev)
    return {"ok": True, "max_dev": float(maxdev)}


def gate_s2():
    """S2: iid-нуль DC+MA(1) (f=0,a=0): mean ρ̂(1) = 0 ± 3·SE.
    UNCENTERED iid: E[ρ̂(k)] = 0 ТОЧНО (симметрия по знаку одной координаты: flip x_1
    ⇒ E[x_1·x_2/D] = 0 ⇒ индуктивно все лаги); −1/(N−1) — target DEMEANED ACF
    (оценка среднего), не наше безцентровое соглашение (ревью r4 C1).
    D13: «срыв на 2.81 SE» при старом target был артефактом wrong target — тот же
    seed 42, B=2000: mean = +0.000593 = 0.52 SE от 0 (empirically проверено).
    Tolerance 3·SE сохранён как консервативный запас."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((BNULL, N))
    rho1 = np.array([rho_metrics(x)["dip"] for x in X])
    mean, sd = float(rho1.mean()), float(rho1.std())
    target = 0.0
    se = sd / float(np.sqrt(BNULL))
    ok = abs(mean - target) <= 3.0 * se
    return {"mean": mean, "sd": sd, "target": target, "se": se, "ok": bool(ok)}


def gate_s3():
    """S3: AR(1) φ=0.5: mean ρ̂(1) ∈ [0.3, 0.6]."""
    rng = np.random.default_rng(43)
    phi = 0.5
    X = np.empty((BNULL, N))
    X[:, 0] = rng.standard_normal(BNULL) * (1.0 / float(np.sqrt(1.0 - phi * phi)))
    e = rng.standard_normal((BNULL, N - 1))
    for t in range(1, N):
        X[:, t] = phi * X[:, t - 1] + e[:, t - 1]
    rho1 = np.array([rho_metrics(x)["dip"] for x in X])
    return {"mean": float(rho1.mean()), "ok": bool(0.3 <= float(rho1.mean()) <= 0.6)}


def gate_s4():
    """S4: period-2: ρ(k) = (−1)^k. Чётный N: r = 0 точно (TZ §1.7);
    нечётный N = 383: Σx = +1 ⇒ r = 1/N точно (краевой член, свойство)."""
    def r_of(n):
        x = np.zeros((n, 1))
        x[:, 0] = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        return field.acf_moments(x, min(K, n - 1))["r"]
    r_even = r_of(384)
    r_odd = r_of(N)
    ok = abs(r_even) < 1e-12 and abs(r_odd - 1.0 / N) < 1e-12
    return {"r_even384": float(r_even), "r_odd383": float(r_odd),
            "expect_odd": float(1.0 / N), "ok": bool(ok)}


def oos_fit(rho, n):
    """OOS-вариант (reportable): f из k=2..48, a из дипа; тест на plat_hi_oos (k=49..96).
    Нормировка (1−k/n) — PER-CELL n (ревью r4: старая версия брала глобальный N=383 —
    неверно для доков с 96 ≤ n < 383). Дип НЕ входит в тест (a из ρ̂(1) ⇒ циркулярно)."""
    kk = np.arange(2, 49)
    f = float(np.mean(rho[1:48] / (1.0 - kk / n)))
    dip_c = float(rho[0]) / (1.0 - 1.0 / n)
    out_of_model = False
    if f <= 0.0:
        if dip_c >= 0.0:
            a = 0.0
        else:
            d = -dip_c
            disc = 1.0 - 4.0 * d * d
            out_of_model = bool(disc <= 0.0)
            a = (1.0 - float(np.sqrt(max(disc, 0.0)))) / (2.0 * d)
            if not np.isfinite(a) or abs(a) >= 1.0:
                a = 0.0
        f = 0.0
    else:
        g = 1.0 - min(max(f, 0.0), 1.0 - 1e-9)
        d = f - dip_c
        if abs(d) < 1e-12:
            a = 0.0
        else:
            disc = g * g - 4.0 * d * d
            out_of_model = bool(disc <= 0.0)
            a = (g - float(np.sqrt(max(disc, 0.0)))) / (2.0 * d)
            if not np.isfinite(a) or abs(a) >= 1.0:
                a = 0.0
    return {"f": float(f), "a": float(a), "out_of_model": out_of_model}


def gate_s5(dh):
    """S5: Fourier-суррогат (positive control): круговой ACF ≤ 1e-6;
    линейный wrap — отчёт («свойство, не баг», урок AT0_2). Per-cell n/k (M3)."""
    n = dh.shape[0]
    k = min(K, n - 1)
    rng = np.random.default_rng(7)
    F = np.fft.fft(dh, axis=0)          # (n, D): ось 0 — частотный бином j
    mag = np.abs(F)
    D = dh.shape[1]
    n_pairs = (n - 1) // 2              # сопряжённые пары (j, n−j), j = 1..n_pairs
    theta = rng.uniform(0.0, 2.0 * np.pi, (n_pairs, D))
    ph2 = np.zeros((n, D))
    ph2[1:n_pairs + 1, :] = theta
    ph2[n - n_pairs:n, :] = -theta[::-1, :]
    X = np.fft.ifft(mag * np.exp(1j * ph2), axis=0).real   # сопряж. симметрия ⇒ вещественный
    # круговой ACF: напрямую по X и по dh (не тождество IDFT(|F|²) на самом dh)
    circ_real = np.empty(k)
    circ_surr = np.empty(k)
    for kk in range(1, k + 1):
        circ_real[kk - 1] = float(np.einsum("ij,ij->i", dh, np.roll(dh, -kk, axis=0)).sum())
        circ_surr[kk - 1] = float(np.einsum("ij,ij->i", X, np.roll(X, -kk, axis=0)).sum())
    c0 = float((dh * dh).sum())
    dev_circ = float(np.max(np.abs(circ_surr - circ_real)) / c0)
    # линейный ACF: wrap-around хвост (свойство, не баг)
    dev_lin = float(np.max(np.abs(rho_metrics(dh, k)["rho"] - rho_metrics(X, k)["rho"])))
    return {"max_dev_circ": dev_circ, "ok": bool(dev_circ <= 1e-6),
            "max_dev_lin_wrap": dev_lin,
            "note": "круговой ACF — точный positive control; линейный — wrap-around, свойство"}


def gate_s8(dh):
    """S8: Σρ̂_full перм-инвариантность (5 сидов, spread < 1e-12). Per-cell n (M3)."""
    n = dh.shape[0]
    base = field.acf_moments(dh, n - 1)["sum_rho_full"]
    rng = np.random.default_rng(99)
    vals = [base]
    for _ in range(5):
        vals.append(field.acf_moments(dh[rng.permutation(n)], n - 1)["sum_rho_full"])
    spread = max(vals) - min(vals)
    return {"spread": float(spread), "ok": bool(spread < 1e-12)}


def gate_s9(real):
    """S9 (якорь): ACF Δh == T0.1 store (≤1e-12) + K/N-сверка (ревью r4 C5.1):
    N_store == n; K-алфавит T0.1 {0,24,32,96}: K=96 ⇔ k=96, K=32 ⇔ 32≤k≤95,
    K=24 ⇔ 24≤k≤31, K=0 ⇒ клетка не должна присутствовать. absent в T0.1 — отчёт.
    Пин — на STORE (данные, бит-в-бит к TZ-шине 23281c64); claim-файл T0_1 сессией
    перезаписывался (коммит 6c1cdf3, r5-прогон) — его sha только отчитывается."""
    fsha = sha256_file(T01_FILE)
    ssha = sha256_file(T01_STORE)
    assert ssha == T01_STORE_SHA, f"S9: T0 store sha {ssha[:16]} ≠ закреплённый {T01_STORE_SHA[:16]}"
    per = json.load(open(T01_STORE))["per_doc"]
    max_dev, n_chk, absent = 0.0, 0, []
    kbad, nbad = [], []
    for (nm, L), rm in real.items():
        m = per[nm]["layers"][str(L)]["h_raw"]["mean"]
        rho = rm["rho"]
        ref = (m or {}).get("rho") or []
        if len(ref) == 0:
            absent.append(f"{nm}|L{L}")
        else:
            L_min = min(len(rho), len(ref))
            max_dev = max(max_dev, float(np.max(np.abs(rho[:L_min] - np.asarray(ref[:L_min], dtype=np.float64)))))
            n_chk += 1
        Ks, Ns = m.get("K"), m.get("N")
        if Ns is not None and Ns != rm["n"]:
            nbad.append(f"{nm}|L{L}: N_store={Ns} n={rm['n']}")
        k = rm["k"]
        if Ks == 0:
            kbad.append(f"{nm}|L{L}: K_store=0, но клетка присутствует")
        elif Ks == 96 and not (k == K):
            kbad.append(f"{nm}|L{L}: K_store=96, k={k}")
        elif Ks == 32 and not (32 <= k <= 95):
            kbad.append(f"{nm}|L{L}: K_store=32, k={k}")
        elif Ks == 24 and not (24 <= k <= 31):
            kbad.append(f"{nm}|L{L}: K_store=24, k={k}")
    ok = max_dev <= 1e-12 and not kbad and not nbad
    return {"n_checked": n_chk, "absent_T01": absent, "max_dev": float(max_dev),
            "k_mismatch": kbad, "n_mismatch": nbad,
            "file_sha256": fsha,
            "file_sha_note": "claim-файл обновлён AT0_1 r5 (коммит 6c1cdf3); пин — на store",
            "store_sha256": ssha, "ok": bool(ok)}


# ---------------------------------------------------------------- main

def main():
    global B1, B2, BNULL, NSEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    t_start_iso = datetime.now().isoformat()
    logline(f"AT0_5 start {t_start_iso} py={sys.version.split()[0]} "
            f"np={np.__version__} smoke={args.smoke}")

    # trackio (AGENTS: обязателен для CPU-прогонов). smoke — тест, не эксперимент.
    tr = None
    if not args.smoke:
        tr_cfg = {
            "started_at": datetime.now().isoformat(),
            "revision": REVISION,
            "script_sha": sha256_file(os.path.abspath(__file__)),
            "seed": (f"multi: SEED1/2/N={SEED1_OFF}/{SEED2_OFF}/{SEEDN_OFF}, "
                     f"OOS={SEED_OOS_OFF}, CAL_A/B={SEED_CAL_A}/{SEED_CAL_B}, "
                     f"FCOV={SEED_FCOV_OFF}, PERM={SEED_PERM_BASE}; формулы per-cell в хедере"),
            "device": "CPU", "model": "AT0_5", "log": os.path.abspath(RUN_LOG),
            "K": K, "N": N, "B1": B1, "B2": B2, "BLK": BLK, "NSEED": NSEED,
        }
        tr = trackio_start(f"at05_gencheck_{REVISION}", tr_cfg)
        tlog(tr, 0, {"started": 1.0})
        logline(f"trackio: run at05_gencheck_{REVISION} init ok")
        # краш — НЕ молча (AGENTS): atexit-страховка — если финиш-точка не была
        # записана (unhandled exception / kill) — status=crash + finish()
        import atexit
        tr_done = [False]

        def _trackio_on_exit():
            if not tr_done[0]:
                try:
                    tlog(tr, time.time() - t0, {"finished_at": datetime.now().isoformat(),
                                                 "status": "crash",
                                                 "error": "unhandled exception / abort (fin-точка не записана)"})
                    tr.finish()
                except Exception:
                    pass
        atexit.register(_trackio_on_exit)

    # G1′: field.py sha-пин + selfcheck
    fsha = sha256_file(os.path.join(ROOT, "prism/field.py"))
    assert fsha.startswith(FIELD_PIN), f"G1′: field.py sha {fsha[:16]} ≠ пин {FIELD_PIN}"
    field._selfcheck()
    logline(f"G1′: field.py sha {fsha[:16]} + _selfcheck ok")

    # G0: данные
    data = {p: sha256_file(p) for p in (GEN, OWN, COR)}
    gen_meta = json.load(open(GEN.replace(".npz", "_meta.json")))
    gen_sha_ok = int(sum(1 for v in gen_meta.values() if v.get("sha_ok")))
    z = np.load(GEN, allow_pickle=True)
    zo = np.load(OWN, allow_pickle=True)
    probe_docs = ["chat_0", "math_0", "code_0"]   # TZ §1.4: 3 строки gen-doc
    decs = {nm: a1.decode_probe(np.asarray(z[f"{nm}_tok"]), nm) for nm in probe_docs}
    dec_own = a1.decode_probe(np.asarray(zo["own_w0_tok"]), "own_w0")  # + 1 own-doc
    logline(f"G0: gen sha {data[GEN][:16]} gen_sha_ok {gen_sha_ok}/64; "
            + "; ".join(f"decode_probe[{nm}]={decs[nm]['lines'][0]['text'][:40]!r}"
                        for nm in probe_docs)
            + f"; decode_probe[own_w0]={dec_own['lines'][0]['text'][:40]!r}")
    cor_z = np.load(COR, allow_pickle=True)
    n_cor_toks = sum(1 for f in cor_z.files if f.endswith("_tok") and cor_z[f].size > 0)
    logline(f"G0 sources: corpus токенов {n_cor_toks}/16 → "
            f"{'ok' if n_cor_toks else 'not_testable (G16, унаследовано)'}")
    assert gen_sha_ok == 64, "G0: gen_sha_ok < 64"

    docs = load_docs()
    assert len(docs) == 96, f"G0: n_docs = {len(docs)} ≠ 96"

    # degen-флаг T0.1 (вердикт-пул = non-degen)
    t01_per = json.load(open(T01_STORE))["per_doc"]
    degen = {nm: bool(t01_per[nm]["degen_tz"]) for nm in docs}
    n_nd = sum(1 for v in degen.values() if not v)
    logline(f"degen (T0.1 флаг): {96 - n_nd}/96; non-degen пул: {n_nd}")

    smoke = args.smoke
    if smoke:
        B1, B2, BNULL, NSEED = 20, 5, 200, 2

    if smoke:
        docnames = sorted(d for d in docs if d.startswith("chat_"))[:2]
        layers = (23,)
    else:
        docnames = sorted(docs)
        layers = LAYERS

    paths, skipped = build_paths({nm: docs[nm] for nm in docnames}, layers, t01_per)
    idx_of = {nm: i for i, nm in enumerate(docnames)}
    logline(f"paths: {len(paths)} клеток из {len(docnames) * len(layers)} "
            f"(skip {len(skipped)}: k < 16 / T0.1 K=0; S1 телескоп — assert в build_paths)")

    # real-метрики per cell (field.acf_moments — S6/S7/S9 источник; per-cell k_i)
    real = {}
    for (nm, L), pth in paths.items():
        k = pth["k"]
        am = field.acf_moments(pth["dh"], k)
        fdc = field.f_dc(pth["dh"])
        rho = am["rho"]
        dhc = pth["dh"] - pth["dh"].mean(axis=0)
        # d_eff_diag (ревью r4 C4): participation ratio per-dim дисперсий (диагональ)
        s2 = (dhc ** 2).mean(axis=0)
        d_eff = float((s2.sum()) ** 2 / float((s2 * s2).sum()))
        # d_eff_full (ревью r5 C2): (trΣ)²/‖Σ‖_F² — ПОЛНАЯ ковариация (cross-dim включён);
        # при сильных cross-dim корреляциях d_eff_full ≪ d_eff_diag
        Sigma = (dhc.T @ dhc) / (pth["n"] - 1)
        trS = float(Sigma.trace())
        froS = float((Sigma * Sigma).sum())
        d_eff_full = float(trS * trS / froS) if froS > 0 else 0.0
        real[(nm, L)] = {"rho": rho, "k": k, "n": pth["n"],
                         "r": am["r"], "f_dc": fdc,
                         "gram_resid": am["gram_resid"],
                         "dip": float(rho[0]),
                         "plat_lo": float(np.mean(rho[PLAT_LO[0] - 1:PLAT_LO[1]])) if k >= PLAT_LO[1] else None,
                         "plat_hi": float(np.mean(rho[PLAT_HI[0] - 1:k])) if k >= PLAT_HI[0] else None,
                         "sumK": float(rho.sum()),
                         "d_eff": d_eff,
                         "d_eff_full": d_eff_full,
                         "bump": float(0.5 * (rho[2] + rho[3]) - fdc) if k >= 4 else None}
    logline(f"real-метрики: {len(real)} клеток")
    tstage(tr, time.time() - t0, "real_metrics")

    gates = {"G1_prime": {"field_sha256": fsha, "ok": True},
             "G0": {"data_sha256": {k: v for k, v in data.items()},
                    "gen_sha_ok": f"{gen_sha_ok}/64",
                    "decode_probes": {nm: decs[nm]["lines"][0]["text"][:60]
                                      for nm in probe_docs},
                    "decode_probe_own": dec_own["lines"][0]["text"][:60],
                    "G16_corpus": "not_testable (0/16 tok)" if n_cor_toks == 0 else "ok",
                    "ok": True},
             "S0": gate_s0(),
             "S2": gate_s2(),
             "S3": gate_s3(),
             "S4": gate_s4()}
    s5_key = (docnames[0], 23) if not smoke else (docnames[0], layers[0])
    gates["S5"] = gate_s5(paths[s5_key]["dh"])
    gates["S8"] = gate_s8(paths[s5_key]["dh"])
    gates["S9"] = gate_s9(real)
    # S6/S7 — по всем клеткам
    g6_max = max(v["gram_resid"] for v in real.values())
    g7_max = max(abs(v["r"] - v["n"] * v["f_dc"]) for v in real.values())
    gates["S6"] = {"max_gram_resid": float(g6_max), "ok": bool(g6_max < 1e-10)}
    gates["S7"] = {"max_dev_r_NfDC": float(g7_max), "ok": bool(g7_max <= 1e-12)}
    for g, key in (("S0", "ok"), ("S2", "ok"), ("S3", "ok"), ("S4", "ok"),
                   ("S5", "ok"), ("S8", "ok"), ("S9", "ok")):
        assert gates[g][key], f"гейт {g} FAILED: {gates[g]}"
    assert gates["S6"]["ok"] and gates["S7"]["ok"], f"S6/S7 FAILED: {gates['S6']} {gates['S7']}"
    logline(f"S0 ok; S2 iid: mean={gates['S2']['mean']:.6f} target={gates['S2']['target']:.6f} "
            f"ok={gates['S2']['ok']}; S3 AR(1): mean={gates['S3']['mean']:.4f} ok={gates['S3']['ok']}")
    logline(f"S4 period-2: r_even384={gates['S4']['r_even384']:.2e} r_odd383={gates['S4']['r_odd383']:.2e} "
            f"(=1/N={gates['S4']['expect_odd']:.2e}); S5 circ dev={gates['S5']['max_dev_circ']:.2e} "
            f"(lin wrap {gates['S5']['max_dev_lin_wrap']:.2e}); S8 spread={gates['S8']['spread']:.2e}")
    logline(f"S9 якорь T0.1: {gates['S9']['n_checked']} клеток, max_dev={gates['S9']['max_dev']:.2e}, "
            f"absent_T01={gates['S9']['absent_T01']}, k_mismatch={gates['S9']['k_mismatch']}, "
            f"n_mismatch={gates['S9']['n_mismatch']}")
    tstage(tr, time.time() - t0, "gates")

    # ЧАСТЬ 1 + ЧАСТЬ 2 + ЧАСТЬ 3
    cells = {}
    scores_cache = {}
    for (nm, L), pth in paths.items():
        idx = idx_of[nm]
        li = layers.index(L)   # слои в seed: 100000 + 5·doc_idx + layer_idx (пре-рег)
        rm = real[(nm, L)]
        k, n = rm["k"], rm["n"]
        # Часть 1
        fit = dcma1_fit(rm["rho"], n, k)
        mvec = pth["dh"].mean(axis=0)
        svec = pth["dh"].std(axis=0)
        m2, s2, dc_dir = fit_scale(mvec, svec, fit["f"])   # r5 M1: M=0 edge — flag
        # D-мерный null из FIT-НУТЫХ (f, a) (ревью р3 + р4 C2): total power сохранён,
        # DC-доля = fit-нутый f — plateau-покрытие не циркулярно
        X = dcma1_sim_d_fit(m2, s2, fit["a"],
                            B1, SEED1_OFF + NSEED * idx + li, n=n)
        mats, simagg = sim_mats(X, k)
        cov = coverage(rm, mats)
        sim3 = {key: (float(np.mean(mats[:, j])) if np.isfinite(mats[:, j]).all() else None)
                for j, key in enumerate(("dip", "plat_lo", "plat_hi"))}
        # OOS-вариант (reportable): fit k=2..48 → тест plat_hi_oos (k=49..96);
        # только для полных клеток (k_i = 96); дип НЕ тестируется OOS (a из ρ̂(1) ⇒ циркулярно)
        if k == K:
            fit_oos = oos_fit(rm["rho"], n)
            m2o, s2o, _ = fit_scale(mvec, svec, fit_oos["f"])
            Xo = dcma1_sim_d_fit(m2o, s2o, fit_oos["a"],
                                 B1, SEED_OOS_OFF + NSEED * idx + li, n=n)
            mats_o = sim_mats(Xo, k)[0]
            real_oos = float(np.mean(rm["rho"][48:96]))
            lo, hi = np.quantile(mats_o[:, 3], (0.025, 0.975))
            cov_oos = {"covered": bool(lo <= real_oos <= hi), "fit_oos": fit_oos,
                       "real_plat_hi_oos": real_oos, "plat_hi_oos_q": [float(lo), float(hi)]}
        else:
            cov_oos = None   # короткая клетка: OOS-полосы нет (reportable)
        # Часть 2
        scores = svd_scores(pth["vhat"])
        scores_cache[(nm, L)] = scores
        ladder, sel, sel_tz = var_ladder(scores)
        ar1 = ar1_fit(pth["nlog"])
        gensim = None
        if abs(ar1["beta"]) >= 1.0:
            gensim = {"skipped": "ar1_unstable", "beta": ar1["beta"]}
        else:
            sol = ladder[sel]["coef"]
            dhs = gen_sim(sel[0], sel[1], sol, ar1, float(pth["nlog"].mean()),
                          B2, SEED2_OFF + NSEED * idx + li, n=n)
            gm, gagg = sim_metrics(dhs, k)
            real3 = [rm["dip"], rm["plat_lo"], rm["plat_hi"]]
            gensim = {"sim_real_gap": {key: (float(np.mean(np.abs(gm[:, j] - real3[j])))
                                            if real3[j] is not None and np.isfinite(gm[:, j]).all()
                                            else None)
                                       for j, key in enumerate(("dip", "plat_lo", "plat_hi"))},
                      "agg": gagg,
                      "r_real_vs_sim": float(rm["r"] - gagg["r"]),
                      "f_dc_real_vs_sim": float(rm["f_dc"] - gagg["f_dc"])}
        cells[(nm, L)] = {
            "real": {key: rm[key] for key in ("r", "f_dc", "dip", "plat_lo", "plat_hi", "sumK", "d_eff", "d_eff_full", "bump")},
            "k": k, "n": n,
            "part1": {"fit": fit, "cov": cov, "simagg": simagg, "sim3_mean": sim3,
                      "dc_dir": dc_dir,
                      "f_fit_vs_real": float(fit["f"] - rm["f_dc"]),
                      "gap": {key: (float(rm[key] - sim3[key])
                                    if rm[key] is not None and sim3[key] is not None else None)
                              for key in ("dip", "plat_lo", "plat_hi")},
                      "oos": cov_oos},
            "part2": {"selected": list(sel),
                      "selected_tz": list(sel_tz),
                      "bic": {f"r{r}p{p}": ladder[(r, p)]["bic"] for r in R_GRID for p in P_GRID},
                      "bic_tz": {f"r{r}p{p}": ladder[(r, p)]["bic_tz"]
                                 for r in R_GRID for p in P_GRID},
                      "r2": {f"r{r}p{p}": ladder[(r, p)]["r2"] for r in R_GRID for p in P_GRID},
                      "ar1": {k: ar1[k] for k in ("beta", "alpha", "R2", "sd")},
                      "gensim": gensim},
        }
    logline(f"клетки: {len(cells)} (часть 1+2)")
    tstage(tr, time.time() - t0, "cells_part12", {"n_cells": len(cells)})

    # Часть 3: negative control
    nulls = []
    for (nm, L), pth in paths.items():
        idx = idx_of[nm]
        li = layers.index(L)
        for si in range(NSEED):
            nulls.append(null_cell(idx, li, pth, scores_cache[(nm, L)], si))
    n_null = len(nulls)
    nulls_full = [c for c in nulls if c["tail_excess"] is not None]
    n_short = n_null - len(nulls_full)
    frac_cov = float(np.mean([c["cov"] for c in nulls]))
    # G8-v3 (а): median |tail_excess| ≤ 3·sd ЭМПИРИЧЕСКОГО null-распределения
    # (пул по клеткам). Параметрическая sim занижает масштаб флуктуации (1.6–5.5×),
    # поэтому sd_null (sim) — только reportable-диагностика калибровки.
    tails = [c["tail_excess"] for c in nulls_full]
    sd_pool = float(np.std(tails)) if tails else None
    med_ratio = (float(np.median(np.abs(tails)) / max(sd_pool, 1e-12))
                 if (tails and sd_pool is not None) else None)
    med_fdc_null = float(np.median([c["fdc"] for c in nulls]))
    sds = [c["sd_null"] for c in nulls_full if c["sd_null"] is not None]
    med_sd_null = float(np.median(sds)) if sds else None
    # G8-v3.1 (б): sim-СРЕДНЕЕ (raw) согласовано с null-статистиками (raw) на
    # эмпирическом масштабе: frac клеток с |null − sim_mean| ≤ 3·sd_pool_stat ≥ G8_COV_MIN.
    # (без base-вычитания: sample f_dc sim перегружен шумом среднего MA(1) — баг v3)
    g8b = {}
    for key, skey in (("dip", "sim_dip"), ("plat_lo", "sim_plat_lo"),
                      ("plat_hi", "sim_plat_hi")):
        pool = [c[key] for c in nulls if c.get(key) is not None]
        okc = [c for c in nulls if c.get(key) is not None and c.get(skey) is not None]
        if not pool or not okc:
            continue
        sd_p = float(np.std(pool))
        gaps = [abs(c[key] - c[skey]) for c in okc]
        g8b[key] = {"frac_within_3sd_emp": float(np.mean([g <= 3 * max(sd_p, 1e-12) for g in gaps])),
                    "sd_pool": sd_p, "median_abs_gap": float(np.median(gaps))}
    g8b_ok = bool(g8b) and all(v["frac_within_3sd_emp"] >= G8_COV_MIN for v in g8b.values())
    sel_dist_null = {}
    for c in nulls:
        key = f"r{c['sel'][0]}p{c['sel'][1]}"
        sel_dist_null[key] = sel_dist_null.get(key, 0) + 1
    g8_ok = (med_ratio is not None and med_ratio <= G8_TAIL_SIG) and g8b_ok
    gates["G8_null"] = {"n_cells": n_null, "n_short_k_lt_33": int(n_short),
                        "median_ratio_tail_over_sd_emp": med_ratio,
                        "sd_pool_tail": sd_pool,
                        "median_sd_null_sim": med_sd_null,
                        "median_fdc_null": med_fdc_null,
                        "g8b_sim_central": g8b,
                        "frac_dcma1_cov_raw": frac_cov,
                        "selected_dist_null": dict(sorted(sel_dist_null.items())),
                        "require": (f"G8-v3.1: (а) median |tail_excess|/sd_emp ≤ {G8_TAIL_SIG} — sd_emp = std ЭМПИРИЧЕСКОГО "
                                     f"null-распределения (пул по клеткам; параметрическая DC+MA(1)-sim занижает флуктуацию "
                                     f"null в 1.6–5.5× — реальные строки Δh не гауссовы по колонкам; sim sd_null — "
                                     f"reportable); (б) sim-СРЕДНЕЕ (TAIL-FIT: f на k=33..96, a из лаг-1; внутри-блок "
                                     f"k<8 сохранена по построению и из fit исключена) RAW согласовано с null-статистиками "
                                     f"(dip, plat_lo, plat_hi) на эмпирическом масштабе: frac клеток с |null−sim| ≤ "
                                     f"3·sd_pool_stat ≥ {G8_COV_MIN} (без base-вычитания: sample f_dc sim перегружен шумом "
                                     f"среднего MA(1) ≈V/n — баг v3); распределение selected (r,p) на null — reportable"),
                        "ok": bool(g8_ok)}
    if not smoke:
        assert g8_ok, f"G8 (negative control) FAILED: {gates['G8_null']}"
    else:
        logline("G8 (smoke: информационно, n_null=4) — assert только в полном прогоне")
    logline(f"G8 null: {n_null} (doc,L,seed); (а) median|tail_excess|/sd_emp={med_ratio:.2f} "
            f"(req ≤ {G8_TAIL_SIG}; sd_emp={sd_pool:.5f}, sd_null_sim med={med_sd_null:.5f}, "
            f"f_dc_null med={med_fdc_null:.5f}); "
            + " ".join(f"(б) {k2}: frac={v['frac_within_3sd_emp']:.2f} med_gap={v['median_abs_gap']:.5f}"
                       for k2, v in g8b.items())
            + f"; frac(raw DC+MA(1) cov)={frac_cov:.3f} (reportable) ok={g8_ok}")
    tstage(tr, time.time() - t0, "nulls", {"n_null": n_null})

    # ЧАСТЬ 1 агрегат + ВЕРДИКТ (ревью r4: primary-пул = non-degen ∩ K96; in-sample GOF;
    # р5 C1: Wilson CI + source-coverage; CI содержит 0.5 ⇒ inconclusive, не branch1/2)
    per_layer = {}
    for L in layers:
        avail = [nm for nm in docnames if (nm, L) in cells]
        nd = [nm for nm in avail if not degen[nm]]
        de = [nm for nm in avail if degen[nm]]
        elig = [nm for nm in nd if cells[(nm, L)]["k"] == K]        # primary-пул (все 3 полосы)
        short = [nm for nm in nd if cells[(nm, L)]["k"] < K]        # sensitivity
        cov_elig = [cells[(nm, L)]["part1"]["cov"]["covered"] for nm in elig]
        cov_short = [cells[(nm, L)]["part1"]["cov"]["covered"] for nm in short]
        cov_de = [cells[(nm, L)]["part1"]["cov"]["covered"] for nm in de]
        oos_elig = [cells[(nm, L)]["part1"]["oos"]["covered"]
                    for nm in elig if cells[(nm, L)]["part1"]["oos"] is not None]
        src = {}
        for nm in elig:
            src[docs[nm]["source"]] = src.get(docs[nm]["source"], 0) + 1
        # C1: source-coverage (кто покрывается, а не только кто в пуле)
        src_cov = {}
        for sname in sorted(set(src)):
            ns = [nm for nm in elig if docs[nm]["source"] == sname]
            cc = [cells[(nm, L)]["part1"]["cov"]["covered"] for nm in ns]
            src_cov[sname] = {"n": len(ns), "n_covered": int(sum(cc)),
                              "frac": (float(np.mean(cc)) if cc else None)}
        d_effs = [cells[(nm, L)]["real"]["d_eff"] for nm in elig]
        d_effs_full = [cells[(nm, L)]["real"]["d_eff_full"] for nm in elig]
        # C1: Wilson 95% CI для доли покрытых
        ci = wilson_ci(int(sum(cov_elig)), len(elig))
        per_layer[L] = {
            "n_nondegen": len(nd),
            "n_eligible_k96": len(elig),
            "n_covered": int(sum(cov_elig)),
            "frac_covered": (float(np.mean(cov_elig)) if cov_elig else None),   # PRIMARY
            "frac_covered_ci95": ci,
            "frac_covered_by_source": src_cov,
            "n_short": len(short),
            "frac_covered_short": (float(np.mean(cov_short)) if cov_short else None),
            "n_degen": len(de), "n_covered_degen": int(sum(cov_de)),
            "frac_covered_degen": (float(np.mean(cov_de)) if cov_de else None),
            "frac_covered_oos": (float(np.mean(oos_elig)) if oos_elig else None),
            "source_split": src,
            "d_eff_median": (float(np.median(d_effs)) if d_effs else None),
            "d_eff_full_median": (float(np.median(d_effs_full)) if d_effs_full else None),
            "n_f_le_0": int(sum(1 for nm in elig if cells[(nm, L)]["part1"]["fit"]["f_le_0"])),
            "n_out_of_model": int(sum(1 for nm in elig if cells[(nm, L)]["part1"]["fit"]["out_of_model"])),
        }
    # C1: 3-х веточный вердикт: CI lo > 0.5 → above; CI hi < 0.5 → below;
    # иначе (CI содержит 0.5) → inconclusive. Пустой пул → branch2.
    layer_state = {}
    for L in layers:
        if per_layer[L]["n_eligible_k96"] == 0:
            layer_state[L] = "empty"
            continue
        ci = per_layer[L]["frac_covered_ci95"]
        if ci is None:
            layer_state[L] = "inconclusive"
        elif ci[0] > FRAC_MIN:
            layer_state[L] = "above"
        elif ci[1] < FRAC_MIN:
            layer_state[L] = "below"
        else:
            layer_state[L] = "inconclusive"
    n_empty = sum(1 for v in layer_state.values() if v == "empty")
    n_below = sum(1 for v in layer_state.values() if v == "below")
    n_inc = sum(1 for v in layer_state.values() if v == "inconclusive")
    fl = {L: per_layer[L]["frac_covered"] for L in layers
          if per_layer[L]["frac_covered"] is not None}
    if n_empty > 0 or n_below > 0:
        branch = "branch2"
        verdict = (f"branch2: fitted DC+MA(1) per-doc НЕ покрывает дип+плато на всех слоях "
                   f"(below: {sorted(L for L in layers if layer_state[L] == 'below')}; "
                   f"empty primary-пул: {sorted(L for L in layers if layer_state[L] == 'empty')}) "
                   f"⇒ структура сверх DC+MA(1) есть, T3 получает работу "
                   f"(in-sample GOF; generative-claim не делается)")
    elif n_inc > 0:
        branch = "inconclusive"
        verdict = (f"inconclusive: CI95 для frac_covered содержит порог {FRAC_MIN} на слоях "
                   f"{sorted(L for L in layers if layer_state[L] == 'inconclusive')} — "
                   f"frac ≈ {FRAC_MIN} в пределах шумa (n_eligible ~50 ⇒ CI ±0.14); "
                   f"branch1/branch2 не выносится (in-sample GOF; generative-claim не делается)")
    else:
        branch = "branch1"
        verdict = (f"branch1: fitted DC+MA(1) per-doc (IN-SAMPLE GOF, не held-out generative transfer: "
                   f"a из ρ̂(1) ⇒ дип циркулярен; OOS — только plat_hi_oos, reportable) покрывает "
                   f"дип+плато (frac ≥ {FRAC_MIN} на каждом слое, CI95 lo > {FRAC_MIN}; "
                   f"primary-пул: non-degen ∩ K96) "
                   f"⇒ «три моды» мертвы как in-sample generative объяснение, T3 остаётся только горб")
    logline(f"ВЕРДИКТ (primary-пул: non-degen ∩ K96; in-sample GOF; C1: CI95 + source): "
            + ", ".join(f"L{L} {fl[L]:.3f} ({per_layer[L]['n_covered']}/{per_layer[L]['n_eligible_k96']}) "
                       f"CI95[{per_layer[L]['frac_covered_ci95'][0]:.3f},{per_layer[L]['frac_covered_ci95'][1]:.3f}] "
                       f"{layer_state[L]}" for L in layers if L in fl)
            + f" → {branch}")
    logline(f"пул per layer: " + "; ".join(
        f"L{L}: elig {per_layer[L]['n_eligible_k96']} {per_layer[L]['source_split']}, "
        f"src_cov { {s: v['frac'] for s, v in per_layer[L]['frac_covered_by_source'].items()}}, "
        f"short {per_layer[L]['n_short']}, degen {per_layer[L]['n_degen']}, "
        f"f_le_0 {per_layer[L]['n_f_le_0']}, out_of_model {per_layer[L]['n_out_of_model']}, "
        + (f"d_eff {per_layer[L]['d_eff_median']:.0f}/{per_layer[L]['d_eff_full_median']:.0f} (diag/full)"
           if per_layer[L]['d_eff_median'] is not None else "d_eff n/a")
        for L in layers))
    fl_oos = {L: per_layer[L]["frac_covered_oos"] for L in layers
              if per_layer[L]["frac_covered_oos"] is not None}
    logline(f"OOS (fit k=2..48, тест plat_hi_oos; reportable): "
            + ", ".join(f"L{L} {v:.3f}" for L, v in fl_oos.items()))

    # лестница/норма — сводка (reportable)
    sel_dist = {}
    for c in cells.values():
        key = f"r{c['part2']['selected'][0]}p{c['part2']['selected'][1]}"
        sel_dist[key] = sel_dist.get(key, 0) + 1
    sel_dist_tz = {}
    for c in cells.values():
        key = f"r{c['part2']['selected_tz'][0]}p{c['part2']['selected_tz'][1]}"
        sel_dist_tz[key] = sel_dist_tz.get(key, 0) + 1
    betas = [c["part2"]["ar1"]["beta"] for c in cells.values() if c["part2"]["ar1"]["R2"] is not None]
    r2s = [c["part2"]["ar1"]["R2"] for c in cells.values() if c["part2"]["ar1"]["R2"] is not None]
    ladder_sum = {"selected_dist": dict(sorted(sel_dist.items())),
                  "selected_dist_tz": dict(sorted(sel_dist_tz.items())),
                  "n_ar1_unstable": int(sum(1 for c in cells.values()
                                            if isinstance(c["part2"]["gensim"], dict)
                                            and c["part2"]["gensim"].get("skipped") == "ar1_unstable")),
                  "ar1_beta_median": float(np.median(betas)) if betas else None,
                  "ar1_R2_median": float(np.median(r2s)) if r2s else None,
                  "note": ("reportable; selected — по primary BIC (logdet, full df, ревью r4 C6); "
                           "TZ BIC — sensitivity; D-sim ≠ D-real (r-проекция) — gap = доля структуры в r проекциях; "
                           "no burn-in (M4): VAR нач. нули (первые p шагов — нач.-условный артефакт, ≤ p/383 ≈ 1% "
                           "ACF-массы, задокументировано в хедере), AR(1) ñ_0 = mean(n_real) (quasi-stationary); "
                           "|β|→1 — transient возможен (part2 reportable, не влияет на вердикт)")}
    logline(f"лестница: selected_dist={ladder_sum['selected_dist']}")
    logline(f"AR(1) норма: beta_med={ladder_sum['ar1_beta_median']:.3f} "
            f"R2_med={ladder_sum['ar1_R2_median']:.3f}")

    # Калибровка (reportable, ревью r4 C4): первые K96 non-degen доки per layer
    cal_cells = []
    for L in layers:
        cands = [nm for nm in docnames
                 if (nm, L) in cells and cells[(nm, L)]["k"] == K and not degen[nm]]
        if cands:
            cal_cells.append((cands[0], L))
    calibration = calibrate_null_on_null(cal_cells, paths, cells, idx_of, layers)
    logline(f"калибровка null-on-null (reportable): " + "; ".join(
        f"{k_}: marg {['%.3f' % v for v in v_['marginal']]}, joint {v_['joint']:.3f}, "
        f"d_eff {v_['d_eff_diag']:.0f}/{v_['d_eff_full']:.0f} (diag/full)" for k_, v_ in calibration.items()))

    # Full-cov null-калибровка (ревью r5 C2, опция 2, reportable): на ВСЕХ eligible
    # клетках (K96 ∩ nondegen) — Gaussian null с ПОЛНОЙ ковариацией L Lᵀ = Σ_c.
    # Сравнение с primary coverage (independent null): |Δ| велико ⇒ CI узок/широк
    # из-за cross-dim структуры (d_eff_full мал). Не-гауссовость (остаток) измеряет
    # G8-v3 эмпирический пул.
    fullcov_cal = {}
    if not smoke:
        for (nm, L), pth in paths.items():
            if cells[(nm, L)]["k"] != K or degen[nm]:
                continue
            fit = cells[(nm, L)]["part1"]["fit"]
            n = pth["n"]
            li = layers.index(L)
            Y, how = fullcov_sim(pth["dh"], fit["f"], fit["a"], B1,
                                 SEED_FCOV_OFF + 5 * idx_of[nm] + li, n=n)
            my, _ = sim_mats(Y, K)
            rm = real[(nm, L)]
            fcov, qs = {}, {}
            for j, key in enumerate(("dip", "plat_lo", "plat_hi")):
                if rm[key] is None:
                    continue
                lo, hi = np.quantile(my[:, j], (0.025, 0.975))
                fcov[key] = bool(lo <= rm[key] <= hi)
                qs[f"{key}_q"] = [float(lo), float(hi)]
            fullcov_cal[f"{nm}|L{L}"] = {"how": how, "covered": bool(all(fcov.values())), **qs}
        # per-layer coverage под full-cov null
        fullcov_per_layer = {}
        for L in layers:
            cc = [fullcov_cal[f"{nm}|L{L}"]["covered"]
                  for nm in docnames
                  if (nm, L) in cells and cells[(nm, L)]["k"] == K and not degen[nm]
                  and f"{nm}|L{L}" in fullcov_cal]
            fullcov_per_layer[L] = (float(np.mean(cc)) if cc else None)
        logline(f"full-cov null-калибровка (r5 C2, reportable): "
                + ", ".join(f"L{L} {v:.3f}" if v is not None else f"L{L} n/a"
                            for L, v in fullcov_per_layer.items())
                + f"; how: {dict((h, sum(1 for c in fullcov_cal.values() if c['how'] == h)) for h in ('chol', 'svd'))}")
    else:
        fullcov_cal, fullcov_per_layer = {}, {}
        logline("full-cov null-калибровка (smoke: пропущено)")
    tstage(tr, time.time() - t0, "calibration", {"n_cal": len(calibration), "n_fullcov": len(fullcov_cal)})

    # G2: полнота клеток
    cell_keys = {"real", "part1", "part2"}
    g2_ok = (len(cells) == len(paths)) and all(set(c) >= cell_keys for c in cells.values()) \
        and all(set(c["part1"]) >= {"fit", "cov", "simagg", "sim3_mean", "gap", "oos"}
                for c in cells.values()) \
        and all(set(c["part2"]) >= {"selected", "bic", "ar1", "gensim"} for c in cells.values())
    gates["G2"] = {"n_cells": len(cells), "expected": len(paths), "ok": bool(g2_ok)}
    assert g2_ok, "G2: неполные клетки"

    # артефакты
    zq = 1.959963985
    qmc = float(np.sqrt(0.025 * 0.975 / B1) / (np.exp(-zq * zq / 2) / np.sqrt(2 * np.pi)))
    store = {"experiment": "AT0_5 T0.5 (генеративная проверка, disk-only)",
             "meta": {"env": {"python": sys.version.split()[0], "numpy": np.__version__},
                      "n_docs": len(docnames), "layers": [int(L) for L in layers],
                      "K": K, "N": N, "smoke": bool(smoke),
                      "skipped": skipped,
                      "k_rule": ("k = min(96, n−1); n = min(T,384)−1 = строк Δh (= N в T0.1 store); "
                                 "primary-пул = K96 ∩ non-degen (T0.1 K-алфавит {0,24,32,96}); "
                                 "short 16≤k<96 — sensitivity; skip: k<16 или T0.1 K=0 (fp16_l1_degraded)"),
                      "pool_note": ("corpus: 0/16 tok (G16 not_testable — только provenance-оговорка; "
                                    "h-данные есть, corpus в primary-пуле; per-source split в per_layer)"),
                      "quantile_mc_error_sigma": qmc,
                      "quantile_mc_note": "se_q(0.025; B1) в единицах σ sim-статистики (B=200)",
                      "B1": B1, "B2": B2, "BLK": BLK, "NSEED": NSEED,
                      "seeds": {"part1": SEED1_OFF, "part2": SEED2_OFF,
                               "null": SEEDN_OFF, "oos": SEED_OOS_OFF,
                               "cal_a": SEED_CAL_A, "cal_b": SEED_CAL_B,
                               "fullcov": SEED_FCOV_OFF, "perm": SEED_PERM_BASE},
                      "seed_note": ("r5 C3: null sim = SEEDN + NSEED·(5·idx+li) + seed_i (уникален по doc/L/seed); "
                                    "block_perm = SEED_PERM + 25·idx + 5·li + seed_i; cal = OFF + 5·idx + li; "
                                    "fullcov = SEED_FCOV + 5·idx + li"),
                      "field_sha256": fsha,
                      "data_sha256": {os.path.basename(k): v for k, v in data.items()},
                      "T01_store_sha256": T01_STORE_SHA,
                      "revision": REVISION,
                      "started_at": t_start_iso,
                      "finished_at": datetime.now().isoformat(),
                      "trackio_run": f"at05_gencheck_{REVISION}" if not smoke else None},
             "gates": gates,
             "prereg": {"verdict": verdict,
                        "file": "docs/AT0_5_prereg.md",
                        "estimand": ("per-doc in-sample GOF: covered = real (dip, plat_lo, plat_hi) ∈ "
                                     "[q2.5, q97.5] sim-распределения fitted DC+MA(1) (D-мерный null из "
                                     "fit-нутых (f,a), total power сохранён, DC-доля = fit-нутый f)"),
                        "rule": (f"branch1 ⇔ frac_covered ≥ {FRAC_MIN} на каждом слое "
                                 f"(primary-пул: non-degen ∩ K96; OOS/generative-claim — не делается)")},
             "aggregates": {"per_layer": per_layer, "ladder": ladder_sum,
                            "G8_null": gates["G8_null"],
                            "calibration": calibration,
                            "calibration_note": ("null-on-null: партия A (B=200) → квантили, партия B "
                                                  "(B=200) → покрытие; marginal ≈ 0.95 если CI калиброван; "
                                                  "joint — эмпирический (метрики коррелированы)"),
                            "fullcov_calibration": fullcov_per_layer,
                            "fullcov_note": ("r5 C2: Gaussian null с ПОЛНОЙ ковариацией (L Lᵀ = Σ_c) на "
                                              "eligible клетках; сравнение с primary coverage (independent null); "
                                              "не-гауссовость (остаток) измеряет G8-v3 эмпирический пул"),
                            "fullcov_cells": fullcov_cal},
             "cells": {f"{nm}|L{L}": c for (nm, L), c in cells.items()}}
    store_clean = field.jclean(store)

    if smoke:
        logline("AT0_5 SMOKE OK (артефакты не пишутся)")
        return

    os.makedirs(STORE_DIR, exist_ok=True)
    store_bytes = json.dumps(store_clean, sort_keys=True, ensure_ascii=False).encode()
    sha = hashlib.sha256(store_bytes).hexdigest()
    store_path = os.path.join(STORE_DIR, f"{sha}.json")
    with open(store_path, "wb") as f:
        f.write(store_bytes)
    logline(f"store: {store_path} (sha {sha[:16]})")
    # C7 (ревью r4): ok строго is True (гейт без ok — FAIL, а не PASS);
    # при FAIL — store + run.log пишутся, claim/manifest — НЕТ, RUN_RC=1
    all_ok = all(g["ok"] is True for g in gates.values() if isinstance(g, dict))
    if not all_ok:
        bad = [k for k, g in gates.items() if isinstance(g, dict) and g.get("ok") is not True]
        logline(f"AT0_5 gates: FAIL ({bad})")
        logline("RUN_RC=1")
        with open(RUN_LOG, "w") as f:
            f.write("\n".join(_LOG) + "\n")
        if tr is not None:
            tr_done[0] = True
            tlog(tr, time.time() - t0, {"finished_at": datetime.now().isoformat(),
                                         "status": "fail", "gates_failed": bad,
                                         "log_md5": hashlib.md5(open(RUN_LOG, "rb").read()).hexdigest()})
            tr.finish()
        raise SystemExit(1)
    claim = {"experiment": store["experiment"],
             "store": os.path.relpath(store_path, ROOT), "store_sha256": sha,
             "meta": store["meta"], "gates": store["gates"], "prereg": store["prereg"],
             "aggregates": store["aggregates"],
             "wall_sec": round(time.time() - t0, 1)}
    with open(CLAIM, "w") as f:
        json.dump(field.jclean(claim), f, ensure_ascii=False, indent=1)
    manifest = {"experiment": "AT0_5", "claim": os.path.relpath(CLAIM, ROOT),
                "claim_sha256": hashlib.sha256(
                    open(CLAIM, "rb").read()).hexdigest(),
                "store": claim["store"],
                "store_sha256": sha, "run_log": os.path.relpath(RUN_LOG, ROOT),
                "wall_sec": claim["wall_sec"],
                "verdict": verdict,
                "date": datetime.now().isoformat()}
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    logline(f"written: {CLAIM}, {MANIFEST}, {RUN_LOG}")
    logline("AT0_5 gates: ALL PASS")
    logline("RUN_RC=0")
    with open(RUN_LOG, "w") as f:
        f.write("\n".join(_LOG) + "\n")
    if tr is not None:
        tr_done[0] = True
        tlog(tr, time.time() - t0, {
            "finished_at": datetime.now().isoformat(), "status": "ok",
            "verdict": branch,
            "log_md5": hashlib.md5(open(RUN_LOG, "rb").read()).hexdigest(),
            "wall_sec": round(time.time() - t0, 1),
            "store_sha": sha[:16]})
        tr.finish()
        logline(f"trackio: finish ok (status=ok, verdict={branch})")


if __name__ == "__main__":
    main()
