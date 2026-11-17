# Ревью AT2c.py vs ТЗ — ТОЛЬКО A5–A8 (код)

Код: experiments/at2/AT2c.py
sha256 fb0b43fdaeccc228763b11f766e29dac87efeb17d78c93fc38af7cb0d665b832 (проверено)
ТЗ: docs/AT2c_prereg.md
sha256 58eb7bb47883df684308fed63a17c367da69b9c811b8f477e4ff0e5aaf7f8a88 (проверено)
Метод: статическое чтение кода + сверка с diagnostic-артефактом/скриптом. py_compile OK.

## A5. S8: воспроизведение AT2b_diag.json — СОВПАДАЕТ (основные пункты)
- Thread-pinning: OMP/OPENBLAS/MKL=4 ДО импортов numpy/torch + torch.set_num_threads(4)
  (строки 45–51) — идентичен диагностике (AT2b_diag_c2c1.py строки 39–45). TZ §7 S8 обязательство — есть.
- S8_REF (строки 119–124) == results/AT2b_diag/AT2b_diag.json c1.stationary_pooled_cd ПОЛНЫМ
  float64 бит-в-бит (сверено python-чтением файла); DIAG_SHA (строка 126) == sha256 артефакта на диске.
- s8_anchor_block (строки 1165–1197) операция-в-операции == C1c-блок диагностики (строки ~528–570):
  Jagg=mean_{d,t}JR f64 → Gagg=Jagg^T Jagg → eigh → top=order[:r], 2-й арм=order[r:2r] (weak_mid,
  desc); Xs=H@V − t-mean (f32); yf=z_U128 − mean_t; full-pool LOO-ridge α=1, intercept без penalty;
  cos +1e-12; da=top−weak; CI=ci_of НЕ-взвешенный mean, B=1000, SEED=0 (ci_of идентичен: diag 423–428).
- "Только при пуле == 32 доков": guard `if N_DOCS_RUN == N_DOCS` (471, 517); smoke (2 дока) —
  skip + "skipped (smoke: pool 2 != 32)" (541) — объявлено в TZ. r=7 гарантирован S5a до pass2
  (ARMSWAP — r=7 фикс). 4 слоя: LAYS=(6,12,18,22). Tol 1e-6 (125); max_diff по CI-эндпоинтам И mean.

### A5 — MINOR-1: S8 не гейтится в PRERUN (пул == 32)
- Строка 535: `if MAIN_RUN and not s8_ok: raise SystemExit(...)` — гейт только в main.
  PRERUN (AT2c_PRERUN=1) работает с пулом 32: якорь считается и попадает в артефакт
  (prerun_c0.json, поле s8_anchor с "ok"), но прогон НЕ стопится при fail.
- TZ §7 S8: «Провал = STOP … В smoke S8 не гейтится (пул 2 дока ≠ 32) — объявлено» —
  исключение объявлено ТОЛЬКО для smoke. Премисса arm-swap (§11) и go/no-go-решение
  (§11.0) стоят на pre-run замере — проваленный там S8 делает замер непроверенным.
- Комментарий кода (515) фиксирует намерение «prerun/armswap — отчёт» — это отклонение
  от текста TZ, а не его интерпретация. Fix: `if not s8_ok: raise` для всех режимов
  с пулом 32 (или явная декларация исключения PRERUN в TZ).

### A5 — NIT-1: ARMSWAP: якорь считается только для L6 и отбрасывается
- В ARMSWAP LAYS=(6,) (строка ~88) → s8_anchor_block (471–472) выполняется только для L6,
  logline печатает только L6, а armswap-артефакт (out-дикт ~762–775) s8-поле НЕ содержит —
  результат вычисляется и молча выбрасывается. Не дефект клетки (arm-swap — exec-фрагмент §11),
  но мёртвая работа + несогласованно с комментарием 515 («armswap — отчёт»).

## A6. Вердикт-дерево — СОВПАДАЕТ
- Порядок: control → (a) → (b) → (c) → (d) → NOT_RESOLVED (строки 880–1005). Все условия
  — на первичной weighted-метрике (wcos из agg; ci_w для CI). weak_mid/uniform — вне дерева.
- (a) SCRATCHPAD (955–963): для k∈{2,4,8}: ci_ok = AND по 4 слоям weighted CI lower > 0
  (weak − max(pca,haar), boot[(L,k)]["weak_vs_maxPCAHaar_wCI"]) И wmean_ok = AND по слоям
  weighted mean (weak − max(pca,haar)) > DELTA_MIN=0.01; SCRATCHPAD при n_a ≥ 2.
  Примечание: в задании «wmean>0» — в TZ §5(a) строго > DELTA_MIN; код следует TZ (строже).
- (b) DEADWEIGHT (965–972): max|wcos(top)| и max|wcos(weak)| ≤ DELTA_MIN по 4×4
  (4 слоя × tree_ks=(1,2,4,8)) — ровно «все (l,k) ∈ 4×4».
- (c) REDUNDANT_BASIS (975–993): все 6 попарных CI {top,weak,pca,haar} (weighted) содержат 0
  на всех (l,k)∈4×4; report-only пары (weak_vs_top, weak_vs_weak_mid) исключены (981–983) —
  корректно: TZ «weak_mid НЕ в дерево». + k=1: top ≥ max(weak,pca,haar) по wcos без значимости (985–989).
  Примечание: «все 7 CI» в задании — среди 4 армов ровно 6 пар; 7-я (weak_vs_top) — знак-инверсия
  top_vs_weak, contains-0 симметричен — на вердикт не влияет. Код следует TZ (6 пар).
- (d) METRIC_UNINFORMATIVE (995–1005): max попарного |wcos-различия| J_ARMS по (l,k) ≤ DELTA_MIN,
  иначе NOT_RESOLVED. Совпадает с TZ §5(d).

## A7. k=0-гейт — СОВПАДАЕТ
- PASS_TO_J_K = control_ok = all(ctrl_wci[L][0] > 0.0 for L in LAYS) (строка 874):
  weighted CI lower > 0 на ВСЕХ 4 J-слоях (AND-over-l), односторонне. TZ §4 — ровно.
- da = cos_top − cos_weak per (d,t) при k=0 (869–871); CI = ci_w(da, w_all[0], rngb) (872):
  per выбор weighted mean по позициям выбранных доков; веса w_all[0]=‖y_c(d,t)‖² зависят
  только от (d,k,t) — ОБЩИЕ для армов (строки 672–675) — правило «одинаковая статистика».
- Fail → SUBSPACE_CONTROL_FAIL (881–884): k≥1 отчитывается, не интерпретируется как ответ T2.
  Unweighted CI (ctrl_ci, 873) также считается и отчитывается (bootstrap-блок артефакта) — TZ §13.
- Армы: k=0 гейт-пара J {top,weak} (869–871); L23 {pca,haar,uniform} per k (ARM_L23, строка ~97);
  J при k≥1 — 6 армов (ARM_MAIN, 428). Код считает ВСЕ 6 J-армов и при k=0 — это обязательство
  TZ §7 S7 (S7_expected = 4×32×625×6 + 32×625×3 = 540000, 625 = Σ_k(128−k) по всем 5 k);
  S7 exact-set check это enforcing (строки ~1013–1030). «J {top,weak} при k=0» из задания
  трактуется как гейт-пара — код соответствует TZ (авторитетный источник).

## A8. Boot — СОВПАДАЕТ
- B=1000: BOOT_B = 100 if SMOKE else 1000 (строка 93; TZ S9: smoke B=100, main B=1000);
  SEED=0 (строка 92); bootstrap-поток rngb = np.random.default_rng(SEED) (строка 789;
  PRERUN/ARMSWAP — свои потоки, строки 715/756; S8-якорь — отдельный fresh поток 1189,
  как в диагностике — default_rng(SEED) внутри C1c-блока).
- Weighted CI = квантили (2.5/97.5) bootstrap-распределения взвешенного среднего; веса
  пересобираются на каждом draw: `w[sel]` по ресемплинговым докам (ci_w, 1137–1145):
  num=(w[sel]*dd[sel]).sum(), den=w[sel].sum() — weighted mean по позициям выбранных доков.
- Unweighted = квантили per-(d,t): `dd[sel].mean()` (ci_of, 1128–1135) — среднее по
  (док,t) позициям ресемплированных доков. Ресэмплинг доков с повтором:
  rngb.integers(0, N_DOCS_RUN, N_DOCS_RUN) — в обеих функциях. TZ §4/§7 S8 — ровно.
- Одна и та же конвенция: control CI, попарные tree-CI (а/c), PRERUN/ARMSWAP CI.

## ИТОГ

VERDICT: REVISE (1 MINOR, 2 NIT; main-путь A5–A8 — корректен)

1. MINOR-1 (A5): S8 не гейтится в PRERUN при пуле 32 — строка 535 (`if MAIN_RUN and not s8_ok`);
   TZ §7 S8 исключает из гейта только smoke («пул 2 дока ≠ 32 — объявлено»); pre-run замер
   (go/no-go + премисса arm-swap) при проваленном S8 становится непроверенным.
   Fix: гейт для всех режимов с N_DOCS_RUN==32.
2. NIT-1 (A5): ARMSWAP — якорь считается только для L6 и не попадает в артефакт
   (471–472, out-дикт armswap) — мёртвая работа.
3. NIT-2 (A6): в задании «все 7 CI» для (c) — в TZ 6 попарных CI среди {top,weak,pca,haar};
   код следует TZ; на вердикт различие не влияет (contains-0 симметричен к знак-инверсии).
   (Аналогично: «wmean>0» в (a) задания vs >DELTA_MIN в TZ — код следует TZ, строже.)

Основание: main-прогон (клетка) по A5–A8 соответствует ТЗ полностью: S8-якорь бит-в-бит
совпадает с C1c-конфигурацией диагностики и референсом в AT2b_diag.json (sha сверены),
дерево/гейт/bootstrap — дословно TZ. REVISE только из-за MINOR-1 в aux-режиме.

NOTES: сомнений при статической сверке не возникло; exec-проверка якоря (bit-identity)
не проводилась — за её пределами (только код vs ТЗ).
REVIEW_AT2C_CODE1B_DONE
