AT2c code review (A9-A12 + protocol)
SHA code: fb0b43fda...665b832 OK
SHA prereg: 58eb7bb47...f8a88 OK

## A9. r_final / S5a + S5b — OK
- Rematching (AT2c.py L388-404): ratios per (L,doc) = max/min d_eff{top,weak,pca,haar};
  max(ratios) > 1.2 -> r_final = int(round(min(min d_eff))) [ожидание 7], иначе r_final = R = 32;
  S5a жёсткий гейт r_final == 7, иначе SystemExit STOP. Дословно AT2b.py L350-355 (унаследовано).
  TZ §2 "ratio > 1.2 -> r = round(min d_eff); S5a: r_final == 7, иначе STOP" — совпадает.
- ARMSWAP-ветка: r_final = 7 фикс, rematching не нужен (TZ §11: "r=7 фикс") — ок.
- S5b (L470-502): per слой 21 вектор (top 7 + weak 7 + weak_mid 7), f64-векторы eigh до f32-каста;
  resid = ||Gagg*v - (v@Gagg v)*v|| / (|lam|+1e-30) (Rayleigh-quotient = eigenvalue в точности f64);
  гейт all(< 1e-9) + SystemExit. TZ §7 S5b: "< 1e-9 (f64)" — совпадает.
  weak-порог nz: lam > 1e-12*lammax + r_weak_eff отчёт + STOP не делать (TZ §2) — совпадает.
  НОТА: в ЗАДАЧЕ ревьюера написано "S5b: Gagg <= 1e-6" — в ТЗ 1e-9; код следует ТЗ (1e-9). Расхождение в сводке задачи, не в коде.
## A10. S1-S4 — OK (1 расхождение с ТЗ в сводке задачи)
- S1 (L111-117, L255-260): 7 sha-файлов (6 jcache L{6,12,18,19,22,23} + pool_ids_L23),
  строгое равенство, SystemExit при провале. "7/7 OK". — совпадает с ТЗ §7 S1.
- S2 (L283-292): гейт mean KL < 1e-5 на 2 дока x 128 позиций; all32 — ТОЛЬКО отчёт
  (mean/min/max в s2_kl_report). TZ §7 S2: "< 1e-5 на 2 дока (гейт) + KL на всех 32 (отчёт)".
  Совпадает. НОТА: в задаче ревьюера "KL <= 1e-6 all32" — в ТЗ гейт 1e-5 (2 дока); код следует ТЗ.
- S3 (L293-308): hi = P(z[t+1], next-token), lo = flip-контроль; пул 32x127;
  гейт mean_hi>0.1 & top1_hi>0.1 & mean_lo<0.05 & top1_lo<0.05 + SystemExit.
  TZ §7 S3: "hi > 0.1/0.1; flip lo < 0.05/0.05; exploratory" — совпадает (комментарий "exploratory" на месте).
- S4 (L311-330): main — строгое равенство total 90/4096 + per-домен wiki 8 / math 41 / code 36 / chat 5,
  SystemExit. Smoke — не гейтится (лог "ожидаемые числа не гейтятся"; пул 2 дока != 32) —
  ТЗ молчит об исключении S4 в smoke (S5a/S8 исключены явно) — NIT (декларация в логе, см. ниже).
- one-hot (L807-855): onehot_med per k + "onehot_non_informative": True (хардкод, дословно AT2b.py L869,
  конвенция D3: разные пространства). TZ §6: "non_informative: true (D3; вне условия (a))" — совпадает.
  "median <= 0.5" из задачи ревьюера в ТЗ нет — код следует ТЗ.
## A11. Thread-pinning / eigh / GEMM — OK (2 расхождения с ТЗ в сводке задачи)
- Pinning (L50-58): OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS = "4"
  ДО импорта numpy/torch; torch.set_num_threads(4) после импорта torch.
  Дословно AT2b.py L19-25 и AT2b_diag_c2c1.py L39-45 — "ТОЖЕ, что в AT2b.py/диагностике" (ТЗ S8) — совпадает.
  НОТА: VECLIB_MAXIMUM_THREADS нигде в линии (ТЗ/AT2b/диагн/AT2c) не устанавливается —
  "VECLIB" в задаче ревьюера за пределами ТЗ; код следует ТЗ.
- eigh: torch.linalg.eigh — явный предписанный ТЗ §2 ("torch.linalg.eigh(Gagg) (lambda, w, по возрастанию)");
  диагностика тоже torch (AT2b_diag L255, L534) — S8 bit-identity сохранён.
  "eigh — numpy" в задаче ревьюера противоречит ТЗ; код следует ТЗ.
- GEMM f32: z_full = H23 @ E.T (f32 torch, L281); проекции/anchor — numpy f32 (H_np @ V),
  идентично диагностике C1c (bit-identity S8). TZ-формулировка "f32 GEMM" (S8) соблюдена по факту.
## Correction (точные строки, grep-верифицировано)
- Pinning: L39-45 (env OMP/OPENBLAS/MKL=4: L39-41; torch.set_num_threads(4): L45).
- SHA_FILES: L101-109; S1-гейт-цикл: L268-272. S2: z_full L296, kl2 L302, гейт L303.
  S3: L309-325 (гейт L318-320). S4: L316-352 (гейт L344-346). one-hot: L807-855.
- N8: L831-840 (см. MINOR-1).
## A12. Trackio — OK
- Run name: at2c_t21c_r1 (L246: f"at2c_t21c_{REVISION}", REVISION="r1") — ТЗ S10 "run at2c_t21c_r1" — совпадает.
- t=0 config (L224-244): started_at (ISO), revision, script_sha, seed=0, device=cpu, model, log,
  layers/k/r=32/alpha/boot_b/n_docs/delta_min=0.01. Все поля из ТЗ S10 + p.5 (started_at, revision,
  script_sha, seed, boot_b, delta_min, r=32) присутствуют. tlog(tr, 0, ...) — первая точка t=0.
- Stage-точки (tstage, step=wall-с от t0): load (L254); subspaces pass1 (phase=pass1_deff, L369);
  subspaces pass2 (phase=pass2_stationary, r_final+rematched, L413); ridge (L571); lk4 (L822);
  verdict (L871); artifact (L975) — ровно набор из ТЗ S10 (load/subspaces c r_final/ridge/lk4/verdict/artifact).
- Финал happy-path (L1129-1135): лог пишется ДО финальной точки; finished_at + status=ok +
  log_md5 (md5 записанного файла) + verdict; tr.finish().
- Crash-path (L248-262, atexit): лог первым (CRASH-строка), затем status=crash + finished_at +
  log_md5 + tr.finish(); guard tr_done (не дублирует). ТЗ: "краш — НЕ молча ... в finally/atexit" —
  atexit покрывает и SystemExit (гейт-Стоп), и unhandled. НОТА (NIT-3): формулировка ТЗ "в finally";
  статус "interrupted" не эмитится (SIGINT -> crash) — допустимый поднабор ok|crash|interrupted.
## C. Протокол — OK
Файлы существуют: docs/AT2c_prereg.md (sha 58eb7bb4...f8a88 = текущий), docs/AT2c_review.md (r1),
docs/AT2c_review_r2.md.._r6.md, docs/AT2c_review_answers.md (+ docs/AT2c_results.md).
VERDICT-маркеры: r1 REVIEW_AT2C_TZ_DONE (REVISE, 1 MAJOR+1 MINOR+4 NIT); r2 _R2_DONE (REVISE);
r3 _R3_DONE (PASS); r4 _R4_DONE (PASS, 1 MINOR pre-existing non-blocking + 2 NIT);
r5 _R5_DONE (PASS — верификация фикса MINOR r4 "31->32 фолда" в текущем ТЗ);
r6 _R6B_DONE (PASS — финальный, на ТЕКУЩЕМ sha ТЗ 58eb7bb4: sha в начале и конце файла).
Ответка покрывает все раунды, включая r6 (S6c-перепроектирование: cos-вклад <= 0.02 +
конфаунд <= DELTA_MIN; код L644-670 реализует ровно это: gate J-армы excl uniform, dcos <= 0.02,
confounding per J_ARMS <= DELTA_MIN, ybar < 1e-5).
MINOR r4 ("31 фолд") закрыт: текущее ТЗ L193 "32 LOO-фолда per k, по 31 train-дока"; код S6 —
32 LOO-итерации per k (for d in range(32), Xf без дока d). Ревью-луп завершён.
Нота: docstring AT2c.py L4-6 описывает луп до r3 PASS (r4-r6 не упомянуты) — NIT-1.
## Замечания

MAJOR: нет.

MINOR-1 (report-only, вне вердикта/гейтов): N8 — ТЗ §6: "cos_c per (l,k,arm) на lk4-hit
(k=0 канон) позициях, ВСЕ 6 MAIN-АРМОВ"; код AT2c.py L831-840: `for arm in J_ARMS` —
только 4 арм (top/weak/pca/haar); weak_mid и uniform в N8 отсутствуют. Комментарий кода
"(T2.1b: армы J_ARMS)" — унаследовано из AT2b.py (L603) без обновления под ТЗ T2.1c.
Не влияет на вердикт (N8 report-only), но артефакт n8 неполный против ТЗ.
Фикс: заменить J_ARMS на ARM_MAIN в цикле N8 (или пометить отклонение в ТЗ).

NIT-1: AT2c.py L4-6 (docstring): ревью-луп записан "r1 REVISE -> r2 REVISE -> r3 PASS" —
фактически r4-r6 существуют (финал r6 PASS на текущем sha). Устаревшая строка.
NIT-2: trackio_start L134: os.environ.setdefault("TRACKIO_DIR", ROOT/.trackio) — AGENTS p.5:
"скрипты его не задают сами". setdefault не переопределяет экспорт run_one.sh (тот же путь),
дословно унаследовано из AT2b.py L88 (прошедшего ревью) — не блокирует.
NIT-3: crash-маркер через atexit (L248-262), а не try/finally (формулировка ТЗ S10 "в finally");
статус "interrupted" не эмитится (SIGINT -> "crash"). Функционально эквивалентно для ok|crash.
NIT-4: DIAG_SHA (L120) — только в ref_source артефакта; runtime-перепроверка sha файла
results/AT2b_diag/AT2b_diag.json не делается. ТЗ требует только числовое сравнение (S8) +
ref_source — формально ок; усиление провенанса — опционально.
NIT-5: S4 в smoke не гейтится (L338-340) — ТЗ явно исключает из smoke только S5a/S8;
для S4-в-smoke исключения в ТЗ нет (пул 2 дока vs 90/4096 — гейт невозможен по построению).
Декларация есть в логе; формализовать в ТЗ или оставить как есть.

НОТА (не дефект кода): сводка этой задачи содержит числа, отличные от ТЗ: S5b "1e-6" (ТЗ: 1e-9),
S2 "KL <= 1e-6 all32" (ТЗ: гейт 1e-5 на 2 дока, all32 — отчёт), "VECLIB" (в ТЗ/линии нет),
"eigh — numpy" (ТЗ §2: torch.linalg.eigh). Код во всех 4 пунктах следует ТЗ — проверено.
## VERDICT: PASS
A9 (r_final/S5a/S5b): совпадает с ТЗ. A10 (S1 7 sha, S2, S3, S4, one-hot): совпадает с ТЗ.
A11 (pinning 4 тр. до импортов; eigh torch по ТЗ; f32 GEMM): совпадает с ТЗ/AT2b/диагностикой.
A12 (trackio: t=0 config полный, 7 stage-точек c r_final в pass2, финал status+log_md5,
crash-маркер): совпадает с ТЗ. C (протокол: r1-r6 + ответка, финал r6 PASS на текущем sha,
маркеры на месте, луп завершён): OK.
1 MINOR (N8: 4 арм против "все 6 main-армов" — report-only, не в вердикт) + 5 NIT.
Ни MAJOR, ни блокирующих отклонений: код не ломает гейты, дерево, S8-якорь (вербально
идентичен C1c-блоку диагностики: тот же eigh/B=1000/SEED=0/ci_of).
REVIEW_AT2C_CODE2_DONE
