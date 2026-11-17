# AT2c (T2.1c) — пре-рег: стационарная state-readout клетка (disk-only)

Дата: 17.11. Статус: НЕИЗМЕНЯЕМО после записи (fix-до-прогона).

**Исход:** T2.1b — `SUBSPACE_CONTROL_FAIL` (docs/AT2b_results.md); 2-й раунд
внешнего ревью (docs/AT2b_review.md, ответы docs/AT2b_review_r2_answers.md)
подтвердил два конфаунда проваленного k=0-контроля и disk-only диагностика
(`experiments/at2/AT2b_diag_c2c1.py`, run `at2b_diag_c2c1_r3`,
`results/AT2b_diag/AT2b_diag.json`, sha 617f971e…) — бит-в-бит воспроизведение
прогона r1 (860000/860000, max|diff| = 0) — дала доказательства:

- **C1 (подтверждён):** per-(l,d,t) вращающийся J-базис + один общий ridge на
  31×128 позиций конфаундит контроль. Стационарный per-layer базис
  (J_agg = mean_{d,t} J_{d,t}): top > weak на ВСЕХ 4 слоях, CI top−weak
  +0.21…+0.36 (диагностика C1c). Разбиение top/weak функционально валидно.
- **C2 (подтверждён, не флиппер):** обучающая цель ridge — сырые z-логиты;
  per-док сдвиг c(d) = ȳf − ȳ(d) ≈ 17–28 при ‖yc_t‖ ≈ 36 сжимает cos_c всех
  арм к ~0. Обучение на per-doc центрированной цели: mean training target = 0
  ⇒ intercept: при k=0 — по сути нулевой (f32-уровень округлений, pre-main
  замер: cos-вклад ≤ 5.2e-8); при k>0 — малая коррекция (cos-вклад max_d
  weighted: k=1 ≤ 2.432e-3, k=8 ≤ 1.44e-2; попарный конфаунд verdict-количества
  (per-arm weighted-wmean) ≤ 7.6e-3 < DELTA_MIN = 0.01; §11.2, r7) — per-док сдвиг
  17–28 исчезает по построению.
- **C3 (частично):** разброд R²_c vs cos_median — следствие C2; weighted cos
  (веса ‖yc_t‖²) — первичная метрика, median — secondary (оба отчитываются).
- Haar-аномалия k=0 — симптом C1 (стационарный фиксированный базис читает
  текущий логит лучше нестационарных J-координат через общий ridge).

**Клетка T2.1c:** та же клетка T2 (state h_l → readout будущего логита;
гипотеза T2 из AT2-пре-рега: «top-J выигрывает на k=1; weak-J выигрывает на
k≥2 сверх PCA/Haar при matched d_eff и над lk4»), но с тремя исправлениями
конструкции (стационарный базис, centered-цель, weighted-метрика) и
наследуемым деревом T2.1b. Цель — сделать k=0-контроль проходимым, чтобы k≥1
можно было интерпретировать как ответ на T2.

## 1. Входы (disk-only, ноль новых форвардов)

Тот же кэш exp27b + модель, что T2.1b (S1 — те же 7 sha):
`/data/prism_kernels/exp27b/results/exp27b_jcache_L{6,12,18,19,22,23}.pt`,
`exp27b_pool_ids_L23.pt`; `/data/Qwen/Qwen3.5-0.8B` (tied E, z-маршрут
z = H_L23 @ Eᵀ, без повторной нормировки). Пул: 32 дока (4 домена × 8), S=128.
Центральный артефакт-референс: `results/AT2b_diag/AT2b_diag.json` (read-only;
гейт S8 — якорь провенанса).

## 2. Базисы

- **r_final:** d_eff rematching ТОЧНО как в T2.1b/диагностике (per (L,doc):
  P-взвешенный Грамиан G = P^{1/2}(JR·JRᵀ)P^{1/2} при t=0, R=32, lift, d_eff
  per (L,doc) для {top, weak, pca, haar}; ratio > 1.2 → r = round(min d_eff)).
  Ожидание 7; **жёсткий гейт S5a: r_final == 7, иначе STOP** (не софт: сравнение
  с T2.1b/диагностикой ломается).
- **top/weak/weak_mid (стационарный per-layer J-базис, ГЛАВНОЕ ИЗМЕНЕНИЕ):**
  per слой L ∈ {6,12,18,22}: `Jagg = mean_{d,t} JR_{d,t}` ([128, 1024], f64);
  `Gagg = Jaggᵀ Jagg` ([1024, 1024], rank ≤ 128); `torch.linalg.eigh(Gagg)`
  (λ, w, по возрастанию); порядок по убыванию λ:
  - `top` — первые r_final (сильнейшие);
  - `weak` — r_final самых слабых НЕ-нулевых (λ > 1e-12·λmax; конвенция
    T2.1/AT2b — «слабая сторона спектра», weak-арм T2-линии; если ненулевых
    < r_final — взять все ненулевые, r_weak_eff отчитать, STOP не делать —
    зафиксировано);
  - `weak_mid` — позиции r_final … 2·r_final по убыванию (конвенция
    диагностики C1c; report-only + S8-якорь; НЕ в дерево).
  Конструкция — ТОЧНО как в диагностике (S8-якорь гарантирует). **Отличие от
  r1-конструкции объявлено:** без P^{1/2}-взвешивания в z-пространстве (P —
  per-(d,t), канонического стационарного аналога нет; взвешенный стационарный
  вариант — новая конструкция, T2.1d+, не эта клетка).
- **pca:** per (l,doc), SVD центрированного H того же дока, первые r_final
  правых сингулярных векторов (как в T2.1b).
- **haar:** QR(G32[:, :r_final]), G32 ~ N(0,1)^{1024×32} — ОДИН
  `rng_base = default_rng(SEED=0)`: сначала draw G (1024×32), затем uniform
  (конвенция T2.1b §3, bit-детерминизм).
- **uniform:** одна глобальная случайная прямая (1 dim), из того же потока
  после G (floor-референс).
- **`*_dc` армы НЕ делаются** (T2.1b-контроль DC-проблемы): цель центрирована
  (per-док) + X t-центрирован ⇒ DC-компонента базиса ортогональна X по
  построению (dot(ones, X_c) = 0 per док) и не влияет на readout; DC-проблема,
  которую `_dc` контролировал в T2.1b, в T2.1c снята целью (§3). Зафиксировано.
- **Асимметрия объявлена (F7-наследник):** top/weak/weak_mid — per-layer
  (stationary), pca — per (l,doc), haar/uniform — глобальные. Meta-блок
  `asymmetry` в артефакте.

## 3. Пайплайн X, цель, estimator, метрика

- **X (один пайплайн для ВСЕХ арм):** `X(l,d,arm) = H(l,d) @ V_arm`
  ([128, r_arm]) → центрирование по t (t=0..127) per (l,doc,arm) → срез [k:].
  (Тот же, что T2.1b §2; `_dc` не существует — §2.)
- **Цель:** `y(d,t) = z_U128[d, t+k]` (t = 0..127−k; k=0 → текущий логит);
  `ȳ_{d,k} = mean_t y(d,t)` per (док, k); `y_c = y − ȳ_{d,k}`.
- **Estimator (ИСПРАВЛЕНИЕ C2):** per (L, k, arm) — LOO-doc ridge (32 фолда,
  α = 1.0, no-penalty на intercept: `A[-1,-1] -= ALPHA`), **обучающая цель =
  y_c** (центрированная). Mean training target = 0 (per-док centering) ⇒
  intercept: **при k=0: b̂ ≈ 0** (в реальной арифметике 1ᵀyf = 0: X центрирован
  на том же наборе позиций, что и фит; в f32 — уровень округлений: pre-main
  замер: cos-вклад ≤ 5.2e-8, |b̂|/max|W| ≤ 3.4e-6). **При k>0: X центрирован
  по t=0..127, а фит — по t=0..127−k ⇒ 1ᵀX_slice ≠ 0 ⇒ b̂ = −(1ᵀXf·W)/n** —
  малая коррекция. **Cos-вклад b̂ в cos_c (pre-main замер, 32 дока, все J-армы,
  max_d weighted-mean |cos(ŷ+b̂·1, y_c) − cos(ŷ, y_c)| — статистика ТЗ r7):
  k=1 ≤ 2.432e-3, k=2 ≤ 4.512e-3, k=4 ≤ 1.006e-2, k=8 ≤ 1.44e-2** (uniform-арм
  до 5.7e-2 — 1-D фича, floor, не в вердикт; r6-замеры были unweighted:
  k=1 ≤ 2.0e-3 … k=8 ≤ 1.26e-3/1.26e-2 — отсюда калибровка порога 2e-2).
  **Попарный конфаунд** (статистика r7: per-arm weighted-wmean, |wmean_a −
  wmean_b| по J_ARMS): ≤ 7.6e-3 < DELTA_MIN — intercept НЕ доминантный
  конфаунд в попарных сравнениях (r6-замер unweighted max_d-pairwise был
  ≤ 8.7e-3); эффект C2-сдвига (17–28 в z-логитах, O(1) в cos-единицах) всё
  ещё ≫ него. Внимание: весовое отношение |b̂|/max|W| — arm-зависимый proxy
  (J-армы: k=0 ≤ 3.4e-6, k>0 ≤ 5.4e-2; L23-контроль до 4.4e-1) — с DELTA_MIN
  (cos-порогом) сравнивается только cos-вклад. Числовая проверка — S6c-гейт
  §11.2 (cos-вклад).
- **Метрика (ИСПРАВЛЕНИЕ C3):** estimator и метрика в ОДНОЙ (центрированной)
  рамке: per позиция `cos_c(t) = cos(ŷ_c(t), y_c(t))`, где ŷ_c = XW + b̂
  (предсказание centered-цели; b̂ ≈ 0 при k=0, малый при k>0 — см. estimator §3).
  Это тождественно формуле T2.1b
  `cos(pred − ȳ, y − ȳ)`, применённой к centered-рамке предсказания
  (вычитание сводится к 0). Per (L,k,arm) отчитывается ДВЕ агрегации:
  - **первичная — weighted cos:** `wcos = Σ_{d,t} w(d,t)·cos_c(d,t) / Σ w(d,t)`,
    `w(d,t) = ‖y_c(d,t)‖²` (веса зависят только от (d,k,t) — ОБЩИЕ для всех
    армов на позиции: разности армов не конфаундятся весами, правило
    «одинаковая статистика на обеих сторонах»);
  - **вторичная — median cos** (конвенция T2.1b) + R²_c per (doc,k) =
    1 − ‖ŷ_c‖²/‖y_c‖² (отчёт).
  **Все условия дерева вердикта и k=0-гейта используют первичную (weighted)
  метрику** (зафиксировано; отклонение от T2.1b, где primary была median —
  это и есть исправление C3).

## 4. k=0 контроль (SUBSPACE_CONTROL)

Контроль — первичная (weighted) метрика: per (L, d, t) diff = w·(cos_top −
cos_weak); per слой статистика = weighted mean по (d,t). Doc-bootstrap
(B=1000, SEED=0, ресэмплинг доков с повтором; per выбор: weighted mean по
позициям выбранных доков) CI. **Pass = НИЖНЯЯ граница CI > 0 на ВСЕХ 4
слоях (AND-over-l; односторонний, конвенция T2.1b C1).** Провал → вердикт
`SUBSPACE_CONTROL_FAIL` (k≥1 отчитывается, не интерпретируется как ответ на
T2). Ожидание: не пререгистрировано (оба исхода информативны); есть
подтверждённое ожидание для weak_mid-пары (C1c) — см. §9; для пары гейта
(weak-tail) — «ожидается, не измерено» + pre-run замер §11.0.

## 5. k≥1 (ответ на T2) — дерево вердиктов (наследовано от T2.1b §7)

KS_RUN = (0,1,2,4,8); k=0 — только гейт §4. **DELTA_MIN = 0.01** (как в
T2.1b; калибровка уровня — блок `delta_min_calibration` в артефакте, по
первичной метрике). Приоритет: **control → (a) → (b) → (c) → (d) →
NOT_RESOLVED**; weak_mid/uniform/ker-J/L23 — контроли, не в вердикт.

- **SUBSPACE_CONTROL_FAIL:** гейт §4 не пройден.
- **(a) SCRATCHPAD (T2 жива):** на **≥2 из k ∈ {2,4,8}**: weak > max(pca,
  haar) per (l,k) **односторонне: нижняя граница doc-bootstrap CI (первичная
  weighted-метрика) > 0**, для ВСЕХ 4 l (AND-over-l), И weighted mean
  (weak − max(pca,haar)) per (L,k) > DELTA_MIN. (T2.1b: median > DELTA_MIN —
  заменено primary по §3; остальное — дословно.)
- **(b) DEADWEIGHT:** на ВСЕХ (l,k) ∈ 4×4: |wcos(top)| ≤ DELTA_MIN И
  |wcos(weak)| ≤ DELTA_MIN.
- **(c) REDUNDANT_BASIS:** на всех (l,k) все парные CI между {top, weak, pca,
  haar} содержат 0 И на k=1 top ≥ max(остальных) по wcos (без значимости).
- **(d) METRIC_UNINFORMATIVE:** max по (l,k) max-парная |wcos-разность|
  между {top, weak, pca, haar} ≤ DELTA_MIN.
- Иначе **NOT_RESOLVED** (смешанные; детали в артефакте).

**Report-only (вне дерева):** parные CI `weak_vs_top` per (l,k) (прямой T2-
контраст «читается ли будущее из weak лучше чем из top» — T2.1b не имел этой
ветки в дереве; добавлено как отчёт, не влияет на вердикт) и weak_vs_weak_mid.

## 6. lk4, N8, L23 (наследованы от T2.1b)

- **lk4:** канон `exp64_at_battery.py:23-30` (S4: 90/4096; wiki 8 / math 41 /
  code 36 / chat 5); отчёт: n_hits per k, U128-coverage per k с n и
  doc-bootstrap CI (B=1000); one-hot vs y_c cos — `non_informative: true`
  (D3; вне условия (a)).
- **N8 (report-only):** cos_c per (l,k,arm) на lk4-hit (k=0 канон) позициях,
  все 6 main-армов, без интерпретации.
- **L23-контроль (report-only, sanity):** pca/haar/uniform per k на
  выходном слое (здесь строится z) — «будущее не читается из выходного
  слоя» (k=0 должен читаться сильнее k≥1).
- **Report-only (новый, C1b из диагностики):** локальный per-(l,t) контроль —
  frac_t_top_gt_weak + mean diff per слой (локальные per-(l,d,t) базисы,
  32 дока, per-doc LOO). **НЕ гейт** — обоснование: (i) базис T2.1c
  стационарный по построению, конфаунд C1 (вращение по t) снят структурно;
  (ii) локальный тест с n=32 per (l,t) шумный (C1b: L6 frac 0.49, L12 0.34 —
  как гейт убил бы прогон на шуме); (iii) он измеряет ЛОКАЛЬНЫЙ дизайн, а не
  стационарный. Роль — robustness-отчёт: если top>weak держится и локально в
  большинстве t — разбиение не артефакт усреднения.

## 7. Гейты (до записи артефакта; провал = артефакта нет)

- **S1** sha: те же 7 (AT2-пре-рег §7; список в T2.1b-пре-реге §1, зафиксирован
  в коде).
- **S2** z-маршрут: mean KL(softmax(z_full)[U128] ‖ P128_RAW) < 1e-5 на 2 дока
  × 128 позиций (гейт) + KL на всех 32 (отчёт: mean/min/max).
- **S3** ориентация: hi (mean P(next-token), top1) > 0.1/0.1; flip lo <
  0.05/0.05; пул 32 дока × 127. **Статус: exploratory** (пул post-hoc из r1;
  слово «пре-регистрирован» не применять).
- **S4** lk4-канон: 90/4096, per-домен 8/41/36/5 (жёсткое равенство).
- **S5a** r_final rematching == 7 (жёсткое; иначе STOP — §2).
- **S5b** eig-закрытость стационарного Gagg: per слой, для векторов top/weak/
  weak_mid (21 вектор per слой): ‖Gagg·v − λ·v‖ / (|λ| + 1e-30) < 1e-9 (f64).
  (Алгебра eigh; sanity-гейт.)
- **S6** P-zero: ridge с нулевыми фичами (32 LOO-фолда per k, по 31 train-дока в каждом) ⇒ W_X = 0
  (rel < 1e-10) И intercept = mean(y_c,f) (rel < 1e-10; для centered-цели
  mean(y_c,f) ≈ 0 — assert на вычисленное значение, как в T2.1b R2E-3;
  assert на ВЕСА, не на cos).
- **S6b** R²-floor: константный предиктор in-sample ⇒ R²_c = 0 per (doc,k);
  TSS_c > 0 и |R²_c| < 1e-12.
- **S7** полнота (явная формула; «заявленные армы ↔ посчитанные клетки»):
  ```
  KS = (0,1,2,4,8); позиций на док = 625 = Σ_k (128−k) = 128+127+126+124+120
  arms_main (4 l) = 6: {top, weak, weak_mid, pca, haar, uniform}
  arms_L23        = 3: {pca, haar, uniform}
  S7_expected = 4×32×625×6 + 32×625×3 = 480000 + 60000 = 540000
  ```
  Гейт: множество ключей per_pos == множество, порождаемое формулой (ТОЧНОЕ
  равенство) И len(per_pos) = 540000. Артефакт отчитывает: `S7_expected`,
  `arms_main`, `arms_L23`, `k`, `weak_mid_in_verdict: false`,
  `uniform_in_verdict: false`.
- **S8** **diagnostic anchor (НОВОЕ, якорь провенанса):** точное воспроизведение
  конфигурации диагностики C1c (stationary top vs weak_mid [r..2r по убыванию],
  centered-цель, full-pool LOO-ridge, НЕ-взвешенный mean CI, B=1000, SEED=0)
  → per-слой CI top−weak сверить с `results/AT2b_diag/AT2b_diag.json`
  (`c1.stationary_pooled_cd`): **max |diff| ≤ 1e-6** (ожидается 0.0 — тот же
  4-тредный thread-pinning, f32 GEMM; **обязательная установка: ТОЖЕ, что в
  AT2b.py/диагностике — OMP/OPENBLAS/MKL_NUM_THREADS=4 и
  torch.set_num_threads(4) ДО импортов numpy/torch** — иначе порядок редукции
  GEMM другой и diff ~1e-5 (поймано в диагностике r1/r2)). Значения-референс (из артефакта,
  полный float64):
  - L6: mean +0.292580, CI [+0.271309, +0.314404]
  - L12: mean +0.235641, CI [+0.209705, +0.265126]
  - L18: mean +0.344870, CI [+0.301765, +0.386617]
  - L22: mean +0.361038, CI [+0.339858, +0.384186]
  Провал = STOP (T2.1c не является корректным расширением просмотренной
  диагностики). В smoke S8 не гейтится (пул 2 дока ≠ 32) — объявлено.
- **S9** детерминизм: SEED=0 всюду (haar, uniform, bootstrap, выборка).
  **Smoke-перезапуск (отдельный шаг, до main-прогона):** smoke ×2
  (`AT2c_SMOKE=1`: L=(6,), k=(0,1), 2 дока, B=100) → md5 детерминированного
  под-артефакта `results/AT2c_smoke/AT2c_smoke_det.json` (metrics, d_eff,
  per_pos, bootstrap, verdict, r_final; без таймстемпов, sort_keys=True)
  **бит-в-бит**. Пара md5 фиксируется в results-доке.
- **S10** trackio: run `at2c_t21c_r1`; step = wall-секунды; t=0 config
  (started_at ISO, device=cpu, model, log, revision=r1, script_sha, seed=0,
  boot_b=1000, delta_min=0.01, r=32 — r_final неизвестен на t=0, он — в
  stage-точке subspaces); stage-точки (load, subspaces — с r_final/rematched,
  ridge, lk4, verdict, artifact); лог-файл пишется ДО финальной точки;
  финальная t=duration c finished_at + status (ok|crash|interrupted) +
  `log_md5` (md5 записанного лог-файла — обязательное поле trackio p.5) в
  `finally`. **Исправление NIT-2 T2.1b:**
  артефактное поле времени сборки называется `assembled_at` (не `started_at`);
  start — только в trackio-config и log-строке.

## 8. Цена

Загрузка ~20 с; per-t eigh ×16384 (4 l × 32 дока × 128 t, eigh 128×128 —
нужен для rematching (t=0) И C1b-report (локальные per-(l,d,t) базисы))
3–8 мин (диагностика: весь прогон 378 с включал это + 10-арм ridge); ridge:
4 l × 5 k × 6 арм + L23 × 5 k × 3 арм = 135 клеток × 32 фолда × мелкие
солверы ≈ 2–4 мин; C1b-report: 4 l × 128 t × 2 арм × 32 дока × tiny-solvers
≈ 1–2 мин; lk4 + bootstrap — минуты. **Итого ≈ 10–20 мин CPU (4 тр.), пик
памяти ~18–20 G** (z_full 4G + p_soft 4G в S2/S3-фазе — высвобождаются до
subspaces; ew_cache per слой ~2.7G, высвобождается per слой; диагностика
прошла с RSS 17.9G). Один прогон; перепуск — только по гейтам/крашу.

## 9. Ожидания (не гейты)

- **k=0 контроль: проход на всех 4 слоях ОЖИДАЕТСЯ, но основание — для другой
  пары армов:** диагностика C1c измеряет top vs **weak_mid** (средний пояс,
  позиции r..2r): НЕ-взвешенный CI +0.21…+0.36. Гейт §4 использует top vs
  **weak-tail** (слабейшие r не-нулевых) — прямой исход для tail-пары не
  измерен ни в одном артефакте. Ожидание: хвост слабее среднего пояса по
  энергии (монотонность порядка собственных значений Gagg: λ_tail < λ_belt)
  ⇒ readout хвоста хуже ⇒ top > weak-tail тем более;
  но это не доказанное тождество — дешёвый **pre-run замер** (§11.0) даст
  прямые цифры до запуска. Если контроль всё же провалится — это РЕЗУЛЬТАТ
  (информативен против T2), не баг: S8 гарантирует, что пайплайн = просмотренная
  диагностика (для weak_mid-пары).
- **k≥1: неизвестно — это и есть вопрос T2.** Уровни ожидается ~0.01–0.05
  (T2.1b-полоса); DELTA_MIN = 0.01 достижим, но не гарантирован.
- Haar/uniform ≈ 0 (центрированная рамка, изотропия) — контроль на артефакте.
- C1b-report: если top>weak держится локально в большинстве t — разбиение не
  артефакт усреднения (отчёт).

## 10. Что НЕ делается

- Никаких правок AT2.py/AT2b.py/диагностики/их артефактов (заморожены;
  диагностика — read-only референс для S8).
- Никаких новых форвардов, GPU, переобучения. Store/кэши — read-only.
- P-взвешенный стационарный базис (новая конструкция — T2.1d+, отдельный
  пре-рег), per-doc стационарные базисы, другие r-правила.
- Локальный per-(l,t) контроль в дереве/гейтах (только report — §6,
  обоснование там).
- T2.2/T2.3/T2.4 — НЕ эта клетка.

## 11. Смоки (до main-прогона, отдельный шаг)

0. **Pre-run замер (обоснование §9, НЕ гейт):** k=0-контроль для пары
   top vs weak-tail на полном пуле (4 l × 32 дока, LOO-ridge, weighted,
   B=1000, SEED=0) — результат фиксируется в results-доке ДО запуска
   main-прогона (дешёвый exec-фрагмент того же кода, ~1–2 мин). Если замер
   показал fail на каком-то слое — решение о запуске принимает супервайзер
   (control-fail информативен, но T2 останется неразрешённым).
1. **Детерминизм (S9):** smoke ×2 → md5 дет-под-артефакта бит-в-бит.
2. **Деградационные (exec-фрагменты, не полный прогон):**
   - S6 (P-zero): нулевые фичи → W_X = 0, intercept = mean(y_c,f) (exec
     точного кодового фрагмента ridge, 1 слой, 1 k, 32 фолда).
   - S6b (R²-floor): TSS_c > 0, |R²_c| < 1e-12 per (doc,k).
   - **Centering (новый designed-сценарий):** |mean_t y_c(d,t)| < 1e-5 per
     (doc,k) (f32); **cos-вклад b̂ (J-армы, L ∈ {6,12,18,22}): per (L,k,arm)
     max_d weighted-mean |cos(ŷ+b̂·1, y_c) − cos(ŷ, y_c)| ≤ 0.02 И попарный
     конфаунд (J_ARMS) ≤ DELTA_MIN**, где попарный конфаунд = max по (L,k)
     от max-по-парам |wmean_a − wmean_b|, wmean_arm = Σ_{d,t} w(d,t)·|dcos_arm(d,t)|
     / Σ w(d,t) (per-arm weighted-wmean — количество, конфаундящее попарные
     wcos-контрасты вердикта; r7-уточнение статистики, см. ниже). max_d-
     попарное различие — report-only (грубая верхняя граница, не конфаунд
     контраста). **r7-уточнение (пост-r3):** r6-реализация считала конфаунд как
     попарное различие max_d-значений; эта же статистика, посчитанная на
     weighted-dcos (по формуле ТЗ), дала 1.03e-2 > DELTA_MIN (порог был
     откалиброван по unweighted-замеру 8.7e-3); verdict-количество (per-arm wmean) = 7.6e-3 < 0.01 — claim
     «intercept НЕ доминантный конфаунд в попарных сравнениях» выполняется;
     r7 фиксирует статистику конфаунда = wmean-разность (порог DELTA_MIN без
     изменений), max_d-значения остаются в гейте (dcos ≤ 2e-2) и в отчёте.
     (pre-main замеры weighted: dcos max_d ≤ 1.44e-2, конфаунд wmean ≤ 7.6e-3;
     пороги: 2e-2 = 1.6×1.26e-2 (r6-калибровка), DELTA_MIN = 0.01 (запас 32%);
     smoke-пул — подмножество 32 доков, замер smoke ≤ 32-док замера). Uniform-арм и L23-контроль — report-only (uniform: 1-D
     фича, intercept механически велик; L23 — не входит в клетку T2). Весовое
     отношение |b̂|/max|W| — report-only. Превышение cos-вклада = STOP —
     claim сломан).
   - **Arm-swap (новый designed-сценарий, exec-фрагмент, 32 дока, L6, k=0,
     2 арма):** пометки top/weak переставлены → k=0-контроль ОБЯЗАН FAIL
     (CI lower < 0). Премисса: pre-run замер §11.0 показал проход top vs
     weak-tail на L6 (lower > 0) — swap инвертирует знак разности (тот же
     подпространства, метки поменяны), lower обязан стать < 0. Если §11.0
     fail — arm-swap не проводится (премисса не выполнена; сам main-прогон
     даст control-fail — информативный исход).
   - S8-anchor в smoke НЕ проверяется (2 дока ≠ 32) — объявлено в §7.

## 12. Провенанс (что на чём стоит)

- Диагностический артефакт `results/AT2b_diag/AT2b_diag.json` (sha 617f971e…,
  git) — валидация V1–V4 (бит-в-бит прогон r1), C1a/C1b/C1c, C2/C3 числа;
  внешнее 2-е ревью — docs/AT2b_review.md + ответы docs/AT2b_review_r2_answers.md.
- T2.1b: docs/AT2b_prereg.md, results-док docs/AT2b_results.md,
  results-ревью docs/AT2b_results_review.md (VERDICT: REVISE, все MINOR
  закрыты, ответка docs/AT2b_results_review_answers.md).
- T2-гипотеза и дерево — docs/AT2_prereg.md §8 + docs/AT2b_prereg.md §7
  (наследуются дословно, кроме замены median → primary weighted по §3).

## 13. Схема артефакта (results/AT2c/AT2c.json)

meta (started_at → trackio, `assembled_at`, script_sha, revision, r_final,
rematched, seed, device, metric: "weighted_cos primary + cos_median secondary",
target: "per-doc centered y_c", basis: "stationary per-layer Jagg"),
gates S1–S8 + S6/S6b детали, s2_kl_report, s5 (r_final, d_eff, gagg_closure),
s7 (S7_expected, len, set_eq, arms), **s8_anchor** (per-L: got_ci, ref_ci,
max_diff, ok; ref_source: sha диагностики), metrics (per "L|k|arm": wcos,
cos_median, R2_mean), per_pos_cos (540000), bootstrap (control weighted CI +
unweighted CI, control_ok, pairwise_CI per (L,k) incl. weak_vs_top report,
floor_uniform, lk4_coverage), lk4 (hit_total, per_dom, n_hits_per_k,
onehot_med, onehot_non_informative), n8, l23, report_local_per_lt (C1b),
delta_min_calibration, verdict + verdict_detail, wall_sec.
Артефакт ~20–25 MB → НЕ в git (правило >10MB); sha + бэкап /tmp/recovered/AT2c/.
