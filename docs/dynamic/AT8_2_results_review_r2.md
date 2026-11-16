# Внешнее ревью ROUND 2: AT8_2 (T8.2) — фиксы по results-ревью r1 + артефакт v2

Ревьюер: внешний субагент (read-only; модель не запускалась; все числа пересчитаны
независимо из store/claim/логов; exec-тесты прогнаны /opt/miniconda3/bin/python3.12).
Нумерация: M-N* — новые MINOR, N-* — новые NIT (r1-NIT-1..5 — исходные замечания,
их закрытие — в разделах A–E).

## Объекты (sha256, проверены перед чтением)
| Объект | sha256 |
|---|---|
| R1-ревью `docs/dynamic/AT8_2_results_review.md` | `6acfdd64317ed808d4895caca3d72774e8825120f405b3d17470f576f6c1ac15` |
| TZ v1.2 `docs/AT8_2_TZ.md` | `2e38f3e00f9495b6434e777931ec1bb793369d0e97c692e990a113b05de39755` |
| `experiments/at8/AT8_2_recovery.py` (r2, будущие прогоны) | `eb34b49a89c6d141fc4fa5e371d949afbf8296c8a314dff3e653f4250e7395b5` |
| `experiments/at8/at8_2_analysis.py` (общий фрагмент) | `3c8212e86f585387f1c2a3238cb45b3e109e00d11cf89bd1a5070d206ae8f056` |
| `experiments/at8/AT8_2_v2_reanalysis.py` | `d76f390713021fc59323090c479a22e14db70871c5d18f35dd5e3b36cb6a873d` |
| `experiments/at8/test_at8_2_resume.py` | `6b14da7066a8004dd91a204413257789874131b0fbdb12d7b58292229135eb94` |
| `experiments/at8/test_at8_2_cells.py` | `a87d4f5cf202087cc396210fd99e660a1bb229933abb339c04c99690e8c97c16` |
| `experiments/at8/test_at8_2_verdict.py` | `f23b67f16162273a38b82968f4e1fa6f046276b27cf93ebe711c808649fce4c2` |
| `results/AT8/AT8_2_RECOVERY_v2.json` | `d073a1f4653d37e969d9bdf5015ea5de50923364fa79d78e266e1ea640541d0a` |
| `results/AT8/AT8_2_v2_manifest.json` | `f4e2396c0871937cb1e0e229945841c03435ce47d0efb53af04f5e70238e4a4c` |
| `logs/AT8/AT8_2_v2_run.log` | `0839e4469882a9d14561a2b07a4fdd9c79d76c7570a3943b14d776875ea3fd77` |
| R1-claim `results/AT8/AT8_2_RECOVERY.json` (не тронут) | `779008c46a770b0c2f96b8113e996170e9382cd88147d1754f8f70533440d8c0` |
| Store `results/AT8/AT8_2_RECOVERY_store.json` (не тронут) | `69f32421a22d7fba55b250c4b65f258ec1b5cfb4daaa7cb6fc34edeebd4365ea` |
| R1-manifest `results/AT8/AT8_2_RECOVERY_manifest.json` | `294db874dd710571f1736a863fbf08b42e63f9aa498cde389746cf3a8dc8482b` |
| R1-лог `logs/AT8/AT8_2_run.log` (не тронут) | `dea5afb3b9733a80567e3d6a1a009ee73ea2642c4c9505ec481ff379f2d49d45` |
| `experiments/kaggle_e64gpu/e64_gpu.py` (S7) | `8c517e18f801fa877dfde9265ed5ce18df1e7d3af705194fc01ff6b150cf7633` |

Замечание по sha R1-ревью: в r2-промте, TZ §9 и docstring v2-скрипта р1-ревью цитируется
как «sha 26acfdd6…», но файл на диске = `6acfdd64…` (расхождение на первый символ —
опечатка при транскрипции; содержимое файла совпадает с r1-ревью по тексту, VERDICT:
REVISE + маркер REVIEW_AT8_2_DONE на месте). См. N-1.

Метод пересчёта: независимая реконструкция ячеек из сырых dip-кривых store (без
at8_2_analysis), затем медианы/регрессия/гейты — 0 расхождений с v2-claim по 872
проверенным ключам (128 ячеек).

---

## A. Закрытие MAJOR-1

**1. TZ v1.2 §5 — OK.** Популяция оси A зафиксирована: «ВСЕ измеримые (doc,W) с ℓ_rec;
not_recovered имеют ℓ_rec=D_max — цензурированное значение, входит в медиану; вариант с
исключением цензурированных (только recovered/no_effect) — вторичный, в артефакте с явной
меткой censored_excluded». Конвенция «медиана доков» для объединённого зафиксирована:
«сначала медиана по W внутри дока, потом медиана по докам … прямой median по всем
(doc,W) — не конвенция». Формулировки в §5 (ключевая таблица) и в блоке «Ось A»
дублируют друг друга без противоречий.

**2. V2-claim W64 — OK.** Независимый пересчёт из store (полная популяция, по 5 доков на
дозу — wiki_4/5/6 не входят: D_max=2/<0): lrec = {2: 60.0, 4: 39.0, 8: 49.0, 16: 62.0} —
точно совпадает с v2 `axis_A.W64.lrec`. `lrec_censored_excluded` = {61.5, 0.0, 49.0,
62.0} — это r1-числа (r1-claim W64.lrec идентичен), теперь явно помечены вторичными.

**3. ratio/b/r2 (W64, полная) — OK.** Пересчёт: ratio = 62/60 = 1.0333, b = 0.7391,
a = 46.9565, r2 = 0.1842 — совпадает с v2-claim и с независимым пересчётом r1-ревьюера
(1.0333). DIFFUSION недостижим (ratio ≤ 1.03, r2 ≤ 0.184).

**4. combined_secondary — OK (разность с r1-ревьюером = конвенция, не ошибка).** Пересчёт
по конвенции TZ v1.2 (медиана по W внутри дока → по докам, 8 доков на дозу): {2: 35.0,
4: 51.0, 8: 60.0, 16: 62.75}, ratio = 1.7929 — совпадает с v2-claim. Прямой median по всем
(doc,W) (контрафактический пересчёт r1-ревьюера): {44, 48, 55, 62}, ratio 1.4091 —
разность полностью объясняется конвенцией, теперь зафиксированной в TZ v1.2 §5; в
v2-claim combined снабжён `note` с конвенцией. combined — справочный: `compute_verdict`
читает только `axis_A["W64"]` (проверено по коду), в вердикте не участвует.

**5. Вердикт по ОСНОВНОЙ (полной) оси A — OK.** Условия ELASTIC по полным числам:
ratio 1.0333 ≤ 2 ✓, frac_to_global 0.2328 ≤ 0.4 ✓, n_pos = 4 ≥ 2 ✓ (все 4 дозы с
медианой > 0), гейты NOT_RESOLVED: n_meas_doc_n = 19 ≥ 6 ✓, not_recovered 55/116 =
0.4741 < 0.5 ✓. match_pre_reg=True. n_pos=4 подтверждено пересчётом.

## B. Закрытие MINOR-1

**6. Гейты — OK.** `compute_verdict` (at8_2_analysis): «мало пар» — `n_meas_doc_n < 6`
счёт по `c["W"] == w_axis` (W=64, основная ось); «большинство not_recovered» —
`n_notrec > len(with_status)/2` по полной измеримой популяции (все W, D_max≥8 — 116
ячеек, та же популяция, что у оси B). TZ v1.2 §5 фиксирует обе популяции + объяснение,
почему W64-чтение (75%) — diagnostic, не гейт (вырождение под шумовым полом, пре-рег §4-ноут).
`axis_W64_notrecovered_share = 0.75` присутствует в v2-claim `diagnostics` (не гейт —
проверено по коду).

**7. Реальные числа — OK.** Пересчёт: not_recovered = 55 из 116 с rec_status = 0.4741 <
0.5 — гейт не сработал (правильно: 47.4%, «чуть ниже большинства» — как в r1-ревью).
n_meas_doc_n (W=64, l_rec>0) = 19 ≥ 6 — пересчитано из store (20 ячеек W64 с l_rec,
одна — l_rec=0).

**8. Границы гейтов — OK (с N-2).** `test_at8_2_verdict.py` прогнан: PASS. Покрывает:
ELASTIC, DIFFUSION, DILUTION; гейт 1: 4 пары → срабатывает, 24 пары → не срабатывает
(точная граница 5/6 — не покрыта, N-2); гейт 2: 0.625 → срабатывает, ровно 0.5 → не
срабатывает (граница покрыта точно, строго больше). Тест 6 воспроизводит v2-реанализ из
артефакта (ratio 1.0333, frac 0.2328, diagnostics побитово) — защита от дрейфа кода.

## C. Закрытие MINOR-2/3 + NIT

**9. trackio в r2-коде — OK.** Config: `seed: "n/a (deterministic forward, no sampling)"`
(строка 171) + revision/script_sha/device/model/log (AGENTS п.3, 5). Stage-точки
`tstage`: model_loaded (235), docs_loaded (246), forwards_done (306), analysis (334),
artifact (393) — 5 крупных этапов (AGENTS п.4). `log_md5` в finish (строка ~468, md5_file
реализован). `setdefault TRACKIO_DIR` убран — комментарий «TRACKIO_DIR задаёт run_one.sh
(AGENTS п.6) — скрипт его не переопределяет». Наблюдение: прямой запуск без run_one.sh
теперь пишет trackio в дефолтный HF-cache (в r1 было setdefault на .trackio — r1-ревью
приняло); r2-поведение соответствует AGENTS п.6 (скрипт не задаёт сам) — зафиксировано
в N-5, чтобы смоук-прогоны шли через run_one.sh.

**10. resume_plan — OK.** Чистая функция (только имена + store, без I/O), вызывается из
main() (строка 255) и управляет пересчётом (ctrl_resume/dose_resume). `test_at8_2_resume.py`
прогнан: PASS — partial-store (wiki_0 полный: 2 ctrl + 4 дозы → n_resume=6; wiki_1 только
ctrl; wiki_2..7 пустые) + fresh run (n_resume=0). Целостность содержимого при resume —
через sha-верификацию (зафиксировано в docstring: h23_sha не пересчитывается — циклично,
как в r1 MINOR-3; для one-shot-прогона приемлемо).

**11. r1 NIT-2/3/5 — закрыты.** `test_at8_2_cells.py` прогнан: PASS — not_finite покрыта
И NaN, И None (ветки, не срабатывавшие в r1-прогоне); plus not_measurable/no_effect/
recovered/not_recovered/rec_measurable(D_max<8)/to_global (оба исхода). r1 NIT-3: hook в
measure() снимается в try/finally (строки 136–143). r1 NIT-5: n_pos≥2 задокументирован в
TZ v1.2 §5 («Доп. консервативный гейт ELASTIC: n_pos≥2 … зафиксирован в r1-прогоне,
сохранён») и в claim.prereg.verdict_thresholds.

**12. R1-снапшот не тронут — OK.** r1-claim sha 779008c4… == r1-manifest.claim_file_sha256;
store sha 69f32421… == r1-manifest.store_file_sha256 (и r1-manifest sha 294db874…, r1-лог
dea5afb3… — все совпадают с таблицей объектов r1-ревью, т.е. файлы не модифицировались).
v2-manifest ссылается на r1: r1_claim_file_sha256 + r1_manifest_file_sha256 — оба
верифицированы. v2-claim.base_run дублирует связь с r1-прогоном (r1_run_name,
r1_wall_s=257.4).

## D. V2-реанализ как артефакт

**13. Hash-цепь v2 — OK.** v2-claim sha d073a1f4… == v2-manifest.claim_file_sha256; store
69f32421… == v2-manifest.store_file_sha256; r1-claim 779008c4… == v2-manifest.
r1_claim_file_sha256; r1-manifest 294db874… == v2-manifest.r1_manifest_file_sha256.
Все 4 звена + e64_gpu 8c517e18… (S7) — True.

**14. Ячейки v2 == r1 побитово — OK.** 128 ячеек, точное равенство (r1-claim cells ==
v2-claim cells); мой независимый rebuild из сырых кривых: 0/872 расхождений с обоими
claim. Fail-loud самопроверка реально в коде (assert n_diff==0 and not extra → STOP,
RC≠0) и сработала: строка в v2-логе «ячейки: 128 (measurable=120) — пересчёт ==
r1-claim (0 расхождений…)».

**15. V2-лог — OK.** Полная трасса: start (17:09:13Z) → sha-верификация (69f32421/
779008c4, «r1 прогон не тронут») → пересчёт ячеек (0 расхождений) → ось A (полная +
censored) → ось B → diag → VERDICT ELASTIC → written (sha d073a1f4) → маркер
`AT8_2_V2_DONE verdict=ELASTIC match_pre_reg=True (wall 0.1s)` — последняя строка файла.

**16. NaN/inf — OK.** v2-claim: 0 NaN, 0 "inf", 0 null (grep по всему файлу).

**17. Числа v2-лога vs v2-claim — OK.** ratio 1.0333, b 0.7391, r2 0.1842, lrec {60,39,49,62},
frac_to_global 0.2328 (n=116), diag (n_pos=4, n_meas_doc_n=19, n_notrec=55, share_full
0.4741, W64 0.75), VERDICT ELASTIC, wall 0.1s — все совпадают.

## E. Общий код-аудит r2

**18. Единый фрагмент — OK.** Оба скрипта импортируют из at8_2_analysis (recovery:
build_cells_from_store/axis_A/axis_B/compute_verdict/resume_plan; v2: + cells_equal).
Копий логики в скриптах нет (grep np.median/polyfit/def lrec/def _fit/def _ratio: 0
попаданий ни в recovery, ни в v2). TODO/FIXME нет (grep: 0). Единственное косметическое:
`h0 = None` после контрольного блока — маркер двухпроходности (h23 не хранится), no-op,
был и в r1 (N-3).

**19. Гейты S1–S8 не сломаны — OK.** S8 in-process (wiki_1, n=8, W=64, d=0, tol 1e-12,
два независимых forward) сохранён; S2b — одна статистика контроля (h0 один раз на док,
control_sha256 теперь читается из h23_sha store — эквивалентно r1, n_dose_reuse=32);
S2 — точное равенство множеств (assert); S4/S5 — finite-проверки на тех же популяциях;
try/finally добавлен (r1 NIT-3), на логику не влияет. S7 — assert sha e64_gpu.py до exec
(8c517e18…).

**20. Резюме — 1 MAJOR + 3 MINOR r1-ревью: ВСЕ закрыты.** MAJOR-1 — полная популяция
оси A (пересчитано из store: 0 расхождений; совпадает с независимым пересчётом
r1-ревьюера), MINOR-1 — оба гейта по явным популяциям с exec-тестами (включая границу
0.5), MINOR-2 — stage-точки/seed/log_md5, MINOR-3 — resume_plan с exec-тестом. r1
NIT-2/3/5 закрыты (п.11). r1 NIT-1 (ограничения S8) — остаётся задокументированным в
TZ §6 S8 + r1-ревью (допустимо, как в r1-ревью).

---

## Новые замечания

### MINOR (не блокирует PASS; исправить 1-строчной правкой ДО коммита)

**M-N1. TZ §9 цитирует устаревший sha v2-claim.** В «История r1 → v2» (TZ v1.2 §9)
написано: «claim v2, sha 3bb0b324…», но v2-claim на диске = `d073a1f4653d37e9…`
(артефакт перегенерирован после правки конвенции combined в at8_2_analysis — v2-лог
даты 17:09:13Z, v2-manifest и r2-промпт уже содержат правильный d073a1f4). 3bb0b324 —
промежуточная версия, которой на диске нет. Фикс: одна строка в TZ §9 (3bb0b324… →
d073a1f4…). До коммита — обязательно (AGENTS п.8: sha в доках = на диске).

### NIT

- **N-1.** Цитата «sha 26acfdd6…» r1-ревью в r2-промте, TZ §9 и docstring
  AT8_2_v2_reanalysis.py не совпадает с файлом на диске (6acfdd64…) — опечатка на
  первый символ при транскрипции. Косметика; при правке TZ §9 (M-N1) можно поправить
  вместе.
- **N-2.** Тест границы гейта «<6 пар» покрывает 4 (срабатывает) и 24 (не срабатывает),
  но не точную границу 5/6. Гейт — однострочный `n_meas_doc_n < 6`, риск низкий.
- **N-3.** `h0 = None` в recovery-скрипте после контрольного блока — no-op (был и в r1);
  маркер двухпроходности, можно оставить как комментарий или убрать. Косметика.
- **N-4.** В recovery-скрипте log-line resume печатает `len(res)`, а не
  `plan["n_resume"]` (docstring resume_plan говорит «n_resume — для лога»); на одном
  пуле значения совпадают. Косметика.
- **N-5.** Убрав `setdefault TRACKIO_DIR`, прямой запуск recovery-скрипта без
  run_one.sh (смоук) напишет trackio-точки в дефолтный HF-cache, а не в .trackio
  (повтор сценария AT2/AT3, против которого r1-комментарий и ставил setdefault).
  R2-поведение соответствует AGENTS п.6 («скрипты его не задают сами»), но будущие
  смоук-прогоны обязаны идти через run_one.sh.

---

## ИТОГ

Все замечания r1-ревью (1 MAJOR + 3 MINOR) закрыты: MAJOR-1 — полной популяцией оси A
(пересчитано из store: 0 расхождений; совпадает с независимым пересчётом r1-ревьюера),
MINOR-1 — явными популяциями гейтов с exec-тестами (включая границу 0.5), MINOR-2 —
stage/seed/log_md5, MINOR-3 — resume_plan с exec-тестом. V2-артефакт (прецедент AT8_1_v2)
цел: hash-цепь из 4 звеньев, r1-снапшот не тронут, лог с полной трассой и маркером,
0 NaN/inf, лог↔claim совпадает. Единый анализный фрагмент без копий; гейты S1–S8 не
сломаны. Вердикт без изменений: **ELASTIC (match pre-reg)** — теперь по TZ-определению.

Осталось: 1 MINOR (M-N1: устаревший sha в TZ §9 — 1 строка до коммита) + 5 NIT.

VERDICT: PASS

REVIEW_AT8_2_R2_DONE
