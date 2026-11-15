# Prompt внешнего ревью: AT0_1 r7-аннотация (code + artifact)

Роль: внешнее ревью (round 3) post-hoc аннотации AT0_1 (T0.1 accounting cell). Ты НЕ автор
правки. Задача — независимая верификация: (1) кода аннотации, (2) артефакта, (3) ответа.
Вердикт по каждому пункту C1–C7 исходного ревью (docs/AT0_1_review.md, round 2) + свои
находки. Формат вывода: docs/dynamic/AT0_1_r7_REVIEW.md (таблица вердиктов + список находок
MAJOR/MINOR + итог PASS/FAIL).

## Что случилось (контекст, коротко)

- AT0_1 — экспериментальная клетка T0.1 (96 доков × 5 слоёв × 3 объекта × 2 статистики = 2880
  клеток; store `results/AT0/store/23281c64cc418ff1.json`, run-скрипт
  `experiments/AT0_1_accounting.py` sha `de305869…`).
- Ревью round 1 (docs/AT0_1_review.md) → правки r5; ревью round 2 (тот же файл, разделы C1–C7)
  → текущая аннотация r7.
- Ревью round 2 нашло: C1 — MA(1)-z тестировал неверный null (mean_band=0 вместо mean_band=f_dc),
  v_tangent z=381 — DC-артефакт; C2 — `tpu_dev` (max_rel_dev_reconstruction) — тавтология
  (U·hn ≡ h), не provenance-гейт; C3 — claim r6 (b0f74312) не воспроизводится из своего
  meta.script_sha256 (de305869 даёт r5-claim); r5-claim sha нигде не записан; «числа прогона
  не менялись» заявлено, не проверено; C4 — SE на ambient D=1024 занижен (d_eff << D);
  C5 — SE_formula «/98» vs код «/88» (K_MAX−8); C6 — T_none_full = (r−1)/2 − (N−1)f_dc
  (Gram-тождество) лежит в одном dict с настоящими excess-столбцами; C7 — self-audit сверял
  не те числа (evidence_own_w0, а не l23_meta_vs_recomputed).
- Исполнитель: перепустил r5 (de305869, threads=4) для восстановления r5-claim (bit-eq store +
  run.log), закрепил r5/r6 claim'ы в results/AT0/store/, написал
  `experiments/at0_1_r7_annotation.py` (аннотация r6→r7 с assert-самоаудитом), запустил через
  run_one.sh, получил r7 claim `results/AT0/T0_ACCOUNTING.json` + manifest
  `results/AT0/AT0_1_manifest.json` + пин `results/AT0/store/claim_r7_<sha>.json`.

## Что ревьюить (файлы, актуальные на момент ревью)

Код:
- `experiments/at0_1_r7_annotation.py` — аннотация r6→r7 (само-аудит = assert'ы).
- `experiments/at0_1_tpu_dev_recompute.py` — r6-скрипт (исторический; C2 — тавтология).
- `experiments/AT0_1_accounting.py` — run-скрипт r5 (sha de305869, НЕ должен быть изменён —
  проверь sha!).
- `prism/field.py` — vpath/dcurve/acf_moments/f_dc (sha в claim gates.G1prime_field_sha256 =
  bcc607c561…; проверь).

Артефакты:
- `results/AT0/T0_ACCOUNTING.json` — r7 claim.
- `results/AT0/AT0_1_manifest.json` — r7 manifest (claim + manifest-блок).
- `results/AT0/store/claim_r5_<sha>.json`, `claim_r6_b0f74312.json`, `claim_r7_<sha>.json`,
  `23281c64cc418ff1.json` (store), `runlog_r5_pre_rerun.log`.
- `results/AT0/AT0_1_run.log` — log перепуска r5.
- `results/AT0/AT0_1_artifact_hashes.json` — пины.

Доки:
- `docs/AT0_1_review.md` (round 2, C1–C7 — исходные требования),
- `docs/AT0_1_review_answer.md` (ответка r6 — история),
- `docs/AT0_1_review_r7_answer.md` (ответка r7 — проверяемые утверждения).
- Лог запуска: `/tmp/at01_r7.log` (stdout r7: self-audit отчёт JSON + маркер
  R7_ANNOTATION_DONE). Р7-claim: `2230226141a9a697fa8ba6118f0a5585efc1b8ea650e8f3a3a377e75a8d04db2`.

## Обязательные проверки (минимум)

1. **Цепочка provenance (C3):**
   - sha256 `AT0_1_accounting.py` == de3058693022320876da8028c4e63dd1ef57ca8fe9dc565f08f49d396a763f20.
   - sha256 store == 23281c64cc418ff1…; sha256 r6-claim == b0f74312416285207ba99ae93ee75ecc446938b8b61ba82a19a04ac36b913994.
   - r5-claim: существует в store/, его sha совпадает с claim.meta.r7_annotation.chain.r5_claim_sha256;
     run.log перепуска == `runlog_r5_pre_rerun.log` (sha e6dcd6fc62aeced0…) — bit-eq.
   - Сам проверь diff r5→r6 и r6→r7 (например, json-сравнением по листам) и сверь с
     задекларированными наборами путей (R6_ADDED/R6_CHANGED; R7_ADDED/R7_REMOVED/R7_CHANGED
     в r7-скрипте). Любая лишняя/недостающая позиция = MAJOR.
   - manifest r7 == claim r7 + manifest-блок (claim_sha256 в блоке == sha claim-файла).
   - Вывод: воспроизводится ли r7 claim из (de305869 + r6-пин + r7-скрипт) без потери цепочки.

2. **C1:** в r7 claim `prereg.4_plateau_vs_fDC.<obj>.null_analytic_ma1`:
   - z_raw_band.medians == 381.3249665142485 / −0.21574175798491185 / −0.21562117252716537
     (v_tangent/h_raw/du_unit — значения r5/r6) — и пересчитай z_raw_band по store для 3–5 доков
     вручную (rho_Tk[8:96], f_dc, N, D, se = sqrt(Σ 1/((N−k)D))/88) — bit-eq?
   - z_excess = s_doc/se (s_doc = mean_band − f_dc): пересчитай для v_tangent — медиана ≈ 10.55
     (ручной счёт ревью 10.5). Проверь формулу E_doc на корректность (T-k нормировка:
     c(k≥2) ≈ ‖m‖²/D, c(0) ≈ (‖m‖²+var)/D ⇒ E[rho_Tk(k≥2)] = f_dc).
   - Есть ли z_raw_band в per_doc_ma1 и bit-eq со store per_doc.null_ma1_z.z (выбери 3 дока).

3. **C2:** `gates.G0a_tpu_hn_field` в r7: нет ключей max_rel_dev_reconstruction*; есть
   reconstruction_roundoff_identity == 3.7939126356469717e-16 + detail со статусом «тождество,
   не гейт». Проверь разбор тавтологии по веткам max(·,1e-12) сам (hn>1e-12 и hn≤1e-12).
   Убедись, что nowhere в claim/answer tpu_dev не назван гейтом.

4. **C4:** d_eff per-doc — participation ratio ковариации строк объекта (zero-mean). Проверь
   пересчёт для 2–3 доков (eigvalsh((x−mean)ᵀ(x−mean))), сравнись с per_doc_ma1.d_eff (bit-eq?
   или rel 1e-12 — уточни, какой tolerance честный). Проверь se_factor = sqrt((1+2ρ₁²)D/D_eff),
   ρ₁ = θ/(1+θ²) (θ из store null_ma1_z). Оценки: D_eff(med) 25–35, фактор 5.5–7.2.
   Проверь аргумент ответа, что r_eff(0.05)=178 (addentum) — другая метрика.

5. **C5:** SE_formula в r7: «/ 88» + пояснение; код (K_MAX−8) не тронут (sha скрипта).

6. **C6:** T_none_full удалён из excess_median/excess_mean; identity_check_G1 содержит
   T_none_full_median/mean, max_abs_sum_rho_full_minus_(r-1)/2, n_docs; значения bit-eq с r6
   claim (−0.5023582212969491 и т.п.); t0_headline_note дополнен (r_v≈24.9, r_du≈0.0048).

7. **C7:** self-audit r7 (лог /tmp/at01_r7.log): все assert'ы прошли (маркер
   R7_ANNOTATION_DONE, RC=0); проверь, что l23_meta_vs_recomputed (max_abs_dev_by_source
   {P2_corpus: 5.5475605402977024e-05, P1_own: 0.00020829041663661396}, worst own_w4) —
   пересчитай по P1_own.npz/P2_corpus.npz + meta-файлам (dcurve(V,1..K)[0] median vs
   curves.23.dip) — bit-eq?

8. **Код r7-скрипта:** ветки/guard'ы (th is None — есть ли покрытие? dead code?);
   json.dump (NaN/None — jclean?); детерминизм (запустишь ли ты скрипт повторно в /tmp —
   sha claim совпадёт? ВНИМАНИЕ: повторный запуск перезапишет results/AT0/T0_ACCOUNTING.json —
   сначала сделай бэкап в /tmp или запущай из копии репо); согласованность формул с
   AT0_1_accounting.py (se, theta, pool, T_none_full); assert'ы реально останавливают написание
   (проверь, что запись — после всех assert'ов).

9. **Ответка (docs/AT0_1_review_r7_answer.md):** каждое число — из артефакта/лога (не из
   головы); нет ли заявлений, не подтверждённых артефактом; согласована ли позиция
   «вердикты P4/P5 не переписаны» с исходным ревью (ревью не требовало переписывать вердикты —
   проверь).

## Ограничения

- НЕ меняй файлы репо (только чтение + вычисления в /tmp).
- Числа в вердикте — только из фактических пересчётов/артефактов.
- Если находишь расхождение — MAJOR/MINOR + точный path + оба числа.
- В конце: «Вердикт: PASS/FAIL» + список MAJOR/MINOR.
