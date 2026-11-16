# AT0_3 r3.3 — code + artifact review (fixreview-REVIEW cycle)

Прочитай ЭТОТ файл — это твоя задача. Ты ревьюируешь код + артефакты AT0_3 r3.3 —
фикс-цикл по docs/AT0_3_fixreview_REVIEW.md (вторая, дополненная часть ревью).

## Протокол (обязателен)
- Ответы в чате ≤200 символов (статус только), без анализа.
- Ревью пиши в docs/AT0_3_r33_REVIEW.md: full-overwrite writes по 2–4KB за раз.
- Код читай через grep/sed, 60–120 строк за раз, по якорям.
- Числа: ключевые пересчитай/перепроверь python3 (json.load артефактов);
  эксперимент НЕ перепускать.
- Строка «VERDICT: PASS | REVISE | FAIL» — перед финальным маркером.
- Каждое замечание: где/что/почему/фикс/серьёзность (blocker | правка | косметика).
- ПОСЛЕДНЯЯ строка файла ТОЧНО: AT0_3_R33_REVIEW_DONE

## Контекст
- Ответимому ревью: docs/AT0_3_fixreview_REVIEW.md (sha256 d9f785c5, 823 строки).
  Минимальный патч (§6): E1 count-aware concentration; E2 Fisher + detector null
  calibration; E3 anti-ratio по чётности; E4 corpus claim-gate; E5 source KS sensitivity;
  E6 p-floor wording; M1 KS diagnostics; M3 copy-rate confound; M4 layer-specific;
  N2 p=2 guard; N3 G2 gate; O3 тело median_diff + парный артефакт; §6.6 script binding.
- Ревьюер в том файле ОТЗЫВАЛ свои два пункта (anti_frac odd-p; K-зависимый tail) —
  код уже был корректен (см. §1 файла).
- Предыдущие циклы: r3.1/r3.2/r3.2.1 (см. docs/AT0_3_*.md). r3.2-артефакт был
  store bba6fde3a7e807d4 (удалён; в git — в коммите 6350d25).

## Что сделано (r3.3)
- Патч experiments/AT0_3_area.py (проверь сам: sha256sum — ждём fb9b6c58…).
- Свежий полный прогон патченого скрипта (disk-only, без пере-форвардов):
  results/AT0/AT0_3_run.log, store results/AT0/store/4d0323e9b32f4c59.json,
  claim results/AT0/AT0_3_AREA.json, manifest results/AT0/AT0_3_manifest.json.
- Смоук (2+2 пул): /tmp/at0_3_smoke_r33.log — forward-линии бит-идентичны r3.2
  (/tmp/at0_3_smoke_r32.log), добавились только новые блоки.

## Перепроверенные числа (проверь ключевые по артефактам)
- per_doc: 96 доков, diff с r3.2 store = 0 (disk-only инвариант; doc-meta тоже).
- Primary НЕ изменился: KS stat 0.482544237207078 p=0.000999;
  median stat −0.09031712287071525 (== median_L − median_C; в коде assert-self-check);
  mean d +0.00442 p=0.876.
- Detection (K=96, dL/dC = 30/0, nL/nC = 41/51):
  Fisher as-or-less-likely = 2.120335e-15 (первичный);
  binomial 2.180818e-07 — sensitivity, бит-идентичен r3.2 p_two_sided_exact.
- Concentration (count-aware базис count(k mod p)/K):
  median 5.4076, min 2.3799, mean 5.4138, frac>1 = 1.0;
  legacy 1/p: 6.0622 / 2.4295 / 5.9084 (оба в aggregate: *_legacy_1overp + per-doc
  conc_1overp_legacy).
- Anti-phase (n=30, p=2 исключён): mean anti_ratio 0.2425, median 0.0324,
  28/30 < 1 к собственной базе; even n=12 (10/12 <1), odd n=18 (18/18 <1);
  per-doc в aggregate.mean_anti_phase.per_doc.
- Source sensitivity (те же B=1000/seed=117):
  gen_only 28/36: KS 0.460317 p=0.003, median −0.0787 p=0.014;
  gen_plus_own 41/37: KS p=0.001; gen_plus_corpus 28/50: KS p=0.001;
  all_K96 41/51: бит-идентичен primary KS (0.482544, 0.000999).
- Detector null calibration (B_null=200):
  loopy: shuffle_full 0.0227 (max 0.1443), keep_short 0.0277 (max 0.2189),
  kill_peak 2/30 = 0.067 (same_p 0/30);
  clean: пол 1/201 ≈ 0.005 (все три нуля).
  max ρ_du(k≥2) K=96: loopy median 0.9399 / q90 0.9610 / max 0.9786;
  clean median 0.0640 / q90 0.3170 / max 0.7409 (< порога 0.9).
- G2: 1440/1440, present 1440, ok=True, absent=0 (гейт в gates.G2_cells).
- Domain claim-gate: corpus 16/16 nondeg, 0 токенов → claim=False,
  provenance_limited=True, descriptive_only=True; math 16/16/16 → claim=True;
  chat 11 nondeg → n<12 → False.
- KS diagnostics: primary max_gap_x=0.530483 crossings=1;
  robust max_gap_x=0.528684 crossings=1 (разные — не copy-paste).
- Script binding: meta.script_sha256 == sha256(experiments/AT0_3_area.py);
  manifest.claim_sha256 == canonical sha (sort_keys) файла claim;
  имя store == sha256[:16] содержимого.
- Verdict: H1 = refuted (directional refutation), H2 = confirmed (ассоциация).
- p=2 guard проверен exec-ом (синтетический p=2: p2_degenerate=True,
  top2_res_share=None, агрегаты None-safe, test не строится при n=1).

## Что проверять
1. Каждый пункт минимального патч-списка РЕАЛИЗОВАН в коде (не только в артефакте):
   E1 (базис count/K per-doc + aggregate; top2 = сумма двух баз),
   E2 (Fisher-формула; binomial как sensitivity; selection_note; denominator_note;
   null-калибровка с 3 суррогатами; max-ρ по стратумам),
   E3 (anti_ratio = anti_frac/(n_anti/p); even/odd split; per-doc),
   E4 (domain_claim_eligible = n_eligible AND token_verified; corpus → descriptive_only),
   E5 (4 пула, те же B/seed), E6 (floor-формулировка: не ранжировать по p),
   M1 (ks_diagnostic у всех трёх объектов), M3/M4 (ноты),
   N2 (p=2: top2=None + flag + исключение из anti-агрегата),
   N3 (G2: expected, гейт, явный absent-список),
   O3 (тело median_diff + assert в obj_test),
   §6.6 (script binding).
2. Числа выше совпадают с артефактами (python3 json.load).
3. Честность wording: H2 = ассоциация (не механистический носитель);
   effect size — count-aware + legacy рядом; detector-null результат
   (loopy shuffle 2.3% / kill_peak 6.7% / clean 0.5%; loopy max-ρ median 0.94
   vs clean 0.06) — интерпретация согласована с числами: loopy-профили имеют
   повышенный ρ-базис, детектор НЕ «независимый» — это отражено в
   selection_note/finding/summary?
4. Регрессий в H1 (primary pool, verdict, confounds, tail_prediction, location) — нет.
5. Согласованность: H2-gate (Fisher p + median conc + frac) vs verdict vs summary;
   summary-числа == артефакт-числа.
6. Смоук-diff (опционально): forward-линии идентичны, новые блоки — только r3.3.

## Ограничения
- НЕ модифицируй код/артефакты — только ревью-файл.
- Blocker ⇒ VERDICT: REVISE (или FAIL) с перечнем.
- Финальная строка файла: AT0_3_R33_REVIEW_DONE
