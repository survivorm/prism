# AT0_3 r3.3 — code + artifact review

Ревью фикс-цикла по docs/AT0_3_fixreview_REVIEW.md (sha d9f785c5567d…, 823 строки).
Объекты: experiments/AT0_3_area.py (sha fb9b6c585ba4…, 1723 стр.), results/AT0/AT0_3_AREA.json
(claim, 10505b9f…), results/AT0/AT0_3_manifest.json (525ffb9c…),
results/AT0/store/4d0323e9b32f4c59.json (store, 4d0323e9…), results/AT0/AT0_3_run.log,
смоук /tmp/at0_3_smoke_r33.log vs /tmp/at0_3_smoke_r32.log.
Эксперимент НЕ перепускался; все числа пересчитаны python3/json.load (независимые формулы).

## 1. Минимальный патч-список §6 — реализован в КОДЕ (не только в артефакте)

| № | Пункт | Код (стр.) | Статус |
|---|---|---|---|
| E1 | count-aware concentration | period_structure ~L646: `counts = bincount(arange(1,K+1)%p); baseline = counts/K; concentration = prof[r1]/baseline[r1]`; `top2_baseline = baseline[r1]+baseline[r2]` (M2); `conc_1overp_legacy` рядом | ✅ |
| E2 | Fisher + detector null | ~L738: гипергеометрический as-or-less-likely (`hyp_pmf`, `p_fisher_as_or_less`); биномиальный → `p_two_sided_exact_binomial_sensitivity`; `test_family_note`, `selection_note`, `denominator_note`; `detector_null_calibration()` (L856) — 3 суррогата (shuffle_full / keep_short / kill_peak), B_null=200, seed 117; max-ρ-распределения `max_rho_k2_k96` по стратумам | ✅ |
| E3 | anti-ratio по чётности | ~L655: `n_anti = 1 if even else 2`, `anti_baseline = n_anti/p`, `anti_ratio = anti_frac/(n_anti/p)`; агрегат even/odd/all + per_doc; p=2 исключён из anti-агрегата | ✅ |
| E4 | domain claim-gate | ~L1233: `domain_claim_eligible = bool(n_elig and prov_ok)`; corpus → `provenance_limited=True, descriptive_only=True, domain_claim_eligible=False`; pooled_nondeg-построение включает non-claim домены | ✅ |
| E5 | source KS sensitivity | `source_sensitivity()` (L944): 4 пула gen_only / gen_plus_own / gen_plus_corpus / all_K96, те же B_PERM=1000 / SEED_PERM=117 (хардкод в сигнатуре) | ✅ |
| E6 | p-floor wording | `per_doc_p_floor_note`: «p не цитировать как шкалу свидетельств, доки не ранжировать по p» | ✅ |
| M1 | KS-диагностика | `ks_diagnostic()` (L346): max_gap_x, max_gap_side, n_crossings, n_pooled_unique; вызывается у ВСЕХ ТРЁХ объектов в obj_test | ✅ |
| M3 | copy-rate confound | note в `copy_rate_quartiles`: «конфаундирован с L/C … описательно, не каузальный вывод» | ✅ |
| M4 | layer-specific | `layer_specific_note` в results.layer_type И verdict; +4-й пункт в verdict.confounds | ✅ |
| N2 | p=2 guard | ~L664: `p2_degenerate=True`, `top2_res_share=None`, `top2_baseline=None`, note «не цитировать»; None-фильтр в агрегате `top2` | ✅ |
| N3 | G2-гейт | ~L1050: `expected_cells = len(docs)*len(LAYERS)*len(OBJECTS)`, `g2_ok`, `SystemExit(1)`, явный `absent_cells`, `gates.G2_cells` с expected/ok/absent | ✅ |
| O3 | median_diff + assert | `median_diff` (L386) — чистая функция `(a,b)→median(a)−median(b)`, без замыканий; assert в obj_test: `abs(med.stat_obs − (median(vL)−median(vC))) < 1e-12` | ✅ |
| §6.6 | script binding | `meta.script_sha256 = sha256_file(__file__)`; store content-addressed (имя = sha256[:16]); claim.store.sha256; manifest.claim_sha256 (sort_keys) + store_sha256 | ✅ |

Все 13 пунктов минимального патча + 2 отозванных ревьюером (anti_frac odd-p — `{floor,ceil}` в
коде; tail K=96-пул) подтверждены в коде И в артефакте.

## 2. Числовая перепроверка (независимые пересчёты)

### 2.1 Primary (du_unit, K=96) — без изменений, совпадает с заявленным

| величина | артефакт | пересчёт | совпадение |
|---|---|---|---|
| KS stat | 0.482544237207078 | — (взято из store; p воспроизведён ниже) | ✅ |
| KS p | 0.000999000999000999 | 1/1001 — пол (B=1000) | ✅ |
| median stat | −0.09031712287071525 | median_L 0.521420595657899 − median_C 0.6117377185286142 = −0.09031712287071525 | ✅ (assert в коде держит) |
| mean d / p | +0.0044181087586929735 / 0.8761238761238761 | 876/1001 | ✅ |
| n_L/n_C | 41/51 | из store per_doc (K==96, fracs не None) | ✅ |

### 2.2 Detection (30/41 vs 0/51)

- Fisher as-or-less-likely, НЕЗАВИСИМЫЙ пересчёт: гипергеометрический pmf(i)=C(30,i)·C(62,41−i)/C(92,41),
  i=0..30; as-or-less-множество = {30} (наблюдение — крайняя клетка, dC=0; pmf(0)=7.68e−11 > pmf(30) —
  в сумму НЕ входит, распределение асимметрично, nL≠nC). p = pmf(30) = **2.120335433704179e−15** —
  бит-идентично артефакту. Это ровно стандартный two-sided Fisher (R fisher.test, as-or-less-likely).
- Биномиальный sensitivity: p0=30/92, as-or-less по Binom(41, p0) = **2.18081796329386e−07** —
  совпадает с артефактом И бит-идентичен r3.2 `p_two_sided_exact` (git 6350d25). ✅

### 2.3 Concentration (count-aware) — полный per-doc пересчёт из raw rho

30 detected loopy K=96 доков: пересчитаны prof, baseline=count(r)/K, concentration, legacy,
top2, dom_residues, anti_frac, anti_ratio, baseline_max — **0 расхождений** (tol 1e−12).
Агрегаты: median **5.407581942242839** / min **2.3798742461958295** / mean **5.413786268177715** /
frac>1 **1.0**; legacy 1/p: 6.062201613464012 / 2.429454959658243 / 5.908370724006618.
Все совпадают с артефактом. Гейт H2 (median>1 ∧ frac≥0.8) жив: 5.41 / 1.0.
Кросс-чек: r3.3 `conc_1overp_legacy` == r3.2 per-doc `concentration` — 30/30, 0 diff.
Справочная строка ревьюера (chat_0, p=15): legacy 4.9306 → count-aware 4.508, база 7/96=0.0729167 —
артефакт: 4.93060173361356 → 4.507978727875255, baseline_max 0.07291666666666667. ✅

### 2.4 Anti-phase (n=30, p=2 исключён — в пуле p=2 нет, guard проверен exec, §2.7)

- all: mean_anti_ratio **0.24253996799868416**, median **0.032374650701588244**, 28/30 < 1. ✅
- even: n=12 (p∈{8,14,16,18,20,24,32}), 10/12 < 1; odd: n=18 (p∈{7,9,11,13,15,17,19,27,31}), 18/18 < 1. ✅
- Проверка чётности состава по p-counts: 1+2+2+4+1+1+1=12 even, 1+1+1+2+4+4+2+1+2=18 odd, Σ=30. ✅
- Взвешенное среднее: (12·0.28061+18·0.21716)/30 = 0.24254 == all.mean. ✅
- Доки с anti_ratio ≥ 1: own_w8 (p=14, 1.0477) и wiki_8 (p=14, 1.1715) — «большинство» в finding
  (28/30) точна; wiki_8 на 17% выше собственной двухклассовой базы — учтено числом, wording не преувеличивает.
- per_doc-список: 30 записей с p/parity/anti_frac/anti_baseline/anti_ratio. ✅

### 2.5 Source sensitivity (B=1000, seed=117 — те же, что primary)

| пул | n_L/n_C | KS stat | KS p | median d | median p |
|---|---|---|---|---|---|
| gen_only | 28/36 | 0.4603174603174603 | 0.002997 | −0.0786551 | 0.013986 |
| gen_plus_own | 41/37 | 0.4528675016479895 | 0.000999 | −0.0818669 | 0.040 |
| gen_plus_corpus | 28/50 | 0.4914285714285714 | 0.000999 | −0.0883161 | 0.002 |
| all_K96 | 41/51 | 0.482544237207078 | 0.000999 | −0.0903171 | 0.000999 |

all_K96 — бит-идентичен primary (stat и p). Направление (L<C) стабильно во всех 4 пулах;
утверждение summary «направление стабильно по составу источников» подтверждено числами. ✅

### 2.6 Detector null calibration (B_null=200, seed=117)

- loopy (n=41): shuffle_full 0.022691 / max 0.144279; keep_short 0.027667 / max 0.218905;
  kill_peak 2/30 = 0.06667 (chat_15 p_obs=8, code_13 p_obs=7), same_p 0/30. ✅
- clean (n=51): shuffle_full = keep_short = 1/201 = 0.004975 (пол); kill_peak = null
  (0 detected → n_kill=0 — «все три нуля» из промпта — корректно: два суррогата на полу,
  третий не определён и честно null, не 0). ✅
- max ρ_du(k≥2), K=96: loopy median 0.9399128 / q90 0.9609772 / max 0.9786032;
  clean median 0.0639656 / q90 0.3170023 / max 0.7409195 (< порога 0.9). ✅
- Интерпретация согласована с числами: clean-максимумы на порядок ниже порога; loopy-медиана
  max-ρ (0.94) сидит НА пороге 0.9 — т.е. у половины loopy доков пик ≥ порога, детектор
  требует ещё и гармонику. Это отражено: selection_note (selection-on-the-statistic),
  finding («АССОЦИАЦИЯ», не независимая/механистическая детекция), summary
  («подтверждён (ассоциация: … — selection_note/detector_null_calibration)»). ✅

### 2.7 p=2 guard — exec-проверка (синтетика, без изменения файлов)

Синтетический rho с ρ(2)=ρ(4)=…=0.95, ρ(nечётные)=−0.05: `detect_period → p=2`.
Кодовая ветка p==2: p2_degenerate=True, top2_res_share=None, top2_baseline=None.
None-безопасность агрегата: `np.array([v for v in … if v is not None]).mean() if size else None`
— при всех-None даёт None. Detection-тест при n=1: guard `0 < d < n` = False → тест не строится. ✅

## 3. Регрессии (H1 / verdict / confounds / tail / location) — нет

Сравнение с r3.2 (git 6350d25):

- **per_doc: 96 доков, 0 diff в cells, 0 diff в doc-meta** (dom/source/degen_tz/loopy_ladder/
  mcos/copy_rate) — disk-only инвариант подтверждён байт-сравнением JSON. ✅
- **Primary/robustness/descriptive**: ks, median, mean, median_L/C — бит-идентичны r3.2.
  Verdict-строки primary/robustness переформулированы («directional refutation, НЕ mechanistic
  refutation») — это запрошенное r3.3-слово, не регрессия; числовая часть та же. ✅
- **H1**: status «refuted» в обоих; KS p=0.000999; медианы L<C. Tail: «refuted (L.tail > C.tail)»
  в обоих. location_contrast — идентичен (k1 0.07376/0.33049, ratio 4.48086; tail 0.89246/0.54342). ✅
- **Confounds**: 3 прежних — дословно, +1 новый (layer_type, M4). ✅
- **Гейты**: G1_gram/G3/Telescope/G0a_tpu_curves/G15 — идентичны r3.2; G2 дополнен
  expected=1440 (=96·5·3), ok=True, absent=[]. ✅
- **Смоук-diff** (/tmp/at0_3_smoke_r32.log vs r33.log): все forward-строки (G15, G2-счёт,
  G1/G3/Telescope, G0a, PRIMARY/ROBUST/DESCRIPT KS/median/mean, k24, tail, стратумы, detected
  p=[8,15,20,27]) — идентичны; новые строки только r3.3-блоки (Fisher-строка, count-aware
  aggregate, detector-null, source-sensitivity, G2 «120/120 … ok=True», домен with_tokens).
  Смоук Fisher-проверка: 4/4 vs 0/4 (симметричный случай nL=nC=4): as-or-less-множество {0,4} (pmf(0)=pmf(4)=1/70) → p=2/70=0.02857 — совпадает с лог-строкой 2.857e-02; биномиальный 1.250e-01 == r3.2-строка. ✅

## 4. Согласованность (gate ↔ verdict ↔ summary ↔ артефакт)

- H2-gate: Fisher p 2.12e−15 ≤ 0.05 ∧ median_concentration 5.4076 > 1 ∧ frac 1.0 ≥ 0.8
  ⇒ h2_ok=True ⇒ status «confirmed» — согласовано с verdict.H2.status и summary
  («подтверждён (ассоциация …)»). Формула гейта в verdict.H2.gate совпадает с кодом. ✅
- summary-числа == артефакт-числа: 30/41, 0/51, Fisher 2.12e−15, median 5.41 (legacy 6.06),
  top2 0.632, tail 0.892/0.543, k1 0.330/0.074 (4.5× = round(4.4809,1)), KS p=1.0e-03. ✅
- Script binding: meta.script_sha256 == sha256(experiments/AT0_3_area.py) == fb9b6c58…;
  имя store == sha256[:16] содержимого (пересчитано); manifest.claim_sha256 ==
  sha256(sort_keys(claim из манифеста)) — пересчитано, совпадает; manifest.store_sha256 ==
  sha256(store blob); claim.store = {path: store/4d0323e9b32f4c59.json, sha256: полный sha}. ✅
- Domain: corpus 16/16 nondeg, with_tokens=0 → n_claim_eligible=True, provenance_limited=True,
  descriptive_only=True, domain_claim_eligible=**False**; math 16/16/16 → True; chat 11 → n<12 →
  False. pooled_nondeg = 37 = 11+8+16+1+1 (arithmetic сходится). ✅

## 5. Замечания

### П1. keep_short: caveat из докстринга не перенесён в артефакт — правка

- Где: `detector_null_calibration` — докстринг (код) vs `results.period_structure.
  detector_null_calibration.note` (артефакт).
- Что: докстринг честно предупреждает, что keep_short сохраняет k=1..8, поэтому «периоды ≤8
  в сохранённом окне детектируются — честно отчитывается». Артефактный note этого caveat НЕ
  содержит: он говорит только «shuffle_full/keep_short — перестановки позиций».
- Почему: keep_short у loopy выше shuffle_full (0.0277 vs 0.0227, max 0.2189), и у 28/30 loopy
  доков keep_short выше пола; самые высокие — code_13 (p_obs=7, 21.9%) и chat_15 (p_obs=8,
  14.4%) — именно детектируемые периоды ≤8 в сохранённом окне. Кроме того, у loopy-профилей
  высокий ρ-базис (медиана max-ρ 0.94 ≈ порог 0.9) — перестановка значений loopy-profile
  порождает ложные (k,2k)-пары чаще, чем clean (0.005). Читатель артефакта, не видящий
  докстринг, может прочитать 2.8% как чистую ложную детекцию.
- Фикс: добавить в артефактный note одно предложение: keep_short > shuffle_full из-за (а)
  сохранённого окна k≤8 (p≤8 детектируются легитимно, сверить с p_obs per-doc) и (б) высокого
  ρ-базиса loopy-профилей (см. detection.max_rho_k2_k96).
- Серьёзность: **правка** (wording-кавеат; числа и selection_note не пострадают, интерпретация
  H2=ассоциация уже консервативна).

### П2. M1: «crossing rank» заменён числом пересечений — косметика

- Где: `ks_diagnostic` (L346) vs требование M1 «KS crossing rank / doc, max CDF gap location».
- Что: реализованы max_gap_x (позиция максимума) + n_crossings (число) + max_gap_side +
  n_pooled_unique; ранги пересечений не отчитаны.
- Почему: цель пункта — защита от copy-paste KS между объектами — достигнута: тройки различны
  (primary 0.530483/c=1, robust 0.528684/c=1, desc 0.597769/c=1). Число пересечений несёт то же
  защитное свойство; ранги — избыточны.
- Фикс: не нужен; при желании — добавить список позиций пересечений.
- Серьёзность: **косметика**.

### П3. run.log: threads=6/6/6 против смоука 4/4/4 — косметика (к сведению)

- Где: results/AT0/AT0_3_run.log строка 2.
- Что: полный прогон шёл с OMP/OPENBLAS/MKL=6 (env был задан), смоук — 4 (setdefault).
- Почему: на детерминизм не влияет — все перестановочные/бутстроп-циклы seeded (117/117),
  числа воспроизведены; threads меняют лишь порядок BLAS-суммаций (в per_doc diff=0).
- Фикс: не нужен.
- Серьёзность: **косметика**.

## 6. Итог

Минимальный патч §6 реализован полностью (13/13 пунктов в коде и артефакте), все ключевые числа
бит-идентичны независимому пересчёту, disk-only инвариант (per_doc 96/96, 0 diff) и смоук-инвариант
удержаны, регрессий H1/verdict/confounds/tail/location нет, wording H1=directional refutation и
H2=association честен и согласован gate↔verdict↔summary↔артефакт. Замечаний блокерного уровня нет;
одна правка (П1, wording keep_short-кавеата) — не блокирует финализацию, но желательна до
публикации артефакта.

VERDICT: PASS

AT0_3_R33_REVIEW_DONE
