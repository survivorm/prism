# AT0_1 r7-аннотация — внешнее ревью (round 3)

Роль: независимый внешнее ревью post-hoc аннотации r7 (C1–C7, docs/AT0_1_review.md round 2).
Ревьюер — не автор правки. Все числа ниже — из фактических пересчётов (независимые скрипты в
/tmp, 6-thread env = run_one.sh) и артефактов; ничего «из головы».
Дата: 2026-11-15. Формат: таблица вердиктов C1–C7 + находки MAJOR/MINOR + итог.

## 0. Входы и sha (проверено)

| файл | sha256 | статус |
|---|---|---|
| `experiments/AT0_1_accounting.py` | `de3058693022320876da8028c4e63dd1ef57ca8fe9dc565f08f49d396a763f20` | ✓ совпал с требованием (НЕ изменён) |
| `prism/field.py` | `bcc607c561f95e3ed7aaf07677d088159aeb572b804c2f32fc9c5f4c26396532` | ✓ = claim `gates.G1prime_field_sha256`; = версия из git 67addba |
| `results/AT0/store/23281c64cc418ff1.json` (store) | `23281c64cc418ff1fedad2a5013c4bf7c8b7dd7680ac8d4fa7cf746c62fd1732` | ✓ |
| `store/claim_r5_d4a4b101.json` | `d4a4b10135e8b63f361f6fe75475cff7d44b6810e9248d03a8bdf96b5b73d94a` | ✓ существует, sha = `meta.r7_annotation.chain.r5_claim_sha256` |
| `store/claim_r6_b0f74312.json` | `b0f74312416285207ba99ae93ee75ecc446938b8b61ba82a19a04ac36b913994` | ✓ |
| `store/claim_r7_2230226141a9a697.json` | `2230226141a9a697fa8ba6118f0a5585efc1b8ea650e8f3a3a377e75a8d04db2` | ✓ = заявленный r7-claim |
| `results/AT0/T0_ACCOUNTING.json` | `2230226141…a8d04db2` | ✓ bit-eq с r7-пин |
| `results/AT0/AT0_1_run.log` | `e6dcd6fc62aeced0c493a851548c70051a4a60ddcd8204fbe0100772fbf54d93` | ✓ bit-eq со `store/runlog_r5_pre_rerun.log` (тот же sha) |
| `results/AT0/t25_ekurve.json` | `4a43873a7571ca71b3bc4d06b213aea01b6f033ea643f77644c8e21d485b7680` | ✓ = claim `meta.t25_ekurve_sha256` |
| TZ `docs/TZ_add_0-3.md` @ git 67addba | `647c03ab05a25642c6f7e647fbd8dbc333c53d6de2ea8555ed3bee80baa8013d` | ✓ = claim `meta.tz_sha256`; подтверждено коммитом 2dab888 («прогон на TZ-sha 647c03ab») |

## 1. Вердикты C1–C7

| пт | вердикт | ключевые доказательства |
|---|---|---|
| **C1** (MA(1)-центрирование) | **PASS** | z_raw_band medians: v_tangent `381.3249665142485`, h_raw `−0.21574175798491185`, du_unit `−0.21562117252716537` — ровно как в r5/r6 и в prompt. Независимый пересчёт `z = ((mean(rho_Tk[8:96]) − f_dc) + f_dc) / se`, `se = sqrt(Σ_{k=9..96} 1/((N−k)D))/88` по ВСЕМ 51×3=153 (док, объект): **0 расхождений** (bit-eq) со store `per_doc.null_ma1_z.z` и с claim `per_doc_ma1.z_raw_band`. z_excess v_tangent: мой median `10.547543645305291` (≈10.55) = claim, согласуется с ручным счётом ревью (≈10.5). E_doc проверен численно: Monte-Carlo (N=383, D=1024, MA(1) a=−0.4 + DC, 30 draw): E[s_doc]=1.5e-4 (sd 5.3e-4), E[mean_band]=0.07223 vs f_dc=0.07208 ⇒ формула «E[rho_Tk(k≥2)] = ‖m‖²/(‖m‖²+var) = f_dc; E[s_doc]≈0» верна. Старая статистика сохранена диагностикой (z_raw_band), E_doc переписан |
| **C2** (tpu_dev тавтология) | **PASS** | В r7 claim ключей `max_rel_dev_reconstruction*` НЕТ. `reconstruction_roundoff_identity` = `3.7939126356469717e-16` (bit-eq с моим пересчётом; worst `own_w6` L5; max_by_source/by_layer — все 7 чисел bit-eq). detail.status: «тождество (roundoff), НЕ provenance-гейт». Разбор веток проверен сам: hn>1e-12: u=h/hn, ‖u‖=1 ⇒ U=u ⇒ U·hn=h; hn≤1e-12: u=h/1e-12, U=h/‖h‖=h/hn ⇒ U·hn=h — в обеих ветках U·hn ≡ h (кроме round-trip divide→multiply). На реальных данных ветка hn≤1e-12 не достигается (мин. норма ряда = 0.854) — формулировка для неё в claim тем не менее корректна. Строка «tpu_dev» в claim/answer — только имя скрипта `at0_1_tpu_dev_recompute.py`; гейтом не назван нигде (единственное «гейт» в claim — «НЕ provenance-гейт») |
| **C3** (provenance-цепочка) | **PASS** | Цепочка de305869 → r5 `d4a4b101` → r6 `b0f74312` → r7 `22302261` замыкается: r5-claim закреплён (файл+sha в chain), перепуск de305869 дал bit-eq store `23281c64` + run.log `e6dcd6fc` (оба файла на диске). Мой независимый diff (свой leaf-коллектор, не функция r7-скрипта): **r5→r6**: 34 added листа / 0 removed / 1 changed — каждый лист покрыт ровно декларированными R6-паттернами, ни одного лишнего/недостающего. **r6→r7**: 744 added / 21 removed / 7 changed — то же самое по R7_ADDED/R7_REMOVED/R7_CHANGED (ни одного extra/uncovered). (Утверждение ответки «30 added / 8 removed / 7 changed» — node-счёт того же diff; leaf-счёт 744/21/7 — согласовано, оба проверены.) manifest r7 = claim + manifest-блок (deep-eq); `manifest.claim_sha256` = sha claim-файла = `22302261…`; manifest_r5/r6 — та же структура. **Детерминизм**: повторный запуск r7-скрипта из копии репо в /tmp (6-thread env) → claim sha `2230226141…` bit-eq, manifest bit-eq, маркер R7_ANNOTATION_DONE. R5-claim не содержит `r6_annotation` (инвариант). Остаток: r6-аннотационный скрипт не закреплён — см. MINOR-3 |
| **C4** (SE на ambient D) | **PASS** | d_eff пересчитан для всех 153 (док, объект): `eigvalsh((x−mean)ᵀ(x−mean))`, participation ratio — **bit-eq с claim в закреплённом 6-thread env** (run_one.sh: OMP/MKL/OPENBLAS=6). Честный tolerance: в другом thread-окружении drift 1–2 ulp (rel ≤ 3.4e-15, мой дефолтный run) — см. MINOR-2; claim НЕ заявляет bit-eq для d_eff (в self_audit его нет), ложных заявлений нет. se_factor = sqrt((1+2ρ₁²)·D/D_eff), ρ₁ = θ/(1+θ²) (θ из store): se_factor_median bit-eq — h_raw `7.247495241936936`, du_unit `7.017361043999897`, v_tangent `5.485633247526761` (оценка ревью 5.5–7.2 ✓). D_eff(med) = 25.016 / 26.017 / 34.536 (min 8.2–9.0, max 52.4–67.9) — в оценке 25–35 ✓. z_excess_eff medians: −0.0431 / −0.0646 / **1.9309** (mean 2.6412, n|z|≥3 = 21/51) — bit-eq. Аргумент про r_eff(0.05)=178: это 5%-пороговый ранг спектра U-состояний (другая метрика, не participation ratio приращений) — контраргумент ответки корректен и зафиксирован в `SE_eff_formula` |
| **C5** (SE_formula /98 vs /88) | **PASS** | Claim: `SE_formula` = «sqrt(Σ_{k=9..96} 1/((N−k)·D)) / 88 … (K_MAX−8 = 88 лагов; в r5/r6 ошибочно /98)» + пояснение про Bartlett. Код не тронут: sha run-скрипта `de305869…` (неизменен); в `p4_plateau` (строка 547) всегда был `/ (K_MAX - 8)` = /88. R5-claim содержит старый текст «/ 98» (проверено) — т.е. правка только в артефакте, как и требовалось |
| **C6** (T_none_full) | **PASS** | `T_none_full` удалён из `excess_median`/`excess_mean` (во всех 3 объектах). `identity_check_G1` содержит T_none_full_median/mean, `max_abs_sum_rho_full_minus_(r-1)/2`, n_docs: мой пересчёт (nondegen, 53 дока): h_raw med `−0.5023582212969491` / mean `−0.5033896820053053` / gram `8.33e-16`; du_unit `−0.5023929429497754`; v_tangent `−12.868956825537097` / `−13.20265274318846` / `1.07e-14` — **все bit-eq** и с claim, и с r6-claim (значения из excess_* r6). `t0_headline_note` дополнен: r_v ≈ 24.9 (fакт: r_median=24.90236364133829), r_du ≈ 0.0048 (0.004811008660178404) — в блоке t0_headline r_median/f_dc_median на месте. Headline r/f_dc — см. MINOR-4 |
| **C7** (self-audit) | **PASS** | Лог `/tmp/at01_r7.log`: RC=0 (RUN_RC=0), маркер R7_ANNOTATION_DONE (1 шт.), self-audit JSON на месте. Все assert'ы прошли (скрипт дошёл до записи — запись после всех assert'ов, проверено по коду). `l23_meta_vs_recomputed`: мой независимый пересчёт (32 TPU-дока: dcurve(V, 1..96)[0] median vs meta curves.23.dip): P1_own `0.00020829041663661396`, P2_corpus `5.5475605402977024e-05`, n_docs 16/16, worst_doc `own_w4` — **bit-eq** с claim/self_audit/логом. G0a evidence_own_w0 (4 числа) — bit-eq. Roundoff (max/worst/by_source/by_layer) — bit-eq. θ (51×3), z_raw_band=r5-статистика (51×3), T_none_full (3×2), pool=51 — воспроизведены (assert'ы + мой пересчёт). Self-audit сверяет ТЕ ЧИСЛА, которые ревью просил (l23_meta — канал годности рядов), а не только evidence_own_w0 |

**Код r7-скрипта (п.8 prompt): PASS**
- Guard `th is None` (no_match-док в пуле): на текущих данных не срабатывает (все 51 доки matched — проверено), протестирован **exec'ом реальной функции** `ma1_recompute` с имитированным no_match-доком: без краха, per_doc запись `{z_excess, z_raw_band, z_excess_eff: null, d_eff}`, агрегаты z_excess_eff не загрязнены (только matched-доки) — см. MINOR-5.
- `deff_rows` guard (вырожденная ковариация) — простой assert, на данных не достигается (все D_eff > 8).
- json.dump: claim/manifest — через `jclean` (NaN/inf→None, numpy-касты) ✓; stdout-отчёт — обычные float/str/int, NaN-риска нет (лог распарсился).
- Детерминизм: повторный запуск (копия репо в /tmp, 6-thread) → claim sha bit-eq `22302261…`, manifest bit-eq ✓.
- Формулы согласованы с `AT0_1_accounting.py`: se (то же fp-выражение `(s_doc+f_dc)/se`, `(n−arange(9,97))`, /88), theta (`0 if rho1==0 else (1−sqrt(1−4ρ1²))/(2ρ1)`), pool (nondegen + K==96 по всем 3 объектам — идентично строкам 893–894), T_none_full — ✓.
- Запись (шаг 5) — строго ПОСЛЕ всех assert'ов (шаги 0–4) ✓; пост-записный re-sha-assert не может сработать на тех же байтах.
- R5-инвариант: «числа прогона не изменились» — теперь ПРОВЕРЕНО (полный diff r6→r7 = ровно декларированные пути; остальные листья bit-eq; перепуск r5 bit-eq store+log).

**Ответка `docs/AT0_1_review_r7_answer.md` (п.9 prompt): PASS**
- Каждое число сверено с артефактом/логом: z_raw medians 381.325/−0.2157/−0.2156 ✓; z_excess medians −0.4109/−0.4958/10.5475, means −0.2016/−0.1909/17.4853 ✓ (claim: −0.4109050469221357 / −0.4958197811856885 / 10.547543645305291 и т.д.); z_excess_eff medians −0.0431/−0.0646/1.9309, means −0.0352/−0.0351/2.6412 ✓; n|z|≥3 = 45/51 и 21/51 ✓; D_eff(med) 25.0/26.0/34.5 (min 8–9, max 52–68) ✓; факторы 7.2/7.0/5.5 ✓; shift f_dc/se: median 0.0667/0.0682 (заявлено 0.067–0.068 ✓), max 0.3803/0.4559 (заявлено 0.38–0.46 ✓); l23 max dev 2.08e-4 ≤ 1e-3 ✓; r_v≈24.9 / r_du≈0.0048 ✓; «14 декларированных r6-путей» (3+3+1+1+1+1+3+1=14 ✓ node-счёт); «30/8/7» ✓ (node-счёт); «30–150×»: 381/10.5475 = 36.1× ∈ [30,150] ✓.
- «Вердикты P4/P5 не переписаны» — согласовано с исходным ревью: §9 «Минимальный патч» ревью не требует переписывать вердикты (только z-столбец/E_doc/аннотации); вердикты r6≡r7 (все 3 «refuted») — проверено bit-eq.
- Несоответствий «число из головы» не найдено.

## 2. Находки

**MAJOR: нет.**

**MINOR:**

1. **Дата r7 в claim неверна на день.** `meta.r7_annotation.date` = «2026-11-14» (и `reconstruction_roundoff_identity_detail.provenance`: «пересчёт r7 (2026-11-14)»), но фактический запуск по mtime: `T0_ACCOUNTING.json` 2026-11-15 02:14:53 (+0300), `/tmp/at01_r7.log` 2026-11-15 02:14:54; `AT0_1_artifact_hashes.json` честно записывает «2026-11-15 (UTC+3)». Косметика, числа не затрагивает; в claim зашита константа.
2. **BLAS-thread-чувствительность производных величин не задокументирована.** per_doc `d_eff`, `z_excess_eff`, `se_factor_median` bit-воспроизводимы только в закреплённом 6-thread окружении (run_one.sh: OMP/MKL/OPENBLAS_NUM_THREADS=6 — проверено: мой 6-thread пересчёт bit-eq по всем 153, дефолтный-thread — drift 1–2 ulp, rel ≤ 3.4e-15). Claim не заявляет bit-eq для d_eff (self_audit покрывает только thread-робастные величины), т.е. ложного утверждения нет; честный tolerance — ~1e-15 rel между окружениями / bit-eq в pinned-окружении. Одной строкой в claim (или в chain-ноуте) это стоит зафиксировать. Заметно НЕ для z_raw_band/z_excess/θ/T_none_full/l23/roundoff — те bit-eq в обоих окружениях (только `eigvalsh` чувствителен к тредам).
3. **r6-аннотационный скрипт не закреплён** (признано в chain-ноуте и в ответке). Переход r5→r6 верифицирован ассертом «diff = ровно декларированные r6-изменения» (проверено независимо: лишних/недостающих позиций нет), но код самой r6-аннотации невоспроизводим (в репо только `at0_1_tpu_dev_recompute.py` — пересчёт чисел, не правка claim). Задокументированный остаточный пробел; r5-claim, r6-claim и r7-claim зафиксированы файлами с sha, так что цепочка значений закрыта.
4. **Headline `r/f_dc` — тождество.** r = N·f_dc точно (f_dc = ‖mean(x)‖²/mean(‖x‖²), r = ‖Σx‖²/c0 = N·f_dc), поэтому r/f_dc ≡ N: medians 383.0 / 383.0 / 312.59 — просто медианный N по объектам. Реализовано ровно как просило ревью («headline = r/f_dc per object»), но величина тривиальна; информативные r_v=24.9024 / r_du=0.004811 — в note (на месте).
5. **Guard `th is None` не покрыт данными** (все 51 доки пула matched; no_match в пуле невозможен на текущих данных). Протестирован exec'ом реальной `ma1_recompute` с имитированным no_match-доком: без краха, `z_excess_eff: null`, d_eff посчитан, агрегаты не загрязнены. Как defensive guard — приемлемо; в claim/ответке пометки «defensive, не достигается на данных» нет.
6. **Дублирование старых полей.** `null_analytic_ma1.median_z/mean_z/n_abs_z_ge_3/theta_median_abs` (r5-поля) остались со значениями, идентичными новому `z_raw_band` (median/mean/n_abs). Это следствие инварианта «остальные листья bit-eq» (удалить нельзя) — осознанно, но дубль цитируемых чисел (381) в двух местах одного dict'а — потенциальная путаница при цитировании.

## 3. Итог

Цепочка provenance закрыта и воспроизведена: de305869 (неизменен) → store 23281c64 + run.log e6dcd6fc (bit-eq перепуск) → r5 d4a4b101 (пин) → r6 b0f74312 (пин, diff = ровно декларированные r6-изменения) → r7 22302261 (пин; повторный запуск r7-скрипта в изолированной копии репо дал bit-eq claim и manifest). Все семь пунктов C1–C7 закрыты фактически (не только в тексте): числа пересчитаны независимо и совпадают bit-eq (C1 z-статистики по 153 (док,объект), C2 roundoff, C4 d_eff/se_factor в pinned-окружении, C6 T_none_full, C7 l23_meta) или верифицированы (C5 /88, C3 diff'ы и manifest, C7 assert'ы/лог/маркер). Вердикты P4/P5 не тронуты (r6≡r7 bit-eq), как и не требовалось.

MAJOR-находок нет. MINOR — 6 (таблица выше), все косметические/документационные; ни одна не меняет ни одного числа в claim.

**Вердикт: PASS**

Минорный список (к закрытию/пожелание, не блокирует):
- MINOR-1: дата «2026-11-14» в claim/answer vs фактические 2026-11-15 02:14 (+0300) (mtime + artifact_hashes.json).
- MINOR-2: зафиксировать в claim, что per_doc d_eff/z_excess_eff/se_factor — bit-eq в 6-thread env, ~1e-15 rel между env.
- MINOR-3: r6-аннотационный скрипт не закреплён (задокументировано; остаток).
- MINOR-4: headline r/f_dc ≡ N (тождество) — реализовано по букве ревью, но тривиально.
- MINOR-5: guard `th is None` — defensive, не достигается на данных (проверен exec'ом).
- MINOR-6: дубль 381-статистики в `median_z` и `z_raw_band` одного dict'а.

## 4. Addendum round 3b (r7 v2, claim 82f44f08)

Дата: 2026-11-15. Замещение: claim `22302261…` (v1) → `82f44f0813f6adde047f11b97568f08ca19807fb41af145b2977f3fd450999c5` (v2).
Входы (проверено): v1-claim `2230226141a9a697…` (на диске, не восстановлен), v2-claim `82f44f08…` = `T0_ACCOUNTING.json` (bit-eq);
`experiments/at0_1_r7_annotation.py` = `91fcc9bbe2df4b9fd828a04ce758caeb077aa2cd6c3817e73b3d5e5b32a5345b`;
`experiments/at0_1_r6_annotation.py` = `0d52eda5f1f4128aebe1bfdce9f3c32532b4a5ec14be9e1172a6757d2bfe508e` = bit-eq `/tmp/recovered/AT0_1_r6/apply_r6_edits.py`;
лог v2 `/tmp/at01_r7b.log`: RUN_RC=0, маркер R7_ANNOTATION_DONE (1 шт.), `r7_claim_sha256 = 82f44f08…`; mtime v2-артефактов 2026-11-15 02:55 (+0300).

- **Числовой diff v1→v2: 13 листьев, только текст** (0 removed; 956 численных листьев в обоих, diff = 0, bit-eq).
  1 added: `meta/r7_annotation/chain/r6_annotation_script_sha256` (новый sha-пин, текст).
  12 changed (все — строки/список строк):
  1. `meta/r7_annotation/date` — 2026-11-14 → 2026-11-15 (MINOR-1)
  2. `gates/G0a_tpu_hn_field/reconstruction_roundoff_identity_detail/provenance` — «(2026-11-14)» → «(2026-11-15)» (MINOR-1)
  3–5. `aggregates/t0_headline/{v_tangent,h_raw,du_unit}/identity_check_G1/headline` — «r/f_dc per object» → «r per object; r/f_dc ≡ N (G3-тождество) — тривиально, не цитировать; r_v ≈ 24.9, r_du ≈ 0.0048» (MINOR-4)
  6. `aggregates/t0_headline_note` — то же (MINOR-4)
  7. `meta/r7_annotation/chain/annotation_script_sha256` — 9ffc1585… → 91fcc9bb… (= фактический sha r7-скрипта; допустимо: chain)
  8. `meta/r7_annotation/chain/r6_claim_note` — «r6-скрипт не закреплён» → «закреплен r7: experiments/at0_1_r6_annotation.py» (MINOR-3)
  9. `meta/r7_annotation/changes` — +1 строка «R3 (round 3 MINOR-1..6)» в конец списка (только добавление)
  10–12. `prereg/4_plateau_vs_fDC/{v_tangent,h_raw,du_unit}/null_analytic_ma1/c1_c4_note` — +тред-нота (MINOR-2) + defensive-note guard (MINOR-5) + дубль-нота «цитировать z_raw_band» (MINOR-6)
  Ни одно числовое значение (z, median, mean, p, d_eff, se_factor, roundoff, gram dev, n_*, body-sha вне chain) не изменилось.

- **MINOR-1: закрыт.** `meta.r7_annotation.date` = «2026-11-15»; `reconstruction_roundoff_identity_detail.provenance` содержит «пересчёт r7 (2026-11-15)»; согласуется с mtime v2-артефактов (2026-11-15 02:55 +0300) и датой в AT0_1_artifact_hashes.json.
- **MINOR-2: закрыт.** c1_c4_note (все 3 объекта): «per_doc d_eff/z_excess_eff и se_factor_median — bit-eq только в pinned-окружении (run_one.sh: OpenBLAS/MKL_NUM_THREADS=6; r7-прогон: OMP_NUM_THREADS=4 — eigvalsh чувствителен к OpenBLAS/MKL-тредам); в другом thread-окружении дрейф 1–2 ulp (≤3.4e-15 rel, проверено независимо, ревью round 3). z_raw_band/z_excess/θ/T_none_full/roundoff/l23_meta — bit-eq в обоих окружениях».
- **MINOR-3: закрыт.** `experiments/at0_1_r6_annotation.py` существует; sha256 = `0d52eda5…e508e` = `meta.r7_annotation.chain.r6_annotation_script_sha256`; bit-eq с `/tmp/recovered/AT0_1_r6/apply_r6_edits.py` (одинаковый sha).
- **MINOR-4: закрыт.** identity_check_G1.headline (3 объекта) + t0_headline_note: «заголовочный T0 — r per object … r/f_dc ≡ N (G3-тождество r = N·f_dc) — тривиально, не цитировать; информативно: r_v ≈ 24.9 (not DC-free), r_du ≈ 0.0048 (DC-suppressed)».
- **MINOR-5: закрыт.** c1_c4_note: «guard th is None (no_match-док в пуле) — defensive, не достигается на данных (все 51 доки пула matched); проверен exec'ом реальной ma1_recompute с имитированным no_match».
- **MINOR-6: закрыт.** c1_c4_note: «Поля median_z/mean_z/n_abs_z_ge_3 (r5) bit-eq z_raw_band.median/mean/n_abs_z_ge_3 — дубль сохранён инвариантом r5-листьев; цитировать z_raw_band».

- **Цепочка: OK.** `T0_ACCOUNTING.json` bit-eq с `store/claim_r7_82f44f0813f6adde.json` (82f44f08…). Manifest = claim v2 + manifest-блок (deep-eq; `manifest.claim_sha256 = 82f44f08…`, `store_sha256 = 23281c64…`); manifest-sha `4ccaa860…` совпадает с `manifest_sha256` в artifact_hashes. `AT0_1_artifact_hashes.json`: `claim_sha256 = 82f44f08…`; pins полные (claim_r5 d4a4b101 / claim_r6 b0f74312 / claim_r7 82f44f08 / manifest_r5/r6 / runlog e6dcd6fc — все sha сверены с файлами на диске); `store 23281c64cc418ff1.json` и `experiments/AT0_1_accounting.py` (de305869…) не тронуты (sha без изменений); chain: run_script de305869…, r5 d4a4b101…, r6 b0f74312…, r6_annotation_script 0d52eda5…, annotation_script 91fcc9bb… (все = фактические файлы); `superseded_by_current` честно сохраняет v1 (22302261, 9ffc1585, manifest d40847a6).
- **Регрессия: OK.** Все 956 численных листьев v1↔v2 bit-eq (diff = 0). Точечно: z_raw_band medians 381.3249665142485 / −0.21574175798491185 / −0.21562117252716537 ✓; z_excess v_tangent median 10.547543645305291 ✓; T_none_full identity −0.5023582212969491 (h_raw) / −12.868956825537097 (v_tangent) ✓ (в identity_check_G1 diff только строка headline); roundoff 3.7939126356469717e-16 ✓ (detail bit-eq, кроме даты в provenance).

- **Вердикт: PASS.** MAJOR: нет, новых MINOR: нет. Все 6 MINOR round 3 закрыты по существу; изменения v1→v2 — только текст/даты/chain-пины; цепочка пинов согласована; чисел прогона не тронуто.
