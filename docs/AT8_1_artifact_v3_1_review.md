# DELTA-ревью AT8_1 артефакта v3.1 (T8.1-r3, per-seed C1 фикс)

**Дата:** 16.11, 19:49 UTC
**Объект:** `results/AT8/AT8_1_THRESHOLD_v3.1.json` + `AT8_1_v3.1_manifest.json` + `logs/AT8/AT8_1_v3_1_run.log`
**Охват:** только дельта v3 → v3.1 (исправленный перепуск по MINOR-1 внешнего ревью артефакта v3) + интактность v1/v2/v3. Не пересмотр целиком (прошлый PASS — `docs/AT8_1_artifact_r3_review.md`).

**SHA прочитанного (все — с диска на момент ревью):**

| Файл | sha256 |
|---|---|
| `results/AT8/AT8_1_THRESHOLD_v3.1.json` | `b10c0f5051dd734acf38fa897c7a9d1976d06b85351e78d73d0c6e21bcf8ccf2` |
| `results/AT8/AT8_1_v3.1_manifest.json` | `d2df61aeb02ef2b44a9d7049266d3bda8da699a5006a04e947a41f3ac55a22b5` |
| `logs/AT8/AT8_1_v3_1_run.log` | `e363e1d1a35de17d0dc9da6a829110130395df881fd09f15b3783bdf378a5454` |
| `results/AT8/AT8_1_THRESHOLD_v3.json` (база) | `c4fcb1ef5ad0d22feabf42f8fad819f120f13787671015ecbd5b9f941910ebd6` |
| `results/AT8/AT8_1_v3_manifest.json` | `cbb62bb0fc5b5b93f71da6b642fcace74bb8bf3214fbcf71af3b42c8f875d0da` |
| `results/AT8/AT8_1_THRESHOLD.json` (claim v1) | `b2923d74324e9180407f143a83fcffeaba22634ffe212cf723f86456457699a0` |
| `results/AT8/AT8_1_THRESHOLD_v2.json` | `bb507dac4d6969814980dc65d3f52b37f009745f7436a9df9b48db4652179157` |
| `results/AT8/AT8_1_manifest.json` / `AT8_1_v2_manifest.json` | `15996723567030b9ed51467ea692d83211f122471aa31dbf36af2688773301fe` / `548eec26f37a5d196fcb91f4c06dd4c274c2fd1bfab2623a577fbe01e439ed09` |
| `experiments/at8/AT8_1_r3_reanalysis.py` (исполняемый v3.1) | `dba84ddacead12cb51cbd1991175bac78381593112ef063a41c1450b9ab68dc5` |
| `results/AT8/store/018c3dac90eb1bba.json` | `28a0c28d94d982398492e19dbe733b984253173c49b0e55de3de1430a53e618d` |
| `docs/TZ_add_4-8.md` (пре-рег §T8.1) / `docs/AT8_1_review.md` (TZ-ревью, основа машины) | `a8b3f752c785880463e58bf1a1875fc47f99d2e8241919bec5377b0aa18707a0` / `56edfeba2028e03732a8544f163306ef29f9ce0b6ce2412b1226e7d01da92e9e` |
| `docs/AT8_1_artifact_r3_review.md` (прошлое ревью, PASS) / `docs/AT8_1_artifact_r3_answer.md` | `3acac879eccac6130b5825b4a7e58c57105510d467125efbdd4f93645c31670d` / `48f3bc1225969e2e67831e4be0bc1cc86e21d5abf9eb625033b686c061dc3982` |

Метод: read-only; независимый пересчёт из store (`cells[s{sd}_lam{λ}].gen.loopy`, медианы по 6 докам),
структурный diff v3↔v3.1, sha-верификация manifest↔диск.

## Проверки D1–D7

| # | Проверка | Статус | Числа / доказательства |
|---|---|---|---|
| D1 | Per-seed C1 числится и верен | **PASS** | Независимый пересчёт из store: s11 λ*=**0.70** → loopy(0.70)=**0.0** → **FALSE**; s12 λ*=**0.70** → **FALSE**; s13 λ*=**0.65** → **FALSE** — точно совпадает с ожидаемым в брифе и с артефактом (`per_seed.s{11,12,13}.c1_loopy = {c1: false, loopy_at_lstar: 0.0}`, `lam_star` = 0.7/0.7/0.65). λ* сверены двумя путями: per-seed argmin dip (медианы MAIN6 из store, tie → меньший λ) = `lam_star` артефакта; per-seed `dip_curve` артефакта == пересчёт из store на всех 11 точках λ (round 6) для всех 3 сидов. `min_dip`: s11 −0.30806002664274124, s12 −0.3058510816118246, s13 −0.3048474685168706 — == store. |
| D2 | Согласованность per-seed loopy ↔ claim v1 | **PASS** | Seed-avg `gen.loopy` по 3 сидам на каждой из 11 точек λ == `curves_seedavg.loopy` claim v1 побитово: 0.0 на λ≤0.75, 1/3 на 0.85, 1.0 на 1.00; **max dev = 0.0e+00** ≤ 1e-6. Selfcheck артефакта `per_seed_loopy_vs_claim_dev = 0.0` — совпадает. |
| D3 | Note честен | **PASS** | Per-seed note (s11/s12/s13) НЕ содержит «per-seed loopy в store отсутствует» (grep: 0 совпадений в `per_seed.*.note`); вместо формулы C1: «per-seed C1 = gen.loopy(λ*_seed) > 0 (per-seed loopy из store: cells[s{sd}_lam{λ}].gen.loopy)» — 3/3. Единственное вхождение фразы во всём артефакте — top-level `note`, где она **цитируется как опровержение**: «v3-note „per-seed loopy в store отсутствует" было ложью» (честная само-документация фикса, часть заявленной дельты). |
| D4 | Дельта только заявленная | **PASS** | Рекурсивный structural diff v3↔v3.1 — ровно 11 путей, все в заявленном наборе: `date`, `machine`, `note`, `revision` (строки) + добавления `selfcheck.{no_nan_nonfinite_count=0, per_seed_loopy_vs_claim_dev=0.0, v1_intact=true, v2_intact=true}` + `per_seed.s{11,12,13}.c1_loopy` (новые блоки) + `per_seed.s{11,12,13}.note` (замена). Всё остальное — побитово равно v3: cond1, seed-avg C1–C3 (`C1=false/0.0`, `C2=true/6.393994571310583`, `C3=false/9.74704615275065`), λ*=0.7, min_dip=−0.3052401224752934, локаторы и gaps (6/0/3), per-seed dip/cond1/C2/C3, F5 Δ=+0.048018192752839195, families (field/leaf inconsistent, mode vacuous), reachability, s9, F6, verdict. |
| D5 | Вердикт не изменён | **PASS** | `outcome=V3_PATHOLOGY`, `prereg_expectation=V3_PATHOLOGY`, `expectation_match=true`; v3: `fired=true` (cond1=True, C1=False, C2=True, C3=False) — == v3. Маркер в логе: `AT8_1_R3_DONE verdict=V3_PATHOLOGY match=True (wall 0.2s)` — присутствует (grep). `recompute_dev = {dip: 0.0, eff_rank: 0.0, nll_holdout: 0.0}` ≤ 1e-9; `u_dip_dev = 0.0`; `no_nan_nonfinite_count = 0`; токенов NaN/Infinity в файле нет (grep). |
| D6 | Провенанс | **PASS** | `script_sha256` (v3.1-manifest) = `dba84ddacead12cb…` = sha256 текущего `experiments/at8/AT8_1_r3_reanalysis.py` на диске ✓. `claim_file_sha256` (v3.1-manifest) = `b10c0f5051dd734a…` = sha256 v3.1-артефакта на диске ✓. Входы интактны: claim v1 `b2923d74324e…`, store `28a0c28d94d9…`, manifest v1 `159967235670…` — все == записанным в manifest'ах и в `input` артефакта (блок `input` v3.1 == v3 побитово). Лог подтверждает sha-верификацию при старте («claim b2923d74324e9180 / store 28a0c28d94d98239 — совпадает с manifest»). |
| D7 | Замороженные не тронуты | **PASS** | v3 = `c4fcb1ef5ad0d22feabf42f8fad819f120f13787671015ecbd5b9f941910ebd6` ✓ (== manifest v3); v2 = `bb507dac4d6969814980dc65d3f52b37f009745f7436a9df9b48db4652179157` ✓ (== manifest v2); v1 = `b2923d74324e9180407f143a83fcffeaba22634ffe212cf723f86456457699a0` ✓ (== manifest v1). Selfcheck `v1_intact=true`, `v2_intact=true` — независимо подтверждены: sha на диске == `claim_file_sha256` соответствующих manifest'ов. Мtime артефактов v1 (15.11 21:00) / v2 (16.11 12:11) / v3 (16.11 21:38) — до прогона v3.1 (19:39). |

## Замечания

**MAJOR:** нет.

**MINOR-1 — v3.1-файлы не закоммичены.** `git status`: `M experiments/at8/AT8_1_r3_reanalysis.py`
(на диске — v3.1-скрипт `dba84dd…`; в git (`7d26167`) — v3-скрипт `6b064565…`), `??` артефакт v3.1,
manifest v3.1, лог v3.1. Цепочка чисел интактна (manifest pin-ит исполнявшийся скрипт, лог + артефакт
согласованы, D6/D7 пройдены), но это тот же разрыв «committed ≠ исполнявшийся», который в v3 был
закрыт коммитом `7d26167` — закрыть коммитом v3.1-набора (скрипт + артефакт + manifest + лог).
Не блокирует дельту.

**NIT-1 — RUN_RC=0 не в самом логе.** Лог завершается маркером (доказательство по правилам проекта:
маркер + файлы на диске — держится); RC=0 зафиксирован в ответке (`./run_one.sh at8_1_r3_1`, 16.11
19:39 UTC, wall 0.2s). Ставить RC-строку в лог — на будущее.

**NIT-2 — `review_sha256` в артефакте.** `56edfeba…` = `docs/AT8_1_review.md` (TZ-ревью раунда 2 —
основа машины v3), а не ревью артефакта (`3acac879…`, PASS) — идентично v3 и раскрыто в docstring
скрипта («Ревью (основание): docs/AT8_1_review.md (sha 56edfeba…)»). Не ошибка; неоднозначность имени
поля зафиксирована.

## Итог

Дельта v3 → v3.1 соответствует заявленной и ничего больше не трогает: per-seed C1 числится и
независимо пересчитан из store (FALSE×3, совпадает с seed-avg C1=FALSE), ложный v3-note заменён
формулой (top-level note цитирует его как опровержение), selfcheck-поля добавлены, все остальные
числа побитово = v3, вердикт V3_PATHOLOGY/match=True не изменён, провенанс (manifest↔скрипт↔артефакт,
входы) и интактность v1/v2/v3 подтверждены sha. Один MINOR (коммит v3.1-набора) — не блокирующий.

**VERDICT: PASS**

REVIEW_DONE
