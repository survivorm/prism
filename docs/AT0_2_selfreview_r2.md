# AT0_2 — само-ревью исполнителя (round 2, после фиксов N2/N4/N5 + багов ревьюера)

Дата: 13.11.2026. Объект: experiments/AT0_2_shape.py (sha 543a1fcc…), docs/AT0_2_prereg.md.
По чек-листу AGENTS.md; независимое ревью — см. docs/AT0_2_REVIEW.md (round 1, BLOCK) и его фиксы.

## Внесено сверх требований round 1 (баги, найденные при само-аудите переработки)

1. **[MAJOR, мой] KeyError в финальном logline (полный прогон):** `a['frac_docs_p_lt_alpha']`
   при том что aggregate отдаёт `frac_cells_p_lt_alpha` (каскад переименования не дочинен
   ревьюером). Срывался бы ПОСЛЕ записи store/claim/manifest, но ДО run.log/«ALL PASS» —
   smoke его не видит (return раньше). Фикс: ключ переименован в logline.
2. **[MINOR, мой] N5 недоделан доменами:** доменный бутстрап в aggregate() бустрапировал
   КЛЕТКИ (per-cell values), а не per-doc means — тот же класс заниженного CI, который N5
   фиксировал на верхнем уровне (внутри-доковая корреляция слоёв, эффективный n ≈ 12 доков,
   а не 60 клеток). Фикс: per-doc mean по L, кластер = док, как у D_real.
3. **[MINOR, мой] N5 недоделан по чек-листу:** отсутствовал `n_docs_unique_p_lt_alpha`
   (чек-лист round 1, п. 3). Добавлен (доки с p_cell < 0.05).

## Чек-лист AGENTS.md (проверено по коду, не на глаз)

1. **Одинаковая статистика на обеих сторонах:** D_real (D_center_l1) и D_null (Dnull_l1_mean) —
   одно и то же расстояние (L1 к тому же C), один срез (все non-absent клетки), одинаковое
   агрегирование (per-doc mean по L → cluster-бутстрап по 96 докам, B=1000, seed=117). Домены —
   тоже per-doc (фикс выше). Wilson/frac_cells — описательные, в вердикт не входят. ✓
2. **Ветки/guard'ы:** `degenerate → return rec` (без "nulls") — aggregate пропускает
   (`"nulls" not in c` / `p_doc is None`), клетка на месте для G2. AR `if not ok.any(): …
   continue` — p_doc=None, пропускается. PC `max()` по пустому pc_cells — ValueError (fail-loud,
   в полном прогоне pc_cells = 1434). G2: `expected` включает absent, `got` = все клетки
   (absent+normal+degen) — совпадает. Финальный logline: `if not a.get("n_docs"): continue` —
   пустой агрегат `{"n_docs": 0}` пропущен. ✓
3. **Каскад фиксов:** `frac_docs_p` — 0 вхождений (grep); все потребители (smoke logline,
   финальный logline) используют ключи, присутствующие в out aggregate(); `n_docs_unique_…`
   выдаётся (описательный). ✓
4. **json.dump:** jclean рекурсивно на store и claim; manifest — только str/int; все
   numpy-значения float()-касты до записи (per_seed, prof5, rho96, CIs, wilson). ✓
5. **Долгий прогон:** запуск через `./run_one.sh at0_2_full .venv/bin/python
   experiments/AT0_2_shape.py` (pidfile-flock — двойной лонч невозможен; nice -n 10;
   mem-wait ≥10G). NOTE: в чек-листе round 1 написано `python3` — беру venv-python, это же
   интерпретатор, на котором прошли smoke и первый прогон.
6. **Контроли:** L1-TPU degraded = ['own_w0','own_w4'] — тот же набор, что AT0_1
   (excluded_L1), dev > 1e-3. Positive control smoke: max|Δρ̂_circ| 2.2e-16, max|Δr_full| 8.5e-14
   (машинная точность) — прибор работает до интерпретации null'ов.
7. **Артефакт vs код:** старые smoke-артефакты (11:14, meta.smoke=True) будут перезаписаны
   полным прогоном; orphaned smoke-store b870e6a791567507.json — удалить ПОСЛЕ успешного
   полного прогона (чек-лист round 1, п. 6; решение: удалить — store с smoke=True и
   meta.n_docs=96 ловушка).
8. **Мелочи:** smoke logline печатает CI как список (косметика, stdout); date в claim/manifest
   «2026-11-13 (UTC+3)» — актуальная; env.threads = OMP из окружения run_one (6), setdefault 4
   не перебивает.

## Наблюдение (не баг, правило пре-рега)

Вердикт «refuted» срабатывает при D_real выше CI **среднего** D_null (бутстрап 96 per-doc
mean'ов) — это уже смещает D_real относительно среднего null-расстояния, а не против
полного seed-распределения (сигн. на клетку — p_cell, описательный). Т.е. правило
пре-рега консервативно к «confirmed», но относительно чувствительно к «refuted».
Smoke: все 27 пар «refuted» с зазором ~3–10× — к этой граничностости нечувствительно.
Правило зафиксировано пре-регом — не трогаю; в run.log вердикты читаются вместе с
frac_cells_p_lt_alpha и per_seed-распределением в store.

## Smoke (после всех фиксов, 13.11)

`--smoke`: exit 0; G1′ пины ok; G0 gen sha f142f297… 64/64; L1-TPU ['own_w0','own_w4'];
G2 30/30; G8 0; G7-inv 0; PC 2.2e-16 / 8.5e-14; «AT0_2 SMOKE OK (артефакты не пишутся)» —
results/AT0/ не модифицированы (mtime сверены). 2m02s.

VERDICT=PASS (для перезапуска; независимое ревью — см. статус ниже)
