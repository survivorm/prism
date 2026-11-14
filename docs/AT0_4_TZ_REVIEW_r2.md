# AT0_4_TZ_REVIEW_r2
Ревьюер: prism-reviewer (closure r2)
1: closed — код: S3-цикл `for S in seeds_run` (full run: SEEDS=(11,12,13), R1_S+R2_S на каждый сид, FAIL→RC=3); TZ §4.1 S3 «ВСЕ 3 сида» + v3.1-нота; gates.S3_finite: seeds=seeds_run, нота «ВСЕ сида».
2: closed — код: try/except ValueError в run_cell → cells[...]= {"degenerate": True}; gnd: degenerate-док → ok=False, degenerate=True; inst_not_testable = n_ok==0 (правило «все доки»); медианы по оставшимся: P1 (reach исключает "degenerate"), P2/P4 (docs_ok), P3/P5 (p3_docs/p5_docs); TZ §4.2 «Degenerate-док (v3.1)».
3: closed — код: HNR_DIV_MULT=100.0, div = per-instance median ‖h‖ (real) > 100×median ‖h‖(R0 real); divergent ⇒ P1/P2/P4 not_testable; TZ §4.2 G_nd4 «жёсткий порог … ⇒ P1–P4 not_testable (v3.1 по р1-ревью)».
4: closed — код: store["prereg_text"] = "docs/AT0_4_prereg.md v3" (store-блок).
Доп: обоснование §12.7 ок (P2/P4 — пороговые сравнения, не p-value-тесты; перестановочный нуль меняет estimand, r1 сам квалифицировал как «добавить», не блокер); a priori vacuous P1 объявлен в TZ §5 (E[Σ‖Δu‖²]=2N=766 ⇒ r_max=4/766≈0.0052=2/383 — арифметика подтверждена).
НОВЫЕ:
- [COSMETIC] строка prereg_text "v3" слегка устарела: в документе теперь §11 (v3) + §12 (v3.1) — точнее "v3.1"; но чинка r1 предписывала ровно "v3" (исполнено как предписано — не чинить).
- [COSMETIC] широкий `except ValueError` в run_cell: в данной конфигурации (K=96 < N=383) единственный достижимый ValueError — условие c0=0 (du_metrics и field.py:86 — одно и то же условие; field.py:83 недостижим при фиксированном K), т.е. безопасно; специализированное исключение было бы робастнее.
VERDICT=PASS
AT0_4_TZ_R2_DONE
