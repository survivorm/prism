# REVIEW AT2c CODE3 (B1–B10) — 2026-11-17
Внешний ревьюер, автономный. Вердикт: **REVISE** (1 MAJOR, 1 NIT).

## Хеши (сверено)
- AT2c.json sha256 a388d52d…904e = /tmp/recovered/AT2c/AT2c.json ✓
- AT2c_run.log md5 11468364ff194145bcbff81c8011cfe2 ✓ (30 строк)
- AT2c_results.md sha256 8df94190…13fc ✓
- script_sha fb0b43fd…b832 = sha256(experiments/at2/AT2c.py) ✓

## B1: PASS
gates: 11 шт., все True (S1_sha…S7_exact_set).
Лог ↔ артефакт: S1 7/7; S2 KL 1.885e-08/5.229e-09; S4 90/4096 {8/41/36/5};
S5a r_final=7 rematched; S5b Gagg max 2.326e-12; S6c ybar_max k0..k8
6.69/6.56/6.37/6.53/6.58e-06, dcos worst (gated J) = 0.01261 (12|k8|weak_mid)
= 1.26e-02, confounding.max = 0.008685 (22|k8) = 8.68e-03; S7 540000/540000
set_eq=True; S8 0.0×4. Всё совпадает.

## B2: PASS
verdict="NOT_RESOLVED", control_ok=True.
a: n_pass=0; b: max_abs_top=0.39994≈0.40, max_abs_weak=0.16389≈0.16;
c: c_ok=False, k1_top_ge_max_wcos {6,12,18,22}=all True;
d: max_pairwise_diff=0.4074669≈0.4075.

## B3: PASS
s8_anchor: L6/L12/L18/L22: max_diff=0.0; got_mean==ref_mean и got_ci==ref_ci
(float-равенство) на всех 4 слоях. (ref медиан в артефакте нет — только
mean/ci; got-медианы на месте.)

## B4: PASS
per_pos_cos: 540000 float'ов, не пусто.
Per (L,k): n = 32·(128−k) — k0:4096, k1:4064, k2:4032, k4:3968, k8:3840,
32 дока, позиции t∈[k,127] — т.е. shape [32, 128−k] ✓ (все 25 пар (L,k),
включая L23; всего 135 ячеек (L,k,arm)).
wcos ∈ [min,max] per_pos: 6|k1|top: 0.3299 ∈ [−0.8999, 0.9432] ✓;
22|k8|weak: 0.0071 ∈ [−0.9029, 0.8745] ✓; 12|k4|pca: −0.0221 ∈ [−0.8764, 0.8524] ✓.

## B5: PASS
delta_min_calibration: max_pairwise_spread_Jarms_wcos=0.4074669≈0.4075
(= d_metric_uninformative.max_pairwise_diff); floor_CI_width_max=0.16020≈0.16;
delta_min=0.01, оба gt-флага false.

## B6: PASS
l23 k1: pca wcos=0.21612≈0.216; haar=0.20917≈0.209; uniform=0.05502≈0.055.

## B7: PASS
n8['6|k0|top'] = 0.71047 ≈ 0.71.

## B8: PASS (1 NIT)
wall_sec=242.5 (точно); assembled_at=2026-11-17T03:27:09.138922.
NIT: финальная строка лога «wall=243.2s» — дельта 0.7s к артефакту
(лог печатается после сборки артефакта); результаты.md верно использует
242.5 из артефакта.

## B9: FAIL — MAJOR (results.md, §7, «Числовой рисунок», таблица wcos)
Сверено с metrics[...]['wcos'] (артефакт), 4 слоя L6/12/18/22:
1) k0 uniform: md «0.04–0.16», факт 0.0447–0.1818 (max: 0.1818 @ L12) — max неверен.
2) k1 haar: md «0.12–0.19», факт 0.1206–0.2292 (max: 0.2292 @ L22) — max неверен.
3) k8 weak: md «0.007–0.045», факт −0.0014–0.0449 (min @ L18) — СМЕНА ЗНАКА.
4) k8 weak_mid: md «0.04–0.08», факт 0.0420–0.0901 (max @ L12) — max неверен.
5) k8 pca: md «−0.05…+0.04», факт −0.0473–0.0659 (max @ L18) — max неверен.
6) k8 haar: md «0.04–0.09», факт −0.0154–0.0935 (min @ L12) — СМЕНА ЗНАКА.
7) k8 uniform: md «0.006–0.046», факт −0.0044–0.0564 — min (ЗНАК) и max неверны.
Остальное §7 — верно: VERDICT/control_ok; все 11 гейтов с цифрами; дерево
(a) n_pass=0, wmeans k2/k4/k8 −0.14…−0.31/−0.15…−0.31/−0.16…−0.32 ✓ (все
16 weak_vs_maxPCAHaar wmean<0); (b) 0.40 @ 18|k1|top, 0.16 @ 6|k1|weak ✓;
(c) граничные CI: 18|k8 top_vs_pca=[−0.0047,0.1167]≈[−0.005,+0.117] ✓,
22|k8 top_vs_haar=[0.0027,0.0725]≈[+0.003,+0.073] ✓, единственный CI top_*
с lower<0 — ровно этот; (d) 0.4075 ✓; S8=0.0 ✓; wall=242.5 ✓;
sha256/md5/trackio-md5 в §8/§7 ✓.

## B10: PASS
trackio prism.db, run_name='at2c_t21c_r1' (9 точек):
t=0: {started:1.0} + stage load(0.7s); configs: started_at=2026-11-17T03:23:06.688969,
revision=r1, script_sha=fb0b43fd…(=sha256 AT2c.py), seed=0, boot_b=1000,
delta_min=0.01, r=32, group=AT0, log-путь ✓.
Stage-точки: subspaces pass1_deff(52.7s), subspaces pass2_stationary
(124.2s, r_final=7, rematched=true) ✓, ridge(224.9), lk4(237.1),
verdict(237.2), artifact(241.6).
Финальная точка (step=243): status="ok", verdict="NOT_RESOLVED",
log_md5="11468364ff194145bcbff81c8011cfe2" = md5 лога ✓,
finished_at=2026-11-17T03:27:09.9.

## Итог
- MAJOR: B9 — таблица wcos §7: 7 неверных границ (3 смены знака в строке k=8)
  против metrics в AT2c.json. Вердикт/дерево/гейты не затрагнуты, но
  «каждое число — из артефакта» нарушено. Локация: docs/AT2c_results.md,
  §7, блок «Числовой рисунок», строки k=0/k=1/k=8.
- NIT: B8 — log «wall=243.2s» vs артефакт 242.5 (0.7s, эффект печати).
- B1–B8, B10: PASS.

## NOTES
- per_pos_cos хранит per-position cos (не wcos); wcos из metrics —
  weighted-mean по позициям, внутри [min,max] ✓.
- Строки k=0/k=1 в md, кроме указанных, совпадают с 4-слойным range.
- L23 (ridge) не входит в таблицу md (у него нет top/weak/weak_mid) —
  согласуется с arms_L23 в s7.

REVIEW_AT2C_CODE3_DONE
