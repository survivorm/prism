# ТЗ: миграция всех экспериментов prism в trackio

**Статус:** v1.9 (аддендум 16.11: завершение миграции — §9) + v1.8 (patch по TZ-ревью round 4, docs/dynamic/trackio_migration_TZ_review_r4.md:
N16 at16 пер-файловая атрибуция (ceil 47 / ceil3 71 / ceil4 144), N17 §3.3 «1 md5-дубль»→2
(3 T4 + 2 CPU/AT0 = 5 ✓), N18 §7.5 empty_metrics + e18_logit_kd_tpu (11 T4 + e27b_early = 12;
step_idx — прогресс-маркер, не экспериментальная метрика). v1.7 — round 3:
N10 MAJOR cell_s-паттерн `, 0\)`→`, (\d+)` (24→192 точки, e32r3), N11 третий Scenario-формат
(e19_destruction_short: 4 заголовка (a)–(d)), N12 stale-счётчики 72/71→75/74, N13 Total time
8→9 строк, N14 P2c-corc коллизии (129 точек / 121 уник. имя), N15 quiescence-проверка
до запуска (живой писатель AT0_2_r6 под md5-гейтом); + заметки r3: at16 ×3, p_same — только
`hf:`, dip e64 129 (tpu_v8) / 130 (×2 лога). v1.6 — round 2: N1 реестр 115/130 (+3 run'а
+5-й md5-дубль), N2 4 строки T4-таблицы на канон max-time, N3 [eval] step без хвостового
«:», N4 dip-паттерн 129/129 (без якоря, P2c/P4-T*), N5 два формата Scenario, N6–N9 stale-
счётчики/эрата. v1.5 — M2 J_norm:, M3 kl-регистр/знак, m2/m3 (не сработали — см. N3/N4,
эрата), B1-правило; v1.4 — M1 CPU-маркеры, M4 spot-check'и, m4–m9).
ЗАКРЫТО (round 4: PASS, 0 MAJOR; N16–N18 закрыты в v1.8)
**Инициатор:** пользователь («собери по логам… поставить trackio, занести все эксперименты; максимально аккуратно, петля субагентного ревью»)
**Дата:** 2026-11-14
**Связано:** `docs/dynamic/0_EXPER_DURATIONS.md` (инвентаризация логов, 2026-11-14)

## 1. Цель

Одноразовая **не-деструктивная** миграция: каждый экспериментальный прогон prism
(**115 прогонов** — полный инвентарь 130 .log-файлов на дату 15.11 04:00, исключая корневой
`.hf_cache/` и `box/`) получает run в trackio (локальный, open-source, SQLite), чтобы все данные
были доступны в едином дашборде. Никаких логов/артефактов не менять; trackio-БД — производный
регенерируемый кэш от логов (в git не коммитится). Перед запуском скрипта — свежий find (правило 7
AGENTS): если инвентарь изменился (новые .log) — пересобрать реестр, не молча афтить.

## 2. Окружение (проверено)

> **v1.9: устарело — см. §9.5.** Строка «`trackio==0.45.1` в `.venv` (Python 3.13.14)» не действует:
> в `.venv` trackio не импортируется (httpx-ошибка); рабочий интерпретатор — `/opt/miniconda3/bin/python3.12`
> (py 3.12.9, trackio 0.45.1 в user-site `~/.local/lib/python3.12/site-packages`).

- `trackio==0.45.1` в `.venv` (Python 3.13.14), `uv pip install` — локальная сеть, HF-токен не нужен.
- API: `trackio.init(project, name, group, config)` → `run.log(metrics, step)` → `run.finish()`.
- Хранилище: `{TRACKIO_DIR}/{project}.db` (SQLite) + `{project}.lock`.
- CLI-верификация: `trackio list runs --project prism --json`, `trackio get metric --project P --run R --metric M`, `trackio status`.
- **Решение по хранилищу:** `TRACKIO_DIR=/home/survivor/work/prism/.trackio` (внутри репо, в `.gitignore`).
  Обоснование: БД — регенерируемый кэш (источник правды = логи + скрипт миграции);
  держать рядом с репо, чтобы не потерять при чистке `~/.cache`; в git не попадает.

## 3. Реестр прогонов (ПОЛНЫЙ: 115 run'ов; сверка: 130 .log = 113 уникальных файлов в run'ах + 5 md5-дублей + 12 исключённых (+1 второй run на общем файле, +1 run из bench_all.txt))

Каждый run: `name` (snake_case), `log` (путь), `device`, `duration_s` (только если в логе/очереди
есть прямое число; иначе `null`), `notes`. Скрипт должен **афтить** реестр против find-списка
(каждый .log либо run, либо дубль, либо исключён) и писать расхождение как ошибку.

### 3.1 GPU T4 (Kaggle, 38 run'ов из 41 файла; 3 md5-дубля — §3.5)

| # | name | log | duration_s (источник) |
|---|------|-----|-----|
| 1 | e17_lora_distill | prism-exp17-lora-distill.log | макс `time` JSON |
| 2 | e16_4_3_v23_fd_check | prism-exp16-4-3-v23-single-pair-fd-check.log | макс `time` |
| 3 | e32r3_layer_profile | prism-exp32r3-layer-profile.log | макс `time` |
| 4 | e50v3_loop_curvature | logs/e50v3_out/prism-exp50v3-loop-curvature-qwen.log | макс `time` (12210.66; маркер `DONE 12192.1s` → duration_marker_s) |
| 5 | e52k_scale_curve_ptq | logs/e52k_v7/prism-exp52k-scale-curve-real-ptq.log | макс `time` (голова лога обрезана Kaggle — пометить) |
| 6 | e27_noise_floor | kaggle_kernel/kernel-exp27/out/prism-exp27-noise-floor.log | макс `time` |
| 7 | e27_noise_floor_v9 | _auto_chain/exp27_v9/prism-exp27-noise-floor.log | макс `time` |
| 8 | e27b_dense_profile | kaggle_kernel/prism-exp27b-dense-profile.log | макс `time` |
| 9 | e39m_flat_ptq | logs/exp39m/run.log | макс `time` |
| 10 | e16_readout_pull | prism-readout-pull.log | макс `time` (4600.17; маркер `[OK] … 4360s` → duration_marker_s) |
| 11 | e16_stage0_targets | prism-exp16-stage0-targets.log | макс `time` (3535.15; маркер `Total time: 3318s` → duration_marker_s; канон дублей §3.5) |
| 12 | e4_k_sensitivity | prism-exp4-k-sensitivity.log | макс `time` |
| 13 | e19_destruction_tpu | kaggle_kernel/kernel-exp19-tpu/prism-exp19-destruction-tpu.log | макс `time` |
| 14 | e18_logit_kd | prism-exp18-logit-kd.log | макс `time` |
| 15 | e11c_defrag_bridge | prism-exp11c-defrag-bridge.log | макс `time` (1760.14; маркер `[OK] … 1545s` → duration_marker_s) |
| 16 | e64_tpu_v8 | logs/e64/tpu_v8/prism-e64-gpu.log | макс `time` |
| 17 | e19_destruction_early | prism-exp19-destruction-tpu.log | макс `time` |
| 18 | e16_stage0_targets_short | logs/prism-exp16-stage0-targets_short.log | макс `time` |
| 19 | e10b_gauge_bridge | prism-exp10b-gauge-bridge.log | макс `time` |
| 20 | e01_coherence | prism-exp01-coherence.log | макс `time` (канон дублей §3.5) |
| 21 | e18_logit_kd_tpu | kaggle_kernel/prism-exp18-logit-kd-tpu.log | макс `time` |
| 22 | at16_ceiling_ceil4 | logs/ceil4_out/prism-at16-ceiling.log | макс `time` |
| 23 | at16_ceiling | logs/ceil_out/prism-at16-ceiling.log | макс `time` |
| 24 | e16_greedy_proto | logs/proto_out/prism-greedy-proto.log | макс `time` |
| 25 | rleaf_random | logs/rleaf_out/prism-r-leaf-random.log | макс `time` |
| 26 | at16_ceiling_ceil3 | logs/ceil3_out/prism-at16-ceiling.log | макс `time` |
| 27 | e32r3_1_tail_recompute | kaggle_kernel/kernel-exp32r3/output_v7/prism-exp32r31-tail-recompute.log | макс `time` |
| 28 | e64_b4_temperature_pool | logs/e64/kag_out/prism-b4-temperature-pool.log | макс `time` |
| 29 | e2_heatmap | prism-exp2-heatmap.log | макс `time` |
| 30 | e6_l18_snr | prism-exp6-l18-snr.log | макс `time` |
| 31 | e19_destruction_short | prism-exp19-destruction.log | макс `time` |
| 32 | e38d_holonomy | prism-exp38d-holonomy-spectrum.log | макс `time` |
| 33 | e16_readout_pull_tpu | prism-exp16-readout-pull-tpu.log | макс `time` |
| 34 | e64_gpu_old | logs/e64gpu_out_old/prism-e64-gpu.log | макс `time` |
| 35 | e16_readout_pull_tpu2 | prism-readout-pull-tpu.log | макс `time` |
| 36 | e17_2nd_canary_tpu | kaggle_kernel/kernel-exp17-2nd-canary/out/prism-exp17-2nd-order-canary-tpu.log | макс `time` |
| 37 | lfm3_textgen | logs/lfm3gen_out/prism-lfm3-textgen.log | макс `time` |
| 38 | probe_input | logs/probe_out/prism-probe-input.log | макс `time` |

**Правило duration для T4:** `duration_s` = МАКС поля `time` по всему JSON-логу (единый смысл:
всё время kernel'а включая NbConvert-экспорт). Явные маркеры (`[OK] ... Ns`, `Total time: Ns`,
`DONE Ns`, `ALL DONE Ns`) извлекаются в `duration_marker_s` + `duration_marker_src`; сверка
marker vs max_time: dev > 30s = warning в отчёте (систематический хвост NbConvert-экспорта,
проверено на e16_readout_pull/e16_stage0/e11c: dev 215–240s). Ничего не округлять.

### 3.2 Табличный Kaggle-лог (1 run)

| # | name | log | duration_s (источник) |
|---|------|-----|-----|
| 39 | e27b_early | prismexp27b.log | макс колонки `Time` (289.3s; 82 записи) |

Формат: артефакт-заголовок Kaggle на 5 строк (`Time\n\t\n#\n\t\nLog Message\n`), затем записи
`^\d+\.\d+s\t(\d+)\t<текст>`; текст может занимать несколько строк (чанк до следующей записи).

### 3.3 CPU (локальные, 75 run'а на 74 уникальных файлах; 2 md5-дубля — §3.5)

| # | name | log | duration_s (источник) |
|---|------|-----|-----|
| 40 | e45_damage_vs_structure | logs/exp45_damage_vs_structure.log | null в своём логе; **нижняя граница 15505s** из queue_round.log (23:00:41→03:19:06) — в config `duration_lb_s` |
| 41 | e45b_leaf_table | logs/exp45b_leaf_table.log | 6079 (queue: 04:18:49→06:00:08) |
| 42 | e45c_gauge_null | logs/exp45c_gauge_null.log | 3583 (queue: 03:19:06→04:18:49) |
| 43 | e46_frame_as_memory | logs/exp46_frame_as_memory.log | 8 (queue: 06:00:09→06:00:17) |
| 44 | e45c_rerun | logs/exp45c_gauge_null.log | 1203 (лог: `wall 1203.0 с`; tail_round 05:15:41→05:35:58=1217s); **второй run на том же файле**: файл содержит ТОЛЬКО вывод rerun (первый прогон из queue перезаписан) — в config |

| 45 | e45_seednull | logs/exp45_seednull.log | 255 (queue: 06:00:18→06:04:33) |
| 46 | e45_verdicts | logs/exp45_verdicts.log | null |
| 47 | e45_gap_scan | logs/exp45_gap_scan.log | 1007.9 (`wall 1007.9 с`) |
| 48 | e45_kappa_seed_null | logs/exp45_kappa_seed_null.log | null |
| 49 | e45b_analysis | logs/exp45b_analysis.log | null |
| 50 | e45f_gauge_decomp | logs/exp45f_gauge_decomp.log | 2373 (`wall 2373 с`) |
| 51 | e45g_cut_stability | logs/exp45g_cut_stability.log | 779 (`wall 779 с`) |
| 52 | e46_verdicts | logs/exp46_verdicts.log | null |
| 53 | e47c_p2_loop_prediction | logs/exp47c_p2_loop_prediction.log | null |
| 54 | e7_cka_baseline | logs/exp7_cka_baseline_n1024.log | 26620 (`[OK] … 26620s`) |
| 55 | e_r11_basis_tie | logs/exp_r11_basis_tie.log | null |
| 56 | e32r2_hessian | logs/exp32r2_hessian.log | 6142 (`total 6142s`) |
| 57 | e32r2_smoke | logs/exp32r2_smoke.log | 170 (`total 170s`) |
| 58 | e32r3_1 | logs/exp32r3_1.log | null |
| 59 | e32tail_pilot | logs_e32tail_pilot.log | null (Triton→CPU fallback) |
| 60 | e35_preflight | logs/exp35_preflight.log | null |
| 61 | e37_run | logs/exp37_run.log | null |
| 62 | e37_run2 | logs/exp37_run2.log | null |
| 63 | e37_run3 | logs/exp37_run3.log | null |
| 64 | e37_run4 | logs/exp37_run4.log | null |
| 65 | e37r_run | logs/exp37r_run.log | null |
| 66 | e37r_agg | logs/exp37r_agg.log | null |
| 67 | e38a_rerun | logs/exp38a_rerun.log | null (`in 7.7s` — время записи JSON, НЕ duration — не извлекать) |
| 68 | e38b_rerun | logs/exp38b_rerun.log | null |
| 69 | e38c_rerun | logs/exp38c_rerun.log | null |
| 70 | e38d_rerun | logs/exp38d_rerun.log | null |
| 71 | e38e_rerun | logs/exp38e_rerun.log | null |
| 72 | e38f_rerun | logs/exp38f_rerun.log | null |
| 73 | e38g_rerun | logs/exp38g_rerun.log | null |
| 74 | e38h_rerun | logs/exp38h_rerun.log | null |
| 75 | e38i_rerun | logs/exp38i_rerun.log | null |
| 76 | e38e_true_j_l23 | logs/E38e_true_J_L23.log | 1374.6 (`wall 1374.6 с`); канон дубля §3.5 |
| 77 | e39a_v2 | logs/archive/exp39a_v2.log | null |
| 78 | e39a_v3 | logs/archive/exp39a_v3.log | null |
| 79 | e39a_v4 | logs/archive/exp39a_v4.log | null |
| 80 | e39a_v5 | logs/archive/exp39a_v5.log | null |
| 81 | e39a_v6 | logs/archive/exp39a_v6.log | null |
| 82 | e39a_v7 | logs/archive/exp39a_v7.log | null |
| 83 | e39a_v8 | logs/archive/exp39a_v8.log | null |
| 84 | e39a_v9 | logs/archive/exp39a_v9.log | null |
| 85 | e39a_v10 | logs/archive/exp39a_v10.log | null |
| 86 | e39a_v11 | logs/archive/exp39a_v11.log | null |
| 87 | e39c_pre_rerun_fix | logs/exp39c_pre_rerun_fix.log | null |
| 88 | e39d_head_route_probe | logs/E39d_head_route_probe.log | 489.7 (`wall 489.7 с`) |
| 89 | e39d_control_lam0_order1 | logs/exp39d_control_lam0_order1.log | null |
| 90 | e39d_floor_seeds12 | logs/exp39d_floor_seeds12.log | null |
| 91 | e39d_stage1_ordering | logs/exp39d_stage1_ordering.log | null |
| 92 | e39j_run | logs/exp39j_run.log | null |
| 93 | e39k_run | logs/exp39k_run.log | 22.6 (`wall 22.6 с`) |
| 94 | e39m_smoke | logs/exp39m_smoke.log | null |
| 95 | e39_supp_pscale_probe | logs/exp39_supp_pscale_probe.log | null |
| 96 | e39_supp_three_objects | logs/exp39_supp_three_objects.log | 356.6 (`wall 356.6 с`) |
| 97 | e01_coherence_v6_draw2 | logs/exp01_coherence_v6.draw2.log | null |
| 98 | e01_coherence_v6_pinned | logs/exp01_coherence_v6.pinned.log | 213 (`[OK] … 213s`) |
| 99 | e51_lfm3 | logs/e51_lfm3.log | null |
| 100 | e51_qwen | logs/e51_qwen.log | null |
| 101 | at0_1 | logs/AT0/AT0_1_run.log | null |
| 102 | at0_2 | logs/AT0/AT0_2_run.log | null |
| 103 | at0_2_r6_stdout | logs/AT0/AT0_2_r6_stdout.log | null |
| 104 | at0_3 | logs/AT0/AT0_3_run.log | null |
| 105 | at0_4 | logs/AT0/AT0_4_run.log | null |
| 106 | at0_4_rebuild | logs/AT0/AT0_4_rebuild.log | null |
| 106b | at0_4_rebuild_r3 | logs/AT0/AT0_4_rebuild_r3.log | null (rebuild r3, 2026-11-14; маркеров wall нет) |
| 106c | at0_4_rebuild_r4 | logs/AT0/AT0_4_rebuild_r4.log | null (rebuild r4, 2026-11-14; маркеров wall нет) |
| 106d | at0_4_rebuild_r5 | logs/AT0/AT0_4_rebuild_r5.log | null (rebuild r5, 2026-11-15; маркеров wall нет) |
| 107 | at0_5 | logs/at0_5_run.log | null |
| 108 | chain_e45b | logs/chain_e45b.log | null |
| 109 | chain_e46 | logs/chain_e46.log | null |
| 110 | stage0 | logs/stage0.log | null (`Device: cpu`, n_samples=2, K=128 — smoke-прогон) |
| 110b | at0_2b | logs/AT0/AT0_2b_run.log | null (R/A-split r2, 2026-11-15; маркеров wall нет) |

Итого CPU: **75 run'а** (74 уникальных файла + второй run на exp45c-файле).

**Правило duration для CPU:** авто-извлечение стандартных маркеров скриптом: `wall ([\d.]+) с`,
`\btotal (\d+)s`, `\[OK\] \S+ (\d+)s` (скрининг по 74 CPU-логам: ложных срабатываний нет,
каждый лог — ≤1 маркера одного семейства; 3 новых лога 106c/106d/110b — по 0 маркерных строк,
проверено r3 N12). Плюс wall-часы queue/tail-логов (e45, e45b, e45c, e46,
e45_seednull — из queue_round.log). «Время записи JSON» (`in 7.7s`) НЕ извлекать. Для e45 —
`duration_lb_s` (нижняя граница). Найденные маркеры — в отчёте со строкой-источником.

### 3.4 iGPU (1 run)

| # | name | источник | metrics |
|---|------|-----|-----|
| 111 | e93_sycl_bench | results/e93/sycl/bench_all.txt | t/s из bench-таблицы (pp512/tg64 по model×precision×fa) — config + финальные метрики; единого duration нет |

**ИТОГО: 115 run'ов** (T4 38 + таблица 1 + CPU 75 + iGPU 1).
Сверка: 130 .log-файлов (find на дату 15.11 04:00) = 113 уникальных файлов в run'ах + 5 md5-дублей +
12 исключённых; run'ов = 113 + 1 (второй run на exp45c-файле) + 1 (e93 из bench_all.txt) = 115.

### 3.5 Байт-идентичные копии (НЕ отдельные run; md5-сверка скриптом → отчёт)

- exp01: `kaggle_kernel/kernel-exp01/out/prism-exp01-coherence.log` = дубль `prism-exp01-coherence.log` (md5 ec9573a5…)
- stage0: `exp16-stage-0-target-jacobian-cache.log` = `exp16-stage0-target-jacobian-cache.log` = `prism-exp16-stage0-targets.log` (md5 adc4008c…) — все три = run #11
- E38e: `logs/E38e_true_J_L23_carrier_cheat.log` = `logs/E38e_true_J_L23.log` (md5 d636bc48…)
- AT0_1: `logs/AT0/store/runlog_r5_pre_rerun.log` = `logs/AT0/AT0_1_run.log` (md5 2b2c3661…;
  пин р5-перепуска — байт-копия run-лога, 5-й дубль, v1.6 N1)

### 3.6 Исключённые файлы (12 из 130, НЕ run'ы; список в отчёте с причиной)

- kaggle_kernel/auto_chain_tpu.log, _push_queue.log, _push_wait.log (3) — статусы-очереди Kaggle
- logs/kaggle_daemon.log, logs/exp32r3_1_status.log (2) — статусы
- logs/queue_round.log, queue_tail.log, tail_round.log (3) — wall-часы очереди (ИСТОЧНИК duration, не run)
- logs/e93/sycl/bench_err.log, smoke_err.log (2) — stderr бенча
- xet-логи (2): kaggle_kernel/kernel-exp17-2nd-canary/out/.hf_cache/…, kaggle_kernel/kernel-exp19-tpu/.hf_cache/… — сетевые логи HF

Вне скоупа find (не входят в 130 и в 12): `box/*` (4 файла — мониторинг/автономный бокс) и корневой
`.hf_cache/` (3 xet-лога). В отчёте список исключений приводится целиком с причинами.

## 4. Извлечение метрик (только из строк логов, ничего не выдумывать)

**Универсальные экстракторы** (применяются ко всем JSON-Kaggle-логам; шаг = int(поля `time`);
каждый паттерн верифицирован по фактическим строкам логов, 0-hit паттерны не включены):
1. `bridge=([\d.]+)` → `bridge` (7 логов)
2. `kl=([-+]?[\d.]+)` (с `re.I`) → `kl` (регистр: в логах и `kl=` (500), и `KL=` (1051);
   всего 1551; знак: e17 `kl=-0.0000`; v1.4-паттерн `\bKL=([0-9.]+)` пропускал оба — M3)
3. `‖ΔW‖=([\d.]+)` → `dose` (3)
4. `L(\d+): (\d+) J done \((\d+)/(\d+)\)` → `j_progress_L{N}` (e27 ×2)
5. `L(\d+) J (\d+)/(\d+)` → `j_progress_L{N}` (e27b-dense)
5b. `J_norm: (\d+)/(\d+)` → `j_progress` (e17, 64 точки; v1.2-паттерн `J_norm (\d+)/(\d+)`
   без двоеточия — 0 совпадений (M2); в v1.4 паттерн удалён без замены — e17 лишился
   j_progress; v1.5 возвращает)
6. `Layer \d+_\d+ done in (\d+)s` → `layer_done_s` (2)
7. `step (\d+)/(\d+):` → `step_idx` (5)
8. `step (\d+): loss=([\d.]+)` → `loss` (e39m, 45 точек)
9. `chat s\d+: loops=\d+ a=… s=… p=… g=… ent=…` → `chat_{a,s,p,g,ent}` (e52k; в логе одна строка —
   середина chat-лупа утеряна Kaggle-стримингом)
10. `[eval] step=?\s*(\d+)(?=[:\s])` → `eval_step_idx` (два формата: `step 1:` — e16-readout-pull
    (1000), `step=0 bridge_loss=…` — e16_4_3_v23 (14, БЕЗ хвостового «:» после числа — v1.5-паттерн
    с обязательным `:` давал 0/14, N3); lookahead `(?=[:\s])` покрывает оба: 1000+14)
11. `P([0-5][a-z]?)(?: ([\w.]+))? dip (-?[\d.]+)` → `p{n}[_{label}]_dip` (e64 ×2: 130 dip-строк
    = 129 tpu_v8 + 1 e64_gpu_old (`P0 dip 0.4515`); структурная роспись ниже — про tpu_v8, 129)
    — ВСЕ с префиксом Kaggle-таймстампа `[   NNNs] ` — якорь `^` не использовать (v1.5-паттерн с `^`
    давал 0/129, N4); структура: `P0 dip 0.5528` (без label), `P1 own_w0…own_w15 dip` (16),
    `P2 cor_w0…cor_w15 dip` (16), `P2c corc_N/corm_N dip` (24 — суффикс «c»), `P3 rnull_s{0,1,2}_w0…w7 dip`
    (24), `P4 T*.{NN}_wN dip` (48 — точка в label: `\w+` → `[\w.]+`); имена: p0_dip, p1_own_wN_dip,
    p2_cor_wN_dip, p2c_corc_N_dip, p2c_corm_N_dip, p3_rnull_sN_wN_dip, p4_T*.NN_wN_dip —
    **129 совпадений / 121 уник. имя**: `p2c_corc_{0..7}_dip` ×2 — два измерительных раунда
    (t≈578–580s: dip −0.1917…, t≈581–584s: dip −0.1602…, значения различаются; разделителя-
    заголовка между раундами в логе нет), step=int(time) → оба раунда = разные точки одной
    серии, 0 потери (N14); отрицательные значения есть — знак в паттерне)
12. **stateful scenario-tagging (e19):** ТРИ формата заголовка: `--- Scenario (X) ---`,
    `--- Scenario: X (baseline) ---` (baseline — двоеточие + суффикс `(baseline)`, v1.5-паттерн
    только со скобками его не ловил, N5) и `--- Scenario \(([a-z])\): <описание> ---`
    (e19_destruction_short: 4 заголовка `(a): random ΔW` / `(b): 10×α*` / `(c): gauge component`
    / `(d): differential component`, после каждого по строке `bridge=…, KL=…`; X = a/b/c/d — N11;
    r2-формулировка «в e19_destruction_short сценариев нет» ложна — их 4). Начальное состояние =
    no_edit. Заголовок задаёт сценарий; следующие строки (до следующего заголовка) пишутся как
    `bridge_{X}`, `kl_{X}`, `dose_{X}` (X ∈ no_edit, a_random, b_target_1x, c_target_10x, d_gauge,
    e_differential; для третьего формата — буквы X = a/b/c/d).
13. `K=([\d.]+): avg=([\d.]+)` → `k_avg` (e4, 7 точек)
14. `(\d+)->(\d+): raw=…, soft=…, dr8=…, dr16=…, dr32=…` → `raw/soft/dr8/dr16/dr32` (e11c, 8 точек)
15. `raw vs (\w+): ρ = ([\d.-]+)` → `rho_{defrag_N/softmax}` (e11c)
16. `(\d+)->(\d+): raw=…, cent=…, soft=…` → `raw/cent/soft` (e10b, 7 точек)
17. `L(\d+) T=128 \('(\w+)', (\d+)\): ([\d.]+)s` → `cell_s_L{L}_{cell}`, step = eps-idx (0..7)
    (e32r3, **192 точки** = 6 слоёв [6,12,18,19,22,23] × 4 клетки (chat/code/math/wiki) × 8 eps;
    в шапке лога «eps: 16 pts», но измеренных idx 0..7 — факт, проверен python: 192 строк;
    v1.6-паттерн с захардкоженным `, 0\)` даёт 24 — N10 MAJOR: 168 из 192 точек флагманского
    e32r3 молча терялись)
18. `L(\d+): s_l=… slope=… quad=… d3r1 raw/gf/fixP = a/b/c d_b=…` → `s_l/slope/quad/d3r1_raw/d_b` (e32r3, e32r3_1_tail)
19. `(\w+) s\d+: \d+L x \d+pos J done \((\d+)/(\d+)\)` → `j_progress` (e01)
20. `L(\d+): C_b=([\d.]+)` → `C_b_L{N}` (e01)
21. `(\S+) (\S+)\|(bare|nudge) (hit|garbage) (\d+)s` → `at16_{hit|garbage}_{variant}` (at16 ×3:
    ceil 47 / ceil3 71 / ceil4 144 + e64_b4 24 = 286; v1.7-атрибуция перепутана — N16)
22. `rand seed (\d+) (\d+)s` → `rand_seed_s`; `trained done (\d+)s` → `trained_done_s` (rleaf)
23. `^\w+_\d+ done$` → `doc_done=1` (e50v3, 64 точек)
24. `(\w+): \{'p_same': …, 'sha': …, 'uniq': …` → `p_same_{src}`, `uniq_{src}` (e16_greedy_proto;
    совпадает ТОЛЬКО строка вида `hf: {…}` (с двоеточием) — 1 из 4; `dup_cached/clean_cached/
    clean_padded {…}` (без двоеточия) не совпадают → метрики только `p_same_hf`/`uniq_hf`, по 1
    точке — r3-заметка)
25. строка начинается с `{` и содержит `"trained":` → flatten `trained_{k}` (rleaf JSON-строка)

**Гарантированная точка:** каждый run пишет `n_lines` (реальный счётчик строк лога) на step=0 —
trackio 0.45.1 выбрасывает run без метрик (проверено: БД не создаётся).
**run_failed (T4):** config/отчёт помечают run, если в логе `[FAIL]` / `Aborted (core dumped)` /
`AssertionError` (10 прогонов: e27×2, e19×2, e18_tpu, e2, e38d, e16_readout_tpu×2, lfm3).

Длительность-маркеры (фактические счётчики по 130 логам, v1.6 N7; v1.7 N13): `[OK] \S+ (\d+)s` (21 строка
по 21 файлу: 16 T4 + 2 CPU + 3 md5-дубля; уник. содержимое 18), `Total time: (\d+)s` (9 строк,
7 уник.; 3318s ×3 = stage0-дубль, 19724 (e16_4_3_v23), 31653 (e17), 2267 (e18), 57 (e19_short),
4359 (e16_readout_pull), 1005 (e16_stage0_short)), `DONE ([\d.]+)s` (2 чистых: e50v3, e52k; остальные 5 строк —
`ALL DONE Ns`), `ALL DONE (\d+)s` (5: 373/376/413/634/775s, at16/rleaf/proto; плюс варианты без
` s` и с dict — в from-строки не извлекать) → `duration_marker_s` (сверка с max_time, см. §3.1).

**Прогресс-бары (tqdm), pip/Downloading/NbConvert/warnings — не извлекать (фильтр по подстрокам:
`it/s`, `MB/s`, `Downloading`, `Collecting`, `Installing`, `Uninstalling`, `WARNING`, `Warning:`,
`Deprecation`, `━`, `Loading weights`, `nbformat`, `torch_xla`, `huggingface`, `[pip]`, `[notice]`,
`Successfully (installed|uninstalled)`, `Found existing`, `Attempting uninstall`).**
Для text-логов без `time` (CPU): metrics НЕ создаются (кроме `n_lines`) — только config + duration
из явных маркеров. Для e93: таблица bench парсится по строкам `| model | … | test | t/s |`;
ключ метрики несёт precision+fa из заголовка секции: `t_s_{f16|f32}_fa{on|off}_{pp512|tg64}`
(4-я секция пустая — бенч оборван; 6 точек).

**Правило провенанса:** каждое значение в trackio должно быть воспроизводимо из строки лога
(скрипт в отчёт пишет `n_lines_scanned`, `n_metrics`, примеры первых/последних извлечённых строк).

## 5. Safety-требования (максимальная аккуратность)

1. **Не-деструктивность:** скрипт читает логи read-only. До и после миграции — md5 всех 130 логов;
   расхождение = аварийный сбой с ненулевым RC. **Quiescence (N15):** перед запуском — проверка,
   что нет живого писателя в регистровые логи (ps/lsof по путям логов; мtime стабильный ≥10 мин);
   известный случай: `AT0_2_shape.py` (r6-свип, 96 доков) активно дописывает
   `AT0_2_r6_stdout.log` (реестр #103) — миграция стартует ТОЛЬКО после завершения r6 (лог стабилен);
   единственный допустимый fallback — явное исключение ЭТОГО ОДНОГО файла из md5-гейта с
   повторной md5-сверкой после завершения r6 (в отчёте с пометкой), остальные 129 — жёстко.
2. **Идиодентичность/контроль перезапуска:** проект `prism` — если `.trackio/prism.db` существует
   и `--fresh` не задан → RC≠0 с сообщением (молчаливого перезаписывания нет).
   `--fresh` удаляет только `.trackio/prism.db` (не трогая остальное в `.trackio`).
3. **git-гигиена:** `.trackio/` в `.gitignore` + исключение `!results/trackio_migration_report.json`
   (иначе `results/*.json` глотает отчёт — проверено `git check-ignore`); коммитятся: скрипт, ТЗ, ревью,
   отчёт, .gitignore, AGENTS.md (секция trackio), run_one.sh (export TRACKIO_DIR).
4. **Цифры только из логов** (AGENTS.md): duration/metrics — из строк лога или явных маркеров очереди;
   оценочные значения (знак `~` в реестре) в trackio НЕ пишутся — пишется точное значение,
   извлечённое скриптом (макс `time` и т.д.); `~` в реестре = ожидание для сверки.
5. **Правило 7 AGENTS.md:** артефакт-отчёт не редактируется вручную после прогона; правки = перепуск.

## 6. Артефакты

- `experiments/trackio_migration.py` — скрипт (реестр + парсеры + экстракторы + верификация).
- `AGENTS.md` — секция «trackio» (новые локальные прогоны пишут в trackio; Kaggle — через бэкилл).
- `run_one.sh` — `export TRACKIO_DIR="$PRISM_DIR/.trackio"` (безопасно для процессов без trackio).
- `.gitignore` — строка `.trackio/`.
- `results/trackio_migration_report.json` — отчёт: все run'ы, n_metrics, duration, md5 логов
  (до/после), список извлечённых маркеров, ошибки. JSON — sanitized (float-касты, no NaN).
- `.trackio/` — trackio-БД (gitignored).
- `docs/dynamic/trackio_migration_TZ_review.md` — ревью ТЗ.
- `docs/dynamic/trackio_migration_REVIEW.md` — внешнее ревью кода+артефакта.

## 7. Верификация (обязательная до коммита)

1. `trackio list runs --project prism --json` → число run'ов = 115 (числу в реестре).
2. Spot-check (строки фиксируются в REVIEW-файле):
   a) метрики — `trackio get metric`: e17_lora_distill (bridge/kl/j_progress/step_idx),
      e16_readout_pull (bridge/KL/dose/eval_step), e27b_dense_profile (j_progress_L),
      e64_tpu_v8 (129 dip-точек / 121 уник. имя — p2c_corc ×2 раунда; p{n}_{label}_dip);
   b) config — `trackio get run`: e32r2_hessian (duration_s=6142), e50v3_loop_curvature
      (duration_s=max_time≈12211, duration_marker_s=12192.1).
3. md5-сверка логов до/после: 0 расхождений.
4. `git status` — в индексе только разрешённые файлы (скрипт, ТЗ, ревью ×2, отчёт, .gitignore, AGENTS.md, run_one.sh).
5. Отчёт: 0 ошибок; все run'ы из реестра присутствуют; T4-run'ы без эксперимент-метрик помечаются
   `empty_metrics: true` (законно пустые: только прогресс-маркеры/краш/проба — e16_stage0×2, e2, e6_l18,
   e38d, e16_readout_tpu×2, e17_canary, lfm3, probe, e18_logit_kd_tpu (11 T4; step_idx (паттерн 7)
   и n_lines — прогресс-маркеры, не экспериментальные метрики — N18) + e27b_early табличный =
   12 всего); warnings — только dev-маркера
   (NbConvert-хвост).

## 8. Порядок работы (протокол)

1. ТЗ (этот файл) → ревью-луп ТЗ (субагент) → патч.
2. Написание скрипта → само-аудит (чек-лист AGENTS.md) → смоук на 2-3 логах в `/tmp/trackio_test`.
3. Quiescence-проверка §5.1 (нет живого писателя в регистровые логи; AT0_2 r6 завершён).
4. `setsid nohup ./run_one.sh trackio_mig ".venv/bin/python experiments/trackio_migration.py" &` (полный прогон;
   обёртка обязательна — память-вэйт run_one.sh foreground, AGENTS п.6).
   **v1.9: устарело — см. §9.5** (интерпретатор `/opt/miniconda3/bin/python3.12` + флаг `--append` + pre-flight).
5. Само-ревью артефакта (п.7).
6. Внешнее ревью (субагент): код + артефакт → `trackio_migration_REVIEW.md`.
7. Коммит (явные пути).

## 9. Аддендум v1.9 (16.11): завершение миграции — «все старые прогоны в trackio»

**Контекст.** Bulk-миграция (реестр 15.11) **ни разу не завершилась**: попытка 15.11 05:37 сбилась на
quiescence (live writer `at0_2_r6`, pid 1198497 — на 16.11 мёртв). За это время: reorg логов
(results/→logs/, 16.11), переименование AT0_8→AT8_1 (16.11), новые прогоны AT0_5 (финал), AT0_6 (×3),
AT0_7, AT2, AT3 (r1/r3), AT8_1 (T4 + v2), и **live-trackio** (AGENTS: обязателен для CPU-прогонов)
написал 10 записей в **два** DB: репо `.trackio/prism.db` (5 run'ов: at0_4_rebuild_r6/r6b,
at05_gencheck_r1, at05_selftest_tmp2, at07_loop_r1) и `~/.cache/huggingface/trackio/prism.db`
(5 run'ов: at05_selftest_tmp, at0_9_t21_r1, at3_margin_r1/r2/r3 — TRACKIO_DIR не был задан).
Задача: загнать ВСЕ прогоны (115 из снапшота + новые) в канонический DB репо, live-записи не терять.

### 9.1. Найденные дефекты (dry-run 16.11, `/opt/miniconda3/bin/python3.12`)

- **N1 (MAJOR, латентный): `is_json_log` не распознаёт формат Kaggle-лога.** Формат: 1-я строка
  `[{…}`, средние `,{…}`, последняя `]` (NDJSON-mассив). `is_json_log` ждёт, что первая непустая
  строка начинается с `{` → ВСЕ T4-логи классифицируются как текст → 0 метрик → счётчик-чек
  (`TZ_EXPECTED_HITS`) падает. Проверено: `is_json_log(logs/prism-exp17-lora-distill.log) = False`,
  но `parse_json_log` даёт 622 события (парсинг-тело работает). 15.11-прогон не дошёл до счётчиков
  (quiescence), поэтому баг выжил 4 ревью-раунда. **Фикс:** принимать первую строку, начинающуюся с `[`
  (`lstrip("[").lstrip(",")` + `rstrip("]").rstrip(",")`). **Верификация фикса:** 116 старых run'ов с
  исправленным детектором воспроизводят ВСЕ 25 счётчиков v1.8 (bridge 1549, kl 1551, dose 1011, …)
  ровно → логи не менялись с снапшота, экстракторы стабильны. (INFO v1.9-r1: детектор —
  over-approximation: xet-логи `[{…}` тоже «JSON»; безвредно — применяется только к ранам
  реестра, xet-логи исключены из реестра.)
- **N2 (MAJOR, латентный): `EMPTY_METRICS_EXPECTED` (v1.8, 12 имён) никогда не проверялся полным
  прогоном и некорректен.** Текст-логи (все CPU/IGPU) по коду НЕ экстрактятся → `empty_metrics=True`
  у всех них. Факт (dry-run 16.11, все 129 run'ов): **101 пустых** (12 старых T4/table + 89 текст-run'ов
  включая 13 новых; e93_sycl_bench НЕ пуст — 6 bench-точек). Полный список — в `EMPTY_METRICS_EXPECTED`
  v1.9 (происхождение: dry-run 16.11; список в ревью-файле для сверки).
- **N3 (инфо): `at8_1` (новый T4-лог) не даёт эксперимент-метрик** (формат выводов «S7: sha…», «u_…»
  не покрывается паттернами §4) → `empty_metrics=True`, `run_failed=False`, `duration_marker` = 841 s
  (`[OK]`-маркер, совпадает с wall 841.0 s из AT8_1-результатов).

### 9.2. Реестр v1.9: 129 run'ов / 145 .log = 127 уник. регистровых + 5 md5-дублей + 13 исключённых

Новые run'ы (13), добавляются в `CPU_RUNS`/`T4_RUNS`:

| name | log | device | note |
|---|---|---|---|
| at0_5 | `logs/AT0/AT0_5_run.log` (путь исправлен; v1.8: `logs/at0_5_run.log`) | CPU | финальный прогон (tracked) |
| at0_5_g8fail | `logs/at0_5_run.log` (старый файл, untracked) | CPU | ранний прогон: G8 negative control FAIL, traceback; остался после reorg |
| at0_6 | `logs/AT0/AT0_6_run.log` | CPU | финальный (claim 41c5e061) |
| at0_6_v2_2 | `logs/AT0/archive/v2.2/AT0_6_run.log` | CPU | архив v2.2 (claim 655594d0) |
| at0_6_v2r1_partial | `logs/AT0/archive/AT0_6_v2r1_partial_run.log` | CPU | partial (обрыв; claim b33e84bc) |
| at0_7 | `logs/AT0/AT0_7_run.log` | CPU | финальный 16.11; ранние 15.11 (×3) без выживших логов — их live-записи `at07_loop_r1` в DB |
| at0_4_rebuild_r6b | `logs/AT0/AT0_4_rebuild_r6b.log` | CPU | post-process r6.1; live-run того же имени в DB (допуск дуплика) |
| at2 | `logs/AT2_run.log` | CPU | T2.1 (бывш. AT0_9); live-run `at0_9_t21_r1` в HF-cache DB |
| at3_r1 | `logs/AT3_run.log` | CPU | T3.1 r1 (wall=270.0s в логе; duration-маркер-паттернов нет) |
| at3_r4 | `results/AT3_run.log` | CPU | T3.1 r4 (wall=479.8s, 16.11 18:19); r4 переписал r3-лог (один файл, r3 не выжил) |
| at8_1_v2 | `logs/AT8/AT8_1_v2_run.log` | CPU | машина v2, disk-only (wall 0.0s) |
| at2b_smoke | `results/AT2b_smoke/AT2b_smoke_run.log` | CPU | smoke AT2b (T2.1b) 16.11 (smoke=True) |
| at2b | `results/AT2b/AT2b_run.log` | CPU | полный прогон AT2b (T2.1b) 16.11 (SUBSPACE_CONTROL_FAIL); untracked; файл перенесён в results/AT2b/ (16.11 18:2x) |
| at8_1 | `logs/AT8/prism-at8-1-threshold.log` | T4 | T4-прогон (kernel JSON-лог) |

Новое исключение (1): `logs/AT8/AT8_1_run.log` — stdout-текст того же T4-прогона at8_1
(канон — kernel JSON-лог; md5 различаются, один прогон).

**Счётчики `TZ_EXPECTED_HITS`: без изменений** (новые run'ы дают 0 raw hits — dry-run: текст-логи
не экстрактятся, at8_1-JSON не покрывается паттернами §4). `RUN_FAILED_EXPECTED`: без изменений
(10; at8_1 — failure-маркеров нет). `MD5_DUPES`: без изменений (5). `EMPTY_METRICS_EXPECTED`: 101 имя
(100 по dry-run + at2b; текст-логи → empty).

**Арифметика (заменено в `check_registry`):** find_count=145, sum_check=145, registry_runs=129.

**Допуск дупликаций имён:** trackio допускает несколько run'ов с одним `run_name`
(прецедент: `at07_loop_r1` ×5 в текущем DB). Миграционные run'ы с именами, совпадающими с live-
записями (at0_4_rebuild_r6b, at0_4_rebuild_r6, at05_gencheck_r1→at0_5, at07_loop_r1→at0_7) — это
«лог-версия» того же прогона с более бедными метриками; live-записи сохраняются (--append),
дупликат — документирован в note-полях. (опечатка v1.9-оригинала «dopпуск dуплика» — исправлено, v1.9-r1.)

### 9.3. `--append` (N4)

`--fresh` (удаление DB) для текущего состояния разрушит 5 live-записей в DB репо. Новый флаг:
- DB существует + `--append` → DB НЕ удаляется, миграционные run'ы добавляются в существующий DB;
  **идемпотентность:** run'ы, уже присутствующие в DB по (name+log), пропускаются (репорт:
  `skipped_existing_runs`) — перезапуск не дублирует записи;
- DB отсутствует + `--append` → поведение как fresh (создание);
- DB существует без флага → FATAL (без изменений).

### 9.4. Консолидация HF-cache DB (N5; v1.9-r1: источник — бэкап)

После append-прогона: backup `~/.cache/huggingface/trackio/prism.db` → `/tmp/recovered/`, затем
копирование строк 5 live-run'ов (at05_selftest_tmp, at0_9_t21_r1, at3_margin_r1/r2/r3; 9 configs —
at3_margin_r1 имеет 3) из **бэкапа** (`configs`, `metrics`, `system_metrics` по `run_id`;
существующие `run_id` в DB репо не трогать) в DB репо. Источник — бэкап (стабильный снапшот),
не живой HF-DB: живой писатель (процесс без `TRACKIO_DIR`) не может повлиять на merge (v1.9-r1
MINOR-фикс; ATTACH с обычным путём — uri-параметры в ATTACH не парсятся). В отчёт: число
скопированных строк по таблице. Цель — все live-записи в каноническом
DB проекта (TRACKIO_DIR по умолчанию = `ROOT/.trackio`).

### 9.5. Прогон (v1.9-r1: pre-flight + формула ожидаемого итога)

**Pre-flight (обязателен, MAJOR-фикс 16.11-r1):** инвентарь живой — параллельная сессия пишет
новые .log (16.11: at2b_smoke 17:50, AT2b full 18:03 — оба появились после dry-run). Замороженные
числа реестра — недолговечный контракт (AGENTS п.7). Порядок:
1. Quiescence: все .log (не только регистровые) старше 10 мин, живых писателей нет
   (проверка скриптом, как в §5.1 — `find ... -newermt '-10 minutes'` + /proc-scan).
   **Прецедент 16.11 (реализовано):** параллельная сессия во время pre-flight (а) перенесла
   `results/AT2b_run.log` → `results/AT2b/AT2b_run.log`, (б) r4-прогон переписал
   `results/AT3_run.log` (r3 → r4) — реестр обновлён (at2b-путь, at3_r4) до прогона.
2. Свежий `find` .log → классификация каждого нового (run/md5-дубль/исключён) →
   обновление реестра + арифметики `check_registry` + `EMPTY_METRICS_EXPECTED`.
3. Тест-прогон на копиях DB (`TRACKIO_DIR=/tmp/... TRACKIO_HF_DIR=/tmp/...`, реальные DB не
   трогать) → RC=0, 0 ошибок, HF-merge с реальными rows → только тогда реальный прогон.

**Прогон:** интерпретатор `/opt/miniconda3/bin/python3.12` (py 3.12.9 — тот же, что live-прогоны;
trackio 0.45.1 в user-site `~/.local/lib/python3.12/site-packages`).
`setsid nohup ./run_one.sh trackio_mig "/opt/miniconda3/bin/python3.12 experiments/trackio_migration.py --append" &`

**Ожидаемый итог (формула, не замороженные числа; v1.9-r1):** RC=0; в DB репо:
`N0` (существующие configs: live-записи) + 129 миграционных + 9 HF-консолидированных = N0+138 configs;
уникальных имён ≈ 133 + (новые имена среди N0/9). Известные дупликатные имена: `at07_loop_r1` (live,
×5), `at0_4_rebuild_r6` (live + миграционный = ×2), `at0_4_rebuild_r6b` (live + миграционный = ×2) —
допущены (прецедент: trackio допускает несколько run'ов с одним run_name). Отчёт: 0 ошибок,
`rc: 0` (поле rc в отчёте — v1.9-r1: выставляется по errors; раньше оставалось 0 всегда),
warnings — только dev-маркера (NbConvert-хвост, ТЗ §3.1). Идемпотентность: повторный `--append`
пропускает run'ы уже в DB по (name+log) — `skipped_existing_runs` в отчёте.

**Аудит-артёфакт:** отчёт `results/trackio_migration_report.json` — в git (исключение в .gitignore,
v1.9-r1 INFO-фикс); 15.11-отчёт — бэкап `/tmp/recovered/trackio_migration_report_1511.json.bak`.

