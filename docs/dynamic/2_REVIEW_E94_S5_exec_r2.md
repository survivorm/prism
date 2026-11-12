# Ревью E94/S5 r2 — проверка закрытия 5 MINOR из r1

ВЕРДИКТ: PASS
F1: закрыт. Стр. 378: `"flips_explained_by_dry": sum(1 for f in flips if f["bucket"] == "EXPLAINED")` — flip ∧ EXPLAINED, не TP конфьюжна. Артефакт r1: 16 при flips_total=16, buckets {EXPLAINED:6144} (UNEXPLAINED=0).
F2: закрыт. Стр. 491–492: `expected_eq_obs_flips: ctrl_mism == flips_set` — равенство НАБОРОВ «doc:pos» (flips_set из flips_detail, ctrl_mism = set(non_explained)); mismatch_positions = sorted(ctrl_mism). Артефакт: 16 позиций; точечно q1_w0:23, q1_w13:53, q1_w13:64 = первые 3 записи flips_detail; полное тождество множеств проверено (set_eq=True на r1 и r2), expected_eq_obs_flips=true.
F3: закрыт. Стр. 529–534: промпт-иды из request.prompt × brk, каждый сверен с серверным текстом sink_r1. Артефакт: prompt_brk_ids = [1,25,198,328,1149,1288,7285,9726,61442,87507] (10); verified=10, все affected=false → все 10 сверены; prompt_brk_ids_unverifiable_from_raw=[]; any_penalty_affected = any(impact) ИЛИ any(verified.affected) — промпт-токены учитываются, =false.
F4: закрыт. lines_cited (стр. 460/артефакт) = [95,3330,3393,3424,3459,3486,3545,3565,3645]: 3393 (не 3394), 3459 (не 3457). Сверено по исходнику при зафиксированных sha1 c9fa73d6…/HEAD 0b8cfb8fe (пересчитал — совпадает с артефактом): 3393 = `int last_n_repeat = std::min((int) ctx->last_tokens.size(), ctx->dry_penalty_last_n);`; 3459–3461 = `if (rep_limit < ctx->dry_allowed_length) { return; }`.
F5: закрыт. Стр. 569: `negative_controls(tf, r1_names, brk)` — все 16 r1-документов. Артефакт: scope={rep:r1, docs:16, positions:6144}, структура variants/scope в JSON; все три caught=true (rat_forward: +17 UNEXPL/+16 FN/+1 FP; penalize_head: +16/+16; no_breakers: +14 UNEXPL/+14 FP), суммы buckets по вариантам = 6144.
РЕГРЕССИЯ: против stdout r1 — EXPLAINED=6144 (обе репы), flips=16, g_need_max=0.994864 (r1; r2=0.9946 — независимый повтор, дрейф ожидаем), vocab mismatches=16, H0_supported=true, unit_test ok. Изменений сверх 5 предписанных правок не найдено: все новые поля соответствуют F1–F5 (non_explained, mismatch_positions+note, prompt_brk_*, расширенный any_penalty_affected, lines_cited, scope 3→16). Числа negative_controls отличаются от r1-3док — прямое следствие предписанного расширения F5.
FINDINGS:
- (вне чек-листа, не регрессия) В файле r1-ревью есть ещё 2 [MINOR], не вошедшие в F1–F5: явный TIE=0 в buckets (в артефакте по-прежнему отсутствует, verdict использует b.get-дефолт) и корпусная локальность vocab-чека (ids_checked=4772 из 248320). В раунд r2 не предписаны; оставлены открытыми.
- Формулировки: файл r1-ревью содержит 3 [MINOR], промпт r2 — 5; сверка выполнялась строго по чек-листу F1–F5 промпта r2.

S5_R2_DONE
