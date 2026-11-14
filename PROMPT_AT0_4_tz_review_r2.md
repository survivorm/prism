# PROMPT: КЛЮЧЕВОЕ TZ-ревью AT0_4 (r2, closure) — prism-reviewer, file-protocol

Ты — внешний ревьюер. Задача: **closure-проверка** — 4 ПРАВКИ из r1-ревью
(docs/AT0_4_TZ_REVIEW.md, VERDICT=PASS_WITH_MINORS) должны быть закрыты в коде и TZ.
Это КРАТКОЕ ревью: 1 строка на пункт. НЕ пересматривать весь TZ (r1 уже PASS).

## Протокол
1. Пиши ТОЛЬКО в `/home/survivor/work/prism/docs/AT0_4_TZ_REVIEW_r2.md`,
   **перезаписывая файл после каждого раздела** (write, полный файл).
2. Финальная строка файла — ровно: `AT0_4_TZ_R2_DONE`
3. Бюджет: ≤ 15 tool calls. Ближе к лимиту — допиши VERDICT + маркер и стоп.
4. Чужие файлы не трогать.

## Читать (только это)
- `docs/AT0_4_TZ_REVIEW.md` (r1: раздел НОВЫЕ — 4 ПРАВКИ)
- `experiments/AT0_4_mix.py` (актуальный код)
- `docs/AT0_4_prereg.md` (§4.1 S3, §4.2 G_nd4/degenerate, §5 P1/P2, §12)

## Пункты (closed/open + 1 строка где)
1. S3: finiteness по ВСЕМ 3 сидам (было: только 11) — код + TZ §4.1.
2. c0=0 (degенеративный Δu): catch в run_cell → метка degenerate (ok=False),
   инстанс not_testable по правилу «все доки»; P1–P5 — медиана по оставшимся.
   (Было: неловленный ValueError → краш.)
3. G_nd4: жёсткий порог — median ‖h‖ > 100×R0×real ⇒ «divergent» ⇒ P1–P4 not_testable
   (было: только отчёт).
4. prereg_text = "docs/AT0_4_prereg.md v3" (было v1) — уже должно быть закрыто.

Доп. (не блокирует): TZ §12.7 — «перестановочный нуль Δu для 0.1» не принят с
обоснованием — ок ли обоснование? TZ P1 — a priori vacuous (r_max≈2/N) объявлен?

## Формат (строго)
```
# AT0_4_TZ_REVIEW_r2
Ревьюер: prism-reviewer (closure r2)
1: closed/open — <где>
2: closed/open — <где>
3: closed/open — <где>
4: closed/open — <где>
Доп: <1 строка>
НОВЫЕ: <≤2, или «нет»>
VERDICT=PASS|REVISE
AT0_4_TZ_R2_DONE
```
VERDICT=PASS только если все 4 closed; REVISE — если хоть одна open (тогда указать где).
