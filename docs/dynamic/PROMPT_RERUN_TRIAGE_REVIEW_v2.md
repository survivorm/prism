# Промт внешнего ревью: docs/dynamic/RERUN_SCRIPT_TRIAGE.md (v2, машинно-проверяемый)

Ты — **ревьюер, read-only**. Твоя задача: проверить 7 утверждений файла-объекта **командами**, а не рассуждением.
Ты НЕ правишь объект, НЕ правишь код, НЕ запускаешь эксперименты, НЕ качаешь модели из сети (модель
`Qwen/Qwen3.5-0.8B` лежит локально). Разрешено: `read`, `bash` (только перечисленные ниже команды и их прямые
варианты), `write` (ровно один файл — итог ревью).

**Жёсткий лимит: 10 минут, ≤ 12 вызовов bash.** Если команда падает — записываешь факт падения и идёшь дальше,
не «исследуешь причину». Никаких `find /`, никаких обходов истории git, никаких новых гипотез.
Сомнение → строка в разделе «не проверено», а не новый эксперимент.

Объект ревью: `docs/dynamic/RERUN_SCRIPT_TRIAGE.md`
sha256 (сверь его первой командой, если не совпал — STOP, напиши `VERDICT: FAIL (sha mismatch)` и маркер):

    a8600e7f844b57b59db2b65489e9db837ab17ff744a2886d4c975e19ccb37279

## Ровно 7 проверок (по одной bash-команде на каждую, из корня репо /home/survivor/work/prism)

**C1. E-75: маска шапки шаблона.** Триаж утверждает: реальный префикс шаблона = 17 символов = 3 токена,
`len(wrap(""))` = 69 символов (в «шапку» попадало до 52 символов тела), и в артефакте
`results/e75/TMPL_FAMILY.json` записаны `P` ∈ {12,13,14,15,16,17,18,19,20,24}, а не 3.

    .venv/bin/python - <<'EOF'
from transformers import AutoTokenizer
import json
tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B",trust_remote_code=True)
w=lambda t: tk.apply_chat_template([{"role":"user","content":t}],add_generation_prompt=True,tokenize=False)
h0=w(""); full=w(" Beginners BBQ Class")
n=lambda s: len(tk(s,add_special_tokens=False)["input_ids"])
print("len_wrap_empty",len(h0),"true_prefix",len(full)-len(" Beginners BBQ Class")+0)
print("P_true",n(full[:full.find("Beginners BBQ Class")]),"P_old_style",n(full[:len(h0)]))
r=json.load(open("results/e75/TMPL_FAMILY.json"))
print("artifact_P",sorted({v.get("P") for v in r["qwen/tmpl"].values()}))
EOF

PASS, если: `len_wrap_empty`=69, `P_true`=3, `P_old_style`>3, `artifact_P` в диапазоне 12..24.
В отчёте процитируй 4 числа.

**C2. `load_pin_texts` отсутствует.** Триаж: `exp78c_own_surplus.py` и `exp78_reanalysis.py` делают
`exec` сегмента `load_pin_texts`, которого нет нигде в репо; плюс они указывают на несуществующий
`experiments/exp78_surplus.py` (вместо `experiments/e71_80/exp78_surplus.py`).

    grep -rnF "load_pin_texts" experiments/ | head -5; echo "defs:"; grep -rnF "def load_pin_texts" experiments/ | wc -l; echo "path:"; grep -nF "experiments/exp78_surplus.py" experiments/e71_80/exp78c_own_surplus.py experiments/e71_80/exp78_reanalysis.py

PASS: `defs:` = 0, и оба файла содержат путь без `e71_80/`.

**C3. `exp78_reanalysis.py` не парсится.** Триаж: SyntaxError на строке 54.

    .venv/bin/python -c "import ast; ast.parse(open('experiments/e71_80/exp78_reanalysis.py').read())" 2>&1 | tail -3

PASS: вывод содержит `SyntaxError` и строку 54.

**C4. `DynamicCache` мутирует на месте (transformers 5.15.1).**

    .venv/bin/python - <<'EOF'
import inspect
from transformers.cache_utils import DynamicCache
src=inspect.getsource(DynamicCache.update)
print(src[:700])
EOF

PASS: в исходнике видно дописывание в существующий `self.key_cache[layer_idx]` (append/cat) при `layer_idx < len(self.key_cache)`, т.е. состояние копится, а не возвращается новый объект.

**C5. Артефактов нет (проверка утверждения триажа «прогоны 65–91 не оставили артефактов»).**

    for n in 65 66 67 68 70 72 73 77 78 79 80 91; do echo -n "e$n: "; ls results/e$n 2>/dev/null | tr '\n' ' '; echo; done; echo -n "e69c: "; ls results/e69c; echo -n "e75: "; ls results/e75; echo -n "e81: "; ls results/e81

PASS, если папки e65..e80/e91 пусты или отсутствуют, а e69c, e75, e81 непусты (перечисли их содержимое в отчёте).

**C6. Системное ломание путей после reorg.** Триаж: `../results`-литералы в `experiments/e*/*.py` — 0 из 80
резолвятся; 21 файл считает `ROOT` как `parent.parent`; 0 файлов используют `parents[2]`.

    echo -n "dre_results_literals: "; grep -rlF "../results" experiments/e*/*.py | wc -l; echo -n "grep_hits: "; grep -roF "../results" experiments/e*/*.py | wc -l; echo -n "parent_parent: "; grep -rlF "parent.parent" experiments/e*/*.py | wc -l; echo -n "parents2: "; grep -rlF "parents[2]" experiments/e*/*.py | wc -l; ls -d results >/dev/null 2>&1 && echo "results_exists"; ls -d /home/survivor/work/results 2>/dev/null || echo "no_sibling_results"

PASS: `parent_parent` ≥ 20, `parents2` = 0, `no_sibling_results` присутствует, `results_exists` присутствует.
(Точное «0 из 80» проверять не надо — это утверждение выше трудоёмко; отметь его в разделе «не проверено», если
числа `grep_hits` не помогают.)

**C7. E-67 (quantolike) — вырожденность.** Триаж: CHSH-тест T2 вырожден по построению (все 4 обсерваблы —
детерминированные функции одной пары `(h_t, h_{t+1})`, значит |S| = 2 всегда), и T1 `g2 = 1 + Cov_1` — это
переименованный dip. Проверь по коду.

    grep -nE "S *=|corr|sign|CHSH|g2" experiments/e61_70/exp67_quantolike.py | head -25

PASS, если из кода видно, что обсерваблы строятся как функции одной пары состояний (и `g2` определяется через ту же
ковариацию). Цитируй 3–6 строк кода.

## Что писать в отчёт

Файл: `docs/dynamic/RERUN_TRIAGE_REVIEW.md`. Структура — строго:

```
# Ревью: RERUN_SCRIPT_TRIAGE.md
SHA256_ОБЪЕКТА: <фактический>
## C1 … C7
по одному блоку на проверку: PASS|FAIL|НЕ ПРОВЕРЕНО + одна строка с фактическим выводом команды (числа/цитата)
## Не проверено
список утверждений триажа, которые не вошли в 7 проверок
## Согласованность классов
1–3 строки: противоречит ли какой-нибудь вердикт класса (А/Б/В/Г/Д) данным проверок C1–C7
## Замечания
нумерованный список; каждое замечание обязано быть либо (а) расхождением числа/факта, либо (б) ссылкой на строку файла
VERDICT: PASS|REVISE|FAIL
RERUN_TRIAGE_REVIEW_DONE
```

Правила вердикта: любое число в триаже, не подтверждённое командой → `REVISE` с указанием строки.
Только стилистика → `PASS` с замечаниями. Объект нечитаем/не соответствует sha → `FAIL`.

Последняя строка файла — ровно `RERUN_TRIAGE_REVIEW_DONE`. После записи файла выведи в чат одну строку
`RERUN_TRIAGE_REVIEW_DONE` и остановись.
