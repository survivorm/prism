# Ревью коммита 8e43bf6 (батч 2 путей + реестр + t23 + A1246)

sha256 объектов (рабочее дерево; для всех рецензируемых путей `git diff 8e43bf6 HEAD` пуст, локальных правок нет):
```
510703fbc8fc7bafcdacd93412aee424a3ea1ad2561b9851ed95e44169590556  artifacts/registry.json
e30b84cba7260ed0084bcbb480aaee39cd7531c6dd3b9c3f74b796177c6c51b2  tools/artifacts.py
8136616f18bd55bc72a43321a045825abd00bdc866a28f809905bce7c25c08b3  docs/dynamic/RERUN_SCRIPT_TRIAGE.md
c0b5b940c5f87ea395ceaf4742d838aafd9cc8b7f364473d4d62fac9a398f125  docs/ARTIFACTS_STORAGE.md
0628333b7dcdc9c005c573bd433d917a820716d5332927c8fa47f03f62e8081c  experiments/e61_70/exp63_t23_ngram.py
d673c587d19ac0441fd1c3da00dd1b49f8b5729a3260d7e0e244f2ce6a2d81e2  experiments/e61_70/exp64_A1246.py
```

## R1. t23 (exp63_t23_ngram.py) — PASS
(а) Точный фрагмент `prompt_ids` извлечён `ast.get_source_segment` из файла и исполнен (не копия),
с `tok=AutoTokenizer(/data/Qwen/Qwen3.5-0.8B)`, cwd=experiments/ (как делает сам файл), домены из
`gen_layers_qwen_meta.json` (4 шт, MAX_SI=15 у всех):
```
chat: pins_used=chat___data_Qwen_Qwen3.5_0.8B__n12__min192__max512__plain.json n_texts=12
code: pins_used=code___data_Qwen_Qwen3.5_0.8B__n12__min192__max512__plain.json n_texts=12
math: pins_used=math___data_Qwen_Qwen3.5_0.8B__n12__min192__max512__plain.json n_texts=12
wiki: pins_used=wiki___data_Qwen_Qwen3.5_0.8B__n16__min64__max256__plain.json  n_texts=16
```
Ни для одного домена `n16__min384__max512` не выбран (его нет — подтверждено выбором fallback'а);
все 4 домена получили существующий файл, длины текстов ≤256.
(б) Точные строки цикла (`ast.get_source_segment` сегменты `si`, `n_pins`, `PIN_CYCLE`, `pr`)
исполнены на `nm='chat_15'`, `cache['chat']` = реальные 12 pin-текстов:
```
si = 15  n_pins = 12  PIN_CYCLE = {'chat': True}
pr = list(cache['chat'][15 % 12]) = cache['chat'][3]  → IndexError нет, первые id [1847, 186301, 11]
```
(в) `pins_used`/`pin_cycle` в артефакте: строки 90–91 файла —
`print(json.dumps({"pins_used": PIN_USED, "pin_cycle": PIN_CYCLE, "res": out}, ...))` и
`json.dump({... "pins_used": PIN_USED, "pin_cycle": PIN_CYCLE, ...}, open("../results/e63/t23_ngram.json","w"))`. PASS.

## R2. A1246 и класс «непарсящихся файлов» — PASS с уточнением
- `git show 8e43bf6^:experiments/e61_70/exp64_A1246.py` → compile: `SyntaxError line 113:
  closing parenthesis ']' does not match opening parenthesis '('` — баг был ДО коммита.
- `py_compile` нового `exp64_A1246.py`: OK.
- Полный `compile()` по дереву `experiments/` (343 .py-файла) на `8e43bf6^` и `8e43bf6`:
```
не парсятся в 8e43bf6^: exp64_A1246.py:113, e71_80/exp78_reanalysis.py:54
не парсятся в 8e43bf6:  e71_80/exp78_reanalysis.py:54
регресс: [] ; починено коммитом: exp64_A1246.py
```
- Уточнение к ожиданию «ровно 2 битых»: **в дереве коммита ровно 1** (exp78). Второй ожидаемый,
`experiments/at8/AT8_2b.py`, — **untracked** (`git status` → `??`, blob'а нет ни в 8e43bf6, ни в 8e43bf6^),
в рабочем дереве не парсится: `line 969: closing parenthesis ']' does not match opening '('`.
Оба битых файла коммитом не тронуты → «не из этого коммита» верного и для AT8_2b; «тоже битые в HEAD~1»
для него непроверяемо через git (файла там нет). Третьих нет ни в дереве, ни в worktree (ровно 2: AT8_2b, exp78). PASS.

## R3. Батч 2 (24 файла с os.chdir) — PASS
Свой ast-скрипт по blob'ам `8e43bf6:` всех 24 изменённых `exp6*`-файлов:
- (а) `_HERE` определён раньше `os.chdir` — 24/24; (б) `chdir` раньше первого `'../'`-литерала — 24/24;
- (в) `sys.path.insert(0,'.')`/`(".")` не осталось (заменён на `_HERE`, напр. exp62_layers.py:11);
- (г) cwd-зависимых root-relative литералов нет. Единственное формальное совпадение шаблона —
  `exp64_mark.py:83` `"results/e64/e64c_markov.json"`, но он склеен через
  `os.path.join(dirname(dirname(abspath(__file__))), ...)` → абсолют, cwd-независим, падает в тот же
  `<repo>/results/e64/`; формально это НЕ root-relative относительный путь;
- (д) для каждого каталога записи (`open(...,'w')`, `np.save/savez`, `json.dump`) есть `makedirs` —
  реальных пропусков 0 (у `exp62_4b`/`exp62_t1` makedirs нет, но они только читают `../results/...`, записей нет);
- исполнение РЕАЛЬНОЙ вставленной шапки из /tmp с подменённым `__file__`:
```
exp64_at_battery.py (стр.7–9): cwd = /home/survivor/work/prism/experiments ; ../results/e64/x → /home/survivor/work/prism/results/e64/x
exp62_gen.py       (стр.10–11): cwd = /home/survivor/work/prism/experiments ; ../results/e62/x → /home/survivor/work/prism/results/e62/x
```
PASS.

## R4. Батарея L23 (единственный прогон) — PASS
Из /tmp, точная шапка `exp62_layers.py` (строки `_HERE…import run`, exec c `__file__`=файл), данные
`/data/prism_kernels/exp62/gen_layers_qwen.npz` (assert sha_ok прошёл), только L23, без json.dump:
```
L23: F1={'R2': 0.913, 'r1': 0.16, 'r2': 0.96} osc=45/64 clip=19 strict=17 lag1=-0.072 wall=55.1s
```
Совпадает с ожидаем `R2=0.913 osc=45/64 clip=19 strict=17 lag1=-0.072`; wall 55.1 с против ≈52 с (±6%). PASS.

## R5. Реестр — PASS
`artifacts/registry.json` (schema 2, hf_repo survivor-m/prism-artifacts, items — dict путь→запись):
```
записей: 50 ; сумма size 3 904 726 027 B = 3.905 ГБ(10^9)
без hf_path: 0 ; уникальных hf_path: 46 ; ключ==local_path: да ; дублей ключей: 0
записей results/e64/tpu*: 12 (tpu/ и tpu_v8/ разведены — schema 2 работал)
записей, отсутствующих на диске: 0
```
- `tools/artifacts.py verify` → `verify: 50 записей, проблем 0` (rc=0).
- Сеть (разрешено только здесь): `tools/artifacts.py verify --hf` →
  `hf: записей 50, уникальных hf_path 46, файлов в репо 47` / `verify --hf: проблем 0` (rc=0).
- `tools/artifacts.py scan --top 100` → `файлов больше 10.0 МБ: 50, суммарно 5634.1MB`,
  `из них НЕ в реестре: 1, суммарно 1729.3MB`, `1729.3MB  results/exp27_jcache.pt  [НЕ В РЕЕСТРЕ]`.
Замечание: под `results/e64/tpu*/` на диске ещё 25 мелких файлов без записей (крупнейший 2.6 КБ) —
в пределах политики >10 МБ, scan их не флагует. PASS.

## R6. Числа в доках против факта — 2 находки в тексте
| заявлено | команда | факт |
|---|---|---|
| 50 записей (ARTIFACTS:88) | python по registry.json | 50 ✓ |
| сумма 3.905 ГБ (ARTIFACTS:88) | сумма size | 3 904 726 027 B = 3.905e9 ✓ |
| 46/47 (ARTIFACTS:89) | verify --hf | «уникальных hf_path 46, файлов в репо 47» ✓ |
| 50 файлов >10 МБ (ARTIFACTS:83) | find results -size +10M | 50 ✓ |
| **суммарно 5.3 ГБ (ARTIFACTS:83)** | scan / find -printf | **5634.1 MB = 5.63 ГБ** ✗ (см. Найдено-1) |
| вне реестра ровно 1 (ARTIFACTS:91) | scan --top 100 | 1, exp27_jcache.pt ✓ |
| 12 файлов tpu* выгружено | registry tpu-записи | 12 ✓ |
| 24 файла батча 2 (RERUN:167,172) | git show + grep chdir | 24/24 ✓ |
| 29/29 компилируются (RERUN:172) | compile glob exp6[24]_* | 29 файлов, 29/29 ✓ |
| SyntaxError 113 в HEAD (RERUN:163) | compile blob 8e43bf6^ | line 113 ✓ |
| **§И «22 файла» (RERUN:151)** | git show / RERUN:167 | **24** ✗ (см. Найдено-2) |
Остатка, где 5.3 ГБ читается как объём реестра, нет: вхождения `5.3` — только строка 83 (объём всех >10МБ
файлов) и строки 94–96 (преднамеренное предупреждение «не путай, раньше стояло „0 расхождений на 5.3 ГБ“»).

## Найдено
1. **Ошибка в тексте**: `docs/ARTIFACTS_STORAGE.md:83` — «в `results/` **50** файлов >10 МБ, суммарно
   **5.3 ГБ**»; факт сегодня: 5634.1 МБ = **5.6** ГБ (тот же `scan`). Строка 94 сама признаёт разброс
   «5.3–5.6», но конкретная цифра в строке 83 устарела/противоречит выводому контролю. Плюс единицы
   в доке смешаны: 3.905 «ГБ» для реестра — десятичные, 5.3 «ГБ» — если GiB (=5.25 GiB), то это другой
   масштаб в соседних строках.
2. **Ошибка в тексте**: `docs/dynamic/RERUN_SCRIPT_TRIAGE.md:151` — заголовок §И «Батч 2 (E-62/64,
   **22 файла**)» против собственного текста §И (строки 167, 172: «24 файла», «у 24/24») и против
   фактического замера (24 файла с `os.chdir`).
3. **Хрупкое, но не сломано**: (а) ожидание «2 битых файла в дереве» верно только для рабочего дерева:
   `experiments/at8/AT8_2b.py` untracked (бит, line 969) — в git его нет, и контроль «не парсится и в
   HEAD~1» для него невыполним; (б) `exp64_mark.py:83` использует литерал `"results/e64/..."`, склеенный
   с абсолютом из `__file__` — cwd-безопасно, но формально совпадает с шаблоном «root-relative литерал»,
   на который сам же контроль и смотрит; (в) `exp62_4b.py:7` читает `../results/e52_4b/exp47b_gen_kinematics.json`,
   которого в репо нет (есть только `.npz` и каталог) — путь после chdir корректен, но файл всё равно упадёт
   на отсутствующем входе (дефект данных, не путей; не ветка этого коммита).
4. Раскладка реестра: 50 записей = 49 файлов results/>10МБ + 1 файл <10МБ
   (`results/e64/tpu/cor_p1.npz`); то есть «все 50 файлов >10 МБ в реестре» было бы неверно —
   доке такой проверки и не нужен (§91 формулирует корректно: не в реестре ровно jcache).

## Не проверено
- Что 46 уникальных hf_path физически скачаться из HF умеют (pull-путь) — `verify --hf` сверяет реестр с HF-стороной (скачивает ли он объекты для сверки — не замерялось); полная regenerability артефактов не тестировалась.
- Полные прогоны 24 файлов батча 2 (разрешён был один прогон — сделан R4); проверка ограничена
  парсингом, путями, шапками и батареей L23.
- `exp64_mark.py:82` мёртвая конструкция `... if False else ...` — предсуществующая, в диффе коммита
  не проверял (за рамками заявленных правок).

VERDICT: REVISE

BATCH2_CODE_REVIEW_DONE
