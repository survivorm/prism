# Ревью коммита 9965ce5 (батч 1: пути, артефакты, триаж)

Объект: `9965ce5c7a241d277c8e6ec20019454b68075937` ("Политика хранения артефактов + триаж перезапусков +
правки путей батча 1"), 30 файлов. Ревью: 20.11, read/bash/write, без экспериментов кроме одной разрешённой
проверки (R3). Рабочее дерево по всем четырём объектам = коммит (`git diff --stat 9965ce5 -- <4 пути>` пуст).

sha256 объектов (`sha256sum`, рабочее дерево = 9965ce5):

```
cf37df664bf0be0ef711a43e0fa07e0b827ff8f428eaf7e70676ad13d8f09f40  tools/artifacts.py
3403df2096dbe0a72cb3115d5fd145c7477e06d34ae14fa867ce0f64864553cc  artifacts/registry.json
e82f10803b152f3753e7d82c5549e335a935d538e8529660da076c24211a647f  docs/dynamic/RERUN_SCRIPT_TRIAGE.md
1d30012e51923609b3e45cbc2095ffd7a42f8d18cd6da8369c4fef01a29bb0be  docs/dynamic/RERUN_TRIAGE_REVIEW.md
```

Дополнительно (провенанс самого триажа): `docs/dynamic/RERUN_CODE_AUDIT.md` =
`f9faf2a32cb99e6455be18701472ec625aaae87ae26c9c19cbed0ce4954e763a` и `docs/dynamic/RERUN_SELFREV_REVIEW.md` =
`32463cfde06da024baa3113bd5ee181496c30724aedb5ab4f634d1cd5b4758c9` — оба совпадают со значениями, напечатанными
в `RERUN_TRIAGE_REVIEW.md` (строки 13 и 25). Текущий `RERUN_SCRIPT_TRIAGE.md` = `e82f1080…`, а в строке 24 того
же файла напечатан `1263acbe…` — расхождение объяснено в строках 18-19 (ревизия, которую проверяло внешнее
ревью, vs ревизия после правки C7 и добавления §З) и совпадает с фактическим sha256 ✓.

---

## R1. chdir-стратегия не ломает остальные пути — PASS (с одной дырой входов, см. находку 1)

Команда: `ast`-обход всех 10 файлов (разбор `Constant`/`JoinedStr` в аргументах
`open/load/dump/save/savez/makedirs/exists/glob/safe_open/from_pretrained` + все строковые литералы с `/`),
резолв относительно `cwd = <repo>/experiments` (то, что даёт вставленный `os.chdir`).

```
### exp62_layers.py   ABS: /data/prism_kernels/exp62/gen_layers_qwen.npz
   REL '../results/e62/layers_qwen.json'  -> <repo>/results/e62/layers_qwen.json   exists=False (пишет сам)
### exp63_d0.py
   REL '../results/e62/layers_qwen.json'      -> <repo>/results/e62/...        exists=False (создаёт exp62_layers)
   REL '../results/e62/layers_qwen_L23.part'  -> <repo>/results/e62/...        exists=False (fallback)
   REL '../results/e63/d0_strict_at_L23omega.json' -> <repo>/results/e63/...   (пишет)
### exp63_d1_tangents.py   REL '../results/e63' (makedirs), '../results/e63/d1_tangent.json'
### exp63_d2_null.py       REL '../results/e63' (makedirs), '../results/e63/d2_nulls.json'
### exp63_t17_dipnull.py   REL '../results/exp47b_gen_kinematics.json' -> exists=True
                           REL '../results/e63/t17_dipnull.json'
### exp63_t1_tokrhythm.py  REL '../results/e63' (makedirs), '../results/e63/t1_tokrhythm.json'
### exp63_t23_ngram.py     REL '../data/calib_pins/'  -> <repo>/data/calib_pins  exists=True
                           REL '../results/e63/t23_ngram.json'
### exp63_t25_ekurve.py    REL '../results/e63/t25_ekurve.json'
### exp63_t5_audit.py      REL '../results/e63/t5_audit.json'
### exp63_t5_scramble.py   REL '../results/e63/t5_scramble.json'
```

- Относительных литералов вида `"results/..."` **без** `../` в этих 10 файлах нет ни одного (ловушки
  `experiments/results/...` не возникает): единственный не-`../` литерал с косой чертой — f-строка
  `f'{MK}/model.safetensors-...'` в `exp63_t1_tokrhythm.py:21`, где `MK` — абсолютный
  `/data/Qwen/Qwen3.5-0.8B`, и файл на месте (`ls` показал
  `model.safetensors-00001-of-00001.safetensors`).
- Абсолютные пути (`/data/prism_kernels/exp62/gen_layers_qwen.npz`, `/data/prism_kernels/exp47b/...`,
  `/data/Qwen/Qwen3.5-0.8B`) от chdir не зависят — проверено в listing'е выше, все `ABS:` строки начинаются
  с `/data/`.
- Все перечисленные выше относительные пути резолвятся под `<repo>/results/...` или `<repo>/data/calib_pins`
  — то есть ровно туда, куда задумано. Единственный вход, которого нет по этому пути: `../data/calib_pins/*n16__min384__max512*`
  (см. «Найдено» №1) и отсутствующий вход E-62 (это и есть предмет триажа, не регресс правки).
- `sys.path`-ловушек нет: локальные импорты (`from exp62_batt_lib import ...`) есть только в
  `exp62_layers, exp63_d0, d1, d2, t25, t5_audit, t5_scramble` — везде добавлен `sys.path.insert(0, _HERE)`;
  в `t17_dipnull, t1_tokrhythm, t23_ngram` локальных импортов нет вообще
  (`grep -nE "^(from|import) (exp|tools|prism)"` — пусто), поэтому отсутствие `sys.path.insert` там безопасно.

## R2. makedirs в правильном месте и в правильном объёме — PASS

Команда 1: AST + позиционная сверка `_HERE =` vs `os.makedirs` в каждом файле.

```
exp62_layers.py        _HERE:True chdir:True makedirs:True syspath:True  order: _HERE L9  < makedirs L11
exp63_d0.py            ... order: _HERE L10 < makedirs L12
exp63_d1_tangents.py   ... order: _HERE L10 < makedirs L60 (предсуществующий, относительный)
exp63_d2_null.py       ... order: _HERE L14 < makedirs L85 (предсуществующий)
exp63_t17_dipnull.py   ... order: _HERE L9  < makedirs L11   syspath:False (локальных импортов нет)
exp63_t1_tokrhythm.py  ... order: _HERE L6  < makedirs L79 (предсуществующий) syspath:False
exp63_t23_ngram.py     ... order: _HERE L8  < makedirs L10   syspath:False
exp63_t25_ekurve.py    ... order: _HERE L10 < makedirs L12
exp63_t5_audit.py      ... order: _HERE L12 < makedirs L14
exp63_t5_scramble.py   ... order: _HERE L14 < makedirs L16
```

`_HERE` во всех десяти = `os.path.dirname(os.path.abspath(__file__))`, то есть `experiments/e61_70`
(подтверждено R3: `sys.path[0] = /home/survivor/work/prism/experiments/e61_70`). Файлов, где `makedirs`
вставлен без `_HERE`, нет: в шести новых вставках аргумент — `os.path.join(_HERE, "..", "..", "results", "e63")`,
то есть `NameError` невозможен.

Команда 2 (объём): `makedirs` вставлен ровно в 6 файлах (`exp63_{d0,t17_dipnull,t23_ngram,t25_ekurve,t5_audit,
t5_scramble}.py`) + `exp62_layers.py` (для `results/e62`) — как заявлено («6 из 9»). Оставшиеся 3 writer'а
(`d1_tangents:60`, `d2_null:85`, `t1_tokrhythm:79`) уже имели инлайновый `os.makedirs("../results/e63")` перед
своим `json.dump` — проверено `sed`, поэтому покрытие «все 9 writer'ов E-63 создают каталог» верно.

Команда 3: `.venv/bin/python -m py_compile <все 10>` → `COMPILE_OK` (падает только в `__pycache__`, gitignored).

## R3. Дешёвая проверка исполнением — PASS

(a) Шапка `exp62_layers.py` (строки 1-13, `exec(compile(...))` из `/tmp`, `__file__` = реальный путь):

```
before: results/e62 exists = True  cwd= /tmp
cwd after   = /home/survivor/work/prism/experiments
sys.path[0] = /home/survivor/work/prism/experiments/e61_70
e62 dir exists = True  (created now: False)          # каталог уже был — пустой
abspath rel    = /home/survivor/work/prism/results/e62/layers_qwen.json
import OK      = /home/survivor/work/prism/experiments/e61_70/exp62_batt_lib.py
```

Так как `results/e62` уже существовал (пустой), «создаётся» проверено отдельно: копия файла в песочницу
`/tmp/sb/experiments/e61_70/` + `exec` тех же строк 1-11 → `created = True`, `results = ['e62']`.

(b) Одна разрешённая батарея на L23, `run()` из `exp62_batt_lib`, данные
`/data/prism_kernels/exp62/gen_layers_qwen.npz`, 64 дока, артефактов не писал:

```
npz load 0.0 s ; docs=64
L23: F1={'R2': 0.913, 'r1': 0.16, 'r2': 0.96} osc=45/64 clip=19 strict=17 lag1=-0.072 | run wall 52.5 s
real 0m52.713s
```

Сверка с §З п.4 (`F1 R2=0.913`, `osc=45/64`, `strict=17`, `lag1=-0.072`, 52.2 с) — все четыре числа и
время совпадают (52.5 с против 52.2 с, ~0.6 %). Контроль чистоты: `ls results/e62/` — пусто,
`results/e63` по-прежнему не существует.

## R4. exp63_d0 совместим с layers_qwen.json — PASS

По коду `experiments/e61_70/exp62_batt_lib.py:145-146`:

```python
        phase={k: (None if v is None else (v["omega"], v["Rbar"], v["Rbar_strict"],
                                           v["n_strict"], v["strict_names"]))
```

то есть `phase[dom]` — кортеж из 5, индекс 4 = `strict_names`. Живой замер (см. R3b):

```
phase keys: ['chat', 'code', 'math', 'wiki']
phase['chat']=(0.5, 0.79, 0.97, 4, ['chat_1', 'chat_2', 'chat_10', 'chat_11'])  len=5 type(v[4])=list
```

`exp62_layers.py:24-31` кладёт `{k: r[k] for k in (... "phase" ...)}` в `res[f"L{L}"]` при `L = 23`
и пишет `../results/e62/layers_qwen.json` → ключ `"L23"` есть, `["L23"]["phase"]` = то, что читает d0
(после JSON-раунда кортеж становится списком, `v[4]` не меняется).

Исполнение **точного** фрагмента (строки 23-29 `exp63_d0.py`, извлечены `splitlines(True)[22:29]`, не копия):

```
--- exact fragment (lines 23-29 of exp63_d0.py) ---
_P = "../results/e62/layers_qwen.json"
if not os.path.exists(_P): _P = "../results/e62/layers_qwen_L23.part"
if not os.path.exists(_P): raise SystemExit(...)
S23 = json.load(open(_P))["L23"]["phase"]
STRICT = {dm: v[4] for dm, v in S23.items() if v}

SystemExit: нет входа: ../results/e62/layers_qwen_L23.part — сначала exp62_layers.py (он пишет layers_qwen.json)
json-ветка OK  STRICT= {'chat': ['chat_1', 'chat_2']}
fallback .part OK STRICT= {'wiki': ['w_1']}
```

Все три ветки (нет обоих → `SystemExit`; есть json → читает; есть только `.part` → fallback) работают.
Заявление «старого имени не производит ни один скрипт» подтверждено: `grep -rn "layers_qwen_L23.part"` даёт
только доки + саму строку 25 `exp63_d0.py`, ни один `.py` его не пишет.

## R5. pull после фикса — PASS (число «5.3 ГБ» к verify не относится, см. R6)

`git show 9965ce5 -- tools/artifacts.py` → **файл создан этим коммитом целиком** (`1 file changed, 223
insertions(+)`, `@@ -0,0 +1,223 @@`). Новый `cmd_pull` (строки 176-205):

```python
    got = Path(hf_hub_download(repo_id=reg["hf_repo"], filename=it["hf_path"], repo_type="dataset",
                               revision=it.get("hf_revision") or "main"))
    # copy2, а не move/replace: ... shutil.move падал с FileExistsError.
    shutil.copy2(got, dst)
    try:
        got.unlink()   # убираем только ссылку из снапшота; блоб в blobs/ остаётся
    except OSError:
        pass
    ok = sha256_16(dst) == it["sha256_16"]
    ...
    return 0 if ok else 1
```

`grep -nE "replace|shutil|copy2|unlink" tools/artifacts.py` → только `import ... shutil ...`, `shutil.copy2`,
`got.unlink()` (внутри `try/except OSError`) и два упоминания в комментарии; `Path.replace` и `shutil.move`
в файле отсутствуют. `copy2` перезаписывает существующую цель, а не падает; `unlink` защищён; несовпадение
sha — предупреждение + `return 1`, процесс не роняется. `py_compile` → `COMPILE_OK`.

`python tools/artifacts.py verify`:

```
verify: 37 записей, проблем 0
real 0m6.231s
```

37 записей и 0 расхождений — воспроизводится (6.2 с на полном чтении: файлы в page cache). Сам pull
не исполнялся: требует сети, которая в промте запрещена (см. «Не проверено»).

## R6. Числа триажа — ЧАСТИЧНО: объём «5.3 ГБ» реестру не принадлежит

```
items= 37  sum bytes= 3372231223 = 3.372 GB / 3.141 GiB     # git show 9965ce5:artifacts/registry.json
with hf_path= 37  with hf_revision= 37  hf_repo= survivor-m/prism-artifacts
ключей с hf_path без локального файла: 0
```

Каталоги: `results/{e55,e56,e63,e65,e66,e67,e68,e70}` — **все MISSING** (8/8, как заявлено); `results/e62`
существует, но пустой. `git log --all --oneline --diff-filter=A -- results/e63` — пусто; то же для
`e62, e55, e67, e70, e65` — по 0 коммитов-добавлений (заявленное «проверено для e62/e63/e55/e67/e70» ✓).

Входные npz:

```
exp47b_gen_kinematics.npz       791 729 646 B = 755.1 MiB
exp48b_lfm3_gen_kinematics.npz 1578 794 007 B = 1505.7 MiB
gen_layers_qwen.npz             233 232 680 B = 222.4 MiB    # 704 ключа, 64 дока (meta.json) — совпало
gen_layers_lfm3.npz             464 MB (заявлено 442 МБ — те же байты в MiB = 442.6)
```

то есть «755 МБ / 1.5 ГБ / 222 МБ» = MiB ✓, число ключей/доков ✓.

**Расхождение.** Заявленная пара «37 записей и 5.3 ГБ» не сходится: сумма `size` по 37 записям =
**3.372 GB (3.14 GiB)**. Величина 5.3 — это объём *всех* файлов >10 МБ в `results/` (`tools/artifacts.py scan`
→ `файлов больше 10.0 МБ: 50, суммарно 5634.1MB` = 5.25 GiB), а не то, что читает `verify`: 13 файлов
(2.26 GB) в реестре **нет** — `results/exp27_jcache.pt` 1729 МБ (по правилу AGENTS «регенерируемые кэши —
в архив не идут», это объяснимо) и 11 файлов `results/e64/tpu*/P*.npz` (520 МБ) +
`results/e50v3_out/qwen35-08b/tokenizer.json` (12.8 МБ) — они gitignored (`*.npz`, `results/e*/qwen35-08b/`),
в HF не выгружены (`grep -c e64 artifacts/registry.json` = 0) и в реестре не числятся. Разбивка замерена:
`всего не в реестре: 13 файлов, 2262 МБ / e64/tpu*: 11 файлов, 520 МБ / jcache: 1729 МБ`.

## R7. C7 в RERUN_TRIAGE_REVIEW.md воспроизводится — PASS

heredoc извлечён из файла машинно (единственный fenced-блок со `chsh`, тело между `<<'PY'` и `PY`), запущен
из корня репо:

```
iid     min=0.001 max=0.095 med=0.026
cumsum  min=0.029 max=1.454 med=0.579
real 0m0.413s
```

Совпадение с напечатанным в файле (строка «Вывод (20.11, 20 зёрен): `iid min=0.001 max=0.095 med=0.026`;
`cumsum min=0.029 max=1.454 med=0.579`») — побайтное. Главное утверждение: максимум по обоим наборам
**1.454 < 2**, ни одного значения > 2 нет; «до 2 не доходит» ✓. Числа первой редакции (`0.013-0.059`, `1.766`)
из текста сняты ✓. `VERDICT: REVISE` (строка 132) и маркер `RERUN_TRIAGE_REVIEW_DONE` (строка 139) на месте;
тип «САМО-РЕВЬЮ, не внешнее» (строка 3) + sha внешней ответки, совпадающий с диском.

---

## Найдено

1. **`experiments/e63_t23_ngram.py:25` — вход, которого нет, и `f[0]` упадёт на нём IndexError.** Правка путей
   привела glob к правильному каталогу, но совпадений ноль:
   `glob.glob('../data/calib_pins/{dom}___data_Qwen_Qwen3.5_0.8B__n16__min384__max512__plain.json')` →
   `wiki 0 [] math 0 [] code 0 [] chat 0 []` (исполнено из `experiments/`). В `data/calib_pins/` лежат
   `*__n12__min192__max512__plain.json` и `*__n16__min64__max256__plain.json` — нужной пары `n16__min384` нет.
   Строка 26 `d = json.load(open(f[0]))` → `IndexError: list index out of range`. Это не регресс коммита
   (до правки путь был бы просто в другом каталоге), но §З п.3 утверждает «входные npz живы… „нет артефакта"
   — это не потеря, а дрейф пути», и для `t23_ngram` это неверно: скрипт остаётся нерабочим и после батча 1.
   Карта восстановления (§З) его не содержит.
2. **`artifacts/registry.json` / `docs/ARTIFACTS_STORAGE.md:86` / сообщение коммита: «verify → 0 расхождений
   на 5.3 ГБ» — 5.3 ГБ не относится к реестру.** Факт: `verify` читает 37 записей = 3.372 GB; 5.3 (5.25 GiB) —
   это 50 файлов >10 МБ из `scan` (`ARTIFACTS_STORAGE.md:81`, там число верно). 11 файлов `results/e64/tpu*/P*.npz`
   (520 МБ) и `results/e50v3_out/qwen35-08b/tokenizer.json` (12.8 МБ) не в git, не в HF, не в реестре — при
   чтении строки 86 их сочтут засчитанными. Риск ровно тот, ради чего введено правило хранения.
3. **В коммите есть правки, которых нет в его описании** (`git show 9965ce5 -- experiments/e61_70`): во всех
   девяти writer'ах E-63 к `json.dump` добавлен `default=float` (`d0:72`, `d1:61`, `d2:99`, `t17:102`, `t1:80`,
   `t23:76`, `t25:91`, `t5_audit:121`, `t5_scramble:80`) — то есть чинилась сериализация numpy-float'ов;
   и в `exp63_d2_null.py` удалена мёртвая строка `U = u * np.asarray(...)[:, None][:, None][:, 0]`. Обе правки
   полезны, но в сообщении коммита и в §З п.5 не заявлены: «правки путей» на деле включают правки сериализации.
4. **Заявление «pull БЫЛ нерабочим (Path.replace → EXDEV, затем shutil.move → FileExistsError)» нельзя
   проверить по истории.** `tools/artifacts.py` создан этим коммитом целиком, промежуточных ревизий (`replace`, `move`)
   в git нет; «проверено в двух режимах, sha OK» тоже невоспроизводимо офлайн. Код текущего состояния корректен
   (R5), но провенанс-fix в коммите — рассказ о незакоммиченном состоянии, а не дифф.

Хрупкое, но не сломано:

5. `exp63_t17_dipnull.py:17` читает `../results/exp47b_gen_kinematics.json` (в корне `results/`, не в
   `results/e52_4b/`) — файл есть, работает; но это единственный вход вне `e*/`-подкаталога и он пережил
   чужую перестановку (`results/e52_4b/exp47b_gen_kinematics.npz` 206 МБ — другая копия).
6. `os.chdir` меняет cwd всего процесса; ни один из 10 скриптов не использует trackio/explog (правило AGENTS
   для новых экспов), и любое будущее «относительное имя по умолчанию» теперь осядет в `experiments/`.
7. Тот же класс путей остался незалеченным: **19 файлов** в `experiments/e61_70/` пишут `"../results/...` и не
   делают `os.chdir` — 15 из них `exp64_*.py` и 4 шт. `exp62_ab.py`, `exp62_gen.py`, `exp62_gen_lfm3.py`,
   `exp62_r2test.py` (замер: `grep '"../results'` ∧ `! grep os.chdir`). `makedirs` у большинства тоже нет.
   Объём батча 1 этого не требовал, но §З обещает порядок «exp62_layers → exp63_* → exp64_*» —
   на шаге exp64 тот же `FileNotFoundError`/запись не в тот каталог вернётся.
8. `sys` импортирован и не используется в `exp63_t1_tokrhythm.py`, `exp63_t17_dipnull.py` (был нужен старому
   `sys.path.insert`) — косметика.

## Не проверено

- Исполнение `tools/artifacts.py pull` (в т.ч. «два режима» из §З п.6) — нужна сеть/`hf_hub_download`; по
  промту сеть запрещена. Проверено чтением кода + `py_compile`.
- Сам факт выгрузки 37 файлов в `survivor-m/prism-artifacts` и `hf_revision` по каждой записи — сеть.
- Числа стриминга из сообщения коммита (82.8 МБ за 11.0 с, ranged 13.1 МБ за 2.8 с, один член 1.58 ГБ npz
  за 0.7 с) — вне R1-R7, сеть.
- Полный пересчёт §З по всем 61 файлу `exp6{2,3,4,5}_*.py` (заявленная база) — проверял только 10 файлов
  батча + glob-вход `t23`; бюджет 20 минут.
- Что `default=float` реально спасает от падения (без него `json.dump` на numpy-типе падает) — не воспроизводил:
  потребовало бы второго прогона батарей.
- Батареи на L1/L5/L11/L17 (только L23, как разрешено).

VERDICT: REVISE

BATCH1_CODE_REVIEW_DONE
