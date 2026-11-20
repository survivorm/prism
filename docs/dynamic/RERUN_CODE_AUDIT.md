# Фаза B: аудит кода перед перезапуском (батч 1: E-55, E-56, E-63, E-80, E-81)

Дата: 2026-11-20. Вход: `docs/dynamic/RERUN_VALUE.md` (что стоит перезапускать), `docs/TZ_rerun_no_artifacts.md`.
Здесь — только то, что проверено **командой**; выводы без команды помечены «(суждение)».

## S1. Системная поломка: reorg 2026-11-16 сломал ВСЕ относительные пути в скриптах

`git log -1 d6ab76b` → 2026-11-16, «Reorg experiments/ по сериям: e01_10/e11_20/…/e81_91 (283+19 файлов
git-mv)». Файлы переехали на уровень глубже, а пути **внутри** файлов остались рассчитанными на старую
глубину. Проверено:

| проверка | команда | результат |
|---|---|---|
| сколько `../results` в подпапках | `git grep -lF '../results' HEAD -- 'experiments/*.py' \| wc -l` | **42 файла** (вариант с кавычкой `'".\./results'` — 39 файлов / 79 вхождений; ранняя цифра 41 была получена неуточнённым шаблоном) |
| сколько из них резолвится в существующий файл | python: `normpath(dirname(f)+литерал)` по HEAD-версиям файлов | **0 из 84 путей** (несуществующие каталоги: `experiments/results`, `experiments/results/e52_4b`, `experiments/results/e62`) |
| `ROOT = Path(__file__).resolve().parent.parent` | `git grep -lF 'parent.parent' HEAD -- 'experiments/*.py' \| wc -l` | **26 файлов** (ранняя цифра 21 — недоучёт: сейчас 17 в рабочем дереве + 6 фиксов батча 1) → указывает на `experiments/`, а не на корень |
| `HERE/../results` | `git grep -lF 'HERE, "..", "results"' HEAD` | 23 файла |
| корректная форма `parents[2]` | `git grep -lF 'parents[2]' HEAD -- 'experiments/*.py' \| wc -l` | **0 файлов** до фикса (6 после фикса батча 1) |

`run_one.sh` делает `cd /home/survivor/work/prism` ⇒ `../results` = `/home/survivor/work/results`,
`parent.parent` = `experiments/`. Ни `/home/survivor/work/results`, ни `experiments/results` на диске
**нет** — то есть после reorg эти скрипты либо упали на чтении входа, либо (для пишущих веток без
предварительного чтения) создали бы мусорный каталог. Его отсутствие — доказательство, что **после
2026-11-16 ни один из них не доходил до записи** (суждение с сильной опорой: `exp56_look.py:25`
делает `os.makedirs` на импорте — каталога нет ⇒ скрипт не запускался).

Следствие для «потерянных» номеров: их артефакты могли родиться только в прогонах **до** 2026-11-16
(когда `experiments/expNN.py` + `../results` = `results/` — верно), а `results/*.json` в git не
попадают по `.gitignore` ⇒ потеря при чистке. Это ровно та картина, которую описал пользователь.

**Контр-проверка (чтобы не спутать с «всё сломалось давно»):** `results/e64/tpu/P1_own.npz` (56 МБ),
`P2_corpus.npz` (30 МБ), `cor_p1.npz` (20 КБ), `results/exp47b_gen_kinematics.json` (76 КБ),
`results/exp53b_servo.json` (7 КБ) — на месте, т.е. входы батча 1 физически есть.

## S2. `.pyc` — не доказательство запуска

Все 354 `__pycache__/*.pyc` имеют mtime `2026-11-16 13:35:32–33` (bulk-компиляция при reorg,
`stat -c '%y'`), поэтому «есть pyc ⇒ скрипт запускался» — ложный вывод. Единственное доказательство
прогона: лог/артефакт с провенансом.

## S3. Вердикты по скриптам батча 1

| номер | скрипты | входы на диске | путь | состояние |
|---|---|---|---|---|
| **E-80** | `exp80_ring.py`, `exp80b_null.py`, `exp80c_cor_null.py` | `P1_own.npz` ✓, `P2_corpus.npz` ✓, `cor_p1.npz` ✓ | `ROOT=parent.parent` → D=`experiments/results/e64/tpu` ✗ | **починить 1 строку** → runnable |
| **E-81** | `exp81_afterglow.py`, `exp81b_replay.py`, `exp81c_heat.py` | те же npz ✓; артефакты `TAIL16/REPLAY/HEAT.json` уже есть | тот же `ROOT` ✗ | починить; **перезапуск не нужен** (есть артефакт) — только реанализ |
| **E-55** | `exp55_stage0.py` (3 hits), `exp55_stage1b.py` (1), `exp53b_servo.py` (2), `exp55_modes/sbi/jrecal.py` (0, но зависят от `exp55_stage0.json`) | `results/exp53b_servo.json` ✓, `results/exp47b_gen_kinematics.json` ✓ | `HERE/../results` ✗ | починить 4 файла → цепочка stage0 → sbi → jrecal → modes |
| **E-56** | `exp56_look.py` (`OUT=HERE/../results/e56` ✗) | ждёт `results/exp55_stage0.json` — **отсутствует** | ✗ | **заблокирован шагом E-55 stage0** |
| **E-63** | 9 скриптов, все с `../results/e63` ✗ | `d2_null`, `t25_ekurve`, `t17_dipnull`, `t1_tokrhythm`, `t5_audit`, `t5_scramble` — входы из `/data/prism_kernels/exp62/*.npz` ✓ и `results/exp47b_gen_kinematics.json` ✓ | ✗ | **6 из 9 runnable после фикса** |
| ↳ | `exp63_d0.py` | `results/e62/layers_qwen_L23.part` — **каталога `results/e62` не существует** | ✗ | **заблокирован** (артефакты E-62 потеряны) |
| ↳ | `exp63_d1_tangents.py` | нужен d0-выход | ✗ | заблокирован вслед за d0 (суждение по коду) |
| ↳ | `exp63_t23_ngram.py` | `data/calib_pins/*__n16__min384__max512__plain.json` — **нет** (в наличии `n12__min192__max512` и `n8__min64__max64`, 10 файлов) | ✗ | **заблокирован** (нет пин-корпуса n16) |

## S4. Дефекты кода, найденные при чтении (чинить в том же коммите, AGENTS п.9)

1. `exp80_ring.py:38-45` — мёртвая функция `signed_acf()`: тело `A=U@plane@np.eye(...) if False else None`,
   затем `Bm=None` в цикле, `return out` (всегда `[]`). Не вызывается, но при «доработке на глаз»
   даст молчаливый пустой результат.
2. `exp80_ring.py:93` — `json.dumps(verdict if False else verd)` — остаток отладки в `print`.
3. `exp63_d2_null.py:28` — двойное присваивание `U`: строка с `[:, None][:, None][:, 0]` перезаписывается
   следующей; мёртвый код с неверной трансляцией (не стреляет только потому, что затёрт).
4. `exp55_stage0.py` docstring честно фиксирует ограничение: контроль «ритм текста» невозможен —
   в артефакте `exp47b` только sha, без декодированного текста. Переносится в отчёт как ограничение,
   а не как дефект.

## S5. План фикса (минимальный, без массового свипа) — **ВЫПОЛНЕНО**

- Якорь: `ROOT = Path(__file__).resolve().parents[2]` (файл в `experiments/eXX_YY/`) — **ровно один
  каталог-уровень**, чтобы скрипт работал из любого CWD.
- E-80/E-81 (6 файлов): `parent.parent` → `parents[2]` (в `exp81c_heat` ещё и `PRISM_CALIB_PIN_DIR`).
  Проверено: `D=results/e64/tpu`, `P1_own.npz` и `P2_corpus.npz` резолвятся (оба `True`).
- E-55 семество (7 файлов): в `HERE/".."/"results"`-джойнах (включая многострочные) добавлен
  ещё один `".."`; в `exp53b_servo.py` литералы `"../results/…"` заменены на `_RES + "/…"`,
  где `_RES` — абсолютный `<repo>/results` от `__file__`.
- E-63 (9 файлов): `sys.path.insert(0,'.')` → `_HERE`, плюс `os.chdir(<script dir>/..)` (= `experiments/`),
  чтобы существующие `"../results/…"` и `"../data/calib_pins/…"` резолвились в корень репо без правки
  каждой строки. Проверено скриптом-чекером по HEAD-версиям файлов: все 84 литерала ведут в `<repo>/results|data` (до фикса было 0 из 84 резолвящихся, сейчас 0 вхождений `../results` в семи файлах цепочки E-55 и в `exp53b_servo`).
- `json.dump` в 9 скриптах E-63 и в `exp56_look.py:114` получил `default=float` (numpy-типы).
- Мёртвый код вычищен: `signed_acf()` и `verdict if False else` в `exp80_ring.py`, двойное
  присваивание `U` в `exp63_d2_null.py`.
- `py_compile` на всех 22 правленых файлах → OK.

## S5b. Трекинг: `trackio` в `.venv` сломан — обёртка отдельным процессом

Факт: `.venv/bin/python -c "import trackio"` → `ImportError: cannot import name 'httpx' from
huggingface_hub.utils` (в `.venv` `huggingface_hub 1.28.0`; в системном `python3` — hf 1.33.0 и
`trackio 0.45.1` импортируется). Поэтому создан `experiments/trackio_run.py`: обёртку гоняет
`python3` (там trackio жив), эксперимент — дочерним процессом `.venv/bin/python`; на t=0 пишется
config (`started_at`, `revision`, `seed`, `script_sha`, `git_commit`, `device`, `model`, `cmd`),
на T — `finished_at/status/rc/duration/log_md5`. Если trackio всё равно недоступен — печатает
`TRACKIO_UNAVAILABLE` и **не** убивает прогон, а провенанс остаётся строкой `RUN_PROVENANCE {json}`
в логе (мигрируется позже, `docs/ARTIFACTS_STORAGE.md` §1: source of truth — логи).

## S6. Что этот аудит меняет в порядке работ

Батч 1 «numpy за час» распадается: **E-80 (3 скрипта)** и **E-55-цепочка (4 файла)** — реально
runnable после фикса; **E-63 — 6 клеток из 9** (d0/d1/t23 заблокированы отсутствующими артефактами
E-62 и пин-корпусом n16); **E-56 ждёт шаг stage0**. Блокировки — это находки, а не помехи: две из них
(`results/e62/*`, `calib_pins n16`) придётся либо восстанавливать прогоном E-62, либо честно
переклассифицировать соответствующие клетки в «непроверяемо без нового прогона модели».

RERUN_AUDIT_DONE
