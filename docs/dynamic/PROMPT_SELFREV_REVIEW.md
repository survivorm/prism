# Промт внешнего ревью: проверка честности само-ревью (RERUN_TRIAGE_REVIEW.md)

Ты — **независимый ревьюер, read-only**. Объект — не эксперимент и не ТЗ, а **само-ревью**: файл
`docs/dynamic/RERUN_TRIAGE_REVIEW.md` (автор — ассистент). Твоя задача: проверить, не выдуманы ли его проверки,
и верны ли его три «находки». Ты НЕ правишь объекты, НЕ запускаешь эксперименты, НЕ качаешь модели
(`Qwen/Qwen3.5-0.8B` лежит локально). Разрешено: `read`, `bash`, `write` (ровно один файл — твой отчёт).

**Лимит: 20 минут, ≤ 20 вызовов bash.** Если команда падает — фиксируешь факт и идёшь дальше. Сомнение → строка
в разделе «не проверено», а не новая гипотеза. Никакого исследования истории git, никаких `find /`.

Корень репо: `/home/survivor/work/prism` (все команды оттуда).

## Объекты и их sha256 (сверь первой командой; расхождение → STOP, `VERDICT: FAIL (sha mismatch)`)

    sha256sum docs/dynamic/RERUN_TRIAGE_REVIEW.md docs/dynamic/RERUN_SCRIPT_TRIAGE.md docs/dynamic/RERUN_CODE_AUDIT.md

Ожидаемые:
- `RERUN_SCRIPT_TRIAGE.md` = `1263acbef8ee1048cbf87a43e5c62b2df381525ab931d8dc901e51e0c46ef5b1`
- `RERUN_CODE_AUDIT.md` = `f9faf2a32cb99e6455be18701472ec625aaae87ae26c9c19cbed0ce4954e763a`
- `RERUN_TRIAGE_REVIEW.md` — сверь и запиши фактический (он мог измениться, это нормально, если он не содержательнее)

## 5 проверок (ровно эти команды; вывод цитируй)

**R1. Проверки C1–C7 в само-ревью не выдуманы.** Возьми из само-ревью 4 числа и воспроизведи их:

    .venv/bin/python - <<'EOF'
from transformers import AutoTokenizer
import json,ast
tk=AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B",trust_remote_code=True)
w=lambda t: tk.apply_chat_template([{"role":"user","content":t}],add_generation_prompt=True,tokenize=False)
n=lambda s: len(tk(s,add_special_tokens=False)["input_ids"])
h0=w(""); full=w(" Beginners BBQ Class")
print("len_wrap_empty",len(h0),"P_true",n(full[:full.find("Beginners BBQ Class")]),"P_old",n(full[:len(h0)]))
r=json.load(open("results/e75/TMPL_FAMILY.json")); print("artifact_P",sorted({v.get("P") for v in r["qwen/tmpl"].values()}))
try: ast.parse(open("experiments/e71_80/exp78_reanalysis.py").read())
except SyntaxError as e: print("syntax_lineno",e.lineno)
print("defs_load_pin_texts", len([1 for _ in __import__("subprocess").run(["grep","-rnF","def load_pin_texts","experiments/"],capture_output=True,text=True).stdout.splitlines()]))
EOF

Сверь с само-ревью: `69`, `3`, `12`, артефакт `{12..20,24}`, SyntaxError на строке `54`, `defs=0`.

**R2. Проверка F3 (числа путей) воспроизводима.**

    git grep -lF '../results' HEAD -- 'experiments/*.py' | wc -l; git grep -oF '../results' HEAD -- 'experiments/*.py' | wc -l; git grep -lF 'parent.parent' HEAD -- 'experiments/*.py' | wc -l; git grep -lF 'parents[2]' HEAD -- 'experiments/*.py' | wc -l

Ожидается по само-ревью: `42`, `84`, `26`, `0`.

**R3. Проверка F2 (CHSH ≤ 2, а не «=2 тождественно») — это математическое утверждение, проверь его численно.**

    .venv/bin/python - <<'EOF'
import numpy as np
def chsh(X,seed=0):
    r=np.random.default_rng(seed)
    D=X-X.mean(0); D/=(np.linalg.norm(D,axis=1,keepdims=True)+1e-9)
    P=np.linalg.qr(r.normal(size=(X.shape[1],4)))[0][:,:4]
    sgn=lambda M: np.where(M>=0,1,-1)
    A=[sgn(D[:-1]@P[:,i]) for i in range(2)]; B=[sgn(D[1:]@P[:,i+2]) for i in range(2)]
    E=lambda u,v: float(np.mean(u*v))
    return abs(E(A[0],B[0])+E(A[0],B[1])+E(A[1],B[0])-E(A[1],B[1]))
rng=np.random.default_rng(1); X=rng.normal(size=(2000,16)); Y=np.cumsum(rng.normal(size=(2000,16))*0.1,axis=0)
print("iid",[round(chsh(X,s),3) for s in range(6)]); print("corr_max_of_20",round(max(chsh(Y,s) for s in range(20)),3))
EOF

Проверь: все значения `≤ 2` (нарушение невозможно), на iid — близко к 0, на коррелированных — заметно больше 0,
и **ни одно не равно 2**. Если хоть одно `= 2.000` — утверждение само-ревью «тождественно не 2» неверно → FAIL на F2.

**R4. Само-ревью не выдаёт себя за внешнее.** Прочти первые 12 строк `RERUN_TRIAGE_REVIEW.md` и подтверди, что там
явно написано «само-ревью / не внешнее» и перечислены 4 провалившихся внешних прогона (id `1c0500a4`, `6320fafd`,
`bb0c1535`, спавн `qwen3.8-flash-next`). Проверь, что файлы этих прогонов действительно отсутствуют:

    ls -la docs/dynamic/RERUN_TZ_REVIEW.md docs/dynamic/CANON_INVENTORY_TZ_REVIEW.md docs/dynamic/ARTIFACTS_STORAGE_REVIEW.md 2>&1 | tail -4

**R5. В само-ревью нет утверждений без команды.** Пройди по разделу «Не проверено» и по «Замечаниям»: каждое
число должно либо иметь команду выше, либо стоять в «не проверено». Три примера на выбор — сверь.

## Что писать

Файл: `docs/dynamic/RERUN_SELFREV_REVIEW.md`, структура строго:

```
# Ревью само-ревью: RERUN_TRIAGE_REVIEW.md
SHA256_РЕВЬЮ: <фактический>
SHA256_ТРИАЖА: <фактический>
SHA256_АУДИТА: <фактический>
## R1 … R5
по блоку: PASS|FAIL|НЕ ПРОВЕРЕНО + фактический вывод (числа/цитата)
## Найдено
нумерованный список: каждое — (а) расхождение факта или (б) строка файла, где число без команды
## Не проверено
VERDICT: PASS|REVISE|FAIL
RERUN_SELFREV_REVIEW_DONE
```

Правило вердикта: любое расхождение чисел между само-ревью и твоим прогоном → `REVISE` (с указанием, какие именно);
выдуманная проверка (число в само-ревью, которое не воспроизводится) → `FAIL`.
Последняя строка файла — ровно `RERUN_SELFREV_REVIEW_DONE`. Затем выведи в чат эту строку и остановись.
