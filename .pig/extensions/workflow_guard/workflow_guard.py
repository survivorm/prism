"""workflow_guard — напоминание о шагах протокола раунда (prism).

Не глюкавая конструкция: НОЛЬ событийных хуков, НОЛЬ блокировок.
Единственный механизм — prompt_guidelines у инструментов (pig кладёт их в
системный промпт) + читалка состояния wf_check по файлам артефактов/ревью.
Ничего не ломает: если расширение не загрузилось, работа продолжается без него.
"""
import json, os, re
from pathlib import Path

from pig_sdk import Extension

ROOT = Path(os.environ.get("PRISM_ROOT", Path.cwd()))

GUIDELINES = [
    "ПРОТОКОЛ (prism, обязательно): TZ -> ревью-луп ТЗ (чистый файл) -> ОДИН исполнитель на ОДИН эксперимент -> САМО-аудит кода ДО запуска -> run через ./run_one.sh ТОЛЬКО -> само-ревью артефакта -> ВНЕШНИЙ ревью-луп (файл docs/dynamic/*REVIEW*) -> коммит.",
    "Внешнее ревью обязан проходить и КОД/артефакт, а не только ТЗ; 'забыл отправить в ревью' = нарушение протокола.",
    "Лонч только ./run_one.sh <name> <cmd>; BUSY = читать лог, не перезапускать; убивать только свои процессы; pkill/pgrep только с [b]racket-trick.",
    "Промпт субагенту — ФАЙЛ; финальный маркер (EXEC_DONE/S*_DONE) обязателен; RC=0 без маркера = задача не выполнена.",
    "Правки через edit с проверкой; перед КАЖДЫМ коммитом grep-verify маркеров; git add только явными путями (без -A).",
    "Одна CPU-экспериментальная задача за раз; память/LA смотреть до лонча (watchdog /tmp/prism_watch.log).",
    "Числа в отчёт только из артефактов/лог-строк; проверка 'процесс реально идёт' по pgrep -x/файлам, а не по намерению.",
]

CHECKLIST = [
    ("tz_clean", "docs/dynamic/*ТЗ* (актуальное) и последний REVIEW-файл ТЗ без P1", None),
    ("runner_file_prompt", None, "промпт-файл исполнителя существует"),
    ("self_audit", None, "само-аудит изменённого exp-кода сделан (чек-лист AGENTS.md)"),
    ("run_one", None, "запуск был только через run_one (лог /tmp/<name>.log, нет дублей)"),
    ("exec_done_marker", None, "в логе исполнителя есть финальный DONE-маркер"),
    ("artifact", None, "артефакт json создан, sanitize/float-касты, sha16 в манифесте"),
    ("self_review_artifact", None, "артефакт перечитан глазами против кода"),
    ("external_review", "docs/dynamic/2_REVIEW_exec_*.md", "внешнее ревью КОДА/артефакта есть"),
    ("review_answer_committed", None, "ответка ревьюеру + правки в том же/следующем коммите"),
]


def _files_exist(patterns):
    hits = []
    for pat in patterns or []:
        d, f = os.path.split(pat)
        hits += [str(p) for p in (ROOT / (d or ".")).glob(f)]
    return sorted(hits)


def new_extension() -> Extension:
    ext = Extension("workflow_guard")

    def wf_check(ctx, args):
        try:
            with open("/tmp/wf_dbg.log", "a") as f:
                f.write(f"called args={args!r}\n")
        except Exception:
            pass
        exp = (args.get("exp") or "").strip()
        state = {"exp": exp or None, "gates": {}}
        rev = _files_exist(["docs/dynamic/2_REVIEW_exec_*.md"])
        art = None
        if exp:
            for cand in list((ROOT / "results").rglob(f"*{exp.upper()}*.json")):
                art = str(cand.relative_to(ROOT))
                break
        for gid, pat, desc in CHECKLIST:
            if gid == "external_review":
                state["gates"][gid] = {"done": bool(rev), "files": rev[:3], "desc": desc}
            elif gid == "artifact":
                state["gates"][gid] = {"done": bool(art), "file": art, "desc": desc}
            else:
                state["gates"][gid] = {"done": False, "desc": desc,
                                       "note": "проверяется человеком/агентом руками"}
        missing = [g for g, v in state["gates"].items() if not v["done"]]
        state["missing"] = missing
        state["next_action"] = ("НЕ интерпретировать результат, пока missing непусты; "
                                "ближайший шаг: " + (missing[-4] if missing else "всё закрыто")) if missing else "всё закрыто"
        state["summary"] = f"exp={exp or '?'}: missing={len(missing)} гейтов"
        return json.dumps(state, ensure_ascii=False, indent=1)  # str -> SDK обернёт в {"content": ...}

    def wf_next(ctx, args):
        return json.dumps({"protocol_order": "TZ -> review(TZ) -> exec(one runner) -> self-audit -> run(run_one) -> self-review artifact -> external review -> commit",
                "checklist": [c[2] for c in CHECKLIST],
                "exp_hint": (args or {}).get("exp")}, ensure_ascii=False, indent=1)

    ext.tool(
        "wf_check",
        "Проверить состояние протокола для эксперимента (артефакт/ревью-файлы по диску). "
        "Возвращает missing-гейты; пустой missing = можно интерпретировать результат. Вызывать ПЕРЕД выводами.",
        {"type": "object", "properties": {"exp": {"type": "string", "description": "имя/код, напр. R92 / e91 / R917"}}, "required": []},
        wf_check,
        prompt_guidelines=GUIDELINES,
    )
    ext.tool(
        "wf_next",
        "Показать порядок шагов протокола и полный чек-лист (TZ->review->exec->self-audit->run_one->self-review->external review->commit).",
        {"type": "object", "properties": {"exp": {"type": "string"}}, "required": []},
        wf_next,
        prompt_guidelines=GUIDELINES,
    )
    return ext
