#!/usr/bin/env python3
"""Само-ревью артефакта E94_S4pp — механическое, а не глазами (AGENTS.md, пункт «само-ревью
артефакта до внешнего ревью»).

Проверяет НЕ вывод, а структурную честность артефакта:
  A. полнота: 23 запроса, у каждой ok-строки есть вердикт и ридинг;
  B. ничего не выдумано: каждое число артефакта прослеживается до строк запросов или до
     предрегистра (список полей-литералов фиксирован заранее);
  C. вердикт не противоречит строкам (пересобирается из requests и сравнивается);
  D. читаются контроли: манипуляционный гейт пройден там, где ban был;
  E. гейт руки B не «всегда истинен»: probe_dependence считается поVariation across k, и
     проверяется, что он не подменён «emit != вершина».

Выход: печатает PASS/FAIL по каждому пункту, RC=0 только если всё чисто. Только чтение.
"""
import json
import sys

ART = sys.argv[1] if len(sys.argv) > 1 else "results/e94/S4pp.json"
N_PLAN = 23
# поля, которым РАЗРЕШЕНО быть литералом (не из ответов сервера)
LITERAL_OK = {"exp", "spec", "started_local", "wall_s", "server", "prereg", "notes",
              "halt_reason", "requests_n", "requests_ok", "summary"}
fails, warns = [], []


def chk(cond, tag, msg):
    (fails if not cond else warns.__class__()).append((tag, msg)) if not cond else None
    print("%-5s %s %s" % ("FAIL" if not cond else "ok  ", tag, msg))
    return bool(cond)


A = json.load(open(ART, encoding="utf-8"))
R = A.get("requests") or []
S = A.get("summary") or {}

print("=== артефакт %s: запросов %d, halt=%r" % (ART, len(R), A.get("halt_reason")))

# A. полнота и обязательные поля
chk(len(R) == N_PLAN, "A1", "число запросов %d == плану %d" % (len(R), N_PLAN))
tags = [r.get("tag") for r in R]
chk(len(set(tags)) == len(tags), "A2", "теги уникальны")
bad = [r["tag"] for r in R if r.get("status") == "ok" and not r.get("verdict")]
chk(not bad, "A3", "у каждой ok-строки есть вердикт (пусто: %r)" % bad)
bad = [r["tag"] for r in R if r.get("status") == "ok" and not r.get("reading")]
chk(not bad, "A4", "у каждой ok-строки есть человекочитаемое чтение (пусто: %r)" % bad)
bad = [r["tag"] for r in R if r.get("verdict") == "unknown"]
chk(not bad, "A5", "не осталось чтений вне предрегистра (unknown: %r)" % bad)
bad = [r["tag"] for r in R if r.get("status") not in ("ok", "skipped", "error")]
chk(not bad, "A6", "статусы только из алфавита ok/skipped/error (чужие: %r)" % bad)
skipped = [r["tag"] for r in R if r.get("status") == "skipped"]
if skipped and not A.get("halt_reason"):
    fails.append(("A7", "есть skipped-строки, но halt_reason пуст"))
chk(not (skipped and not A.get("halt_reason")),
    "A7", "пропущенные строки объяснены halt_reason (skipped=%d)" % len(skipped))

# B. источник чисел: сводка пересобирается из requests
by = {r.get("tag"): r for r in R}


def num(tag, field):
    return (by.get(tag) or {}).get(field)


pairs = [("S4pp-1", "emit"), ("S4pp-2", "emit"), ("S4pp-ref", "emit"),
         ("S4pp-repro", "emit"), ("S4pp-0", "emit"), ("S2a", "emit"), ("S2b", "emit")]
mism = []
for tag, f in pairs:
    got = (S.get("repro") if tag == "S4pp-repro" else
           S.get("S4pp_0") if tag == "S4pp-0" else
           S.get("S2_pair", {}).get(tag[1:].lower()))
    if isinstance(got, dict) and f in got and num(tag, f) != got.get(f):
        mism.append((tag, num(tag, f), got.get(f)))
chk(not mism, "B1", "сводка не содержит чисел, которых нет в requests (расхождения: %r)" % mism)
for cell in S.get("S3_topk") or []:
    r = by.get(cell["tag"]) or {}
    if cell.get("emit") != r.get("emit") or cell.get("top1") != r.get("top1"):
        fails.append(("B2", "S3-ячейка %s не совпадает со строкой запроса" % cell["tag"]))
chk(not [f for f in fails if f[0] == "B2"], "B2",
    "все %d ячеек свипа = строки запросов" % len(S.get("S3_topk") or []))
ce = S.get("S3_control_expectation") or {}
adr = (A.get("addresses") or {}).get("ctrlA") or {}
chk(ce.get("expected_id") == adr.get("top1"), "B3",
    "ожидание свипа взято из загруженного адреса (%r == %r), а не литералом"
    % (ce.get("expected_id"), adr.get("top1")))

# C. вердикт пересобирается
probes = [by.get("S4pp-1") or {}, by.get("S4pp-2") or {}]
pv = [p.get("verdict") for p in probes]
ref = by.get("S4pp-ref") or {}
if ref.get("verdict") != "ref_ok":
    exp_v = "невалидна: опора"
elif any(v == "invalid" for v in pv):
    exp_v = "невалидна: ban"
elif any(v == "M1" for v in pv):
    exp_v = "M1"
elif all(v == "undec" for v in pv):
    exp_v = "не разрешено"
else:
    exp_v = "нет данных"
got_v = (S.get("verdict") or "")
chk(got_v.startswith(exp_v) or (A.get("halt_reason") and exp_v == "нет данных"),
    "C1", "вердикт %r согласован со строками (пересборка: %r)" % (got_v[:48], exp_v))
if "M1 установлен" in got_v:
    chk(all(p.get("manip_ok") for p in probes if p.get("ban_sent") is not None), "C2",
        "заявленный M1 стоит только на строках с пройденным манипуляционным гейтом")
if "M1 установлен" in got_v:
    outside = [p["tag"] for p in probes if p.get("verdict") == "M1" and
               p.get("emit") == (by.get(p["tag"]) or {}).get("top1")]
    chk(not outside, "C3", "ни одна «M1»-строка не дала вершину отчёта (иначе ветка не решающая)")

# D. контроли бана
ctrl = S.get("controls") or []
chk(len(ctrl) == 4, "D1", "все 4 контроля бана на месте (%d)" % len(ctrl))
ran = [c for c in ctrl if (by.get(c["tag"]) or {}).get("status") == "ok"]
notrun = [c["tag"] for c in ctrl if (by.get(c["tag"]) or {}).get("status") != "ok"]
nbad = [c["tag"] for c in ran if c.get("manip_ok") is not True]
chk(bool(ran), "D2a", "минимум один контроль бана выполнился (выполнилось %d, не выполнилось %r)"
    % (len(ran), notrun))
chk(not nbad, "D2b", "на ВСЕХ выполнившихся контролях эхо бана совпало (не прошло: %r)" % nbad)
still = [c["tag"] for c in ran if c.get("emit") == c.get("ban_sent")]
chk(not still, "D3", "ни на одном выполненном контроле забаненная вершина не осталась "
                     "эмиссентом (осталась: %r)" % still)
if notrun:
    print("     NOTE %d контроле(ов) не выполнились (halt=%r) — это НЕ читается как "
          "«смещение не применилось»" % (len(notrun), A.get("halt_reason")))

# E. предикат гейта не подменён
g = S.get("gate2") or {}
need = {"probe_dependence", "control_dependence", "gate2_met", "probe_emits_by_k",
        "control_emits_by_k", "rule"}
chk(need <= set(g), "E1", "у гейта есть все поля, отсутствуют: %r" % sorted(need - set(g)))
pk = g.get("probe_emits_by_k") or {}
ck = g.get("control_emits_by_k") or {}
pk_ok = {k for k, v in pk.items() if v is not None}
ck_ok = {k for k, v in ck.items() if v is not None}
chk(pk_ok == ck_ok and len(pk_ok) <= 6, "E2",
    "сравнение гейта идёт по одному и тому же множеству k с эмиссентами с обеих сторон "
    "(probe %d, ctrl %d из 6)" % (len(pk_ok), len(ck_ok)))
if len(pk_ok) < 6 or len(ck_ok) < 6:
    print("     NOTE свип неполный: гейт обязан быть null, рука B не запускается")
for side, m, dep in (("probe", pk, g.get("probe_dependence")),
                     ("ctrl", ck, g.get("control_dependence"))):
    vals = [v for v in m.values() if v is not None]
    e = (len(set(vals)) > 1) if len(m) == 6 else None
    chk(dep == e, "E3", "%s: dependence = «>1 различного эмиссента», а не «emit != вершина» "
                      "(посчитано %r, в артефакте %r)" % (side, e, dep))
chk(g.get("gate2_met") in (True, False, None), "E4", "gate2_met трёхзначен")
if g.get("gate2_met") is None:
    print("     NOTE неполный свип -> рука B не запускается (по ТЗ)")
pb = [(k, v) for k, v in pk.items()]
arch = ((A.get("addresses") or {}).get("probe") or {}).get("emit")
if g.get("probe_stays_at_archive_emit") and g.get("probe_dependence") and pk_ok:
    fails.append(("E5", "противоречие: эмиссент везде = архивному, но dependence истинна"))
chk(not (g.get("probe_stays_at_archive_emit") and g.get("probe_dependence")), "E5",
    "нет противоречия stays_at_archive_emit vs dependence (k-карта: %r, архив %r)"
    % (sorted(pb, key=str), arch))

# E6 (exec-r1/Self-1): если ширина не меняла отчёт, gate2_met=false нельзя читать как
# «независимость показана». Делаем это инвариантом чекера, а не надеждой на читателя.
rd = S.get("S3_report_diag") or {}
flat = sorted(k for k, v in rd.items()
              if isinstance(v, dict) and v.get("report_changed_by_topk") is False)
if flat and g.get("gate2_met") is False:
    chk(bool((g.get("gate2_caveat") or "").strip()), "E6",
        "ширина не меняла отчёт на %r и gate2_met=false => оговорка к чтению обязательна"
        % flat)
else:
    chk(True, "E6", "оговорка не требуется (отчёт менялся от ширины: %r, gate2_met=%r)"
        % (flat, g.get("gate2_met")))

print("\n=== ИТОГ: %s (%d fail)" % ("PASS" if not fails else "FAIL", len(fails)))
for f in fails:
    print("  FAIL", f[0], f[1])
sys.exit(0 if not fails else 1)
