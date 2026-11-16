#!/usr/bin/env python3
"""Само-ревью артефакта E94_S4pp — механическое, а не глазами (AGENTS.md, пункт «само-ревью
артефакта до внешнего ревью»).

Проверяет НЕ вывод, а структурную честность артефакта:
  A. полнота: 24 запроса (r7: +S4pp-noop), у каждой ok-строки есть вердикт и ридинг;
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
N_PLAN = 24
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
           S.get("S2_pair", {}).get(tag))
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
elif any(v == "branchA" for v in pv):
    exp_v = None  # r7/MAJOR-2: агрегат не предрегистрирован — пересказ по точкам
elif all(v == "undec" for v in pv):
    exp_v = "не разрешено"
else:
    exp_v = "нет данных"
got_v = (S.get("verdict") or "")
if exp_v is None:
    # r7/MAJOR-2: пересказ по точкам + «не предрегистрирован» + отсутствие «противоречит»
    chk("не предрегистрирован" in got_v and "противоречит" not in got_v
        and all(("S4pp-%d: " % (i + 1)) in got_v for i in (0, 1)),
        "C1", "вердикт = пересказ по точкам без агрегата (r7/MAJOR-2): %r" % got_v[:60])
else:
    chk(got_v.startswith(exp_v) or (A.get("halt_reason") and exp_v == "нет данных"),
        "C1", "вердикт %r согласован со строками (пересборка: %r)" % (got_v[:48], exp_v))
if "противоречит" in got_v:
    fails.append(("C1b", "слово 'противоречит' в вердикте (r7/MAJOR-2: убрано)"))
chk("противоречит" not in got_v, "C1b",
    "«противоречит» отсутствует в вердикте (r7/MAJOR-2)")
for p in probes:
    if p.get("verdict") == "branchA":
        chk(p.get("manip_ok") is True, "C2",
            "«branchA»-строка %s прошла манипуляционный гейт" % p.get("tag"))
        chk(p.get("emit") != p.get("top1"), "C3",
            "«branchA»-строка %s не дала вершину отчёта (иначе ветка не решающая)" % p.get("tag"))

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

# F. r7-ремедиация: tamper, noop, S4pp_0, spread, r8-FINDINGS
tr = A.get("tamper_rehearsal") or {}
chk(tr.get("ok") is True, "F1",
    "tamper-репетиция в артефакте и прошла (rc=%r halt=%r assertion=%r servers=%r/%r)"
    % (tr.get("rc"), tr.get("halt_reason"), tr.get("assertion"),
       tr.get("servers_before"), tr.get("servers_after")))
noop = by.get("S4pp-noop") or {}
chk(noop.get("status") == "ok" and noop.get("verdict") in ("noop_out", "third_branch", "invalid"),
    "F2", "S4pp-noop на месте с предрегистрированным чтением (verdict=%r)"
    % noop.get("verdict"))
if noop.get("status") == "ok" and noop.get("ban_sent") is not None:
    chk(noop.get("manip_ok") is True, "F2b",
        "S4pp-noop: эхо бана совпало с отправленным (manip_ok=%r)" % noop.get("manip_ok"))
s0 = S.get("S4pp_0") or {}
chk(s0.get("manip_ok") in (True, False), "F3",
    "S4pp_0.manip_ok в сводке (r7/MINOR-2, manip_ok=%r)" % s0.get("manip_ok"))
chk((by.get("S4pp-0") or {}).get("manip_ok") == s0.get("manip_ok"), "F3b",
    "S4pp_0.manip_ok сводка == строка")
for side in ("probe", "ctrlA"):
    d = (S.get("S3_report_diag") or {}).get(side) or {}
    chk("top1_lp_spread" in d, "F4",
        "%s: top1_lp_spread записан (r7/MINOR-1, spread=%r)" % (side, d.get("top1_lp_spread")))
# r8/FINDINGS: F3 (branchA + verdict_mechanism), F4 (S2 informative), F6 (ref addr),
# F7 (n_tl в сводке probes)
chk(all(isinstance(pp.get("verdict_mechanism"), str) and pp["verdict_mechanism"]
        for pp in (S.get("probes") or [])), "G1",
    "probes: verdict_mechanism записан на обеих точках (r8/F3)")
chk(all(pp.get("n_tl") is not None for pp in (S.get("probes") or [])
        if (by.get(pp.get("tag")) or {}).get("status") == "ok"), "G2",
    "probes: n_tl построчно в сводке (r8/F7)")
chk((S.get("S2_pair") or {}).get("informative") in (True, False), "G3",
    "S2_pair.informative записан (r8/F4, informative=%r)" % (S.get("S2_pair") or {}).get("informative"))
ref = S.get("ref") or {}
chk(ref.get("addr") and ref.get("top1") is not None and "ban_sent" in ref, "G4",
    "ref: addr + top1 + ban_sent в сводке (r8/F6, r9/F6, addr=%r ban_sent=%r)"
    % (ref.get("addr"), ref.get("ban_sent")))
# r9/FINDINGS: N1 (самодостаточность решающей руки), F5 (признаки гейтов),
# F4 (nonvacuous), F1 (подпись spread), F2 (chain), N2 (формулировка M1)
noop_s = S.get("S4pp_noop") or {}
chk(noop_s.get("addr") and "ban_sent" in noop_s and "chain" in noop_s, "H1",
    "S4pp_noop: addr + ban_sent + chain в сводке (r9/N1, addr=%r ban_sent=%r chain=%r)"
    % (noop_s.get("addr"), noop_s.get("ban_sent"), noop_s.get("chain")))
tam = (A.get("tamper_rehearsal") or {})
chk(tam.get("lit_unique") is True and tam.get("out_lit_unique") is True, "H2",
    "tamper: признаки сработавших гейтов (r9/F5, lit=%r out=%r)"
    % (tam.get("lit_unique"), tam.get("out_lit_unique")))
chk("report_nonvacuous" in (S.get("S2_pair_diff") or {}), "H3",
    "S2_pair_diff.report_nonvacuous записан (r9/F4, %r)"
    % (S.get("S2_pair_diff") or {}).get("report_nonvacuous"))
for side in ("probe", "ctrlA"):
    chk("top1_lp_spread_note" in ((S.get("S3_report_diag") or {}).get(side) or {}), "H4",
        "%s: top1_lp_spread_note подписан (r9/F1)" % side)
chk(all(isinstance(pp.get("chain"), dict) and pp["chain"].get("top_k") == 400
        for pp in (S.get("probes") or [])), "H5",
    "probes: chain — dict с GATE-полями (r10/M1, top_k=%r)"
    % [pp.get("chain", {}).get("top_k") for pp in (S.get("probes") or [])])
chk(all("sent_logit_bias" in pp for pp in (S.get("probes") or []))
    and "sent_logit_bias" in noop_s, "H7",
    "sent_logit_bias в сводке: probes=%r noop=%r (r10/M1)"
    % ([pp.get("sent_logit_bias") for pp in (S.get("probes") or [])],
       noop_s.get("sent_logit_bias")))
# r10/M1: дифф цепей noop vs S4pp-2 — ровно no-op пара (r11/F1: top_n_sigma=-1.0
# на обеих руках = серверный дефолт, noop-плечо чистый no-op по проводу)
if noop_s.get("chain") and len(S.get("probes") or []) == 2:
    p2 = S["probes"][1].get("chain") or {}
    diff = {k for k in set(p2) | set(noop_s["chain"])
            if p2.get(k) != noop_s["chain"].get(k)}
    chk(diff == {"top_k", "dry_multiplier"}, "H8",
        "дифф цепей noop vs S4pp-2 ровно no-op пара (r11/F1): %r" % sorted(diff))
# r11/F4 + r12/F9: chain — сверка с РЕАЛЬНЫМ эхом каждой руки, ДВУСТОРОННЯЯ
# (не реконструкция с реконструкцией): (a) каждое поле chain == поле echo того
# же запроса (включая новые ключи sent_extra); (b) каждое поле echo, не вошедшее
# в chain, — либо серверное поле (белый список), либо поле sent_extra.
# r13/F1: белый список расщеплён — logit_bias (реально ОТПРАВЛЯЕТСЯ как
# sent_logit_bias) из белого списка УБРАН: сверяется вперёд, в нормализованном
# виде [[t, b]] ↔ [{token, bias}]; n_probs (20 — в body(), эхо = 20 на всех 24
# строках) вносится в базу и сверяется. xtc_threshold — серверное дефолтное
# поле (0.1), не отправлялось (xtc=0.0 покрыт xtc_probability в базе); 
# cache_prompt — отправляется False, но сервер нормализует эхо в null (не
# сверяемо по равенству) — оба остаются в _SERVER_ONLY.
_CHAIN_BASE = {"temperature": 0.0, "top_k": 400, "top_p": 1.0, "min_p": 0.0,
               "typical_p": 1.0, "xtc_probability": 0.0, "repeat_penalty": 1.0,
               "seed": 8181, "presence_penalty": 0.0, "frequency_penalty": 0.0,
               "repeat_last_n": 0, "dry_multiplier": 1.0, "dry_base": 1.0,
               "top_n_sigma": -1.0, "mirostat": 0.0, "n_probs": 20}
_SERVER_ONLY = {"post_sampling_probs", "cache_prompt",
                "backend_sampling", "samplers",
                "speculative.types", "xtc_threshold"}
# r13/F8: объём зафиксирован: 24×16 (база) + 24 (logit_bias) + 2×2 (sent_extra
# у S2a/S2b) = 412; «пропавшая рука» не пройдёт H10
_NCHK_EXPECT = N_PLAN * (len(_CHAIN_BASE) + 1) + 4
_mism, _unk, _nchk, _per = [], [], 0, []
for r in (A.get("requests") or []):
    c = dict(_CHAIN_BASE)
    c.update(r.get("sent_extra") or {})
    echo = r.get("echo") or {}
    n_req = 0
    for f, v in c.items():  # c = база ∪ sent_extra: все присланные поля
        _nchk += 1
        n_req += 1
        if echo.get(f) != v:
            _mism.append((r.get("tag"), f, v, echo.get(f)))
    # r13/F1: logit_bias — вперёд, нормализованно (None → [])
    sent = r.get("sent_logit_bias")
    exp_bias = ([[t, b] for t, b in sent] if sent else [])
    lb = echo.get("logit_bias")
    got_bias = ([[d.get("token"), d.get("bias")] for d in lb]
                if isinstance(lb, list) else None)
    _nchk += 1
    n_req += 1
    if got_bias != exp_bias:
        _mism.append((r.get("tag"), "logit_bias", exp_bias, got_bias))
    for f in echo:  # обратный проход: неизвестные поля echo
        if f not in c and f not in _SERVER_ONLY and f != "logit_bias":
            _unk.append((r.get("tag"), f))
    _per.append(n_req)
chk(not _mism and not _unk and len(R) == N_PLAN and _nchk == _NCHK_EXPECT, "H10",
    "chain == реальное эхо, двусторонне, объём зафиксирован (r11/F4, r12/F9, r13/F1/F8): "
    "запросов %d, %d полей (ожидается %d; макс %d/запрос), mismatches=%r, unknown=%r"
    % (len(R), _nchk, _NCHK_EXPECT, max(_per) if _per else 0,
       _mism[:3], _unk[:3]))
# r10/M2: informative согласовано с report_nonvacuous (and-логика):
# informative ⇒ nonvacuous всегда; при живых echo-флагах — равно
nonvac = (S.get("S2_pair_diff") or {}).get("report_nonvacuous")
inf = (S.get("S2_pair") or {}).get("informative")
chk((inf in (False, None)) or nonvac is True, "H9",
    "informative => report_nonvacuous (r10/M2: %r / %r)" % (inf, nonvac))
bA = [pp for pp in (S.get("probes") or []) if pp.get("verdict") == "branchA"]
if bA and noop_s.get("verdict") == "third_branch":
    chk(all(pp["verdict_mechanism"].startswith("M1 НЕ УСТАНОВЛЕН") for pp in bA), "H6",
        "verdict_mechanism согласован с reading (r9/N2): %r"
        % (bA[0].get("verdict_mechanism") or "")[:60])
# r15/M2: финальная динамическая note (g12/spread) — ПЕРЕСОБИРАЕТСЯ из значений
# этого артефакта и сравнивается ПОЛНЫМ СТРОКОВЫМ ЧЕКОМ: подмена текста note
# (цифр, ratio) больше не пройдёт аудит
_g12 = None
for r in R:
    if r.get("tag") == "S4pp-2":
        _g12 = (r.get("base") or {}).get("g12")
_sp_p = ((S.get("S3_report_diag") or {}).get("probe") or {}).get("top1_lp_spread")
_sp_c = ((S.get("S3_report_diag") or {}).get("ctrlA") or {}).get("top1_lp_spread")
_exp_note = None
if _g12 and _sp_p:
    _exp_note = ("отношение решающего флипа к шуму, этот артефакт: g12(S4pp-2)=%s, "
                 "top1_lp_spread: probe=%s, ctrlA=%s — ratio ~%dx "
                 "(g12/spread probe); spread — SYCL-шум без зафиксированной "
                 "границы (r8/F1), значение вычислено из этого артефакта; "
                 "флип остаётся одиночным во всех прогонах (24/24, эмиссии "
                 "идентичны, сверка в пакетах)"
                 % (_g12, "%.3e" % _sp_p,
                    ("%.3e" % _sp_c) if _sp_c else "н/д", round(_g12 / _sp_p)))
_notes = A.get("notes") or []
chk(_exp_note is not None and _notes and _notes[-1] == _exp_note, "H11",
    "финальная note == пересборка из значений артефакта, строковый чек (r15/M2): %r"
    % (_notes[-1][:60] if _notes else None))

print("\n=== ИТОГ: %s (%d fail)" % ("PASS" if not fails else "FAIL", len(fails)))
for f in fails:
    print("  FAIL", f[0], f[1])
sys.exit(0 if not fails else 1)
