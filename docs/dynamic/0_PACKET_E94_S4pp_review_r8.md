# ПАКЕТ: внешнее ревью результата E94/S4pp v2 (r8 — ремедиация BLOCK r7). Репозиторий НЕ читать — всё нужное здесь.

Предыдущее ревью результата (r7, `2_REVIEW_E94_S4pp_exec_r7.md`): BLOCK, 2 MAJOR + 4 MINOR.
Ремедиация предрегистрирована (spec, раздел «ИСПРАВЛЕНИЯ ПО r7-РЕВЬЮ», коммит d783f59 —
ДО кода и прогона), исполнена, прогон v2 (RUN_RC=0, 24/24, wall=48s), само-ревью
`e94_s4pp_audit.py` — PASS 0 fail (33 чека). Ниже: код ремедиации (A–D), выжимка
артефакта (E), построчные эхо (F — ответ на r7/MINOR-3/4), сверка findings r7 (G),
утверждения исполнителя для проверки (H).

## A. Запрос S4pp-noop (MAJOR-1: изоляция третьей ветки)

Константа no-op цепи:
```
   1| # r7/MAJOR-1: заведомо no-op цепь для S4pp-noop (доказательства в предрегистрации,
   2| # раздел ИСПРАВЛЕНИЯ ПО r7-РЕВЬЮ): top_k=0 (top_k_impl: k <= 0 → return,
   3| # llama-sampler.cpp:325), dry_multiplier=0.0 (dry_enabled = multiplier != 0.0, :3648),
   4| # top_n_sigma=0.0 (n <= 0.0f → empty, :3301); xtc=0.0 уже в body() (p <= 0 → empty, :2417).
   5| NOOP_EXTRA = dict(top_k=0, dry_multiplier=0.0, top_n_sigma=0.0)
```
Вызов (порядок: сразу после S4pp-2):
```
   1|            ban_tok=bases["probe2"]["emit"])
   2|         # r7/MAJOR-1: изоляция третьей ветки (no-op цепь на адресе S4pp-2)
   3|         do("S4pp-noop", "r7/MAJOR-1: бан эмиссента на адресе S4pp-2 при заведомо no-op "
   4|                         "цепи (top_k=0, dry=0.0, top_n_sigma=0.0)", "probe2", "noop",
```

Чтения (предрегистрация, spec):
- emit ∉ {4383, 3039} → третья ветка НЕ подтверждена, ветка §A для S4pp-2 в силе (M1/M2).
- emit == 3039 (top1) → ТРЕТЬЯ ВЕТКА ЖИВА, «M1 установлен» по S4pp-2 СНИМАЕТСЯ.
- emit == 4383 / manip_ok=false → невалидна.
Ожидание по канону S5/S6: emit == 3039.

Классификатор (ветки top1ban + noop):
```
   1|     if kind == "top1ban":
   2|         # r4/MINOR-2: на этом адресе A и так не была бы эмиссентом (здесь флип), поэтому
   3|         # строка НЕ доказывает применимость смещения (это делают ctrl-ветки) и не различает
   4|         # M1/M2 — она про то, на какой набор действует ban.
   5|         # r7/MINOR-2: «сработал» ничем не был обеспечен — эмиссент 310 = базовый B,
   6|         # поэтому чтение: «применён, эффекта нет, контроль применимости этой строкой не
   7|         # покрыт» (manip_ok — в строке и в сводке).
   8|         if e == A:
   9|             return ("ctrl_fail", "вершина отчёта осталась эмиссентом despite ban — "
  10|                                "невалидно для всего бана: смещение не применилось")
  11|         if e == B:
  12|             return ("nondec_snip", "бан вершины отчёта применён (manip_ok по эхо), эффекта на "
  13|                                    "эмиссент нет (эмиссент %s = базовый B %s); НЕ РАЗЛИЧАЕТ "
  14|                                    "и НЕ контроль применимости: на этом адресе A и так не "
  15|                                    "эмитировалась, вывод о механизме запрещён" % (e, B))
  16|         return ("nondec_snip", "бан вершины отчёта применён (manip_ok по эхо), эмиссент %s "
  17|                                "∉ {A %s, B %s} — новый класс, описывать отдельно"
  18|                                % (e, A, B))
  19|     if kind == "noop":
  20|         # r7/MAJOR-1: изоляция третьей ветки. С no-op цепью сдвигать эмиссию нечему:
  21|         # возврат на вершину отчёта = третья ветка жива; эмиссент вне {B, A} = третья
  22|         # ветка не подтверждена, ветка §A для S4pp-2 в силе.
  23|         if e == A:
  24|             return ("third_branch", "ТРЕТЬЯ ВЕТКА ЖИВА: с no-op цепью эмиссент вернулся на "
  25|                                     "вершину отчёта %s — штатная цепь (dry=1.0) была тем, "
  26|                                     "что сдвигало эмиссию; «M1 установлен» по S4pp-2 "
  27|                                     "СНИМАЕТСЯ, строка читается «сдвиг цепью (не "
  28|                                     "различает)»" % A)
  29|         return ("noop_out", "ТРЕТЬЯ ВЕТКА НЕ ПОДТВЕРЖДЕНА: с no-op цепью сдвигать эмиссию "
  30|                             "нечему (эмиссент %s ∉ {ban %s, top1 %s}); различающая ветка §A "
  31|                             "для S4pp-2 в силе (вилка M1/M2)" % (e, ban_tok, A))
```

## B. Вердикт — пересказ по точкам (MAJOR-2)

```
   1|         # r7/MAJOR-2: агрегат для пар кроме (undec, undec) предрегистрирован не был —
   2|         # вердикт = пересказ по точкам (тексты зафиксированы в предрегистрации r7);
   3|         # r7/MAJOR-1: точка S4pp-2 сопровождается чтением S4pp-noop (третья ветка).
   4|         noop_v = (find("S4pp-noop") or {}).get("verdict")
   5| 
   6|         def _pt(p, i):
   7|             if p.get("verdict") == "M1":
   8|                 base = "сработала решающая ветка §A (эмиссент %r вне {B, A})" % p.get("emit")
   9|                 if i != 1:
  10|                     return base
  11|                 if noop_v == "third_branch":
  12|                     return base + ("; НО ТРЕТЬЯ ВЕТКА ЖИВА: S4pp-noop (no-op цепь) вернул "
  13|                                    "эмиссию на вершину отчёта — «M1» по этой строке "
  14|                                    "СНЯТ, читается «сдвиг цепью (не различает)»")
  15|                 if noop_v == "noop_out":
  16|                     return base + ("; изолирован от третьей ветки: S4pp-noop (no-op "
  17|                                    "цепь) не вернул эмиссию на вершину отчёта")
  18|                 return base + (" с оговоркой: не изолирован от третьей ветки (сдвиг "
  19|                                "цепью) — см. строку S4pp-noop")
  20|             return ("не различает (эмиссент = вершина отчёта %r; совместимо с M1 и M2)"
  21|                     % p.get("top1"))
  22|         if pv.count("M1") == 2:
  23|             verdict = ("S4pp-1: %s; S4pp-2: %s; обе точки дали эмиссент вне {B, A}. "
  24|                        "Агрегат (M1, M1) не предрегистрирован — это пересказ, не "
  25|                        "агрегатное чтение." % (_pt(probes[0], 0), _pt(probes[1], 1)))
  26|         else:
  27|             t = [_pt(p, i) for i, p in enumerate(probes)]
  28|             verdict = ("S4pp-1: %s; S4pp-2: %s. Агрегат для этой пары не "
  29|                        "предрегистрирован — это пересказ, не агрегатное чтение."
  30|                        % (t[0], t[1]))
  31|     elif all(v == "undec" for v in pv):
  32|         verdict = ("не разрешено: обе точки дали вершину отчёта — совместимо с M1 и M2; "
  33|                    "вывод о механизме запрещён; рука B — по пункту 2 гейта (0_DECISION_E94_ARM_B.md)")
  34|     else:
  35|         verdict = "нет данных (части запросов не выполнились), halt=%s" % halt
  36| 
  37|     s3 = [r for r in recs if r["tag"].startswith("S3-")]
  38|     k1 = find("S3-k1-probe") or {}
```

## C. Tamper-репетиция инлайн (MINOR-5): доказательство fail-fast В АРТЕФАКТЕ

```
   1|     # r7/MINOR-5: доказательство fail-fast (мир tamper) — в артефакте, а не только в логе.
   2|     # Копия скрипта с заведомо неверным литералом пробы обязана умереть в load_addr
   3|     # ДО лонча сервера: rc=3, halt_reason="crash", AssertionError в trace, ноль
   4|     # процессов llama-server, ноль запросов. Рекурсии нет: tamper-копия крашится в
   5|     # load_addr ДО своей tamper-ветки. При любом расхождении — halt ДО запросов.
   6|     tamper = {}
   7|     tpdir = "/tmp/e94_s4pp_tamper"
   8|     try:
   9|         os.makedirs(tpdir, exist_ok=True)
  10|         src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
  11|                                 "exp94_s4pp.py"), encoding="utf-8").read()
  12|         lit = "exp_emit=310, exp_top1=318, exp_rank=" + "2"  # split: полный литерал не
  13|         # встречается в источнике (там только ADDR) — иначе count(lit) == 2 и гейт
  14|         # ложно «не уникален»
  15|         if src.count(lit) != 1:
  16|             tamper = dict(ok=False, why="литерал пробы не уникален в скрипте")
  17|             halt = "tamper-репетиция: литерал пробы не уникален — гейт не проверяем"
  18|         else:
  19|             tp = os.path.join(tpdir, "tampered_exp.py")
  20|             with open(tp, "w", encoding="utf-8") as f:
  21|                 f.write(src.replace(lit, "exp_emit=999999, exp_top1=318, exp_rank=2")
  22|                          .replace('OUT = "results/e94"', 'OUT = "%s"' % tpdir))
  23|             before = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True,
  24|                                     text=True).stdout.split()
  25|             p = subprocess.run([sys.executable, tp],
  26|                                env=dict(os.environ,
  27|                                         PYTHONPATH=os.path.dirname(os.path.abspath(__file__))),
  28|                                capture_output=True, text=True, timeout=300)
  29|             after = subprocess.run(["pgrep", "-x", "llama-server"], capture_output=True,
  30|                                    text=True).stdout.split()
  31|             art_t = ({} )
  32|             if os.path.exists(os.path.join(tpdir, "S4pp.json")):
  33|                 art_t = json.load(open(os.path.join(tpdir, "S4pp.json")))
  34|             out = (p.stderr or "") + (p.stdout or "")
  35|             tamper = dict(
  36|                 rc=p.returncode,
  37|                 halt_reason=art_t.get("halt_reason"),
  38|                 assertion="AssertionError" in out and "уплыл" in out,
  39|                 servers_before=before, servers_after=after,
  40|                 has_requests_key=("requests" in art_t),
  41|             )
  42|             tamper["ok"] = (p.returncode == 3 and art_t.get("halt_reason") == "crash"
  43|                             and tamper["assertion"] and before == after
  44|                             and not tamper["has_requests_key"])
  45|             if not tamper["ok"]:
  46|                 halt = ("tamper-репетиция не прошла (ждём rc=3/crash/AssertionError/"
  47|                         "0 серверов/0 запросов): %r" % tamper)
  48|     except Exception as e:
  49|         tamper = dict(ok=False, why=repr(e)[:200])
  50|         halt = "tamper-репетиция упала: %r" % e
  51|     print("[tamper] %s" % tamper, flush=True)
  52| 
```

## D. Сводка: spread + S4pp_0 (MINOR-1, MINOR-2)

```
   1|     def s3diag(rows, sfx):
   2|         rs = [r for r in rows if r["tag"].endswith(sfx)]
   3|         ntl = sorted({r.get("n_tl") for r in rs}, key=lambda x: (x is None, x))
   4|         heads = sorted({str(r.get("ids_head") or []) for r in rs})
   5|         # r7/MINOR-1: численная нестабильность (top1_lp гуляет по k до 5.43e-4) —
   6|         # записано как пол для любых g12-сравнений этой руки.
   7|         lps = [r.get("top1_lp") for r in rs if r.get("top1_lp") is not None]
   8|         return dict(
   9|             n_rows=len(rs), distinct_n_tl=ntl, distinct_ids_head=len(heads),
  10|             top1_lp_spread=(max(lps) - min(lps)) if lps else None,
  11|             distinct_emits=sorted({r.get("emit") for r in rs}, key=lambda x: (x is None, x)),
  12|             report_changed_by_topk=bool(len(ntl) > 1 or len(heads) > 1),
  13|             reading=("отчёт меняется от top_k — свип валиден как проверка ширины"
  14|                      if (len(ntl) > 1 or len(heads) > 1) else
  15|                      "ОТЧЁТ НЕ МЕНЯЕТСЯ от top_k (n_tl=%s при всех k, порядок id один) — "
  16|                      "свип не показал, что ширина применяется; при таком диагнозе "
```
```
   1|         S4pp_0=dict(emit=(find("S4pp-0") or {}).get("emit"),
   2|                     manip_ok=(find("S4pp-0") or {}).get("manip_ok"),  # r7/MINOR-2
   3|                     verdict=(find("S4pp-0") or {}).get("verdict"),
   4|                     reading=(find("S4pp-0") or {}).get("reading")),
   5|         S4pp_noop=dict(emit=(find("S4pp-noop") or {}).get("emit"),
   6|                        top1=(find("S4pp-noop") or {}).get("top1"),
   7|                        manip_ok=(find("S4pp-noop") or {}).get("manip_ok"),
   8|                        verdict=(find("S4pp-noop") or {}).get("verdict"),
   9|                        reading=(find("S4pp-noop") or {}).get("reading")),
  10|         repro=dict(emit=(find("S4pp-repro") or {}).get("emit"),
```

## E. Выжимка артефакта results/e94/S4pp.json (числа как в файле)

- exp=E94_S4pp, started_local=2026-11-12 13:50, wall_s=48.0, halt_reason=null,
  requests_n=24, requests_ok=24.
- **summary.verdict**: «S4pp-1: не различает (эмиссент = вершина отчёта 318; совместимо
  с M1 и M2); S4pp-2: сработала решающая ветка §A (эмиссент 5522 вне {B, A}); НО ТРЕТЬЯ
  ВЕТКА ЖИВА: S4pp-noop (no-op цепь) вернул эмиссию на вершину отчёта — «M1» по этой
  строке СНЯТ, читается «сдвиг цепью (не различает)». Агрегат для этой пары не
  предрегистрирован — это пересказ, не агрегатное чтение.»
- S4pp-1: emit=318, top1=318, rank=1, manip_ok=true, verdict=undec.
- S4pp-2: emit=5522, top1=3039, rank=4, manip_ok=true, verdict=M1 (строка; чтение
  «M1» снято вердиктом по S4pp-noop — см. G1).
- **S4pp-noop: emit=3039, top1=3039, rank=1, manip_ok=true, verdict=third_branch.**
- S4pp-ref: emit=310 (=B), verdict=ref_ok. S4pp-repro: emit=8810, repro_ok.
- S4pp-0: emit=310, manip_ok=true, verdict=nondec_snip («бан применён, эффекта нет,
  контроль применимости этой строкой не покрыт»).
- Контроли бана (4): все manip_ok=true, moved=true (эмиссент ушёл с вершины).
- S2_pair: S2a emit=310 n_tl=0, S2b emit=310 n_tl=0 (post_sampling_probs=true, T=0 →
  пустой отчёт с обеих сторон; differs=false).
- S3 свип (12 строк): probe emit=310 при всех k∈{1,2,3,4,8,400}; ctrlA emit=488 при
  всех k. S3_report_diag: report_changed_by_topk=false на обоих адресах;
  **top1_lp_spread: probe=3.33e-04, ctrlA=1.95e-04** (r7/MINOR-1).
- **S3_control_expectation: expected_id=488, n_moves=0, moved=[]** (r7/MINOR-3).
- **k1_note: probe_addr «k=1 на адресе пробы дал 310; адрес конфаунден ... НЕ диагностика
  ветки k=1», decisive «ветка k=1 на контроле даёт вершину отчёта — безусловного дефекта
  нет»** (r7/MINOR-3).
- gate2: probe_dependence=false, control_dependence=false, gate2_met=false,
  probe_emits_by_k={1:310,2:310,3:310,4:310,8:310,400:310}, gate2_caveat про
  S3_report_diag (как в v1).
- **tamper_rehearsal: rc=3, halt_reason="crash", assertion=true (AssertionError + «уплыл»
  в stderr), servers_before=[], servers_after=[], has_requests_key=false, ok=true.**

## F. Построчные эхо (ответ на r7/MINOR-4: приписка «эхо S4pp-2» доказуема)

S4pp-2 (стандартная цепь):
```
{
 "temperature": 0.0,
 "top_k": 400,
 "top_p": 1.0,
 "min_p": 0.0,
 "typical_p": 1.0,
 "xtc_probability": 0.0,
 "seed": 8181,
 "cache_prompt": null,
 "backend_sampling": false,
 "post_sampling_probs": false,
 "n_probs": 20,
 "logit_bias": [
  {
   "bias": -100.0,
   "token": 4383
  }
 ],
 "samplers": [
  "penalties",
  "dry",
  "top_n_sigma",
  "top_k",
  "typ_p",
  "top_p",
  "min_p",
  "xtc",
  "temperature"
 ],
 "speculative.types": "none",
 "dry_multiplier": 1.0,
 "dry_base": 1.0,
 "top_n_sigma": -1.0,
 "repeat_penalty": 1.0,
 "presence_penalty": 0.0,
 "frequency_penalty": 0.0,
 "repeat_last_n": 0,
 "mirostat": 0,
 "xtc_threshold": 0.10000000149011612
}
```
sent_extra=null (стандартный payload)

S4pp-noop (no-op цепь — отправлено и подтверждено сервером):
```
{
 "temperature": 0.0,
 "top_k": 0,
 "top_p": 1.0,
 "min_p": 0.0,
 "typical_p": 1.0,
 "xtc_probability": 0.0,
 "seed": 8181,
 "cache_prompt": null,
 "backend_sampling": false,
 "post_sampling_probs": false,
 "n_probs": 20,
 "logit_bias": [
  {
   "bias": -100.0,
   "token": 4383
  }
 ],
 "samplers": [
  "penalties",
  "dry",
  "top_n_sigma",
  "top_k",
  "typ_p",
  "top_p",
  "min_p",
  "xtc",
  "temperature"
 ],
 "speculative.types": "none",
 "dry_multiplier": 0.0,
 "dry_base": 1.0,
 "top_n_sigma": 0.0,
 "repeat_penalty": 1.0,
 "presence_penalty": 0.0,
 "frequency_penalty": 0.0,
 "repeat_last_n": 0,
 "mirostat": 0,
 "xtc_threshold": 0.10000000149011612
}
```
sent_extra:
```
{
 "top_k": 0,
 "dry_multiplier": 0.0,
 "top_n_sigma": 0.0
}
```
(серверный дефолт top_n_sigma=-1.0 виден на S4pp-2: поле в каноническом payload не
отправлялось; на S4pp-noop отправлено 0.0 и эхо подтверждает 0.0.)

## G. Сверка findings r7

- G1 (MAJOR-1, третья ветка): исполнено — запрос S4pp-noop, чтения предрегистрированы
  ДО прогона. ИСХОД: emit=3039=top1 → ТРЕТЬЯ ВЕТКА ЖИВА, «M1 установлен» по S4pp-2
  снят (это видно в verdict и reading строки). Ожидание по канону S5/S6 подтвердилось.
  Оговорка: машинный verdict строки S4pp-2 остался «M1» (чтение §A); интерпретация
  снята в вердикте/reading. Если ревьюер требует изменить и машинный verdict строки —
  это MINOR уровня.
- G2 (MAJOR-2, агрегат): исполнено — verdict = пересказ по точкам, «агрегат не
  предрегистрирован», «противоречит» отсутствует (проверено аудитом C1/C1b).
- G3 (MINOR-1, spread): исполнено — top1_lp_spread в S3_report_diag.
- G4 (MINOR-2, S4pp_0): исполнено — manip_ok в сводке, reading переформулирован.
- G5 (MINOR-3, n_moves/k1_note в выжимке): исполнено — поля в артефакте (были и в v1),
  теперь В ПАКЕТЕ (E).
- G6 (MINOR-4, построчное эхо): исполнено — эхо в каждой строке requests (v1 тоже),
  теперь В ПАКЕТЕ (F) + расширено (dry_multiplier, top_n_sigma, penalties, mirostat).
- G7 (MINOR-5, tamper в артефакт): исполнено — tamper_rehearsal в артефакте (C, E),
  прогоняется инлайн перед лончем сервера; при сбое — halt ДО запросов.

## H. Утверждения исполнителя, которые надо проверить (только по пакету)

1. No-op цепь действительно no-op: top_k=0 (top_k_impl k<=0→return), dry=0.0
   (dry_enabled=multiplier!=0.0), top_n_sigma=0.0 (n<=0→empty), xtc=0.0 (p<=0→empty),
   penalties нейтральны (1.0/0.0/0.0/0) — все подтверждены эхом F (сервер отдал ровно
   эти значения). Верно ли, что при T=0 и таком эхе цепь не может сдвигать эмиссию?
2. Исход emit=3039 (top1) на S4pp-noop при manip_ok=true читается как «третья ветка
   жива». Есть ли в этом чтении дыра (например, бан 4383 сам по себе мог вернуть
   эмиссию на 3039, не цепь)?
3. «M1» в verdict строки S4pp-2 при снятой интерпретации — допустимо ли оставить
   машинный verdict «M1» (чтение §A) и снять его только в сводке, или это
   самопротиворечие артефакта?
4. Числа E: согласованы ли с построчными фактами (emit 5522/3039/310/318/488;
   n_tl=20; spread; n_moves=0)?
5. tamper_rehearsal: достаточно ли доказательств fail-fast (rc=3, crash,
   AssertionError, 0 серверов до/после, 0 запросов) или чего не хватает?

## I. Формат ответа (строго, <= 4000 символов, цитаты <= 2 строк)

```
ВЕРДИКТ: PASS | PASS_WITH_MINORS | BLOCK
H1: <да/нет/частично + <=3 строки>
H2: <то же>
H3: <то же>
H4: <то же>
H5: <то же>
FINDINGS: - [BLOCK|MAJOR|MINOR] <что> <что делать>   (<=8)
```
Отвечать только по пакету. Если данных нет — так и написать. Маркер финальной
строки: S4PP_R8_DONE.
