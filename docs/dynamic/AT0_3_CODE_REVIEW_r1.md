# КОД-ревью AT0_3 eigjac v5.3 (r1)
Дата: 2026-11-18
Скоп: src/prism_sycl.cpp (eigjac-секция 1305–1618, свип-луп 1541–1567), prism_sycl.h:180-183,
py/prism_sycl.py (eigjac 571–644 + биндинг 148-150), py/test_new_ops.py (eigjac-тесты 171–421);
TZ v5.3 §2.3/2.5/2.7. gemm/eigtop/eigbot не читали, код не запускали (статическое ревью).
VERDICT: REVISE

Findings:
- C1 | MINOR | py/test_new_ops.py:394-421 — test_eigjac_convergence_jcache НЕ реализует TZ §2.7
  m-28 контракт: TZ требует ОДИН прогон max_sweeps=16 с буфером offdiag_curve (L6 d0t0, L23 d0t0)
  и assert sweeps_to(1e-5) ≤ 16; тест — 7 отдельных прогонов (k=1..64) без curve-буфера и с
  более слабым assert off<1e-4 при k=64. Критерий «≤16 sweep'ов до 1e-5» не верифицируется.
  Фикс: один прогон max_sweeps=16 + curve=, assert np.argmax(curve<1e-5) ≤ 15 для двух jcache.
- C2 | MINOR | py/test_new_ops.py:274-281 — curve-регрессия (стоп по max_sweeps) не проверяет,
  что fill не перезаписал per-sweep позиции: ассерты только curve2[0]>0 и curve2[7]>=curve2[8]−1e-15.
  Вariant «fill [0..curve_len) = off[0]» (подозреваемый баг, его НЕТ в коде, cpp:1559) ПРОШЁЛ бы
  этот тест (curve2[0]==curve2[7]==off2[0]>0). Фикс: добавить assert curve2[0] > off2[0]
  (матрица не сходится до sweep 1 — пер-сweep значения обязаны отличаться от финала).
- C3 | NIT | src/prism_sycl.cpp:~1440 + py/prism_sycl.py:604-607 — guard tot>0 (zero-матрица)
  и ValueError curve.size<max_sweeps — designed-сценарии без теста (fudge-guard-тест покрывает
  только ротационный guard). Фикс: маленький тест: A=0 (off=0, без NaN) + curve малого size.
- C4 | NIT | py/test_new_ops.py:198-212 — ассерты jcache-теста (subdist 5e-2, ortho 1e-4,
  off<1e-4) вне выреза по бюджету: relλ-ассерты (1e-3/1e-2, cluster-критерий 5%, py:183-193) и
  max_sweeps=64 подтверждены, остальные допуски сверены со структурой, не построчно.

Checklist А–З:
А. Детерминизм — OK. WI = своя матрица (b=b0+id[0], A_/V_/dOff[b]/dDiag[b·n+i]); локальной
  памяти нет (local_bytes=0, cpp:1517); offdiag-редукция — последовательный i,k-цикл в одном
  WI; чанки [b0,b0+chunk) независимы по матрицам.
Б. Ротация — OK. th=sign(t)/(|t|+√(1+t²)), t=(dd−a)/(2bq), c=1/√(1+th²), s=th·c (cpp:1388-1390);
  кросс-строки/колонки и диагонали a−th·bq / dd+th·bq — проверено числом на [[1,1],[1,3]]:
  формулы дают 2∓√2 (0.5858/3.4142) — точно; V·Pᵀ ВКЛЮЧАЯ блок {i,j}×{i,j} (cpp:1415-1420, без
  skip — задокументированный баг-фикс v5.2). Fudge-guard: 0.1·1.1920929e-7·min(|a|,|dd|),
  skip при bq==0 или |bq|≤tiny — совпадает с TZ.
В. Сортировка+MGS — OK. Порядок в eigjac_impl: sort_eigpairs_f64_host → eigjac_mgs_host
  (cpp:1570-1573) = TZ (1)→(2). stable_sort по индексам, пары (λ,столбец) вместе; MGS 1 проход
  f64, кол по кол (проекции на 0..jj−1 + ре-норм.), write-back f32. Порядок влияет на точный
  результат MGS (зависит от порядка колонок), но не на корректность — TZ-порядок соблюдён.
Г. Sweep-луп и стоп — OK. После каждого sweep: D2H off64 (cpp:1547-1548), стоп когда ВСЕ
  off64[b] < tol ИЛИ max_sweeps (cpp:1550-1553); curve[sw−1]=off64[0] в лупе (cpp:1549).
  ПОДОЗРЕВАЕМЫЙ БАГ ОТСУТСТВУЕТ: fill — k=(stop_sw>0)?stop_sw:max_sweeps (cpp:1559), т.е. при
  стопе по max_sweeps (stop_sw=0) fill [max_sweeps..curve_len), per-sweep [0..k−1] не
  затираются; curve[k−1]==offdiag[0] — по построению (из того же off64[0], по которому решён
  стоп). TZ F3 соблюдён. (См. C2: тест это не верифицирует.)
Д. f32 device / f64 host — OK. dOff/dDiag — Dev<float> (cpp:1524-1525, 4·batch B); host:
  off32→f64 (cpp:1548), diag32→f64 (cpp:1564-1567) для vals/sort; hOff/hVals f64 (ABI
  double*). offdiag_rel f32-редукция — TZ F11 (bound ~2e-3, максимум лишний sweep) — честно.
Е. C-ABI ↔ python — OK. .h:180-183 ≡ .cpp:1620-1623 ≡ биндинг argtypes [c_p,c_i64×3,
  c_double,c_d,c_p,c_d,c_d,c_i64] (py:148-150; c_p=POINTER(c_float), c_d=POINTER(c_double)
  py:102-103) — 10 аргументов, типы совпадают. curve=None → c_ptr=None (ctypes → NULL),
  c_len=0; C проверяет hCurve!=nullptr (cpp:1509-1511) перед использованием curve_len —
  согласовано; C требует curve_len>=max_sweeps, python требует c_arr.size>=max_sweeps —
  идентично.
Ж. Тесты §2.7 — в основном OK, 2 MINOR (C1, C2). На месте: curve-контракт F3 (неубывание,
  curve[k−1]==off[0], fill, стоп-по-max_sweeps-регрессия), sweep-time guard (n=128 чанки
  {1,16,64,256,1024,4096}, n=256 чанк C/4=256, гейты ≤4с, fail-loud C≥256, py:330-336),
  fudge-guard (a)+(b) designed, det бит-в-бит (vals/vecs/off, array_equal), jcache
  (relλ 1e-3/1e-2, max_sweeps=64; subdist/ortho/off — C4), random-SPD (relλ 1e-3,
  subdist 1e-3, ortho 1e-4, off<1e-4, max_sweeps=64).
З. AGENTS.md-ловушки — NIT (C3). Guard'ы: fudge — протестирован; tot>0 и curve_len<max_sweeps
  — нет (C3). json: eigjac_tuning.json (float(np.median)), eigjac_convergence.json
  (float()/round, None→null) — numpy-типы/NaN не попадают; eigjac_jcache.json — построение
  res в вырезанном хвосте (C4). Сравнений/статистики в C-коде нет — OK.

Краткое обоснование VERDICT (3-5 строк):
C-ядро чистое и соответствует TZ v5.3: математика ротации подтверждена числовой проверкой,
подозреваемый curve-fill баг (перепись [0..curve_len) при стопе по max_sweeps) отсутствует —
k=(stop_sw>0)?stop_sw:max_sweeps (cpp:1559), детерминизм по строению, ABI и python-биндинг
согласованы, f32/f64-расклад по TZ F11. REVISE (не FAIL): функциональных дефектов в коде нет,
но два теста не закрывают TZ §2.7: m-28-контракт «1 прогон, curve, sweeps_to(1e-5)≤16» заменён
слабым замером (C1), а curve-регрессия слепая к ровно тому багу, от которого себя, по комментарию,
защищает (C2). Фиксы — несколько строк в test_new_ops.py, C-код не трогаем.

CODE_REVIEW_AT03_r1_DONE
