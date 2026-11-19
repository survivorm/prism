# AT0_3 — СТРОГОЕ селф-ревью кода SYCL (r4-пре-ревью)

Дата: 19.11.2026. Ревизия предмета: рабочее дерево после закрытия findings r3 +
doc-ревью (состояние `git status` на момент ревью: M `sycl_prism/py/prism_sycl.py`,
M `sycl_prism/py/test_new_ops.py`, M `sycl_prism/src/prism_sycl.cpp`,
M `docs/TZ_sycl_new_ops.md`, ?? `docs/SYCL.md`, ?? `results/AT0_3/{tuning,
eigtop_eigbot_jcache,gagg_composition,svd_top_hdocs}.json`).

Метод: чтение `py/prism_sycl.py` (815 стр., целиком), `src/prism_sycl.cpp`
(1703 стр., целиком), `py/prism_sycl_ref.py` (арности), `prism/backend.py`
(потребитель), `tests/test_backend.py`; затем ЭМПИРИЧЕСКИЕ проверки на живом
iGPU (мелкие матрицы, секунды) — каждый пункт ниже имеет воспроизведение,
а не рассуждение. Числа — из вывода команд (приведены).

Селф-ревью НЕ заменяет внешнее: находки M1/M2 → правки → внешнее ревью r4.

---

## MAJOR

### M1. `svd_top` возвращает view с ОТРИЦАТЕЛЬНЫМИ шагами → `Backend.svd_top` падает

`py/prism_sycl.py:701,708,730,731`:
```
U_out  = U0[:, ::-1]        # (m<n): шаг по столбцам −4
Vh_out = V[:, ::-1].T       # (m<n): (−4, 16)
U_out  = U[:, ::-1]         # (m≥n)
Vh_out = V0[:, ::-1].T      # (m≥n)
```
`torch.from_numpy` (документированный потребитель —
`prism/backend.py:172`) НЕ принимает отрицательные шаги.

Воспроизведение (живой iGPU):
```
$ python /tmp/chk_svd_strides.py        # A (8,12), r=4
U:  shape=(8, 4) strides=(16, -4)  C_contig=False
Vh: shape=(4, 12) strides=(-4, 16)  C_contig=False
torch.from_numpy(U) -> ValueError At least one stride in the given numpy array is negative

$ PRISM_BACKEND=sycl python -c "... b.svd_top(torch.randn(10,6), 4) ..."
B.svd_top sycl -> ValueError : At least one stride ... negative strides are not supported
```
Существующий тест в репо КРАСНЫЙ (не входил в AT0_3-сьют `py/test_new_ops.py`):
```
$ pytest tests/test_backend.py -q -k svd_top
FAILED tests/test_backend.py::test_sycl_svd_top - ValueError: At least one stride ...
1 failed, 1 passed, 11 deselected
```
Почему не поймано: `test_svd_top_hdocs`/`test_svd_top_degenerate` сравнивают
ЗНАЧЕНИЯ (numpy), не флаги; интеграционный тест backend не запускался.

Числа при этом верны (проверено, r = min(m,n)):
```
(8,12) r=8:  rel_S=6.3e-08 recon_rel=1.0e-07 orthU=6e-08 orthV=4e-07
(12,8) r=8:  rel_S=6.2e-08 recon_rel=1.2e-07
(16,16) r=16: rel_S=2.2e-07 recon_rel=1.2e-07
(1,1) r=1:   rel_S=2.3e-08, C_contig=True (вырожденная размерность маскирует баг)
```
Fix: `np.ascontiguousarray(...)` на U и Vh в ОБЕИХ ветках (+ тест-ассерт
`flags["C_CONTIGUOUS"]` и вызов `torch.from_numpy` в `test_svd_top_hdocs`).
Замечание: reference-ветка (`ref.svd_top`) возвращает положительные шаги —
то есть расхождение бэкендов по контракту выходов.

### M2. `svd_top` заявляет `m,n ≤ 1024`, но внутренний `eigjac` — `n ≤ 256`

`py/prism_sycl.py:680` (guard svd_top: `1<=m<=1024, 1<=n<=1024`) против
`py/prism_sycl.py:603` (`eigjac`: `1<=n<=256`). Ложный контракт в четырёх местах:
`docs/TZ_sycl_new_ops.md:311`, `docs/SYCL.md:128`, docstring `ps.svd_top`,
docstring `Backend.svd_top` (`prism/backend.py:159`).

Воспроизведение:
```
svd (300,400) r=2 -> RuntimeError: prism_sycl: eigjac: PS_ERR_ARG — ограничения
   (1<=batch<=65536, 1<=n<=256, ...), получено batch=1 n=300 ...
svd (257,300) r=2 -> то же
svd (255,255) r=4 -> OK (rel_S=1.0e-06)
```
Падение громкое, но неинформативное (ошибка «чужого» слоя) и противоречит TZ:
TZ §1 (стр. 24–25) прямо описывает путь «брать МАЛУЮ сторону + lift:
A [128,1024] — G = A·Aᵀ (128×128), eigjac(G) — никаких eig 1024×1024».
То есть честный лимит — `max(m,n) ≤ 1024` И `min(m,n) ≤ 256` (малая сторона
идёт в eigjac, большая — в gemm). A[128,1024] этому удовлетворяет (и покрыт
тестом), 300×400 — нет.

Fix: guard svd_top → `1<=m<=1024, 1<=n<=1024, min(m,n)<=256` + сообщение на
уровне svd_top; синхронно TZ §2.4, SYCL.md §2.4, оба docstring; тест на границу
(256/257).

---

## MINOR

### m1. `eigjac(curve=...)`: f32/список → ТИХИЙ no-op при контракте «заполняется на месте»

`py/prism_sycl.py:611`: `c_arr = np.asarray(curve, dtype=np.float64)` — для
f32-массива/списка создаёт КОПИЮ, заполняется копия, буфер пользователя не
меняется; исключения нет.
```
curve f32: заполнена? НЕТ (silent no-op)
curve f64: ненулевых точек = 12, первые 4: [0.535881 0.23357 0.065977 0.005494]
```
Fix: требовать `isinstance(curve, np.ndarray) and curve.dtype == np.float64`
(как уже сделано для C-contiguity) + тест «f32 → ValueError».

### m2. Комбинированные лимиты памяти не проверяются (падает поздно и неинформативно)

- `gemm`: per-dim лимиты допускают M=65536, N=262144 ⇒ `M*N*4 ≈ 68 GB` (плюс
  `K*N*4`) — `malloc_device failed` без указания объёма (`cpp:1078`).
- `eigjac`: `batch*n*n*4` на устройстве ДВА буфера (dA, dVT) + три таких же
  host-вектора (`vt32`, `vecs32`, `out_vecs`, `cpp:1631-1644`): для
  batch=65536, n=256 — 17 GB × 3.
- `eigbot` (py:518-527): host-Cholesky в f64 — `a.astype(f64)` + `mu*I` +
  сумма + `L` + `L32` ≈ 8× размера f32-входа; batch=16384, n=128 → ~8 GB пик.
  Не задокументировано.
Fix: численный guard (`batch*n*n ≤ N_max`, `M*N ≤ N_max`) с сообщением в
элементах/байтах; в docstring — оценка памяти.

### m3. Формула local-бюджета не соответствует ядру (консервативна, но врёт про состав)

`py/prism_sycl.py:545` и `cpp:1292`: `2*n*r*4 + r*r*4 + 64*4`.
Фактически в `eig_iterate` (`cpp:1121-1123`): `lv, lw` по `n*r` (итого
`2*n*r*4`) + `sh` = `(WG+1)` = 65 элементов (`65*4`). Члена `r²*4` в ядре НЕТ.
Разница: declared − real = `4r² − 4 ≥ 0` при r≥1 ⇒ проверка консервативна
(ложных пропусков нет; при r=1 равенство). Но в доках (SYCL.md:109) формула
выдаётся за точную.
Fix: `2*n*r*4 + (WG+1)*4` в py+cpp+доках (или явная пометка «оценка с запасом»).

---

## NIT

- **n1 (мой regression из r3-фикса):** `cpp:1269` — `hResid[b] = ...` без отступа
  (колонка 0) внутри цикла `for (b)`, между комментарием на 4 пробелах и телом.
- **n2:** дублированный устаревший комментарий у `EIGJAC_CHUNK`
  (`cpp:1357-1362`: «правило TZ: C = наибольшее чанк с t_sweep ≤ 4с → **256**»)
  прямо над корректным блоком (`cpp:1363-1369`: «⇒ C=64»). Взаимоисключающие.
- **n3:** мёртвая ветка `cpp:1577-1581`: `const size_t local_bytes = 0;
  if (local_bytes > 65536) {...}` — всегда false (компилятор вправе предупредить).
- **n4:** `cpp:1546` `std::stable_sort` по λ без защиты от NaN: компаратор
  `a < b` с NaN нарушает strict weak ordering (формально UB). Практически
  проверено — не падает и даёт NaN-выход (`off=[..., nan, ...]`,
  `λ=[nan ...]`, `V` содержит NaN), но защита дешёвая (NaN → в конец).
- **n5:** `py:702` `U0[:, nz] = U_r[:, nz].astype(np.float32)` — лишняя копия
  (U_r уже f32); достаточно `np.asarray`/прямого присваивания.
- **n6 (pre-existing, вне дельты AT0_3):** `permutation_null` при малом N:
  `K = min(K, N-1)` может дать K<9 ⇒ `ks` = [3,4] ⇒ `rho[:, 2:].mean()`
  по пустому → `band = [nan nan]` + RuntimeWarning (тихий NaN).
  Воспроизведение: `N=4, K=96 → band=[nan nan]`.
- **n7 (pre-existing, вне дельты):** `acf_moments(K > N-1)`: SYCL-ветка →
  `RuntimeError: неверные аргументы` (cpp PS_ERR_ARG), reference-ветка →
  `ValueError: acf_moments: K=8 вне диапазона [1, 4] для N=5`. Разные типы и
  качество сообщений между бэкендами.
- **n8:** `eigjac_impl` не проверяет симметрию входа (TZ §2.3 требует
  симметричный A). Не-симметричный A даёт молча неверный результат (ядерный
  sweep использует A[k,i]=A[i,k] как инвариант). Хост-проверка O(n²) дёшева
  относительно самого sweep'а — стоит добавить (или явно записать
  «контракт, не проверяем» в TZ §2.3).

---

## Проверено и ПОДТВЕРЖДЕНО корректным (не находки)

- `_eig_check_args` вызывается до host-Cholesky в eigbot (`py:514`) и внутри
  `_eig_call` (`py:553`) — порядок сохранён, сообщения идентичны, C++ backstop
  не тронут.
- Дедуп не изменил поведение: guard'ы те же (`batch/n/r/max_iter` + local-бюджет).
- `_f32` действительно делает `ascontiguousarray` ⇒ `svd_top`-вход корректен для
  view/`.T`/f64 (`Hc32.T` из теста работает).
- Формулы Jacobi: `θ=(A_jj−A_ii)/(2A_ij)`, `t=sign/(|t|+√(1+t²))`,
  `A'_ii=a−t·bq`, `A'_jj=d+t·bq`, `A'_ij=0` — совпадают с NR; fudge-guard XLA
  (`0.1·eps32·min(|a|,|d|)`) на месте; `continue` не нулирует не-малую пару.
- Двойной MGS в `eig_iterate`: все `wg_sum`/`barrier` вне `if (lane < r)` и вне
  ветки вырожденной колонки (n2v — broadcast) ⇒ рассинхронного барьера нет;
  слот `sh[WG]` (race-фикс 18.11) отделён от partial'ов.
- `gemm` double-buffer: барьеры стоят так, что WAR/RAW по `ta/tb` закрыты
  (запись в буфер идёт после барьера, читавший его раунд уже завершён);
  границы `row < M`, `col0+lc < Nc`, `c0` в индексе C — корректны.
- `gemm` чанкинг: `Nc` пересчитывается от `N-c0`, `nch ≥ TC`; крайний случай
  65536×4096×64 = 34.4 GFLOP (документировано, в 3–5× запасе).
- `eigjac` curve-контракт: `curve[s−1]` per-sweep, fill `[k..len)` = `offdiag[0]`
  — выполнено (12 точек, монотонно убывают); стоп по `max_sweeps` даёт ЧЕСТНЫЙ
  `offdiag=0.38` (не ноль).
- Вырожденные входы: `n=1` (нет ротаций, off=0), нечётное `n=3,5`
  (relλ 7e-08/2e-07, orth 1.2e-07), нулевая матрица (`S=[0,0,0]`, `U=Vh=0` —
  «явный сигнал», как заявлено).
- NaN-политика: `eigjac.offdiag=nan`, `eigtop.resid=nan` — «честный сигнал»,
  не тихий ноль (guard `nrmA != 0.0` работает).
- Детерминизм `eigjac`: один work-item/матрица, фиксированный порядок ротаций,
  float4-путь не меняет per-element арифметику.
- `Dev<T>(q, 0)` (eigtop-ветка без `L`) — p=nullptr, guard `inverse && !dL.p`
  не срабатывает; `memcpy` по нулевому размеру не выполняется.
- Арности reference-веток совпадают: `ref.eig_top/eig_bottom` → 3-tuple
  (vals, vecs, resid) — как SYCL.

---

## Итог селф-ревью

| ID | Класс | Файл:строка | Статус |
|---|---|---|---|
| M1 | MAJOR | py:701,708,730,731 | ПОДТВЕРЖДЁН (красный тест в репо) |
| M2 | MAJOR | py:680 vs 603; TZ:311; SYCL.md:128 | ПОДТВЕРЖДЁН |
| m1 | MINOR | py:611 | ПОДТВЕРЖДЁН |
| m2 | MINOR | cpp:1078,1631; py:518 | ПОДТВЕРЖДЁН (расчёт) |
| m3 | MINOR | py:545; cpp:1292 | ПОДТВЕРЖДЁН (сверка с ядром) |
| n1 | NIT | cpp:1269 | мой regression |
| n2 | NIT | cpp:1357-1369 | дубль/противоречие |
| n3 | NIT | cpp:1577 | мёртвая ветка |
| n4 | NIT | cpp:1546 | NaN vs stable_sort |
| n5 | NIT | py:702 | лишняя копия |
| n6,n7 | NIT | py:~800, 350 | pre-existing, вне дельты |
| n8 | NIT | cpp:1558 | нет проверки симметрии |

Вердикт: **REVISE** — правки M1/M2 обязательны, m1–m3 и n1–n5 закрываются тем
же коммитом; после правок — внешнее ревью r4 (дельта) + перепрогон
`py/test_new_ops.py` И `tests/test_backend.py` (интеграция, её не было в сьюте).

SELFREVIEW_R4_DONE
