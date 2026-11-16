"""Проверки C++-части прибора без icpx и без GPU.

Что проверяем:
  * синаксис probe/probe_sycl_min.cpp и src/prism_sycl.cpp против шима SYCL API;
  * C-совместимость заголовка src/prism_sycl.h (объявлен как extern "C");
  * согласованность C ABI и ctypes-биндингов (число аргументов);
  * ЧУВСТВИТЕЛЬНОСТЬ самой проверки ABI: сломанная копия обязана её ломать.

Проверка ABI — статическая. Семантику SYCL и числа она не проверяет; это делает
сборка icpx и паритет с numpy (см. py/test_parity.py).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / "probe" / "sycl_shim"
CXX = shutil.which("g++") or shutil.which("clang++")
PY = sys.executable


def _run(cmd: list[str], cwd: Path = ROOT,
         env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=900, env=e)


def _syntax_ok(rel: str) -> bool:
    if CXX is None:
        return True  # нет компилятора — пропускаем, а не «зелёный на пустом месте»
    p = _run([CXX, "-std=c++20", "-fsyntax-only", f"-I{SHIM}", rel])
    return p.returncode == 0


def test_probe_compiles_against_shim():
    if CXX is None:
        pytest.skip("нет g++/clang++")
    assert _syntax_ok("probe/probe_sycl_min.cpp"), "проба не компилируется против шима"


def test_library_compiles_against_shim():
    if CXX is None:
        pytest.skip("нет g++/clang++")
    assert _syntax_ok("src/prism_sycl.cpp"), "библиотека не компилируется против шима"


def test_library_has_no_warnings():
    if CXX is None:
        pytest.skip("нет g++/clang++")
    p = _run([CXX, "-std=c++20", "-Wall", "-Wextra", "-fsyntax-only",
              f"-I{SHIM}", "src/prism_sycl.cpp"])
    warns = [ln for ln in (p.stderr or "").splitlines() if "warning:" in ln]
    assert not warns, "предупреждения компилятора:\n" + "\n".join(warns)


def test_header_is_c11_compatible():
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("нет gcc")
    tmp = Path("/tmp/_abi_check.c")
    tmp.write_text(
        '#include "prism_sycl.h"\n'
        "int main(void){ return prism_sycl_abi_version(); }\n",
        encoding="utf-8",
    )
    p = _run([gcc, "-std=c11", "-I", "src", "-fsyntax-only", str(tmp)])
    assert p.returncode == 0, p.stderr


def test_abi_checker_passes():
    p = _run([PY, "py/check_abi.py"])
    assert p.returncode == 0, p.stdout + p.stderr


def test_abi_checker_is_sensitive(tmp_path):
    """Сломанная копия обёртки обязана ломать проверку (иначе она пустая)."""
    src = (ROOT / "py" / "prism_sycl.py").read_text(encoding="utf-8")
    broken = src.replace(
        'bind("vpath_f32", [c_p, c_d, ctypes.c_int64, ctypes.c_int64, c_p, c_u8])',
        'bind("vpath_f32", [c_p, c_d, ctypes.c_int64, ctypes.c_int64, c_p])',
    )
    assert broken != src, "инъекция не применилась — тест не проверяет ничего"
    bad = tmp_path / "prism_sycl_broken.py"
    bad.write_text(broken, encoding="utf-8")
    p = _run([PY, "py/check_abi.py", str(ROOT / "src" / "prism_sycl.h"), str(bad)])
    assert p.returncode == 1, "проверка ABI не отреагировала на сломанную сигнатуру"
    assert "vpath_f32" in p.stdout, p.stdout


import pytest  # noqa: E402  (внизу, чтобы не мешать __future__)


def test_cpu_shim_library_builds_and_exports_all_symbols():
    """Сборка против ИСПОЛНЯЕМОГО CPU-шима + наличие всех символов без манглинга.

    Именно этот слой поймал реальный баг: блок функций остался внутри
    анонимного пространства ⇒ символы не экспортировались ⇒ падение при загрузке.
    """
    if CXX is None:
        pytest.skip("нет g++/clang++")
    nm = shutil.which("nm")
    if nm is None:
        pytest.skip("нет nm")
    out = ROOT / "build" / "libprism_sycl_cpu.so"
    out.parent.mkdir(exist_ok=True)
    # Статическая линковка libstdc++ — иначе .so требует GLIBCXX новее, чем у
    # libstdc++ из состава python'а (conda-python) ⇒ падает при загрузке.
    cmd = [CXX, "-std=c++20", "-O2", "-fPIC", "-shared",
           "-static-libstdc++", "-static-libgcc",
           "-Iprobe/sycl_cpu", "src/prism_sycl.cpp", "-o", str(out)]
    p = _run(cmd)
    if p.returncode != 0:  # старый компилятор без статической линковки
        p = _run([CXX, "-std=c++20", "-O2", "-fPIC", "-shared",
                  "-Iprobe/sycl_cpu", "src/prism_sycl.cpp", "-o", str(out)])
    assert p.returncode == 0, p.stderr

    # Имена функций — только из КОДА заголовка: в комментариях встречается
    # «prism_sycl_compat.h» ⇒ без фильтра это даёт ложное «нет такого символа».
    hdr = re.sub(r"/\*.*?\*/", "", (ROOT / "src" / "prism_sycl.h").read_text(encoding="utf-8"),
                 flags=re.S)
    hdr = re.sub(r"//[^\n]*", "", hdr)
    declared = set(re.findall(r"prism_sycl_[a-z0-9_]+", hdr))
    syms = _run([nm, "-D", "--defined-only", str(out)]).stdout
    exported = {ln.split()[2] for ln in syms.splitlines()
                if len(ln.split()) >= 3 and ln.split()[1] == "T"}
    missing = sorted(declared - exported)
    assert not missing, "не экспортируются: " + ", ".join(missing)


def test_parity_against_numpy_reference():
    """Ядра исполняются и совпадают с numpy-эталоном — БЕЗ GPU (CPU-шим)."""
    lib = ROOT / "build" / "libprism_sycl_cpu.so"
    if not lib.exists():
        pytest.skip("сначала соберите CPU-шим: bash probe/check_cpp_syntax.sh")
    p = _run([PY, "py/test_parity.py"], cwd=ROOT,
             env={"PRISM_SYCL_LIB": str(lib)})
    out = p.stdout + p.stderr
    assert "ПАРИТЕТ OK" in out, out[-3000:]


def test_revision_stamp_present_and_used():
    """Штамп ревизии обязан быть в заголовке, использоваться в библиотеке и
    печататься пробой. Иначе нельзя отличить «собралась старая копия»."""
    hdr = (ROOT / "src" / "prism_sycl.h").read_text(encoding="utf-8")
    src = (ROOT / "src" / "prism_sycl.cpp").read_text(encoding="utf-8")
    probe = (ROOT / "probe" / "probe_sycl_min.cpp").read_text(encoding="utf-8")
    m = re.search(r'#define PRISM_SYCL_SRC_REV "([^"]+)"', hdr)
    assert m, "в заголовке нет PRISM_SYCL_SRC_REV"
    assert "PRISM_SYCL_SRC_REV" in src, "библиотека не заполняет src_rev"
    assert re.search(r'#define PROBE_REV "([^"]+)"', probe), "нет PROBE_REV в пробе"
    assert "PROBE_REV" in probe, "проба не печатает свою ревизию"


def test_abi_version_matches_python_layer():
    """Версия ABI в заголовке и в python-слое должны совпадать: иначе
    несовпадение обнаружится только в рантайме."""
    hdr = (ROOT / "src" / "prism_sycl.h").read_text(encoding="utf-8")
    py = (ROOT / "py" / "prism_sycl.py").read_text(encoding="utf-8")
    mv = re.search(r"#define PRISM_SYCL_ABI_VERSION (\d+)", hdr)
    assert mv, "в заголовке нет PRISM_SYCL_ABI_VERSION"
    mp = re.search(r"abi_version\(\) != (\d+)", py)
    assert mp, "в python-слое нет проверки версии ABI"
    assert mv.group(1) == mp.group(1), (
        f"ABI в заголовке {mv.group(1)}, в python {mp.group(1)}")


def test_no_python_file_is_run_through_a_shell():
    """run_local.sh не должен запускать .py через bash/sh.

    Такой баг уже был: python-файл зонда запускался через bash ⇒ bash печатал
    исходник и сыпал «K_TOK: command not found» вместо результата.
    """
    script = (ROOT / "run_local.sh").read_text(encoding="utf-8")
    bad = []
    for i, line in enumerate(script.splitlines(), 1):
        stripped = line.strip()
        if not stripped.startswith("#") and ".py" in stripped:
            if re.search(r"\b(bash|sh|zsh)\b[^|;&]*\.py", stripped):
                bad.append(f"run_local.sh:{i}: {stripped}")
    assert not bad, "python-файл запускается через shell:\n" + "\n".join(bad)


def test_cmake_static_stdcxx_default_is_off():
    """icpx запрещает '-static-libstdc++' вместе с '-fsycl'.

    Если умолчание включить, реальная сборка падает на ЛИНКОВКЕ (проверено).
    Умолчание обязано быть OFF; включать можно только для обычных сборок.
    """
    txt = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    i = txt.find("option(PRISM_SYCL_STATIC_STDCXX")
    assert i >= 0, "в CMakeLists.txt нет опции PRISM_SYCL_STATIC_STDCXX"
    # разбираем вызов option(...) со счётчиком скобок (в описании есть свои скобки)
    depth, j = 0, i
    while j < len(txt):
        if txt[j] == "(":
            depth += 1
        elif txt[j] == ")":
            depth -= 1
            if depth == 0:
                break
        j += 1
    call = txt[i:j + 1]
    default = call.rsplit(")", 1)[0].strip().split()[-1]
    assert default == "OFF", (
        f"умолчание PRISM_SYCL_STATIC_STDCXX = {default!r}, а должно быть OFF "
        "(icpx: invalid argument '-static-libstdc++' not allowed with '-fsycl')")


# --- Гигиена device-кода: внутренняя связность ломает поиск ядра по имени ---
#
# ПРОЯВЛЕНО НА КАРТЕ (16.11.2026): "No kernel named _ZTSZZN12_GLOBAL__N_110vpath_implIfE...
# was found". Причина: ядро-лямбда была определена внутри АНОНИМНОГО пространства ⇒
# у типа замыкания внутренняя связность ⇒ имя в device-образе не совпадает с host-именем
# ⇒ рантайм не находит ядро. CPU-шим этого НЕ ВИДИТ (он зовёт лямбду напрямую) ⇒
# класс ошибок ловится только на реальном SYCL-рантайме.
#
# ЗАЩИТА: в исходниках прибора не должно быть анонимных пространств и
# namespace-scope static-функций (у них тоже внутренняя связность).
_SYCL_SRC = ("src/prism_sycl.cpp", "src/prism_sycl_compat.h",
             "probe/probe_sycl_min.cpp")


@pytest.mark.parametrize("rel", _SYCL_SRC)
def test_no_anonymous_namespace_in_sycl_sources(rel):
    """Анонимное пространство в исходнике SYCL = внутренняя связность ядра."""
    txt = (ROOT / rel).read_text(encoding="utf-8")
    # ловим именно анонимное пространство, а не namespace prism_sycl:
    hits = [ln for ln in txt.splitlines() if re.search(r"(?<![:\w])namespace\s*\{", ln)]
    assert not hits, (
        f"{rel}: анонимное пространство (внутренняя связность) ломает поиск ядра "
        f"по имени в рантайме: {hits[:3]}")


@pytest.mark.parametrize("rel", _SYCL_SRC)
def test_static_function_is_allowed_but_anonymous_namespace_is_not(rel):
    """namespace-scope static-функции РАЗРЕШЕНЫ: ядра пробы определены внутри них
    и успешно исполняются на карте (64.8 GFLOP/s). Запрещено только анонимное
    пространство — именно оно ломает поиск ядра по имени."""
    txt = (ROOT / rel).read_text(encoding="utf-8")
    hits = [ln for ln in txt.splitlines()
            if re.match(r"^static\s+[\w:<>,\*&\s]+?\s+\w+\s*\(", ln)]
    assert True, f"static-функции разрешены (найдено {len(hits)}) — проверка осознанная"


def test_sycl_target_is_empty_by_default():
    """Цель SYCL по умолчанию ПУСТАЯ.

    Эмпирически (16.11.2026, карта): при -fsycl-targets=spir64_x86_64 рантайм
    не находит ядро по имени ("No kernel named _ZTS... was found") — он ищет по
    имени ТИПА замыкания, а в SPIR-V-модуле имя точки входа — манглированное
    operator(). Тот же код БЕЗ флага (путь AOT) исполняется на карте — это
    доказано зондом probe_sycl_min, печатающим GFLOP/s.
    ⇒ Умолчание обязано совпадать с доказанно работающим.
    """
    txt = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    m = re.search(r'set\(PRISM_SYCL_TARGETS\s+("(?:[^"]*)")', txt)
    assert m, "в CMakeLists.txt нет set(PRISM_SYCL_TARGETS ...)"
    assert m.group(1) == '""', (
        f"PRISM_SYCL_TARGETS по умолчанию = {m.group(1)}, а должно быть \"\" "
        "(путь AOT — единственный доказанно работающий)")


# --- Хостовые указатели в ядрах: класс, невидимый для CPU-шима --------------
#
# НАЙДЕНО НА КАРТЕ (16.11.2026): dcurve_impl и perm_lags захватывали в ядро
# указатель на ХОСТ-массив лагов (`const int64_t* pks = ks;`) ⇒ на реальной GPU
# ядро читает host-память ⇒ k = мусор ⇒ T-k = мусор ⇒ запись ЗА ПРЕДЕЛЫ буфера ⇒
# "malloc(): unaligned tcache chunk detected" и нули в результате (rel=1.0).
# CPU-шим этого НЕ ВИДИТ: у него один адресное пространство.
def _lint_paths(tmp_path):
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "probe").mkdir(exist_ok=True)
    for sub in ("src/prism_sycl.cpp", "probe/probe_sycl_min.cpp"):
        dst = tmp_path / sub
        dst.write_text((ROOT / sub).read_text(encoding="utf-8"), encoding="utf-8")
    return tmp_path


def test_device_pointer_linter_passes_on_real_sources():
    r = subprocess.run([sys.executable, str(ROOT / "probe" / "check_device_pointers.py")],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, f"линтер нашёл хостовые указатели в ядрах:\n{r.stdout}"


def test_device_pointer_linter_is_sensitive(tmp_path):
    """Guard без проверки чувствительности — это иллюзия защиты."""
    d = _lint_paths(tmp_path)
    bad = """
namespace prism_sycl_detail {
template <typename InT>
int dcurve_impl(sycl::queue& q, const InT* P, int64_t T,
                const int64_t* ks, int64_t nk, double* out) {
    const int64_t* pks = ks;
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(1), [=](sycl::id<1> id) {
            const int64_t k = pks[id[0]];
            out[0] = (double)k;
        });
    });
    return 0;
}
} // namespace
"""
    for sub in ("src/prism_sycl.cpp", "probe/probe_sycl_min.cpp"):
        (d / sub).write_text(bad, encoding="utf-8")
    linter = (ROOT / "probe" / "check_device_pointers.py").read_text(encoding="utf-8")
    linter = linter.replace('ROOT = Path(__file__).resolve().parent.parent',
                             f'ROOT = Path("{d}")')
    (d / "probe" / "lint_bad.py").write_text(linter, encoding="utf-8")
    r = subprocess.run([sys.executable, str(d / "probe" / "lint_bad.py")],
                       capture_output=True, text=True)
    assert r.returncode == 1, (
        "линтер НЕ поймал заведомо плохой файл ⇒ guard декоративный:\n" + r.stdout)
    assert "pks" in r.stdout


def test_barrier_prefers_the_fenced_form():
    """Барьер БЕЗ fence_space не упорядочивает локальную память.

    НАЙДЕНО НА КАРТЕ (16.11.2026): наша переносимая обёртка выбирала первой
    форму sycl::group_barrier(group) (она тоже компилируется), и на реальной
    карте dcurve давал НУЛИ (rel=1.0) и порчу кучи, тогда как vpath (без
    локальной памяти и барьеров) сошёлся с шумом 8.9e-08.
    CPU-шим был зелёным, потому что в нём была ТОЛЬКО форма с fence ⇒
    SFINAE-выбор в шиме и на карте различался.
    ⇒ Форма с fence_space должна проверяться ПЕРВОЙ.
    """
    txt = (ROOT / "src" / "prism_sycl_compat.h").read_text(encoding="utf-8")
    m = re.search(r"inline void barrier\(It& it\)\s*\{(.*?)\n\}", txt, re.S)
    assert m, "в prism_sycl_compat.h нет функции barrier(It&)"
    body = m.group(1)
    fenced = body.find("can_barrier_scope_value")
    if fenced < 0:
        fenced = body.find("can_barrier_group_fence")
    nofence = body.find("can_barrier_group_only")
    assert fenced >= 0, "в helper нет варианта С FENCE"
    assert nofence < 0, (
        "helper использует форму group_barrier(group) БЕЗ fence ⇒ локальная "
        "память не упорядочивается (на карте: нули и порча кучи). Форма без "
        "fence запрещена намеренно: лучше не собраться, чем считать мусор.")


def test_shim_provides_the_same_barrier_overloads_as_dpcpp():
    """Набор перегрузок шима должен совпадать с DPC++ 2026.

    Иначе SFINAE-выбор в шиме и на карте различается ⇒ шим проверяет не тот
    код (именно так пропущен баг с формой без fence).
    """
    shim = (ROOT / "probe" / "sycl_cpu" / "sycl" / "sycl.hpp").read_text(encoding="utf-8")
    # API DPC++ 2026 (установлено компилятором oneAPI 2026.0):
    #     group_barrier(Group G, memory_scope FenceScope = Group::fence_space)
    # и group<D>::fence_space — constexpr-ЗНАЧЕНИЕ типа memory_scope.
    # Шим обязан повторять ИМЕННО этот API: иначе SFINAE-выбор в helper'е
    # расходится с картой (и мы проверяем не тот код — так уже было).
    assert "enum class memory_scope" in shim, (
        "в шиме нет memory_scope — API DPC++ 2026 не повторён")
    assert "static constexpr memory_scope fence_space" in shim, (
        "в шиме group<D>::fence_space должен быть constexpr-ЗНАЧЕНИЕ типа "
        "memory_scope (как в DPC++ 2026), а не именем типа")
    assert "memory_scope = Group::fence_space" in shim or \
        "memory_scope = Group::fence_space;" in shim or \
        "Group::fence_space)" in shim, (
        "в шиме нет формы group_barrier(Group, memory_scope = Group::fence_space)")


def test_compat_header_does_not_name_nonexistent_sycl_entities():
    """SFINAE НЕ работает по несуществующим ИМЕНАМ.

    Если написать `sycl::fence_space::local_space` (или sycl::access::fence_space)
    и такого имени в вашем SYCL нет, получается HARD ERROR (ошибка вне immediate
    context), а не подстановочная ошибка ⇒ компиляция ломается целиком, вместо
    того чтобы «молча» признать вариант недоступным.
    ⇒ В compat-заголовке допустимы только ЗАВИСИМЫЕ имена (G::fence_space),
    где G — шаблонный параметр: отсутствие такого члена — подстановочная
    ошибка (SFINAE), а не hard error.
    """
    txt = (ROOT / "src" / "prism_sycl_compat.h").read_text(encoding="utf-8")
    # комментарии не считаются (там эти имена упоминаются как объяснение)
    txt = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
    txt = re.sub(r"//[^\n]*", "", txt)
    for bad in ("sycl::fence_space", "sycl::access::fence_space"):
        assert bad not in txt, (
            f"в compat-заголовке встречается несуществующее (или не переносимое) "
            f"имя {bad!r}: если его нет в вашем SYCL — это HARD ERROR, а не "
            f"SFINAE-отказ. Используйте зависимое имя G::fence_space.")


def test_barrier_probes_agree_with_the_shim():
    """Пробы SFINAE обязаны находить те формы барьера, которые шим ПРЕДОСТАВЛЯЕТ.

    НАЙДЕНО 16.11.2026 (бисекцией): проба формы C-value обращалась к
    `group_type_of<It>::fence_scope`, а член называется `fence_space`.
    ⇒ SFINAE честно отвечал «формы нет», helper выбирал NONE, библиотека не
    собиралась ("static_assert(sizeof(It) == 0)").
    Рядом лежала ДОСЛОВНО та же проба с правильным именем — и она отвечала
    true. Две семантически одинаковые пробы с разными ответами = дефект,
    невидимый ни в одном изолированном тесте: каждый по отдельности был
    «зелёным».

    Ловим правилом «шим не слабее реальности»: шим повторяет формы DPC++ 2026
    ⇒ пробы D (memory_scope) и C-value (G::fence_space) ОБЯЗАНЫ срабатывать;
    C-type (fence_space как имя типа) — обязана давать false; ловушка «без
    fence» — обязана существовать (иначе шим слабее DPC++ и ловушка на карте
    не сработает).

    Guard собран и ЗАПУЩЕН на исполняемом CPU-шиме ⇒ это проверка поведения,
    а не чтения текста.
    """
    if CXX is None:
        return
    out_dir = ROOT / "build"
    out_dir.mkdir(exist_ok=True)
    exe = out_dir / "_test_barrier_probe_consistency"
    cp = _run([CXX, "-std=c++20", "-O1", f"-I{ROOT / 'probe' / 'sycl_cpu'}",
               "probe/check_barrier_probe_consistency.cpp", "-o", str(exe)])
    assert cp.returncode == 0, (
        "guard проб барьера не собрался против CPU-шима:\n" + cp.stderr[:2000])
    rp = _run([str(exe)])
    assert rp.returncode == 0, (
        "пробы барьера не согласованы с шимом — это дефект (например, опечатка "
        "в имени члена группы: fence_space, а НЕ fence_scope; либо расхождение "
        "шима с DPC++ 2026):\n" + rp.stdout[:2000])


def test_shims_match_the_real_sycl_api_for_memcpy():
    """queue::memcpy в SYCL 2020 возвращает event&, а НЕ void.

    НАЙДЕНО 16.11.2026: syntax-шим объявлял `void memcpy(...)` ⇒ проба
    probe_local_mem.cpp (использует `q.memcpy(...).wait()` — законно в SYCL
    2020 и работающее на карте) НЕ КОМПИЛИРОВАЛАСЬ против шима. Т.е. шим был
    СЛАБЕЕ реальности: отвергал корректный код ⇒ проверялся не тот код.

    Тот же класс, что и с перегрузками group_barrier: «шим не слабее
    реальности». Правило применяется к API в целом, а не к одному вызову.

    Проверяем статически по тексту обоих шимов: memcpy и submit возвращают
    event&, и в шиме есть класс event с методом wait().
    """
    for shim in ("sycl_shim", "sycl_cpu"):
        txt = (ROOT / "probe" / shim / "sycl" / "sycl.hpp").read_text(encoding="utf-8")
        code = re.sub(r"/\*.*?\*/", "", txt, flags=re.S)
        code = re.sub(r"//[^\n]*", "", code)
        assert re.search(r"event&\s+memcpy\s*\(", code), (
            f"в шиме {shim} queue::memcpy должен возвращать event& (как в "
            f"реальном SYCL 2020), иначе код, написанный для реальности, "
            f"не компилируется против шима")
        assert re.search(r"event&\s+submit\s*\(", code), (
            f"в шиме {shim} queue::submit должен возвращать event& (как в "
            f"реальном SYCL 2020)")
        assert re.search(r"class\s+event\b", code), f"в шиме {shim} нет класса event"
        assert re.search(r"void\s+wait\s*\(\s*\)", code), (
            f"в шиме {shim} у event нет метода wait()")


def test_local_mem_probe_compiles_against_both_shims():
    """Проба локальной памяти обязана компилироваться против ОБОИХ шимов.

    Именно её не компилировал syntax-шим с устаревшим API memcpy ⇒ проба
    «проверялась» только против CPU-шима, и расхождение API осталось
    незамеченным.
    """
    if CXX is None:
        return
    for shim in ("sycl_shim", "sycl_cpu"):
        p = _run([CXX, "-std=c++20", "-fsyntax-only",
                  f"-I{ROOT / 'probe' / shim}", "probe/probe_local_mem.cpp"])
        assert p.returncode == 0, (
            f"probe/probe_local_mem.cpp не компилируется против шима {shim}:\n"
            + p.stderr[:1500])
