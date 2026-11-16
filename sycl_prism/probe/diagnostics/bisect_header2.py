#!/usr/bin/env python3
"""Бисекция №2: собираем namespace блок за блоком (как в header) и смотрим,
на каком шаге can_barrier_scope_value<It>::value переключается с 1 на 0.

Стратегия «наращивания»: стартуем с заведомо работающей пробы (как в
bisect_sfinae.cpp) и добавляем блоки header'а по одному.
"""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # корень дерева sycl_prism
SHIM = ROOT / "probe" / "sycl_cpu"
TMP = Path("/tmp/bs2")
TMP.mkdir(parents=True, exist_ok=True)

ALIAS = """
template <class It>
using group_type_of = std::remove_cv_t<std::remove_reference_t<
    decltype(std::declval<It&>().get_group())>>;
"""

PROBE = """
template <class It, class = void>
struct can_barrier_scope_value : std::false_type {};
template <class It>
struct can_barrier_scope_value<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           group_type_of<It>::fence_space))>> : std::true_type {};
"""

BLOCKS = {
    "fence_scope_is_type": """
template <class G, class = void>
struct fence_scope_is_type : std::false_type {};
template <class G>
struct fence_scope_is_type<G, std::void_t<typename G::fence_space>>
    : std::true_type {};
""",
    "has_fence_scope_value": """
template <class G, class = void>
struct has_fence_scope_value : std::false_type {};
template <class G>
struct has_fence_scope_value<G, std::void_t<decltype(G::fence_space)>>
    : std::true_type {};
""",
    "can_barrier_memory_scope": """
template <class It, class = void>
struct can_barrier_memory_scope : std::false_type {};
template <class It>
struct can_barrier_memory_scope<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           sycl::memory_scope::work_group))>> : std::true_type {};
""",
    "can_barrier_group_fence": """
template <class It, class = void>
struct can_barrier_group_fence : std::false_type {};
template <class It>
struct can_barrier_group_fence<It, std::void_t<decltype(sycl::group_barrier(
           std::declval<It&>().get_group(),
           group_type_of<It>::fence_space::local_space))>> : std::true_type {};
""",
    "can_barrier_group_only": """
template <class G, class = void>
struct can_barrier_group_only : std::false_type {};
template <class G>
struct can_barrier_group_only<G, std::void_t<decltype(sycl::group_barrier(
                                std::declval<const G&>()))>> : std::true_type {};
""",
}

ORDER = [
    "fence_scope_is_type",
    "has_fence_scope_value",
    "can_barrier_memory_scope",
    "can_barrier_group_fence",
    "can_barrier_group_only",
]


def build(parts) -> str:
    return "\nnamespace probe_ns {\n" + ALIAS + "".join(parts) + PROBE + "\n}\n"


def run(tag: str, ns_text: str) -> str:
    hdr = TMP / f"h_{tag}.h"
    hdr.write_text(ns_text, encoding="utf-8")
    src = TMP / f"p_{tag}.cpp"
    src.write_text(
        "#include <sycl/sycl.hpp>\n"
        f'#include "{hdr}"\n'
        "#include <cstdio>\n"
        "#include <type_traits>\n"
        "#include <utility>\n"
        "int main() {\n"
        "  using It = sycl::nd_item<1>;\n"
        '  std::printf("%d\\n", (int)probe_ns::can_barrier_scope_value<It>::value);\n'
        "  return 0;\n"
        "}\n",
        encoding="utf-8",
    )
    binp = TMP / f"b_{tag}"
    r = subprocess.run(["g++", "-std=c++20", f"-I{SHIM}", str(src), "-o", str(binp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        errs = [line for line in r.stderr.splitlines() if "error" in line]
        return "ERR: " + (errs[0] if errs else "?")
    out = subprocess.run([str(binp)], capture_output=True, text=True)
    return out.stdout.strip() or "EMPTY"


print("только проба               :", run("p0", build([])), "(ожидаем 1)")
parts = []
for name in ORDER:
    parts.append(BLOCKS[name])
    print(f"+ {name:28s}:", run(name, build(parts)))
