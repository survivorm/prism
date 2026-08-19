#!/usr/bin/env python3
"""Add params before json.dump in each experiment."""

import ast
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT, "experiments")


def get_params(filepath):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            defaults = node.args.defaults
            start = len(node.args.args) - len(defaults)
            names = []
            for i, arg in enumerate(node.args.args):
                if i >= start:
                    names.append(arg.arg)
            return names
    return []


def find_dump(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if "json.dump(" not in line:
            continue
        if '"params"' in line or "'params'" in line:
            continue
        m = re.search(r"json\.dump\((\w+)", line)
        if m:
            var_name = m.group(1)
            insert_at = i
            for j in range(i - 1, max(i - 5, -1), -1):
                if "os.makedirs" in lines[j]:
                    insert_at = j + 1
                    break
                elif lines[j].strip().startswith("with open"):
                    insert_at = j
                    break
            return insert_at, var_name
    return None, None


def has_params(filepath):
    with open(filepath) as f:
        content = f.read()
    return bool(re.search(r'\["params"\]\s*=\s*\{', content))


def main():
    for fname in sorted(os.listdir(EXP_DIR)):
        if not fname.startswith("exp") or not fname.endswith(".py"):
            continue
        fpath = os.path.join(EXP_DIR, fname)
        if has_params(fpath):
            print(f"SKIP {fname}: already has params")
            continue
        params = get_params(fpath)
        if not params:
            print(f"SKIP {fname}: no params")
            continue
        params = [p for p in params if p != "results_dir"]
        if not params:
            print(f"SKIP {fname}: only results_dir")
            continue
        insert_at, var_name = find_dump(fpath)
        if insert_at is None:
            print(f"SKIP {fname}: no json.dump")
            continue
        param_strs = ", ".join(f"{p}: {p}" for p in params)
        params_line = f'    {var_name}["params"] = {{{param_strs}}}\n'
        with open(fpath) as f:
            lines = f.readlines()
        lines.insert(insert_at, params_line)
        with open(fpath, "w") as f:
            f.writelines(lines)
        print(f"OK {fname}: {var_name}")


if __name__ == "__main__":
    main()
