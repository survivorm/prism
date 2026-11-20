#!/usr/bin/env python3
"""Единый реестр артефактов prism: git получает только манифест, файлы — в HF/локально.

Команды:
  scan                     — что лежит локально и весит больше порога (кандидаты на вынос)
  gate                     — предкоммит-гейт: падает, если в git staged файл > порога
  add <path> --exp E65     — внести файл в реестр (sha256, размер, путь восстановления)
  verify                   — сверить локальные файлы с реестром (sha256)
  push [--repo owner/name] — выгрузить невыгруженное в HF dataset repo
  pull <name>              — восстановить файл из HF по реестру

Реестр — artifacts/registry.json (в git, компактный). Он НЕ хранит данные, только провенанс.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REG = ROOT / "artifacts" / "registry.json"
LIMIT_MB = 10.0
SCAN_ROOTS = ["results", "data", "logs"]


def load_reg():
    if REG.exists():
        return json.loads(REG.read_text(encoding="utf-8"))
    return {"schema": 1, "hf_repo": None, "items": {}}


def save_reg(reg):
    REG.parent.mkdir(parents=True, exist_ok=True)
    REG.write_text(json.dumps(reg, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def sha256_16(p, chunk=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def human(n):
    return f"{n / 1e6:.1f}MB" if n >= 1e6 else f"{n / 1e3:.0f}KB"


def cmd_scan(a):
    rows = []
    for r in SCAN_ROOTS:
        base = ROOT / r
        if not base.exists():
            continue
        for dp, dn, fn in os.walk(base):
            for f in fn:
                p = Path(dp) / f
                try:
                    sz = p.stat().st_size
                except OSError:
                    continue
                if sz > a.mb * 1e6:
                    rows.append((sz, str(p.relative_to(ROOT))))
    rows.sort(reverse=True)
    print(f"файлов больше {a.mb} МБ: {len(rows)}, суммарно {human(sum(s for s, _ in rows))}")
    for sz, p in rows[: a.top]:
        print(f"  {human(sz):>9}  {p}")
    return 0


def staged_big(mb):
    try:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError:
        return []
    big = []
    for rel in [l.strip() for l in out.splitlines() if l.strip()]:
        p = ROOT / rel
        if p.is_file() and p.stat().st_size > mb * 1e6:
            big.append((p.stat().st_size, rel))
    return big


def cmd_gate(a):
    big = staged_big(a.mb)
    if not big:
        print(f"gate OK: в staged нет файлов > {a.mb} МБ")
        return 0
    print(f"gate FAIL: {len(big)} файл(ов) > {a.mb} МБ в staged — в git нельзя:")
    for sz, rel in big:
        print(f"  {human(sz):>9}  {rel}")
    print("\nвынести: python tools/artifacts.py add <path> --exp E<NN>  &&  python tools/artifacts.py push")
    print("затем: git restore --staged <path>  (файл остаётся на диске, в реестре — его sha)")
    return 1


def cmd_add(a):
    p = Path(a.path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.is_file():
        print(f"нет файла: {p}")
        return 2
    sz = p.stat().st_size
    reg = load_reg()
    key = a.name or p.name
    reg["items"][key] = {
        "local_path": str(p.relative_to(ROOT)) if str(p).startswith(str(ROOT)) else str(p),
        "size": sz,
        "sha256_16": sha256_16(p),
        "exp": a.exp or "",
        "note": a.note or "",
        "hf_path": None,
        "hf_revision": None,
    }
    save_reg(reg)
    print(f"внесено в реестр: {key}  {human(sz)}  sha {reg['items'][key]['sha256_16']}")
    if sz <= a.mb * 1e6:
        print(f"  подсказка: файл <= {a.mb} МБ — его можно держать в git; реестр всё равно полезен")
    return 0


def cmd_verify(a):
    reg = load_reg()
    bad = 0
    for key, it in sorted(reg["items"].items()):
        p = ROOT / it["local_path"] if not os.path.isabs(it["local_path"]) else Path(it["local_path"])
        if not p.exists():
            print(f"  НЕТ ЛОКАЛЬНО: {key} ({it['local_path']})" + (" — pull" if it.get("hf_path") else " — только в HF? нет hf_path"))
            bad += 1
            continue
        got = sha256_16(p)
        if got != it["sha256_16"]:
            print(f"  РАСХОЖДЕНИЕ sha: {key} реестр {it['sha256_16']} файл {got}")
            bad += 1
    print(f"verify: {len(reg['items'])} записей, проблем {bad}")
    return 1 if bad else 0


def cmd_push(a):
    reg = load_reg()
    repo = a.repo or reg.get("hf_repo")
    if not repo:
        print("нужен --repo owner/name (dataset) или заполни hf_repo в artifacts/registry.json")
        return 2
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("нет huggingface_hub: .venv/bin/pip install huggingface_hub")
        return 2
    api = HfApi()
    todo = [(k, it) for k, it in reg["items"].items() if not it.get("hf_path")]
    if a.dry_run:
        print(f"[dry-run] repo={repo}, к выгрузке {len(todo)} из {len(reg['items'])}")
        for k, it in todo:
            print(f"  {k}  {human(it['size'])}  → {it['exp'] or 'misc'}/{k}")
        return 0
    api.create_repo(repo_id=repo, repo_type="dataset", private=True, exist_ok=True)
    for k, it in todo:
        p = ROOT / it["local_path"] if not os.path.isabs(it["local_path"]) else Path(it["local_path"])
        if not p.exists():
            print(f"  пропуск {k}: нет файла локально")
            continue
        dst = f"{it['exp'] or 'misc'}/{k}"
        api.upload_file(path_or_fileobj=str(p), path_in_repo=dst, repo_id=repo, repo_type="dataset")
        commit = api.dataset_info(repo).sha
        it["hf_path"], it["hf_revision"] = dst, commit
        save_reg(reg)
        print(f"  выгружено {k} → {repo}:{dst}")
    reg["hf_repo"] = repo
    save_reg(reg)
    return 0


def cmd_pull(a):
    reg = load_reg()
    it = reg["items"].get(a.name)
    if not it:
        print(f"нет в реестре: {a.name}")
        return 2
    if not it.get("hf_path"):
        print(f"{a.name} не выгружен в HF (hf_path пуст)")
        return 2
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("нет huggingface_hub")
        return 2
    dst = Path(a.dest) if a.dest else (ROOT / it["local_path"])
    dst.parent.mkdir(parents=True, exist_ok=True)
    got = Path(hf_hub_download(repo_id=reg["hf_repo"], filename=it["hf_path"], repo_type="dataset",
                               revision=it.get("hf_revision") or "main"))
    # copy2, а не move/replace: кэш HF (/tmp, /home) и репо часто на разных ФС (EXDEV), а если
    # артефакт уже лежит локально — shutil.move падал с FileExistsError.
    shutil.copy2(got, dst)
    try:
        got.unlink()   # убираем только ссылку из снапшота; блоб в blobs/ остаётся
    except OSError:
        pass
    ok = sha256_16(dst) == it["sha256_16"]
    print(f"восстановлено: {dst}  sha {'OK' if ok else 'НЕ СОВПАЛ'}")
    if not ok:
        print(f"  реестр {it['sha256_16']} файл {sha256_16(dst)} — файл оставлен, но помечен как битый")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mb", type=float, default=LIMIT_MB, help="порог в МБ (по умолчанию 10)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan"); s.add_argument("--top", type=int, default=30); s.set_defaults(fn=cmd_scan)
    s = sub.add_parser("gate"); s.set_defaults(fn=cmd_gate)
    s = sub.add_parser("add"); s.add_argument("path"); s.add_argument("--name"); s.add_argument("--exp"); s.add_argument("--note"); s.set_defaults(fn=cmd_add)
    s = sub.add_parser("verify"); s.set_defaults(fn=cmd_verify)
    s = sub.add_parser("push"); s.add_argument("--repo"); s.add_argument("--dry-run", action="store_true"); s.set_defaults(fn=cmd_push)
    s = sub.add_parser("pull"); s.add_argument("name"); s.add_argument("--dest"); s.set_defaults(fn=cmd_pull)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
