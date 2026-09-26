# -*- coding: utf-8 -*-
"""清理编译中间产物 + 重算 latex-project.json 的 SHA256 清单（保留原有 schema 字段）。"""
import hashlib
import json
from pathlib import Path

D = Path(r"E:\读研\26届数学建模比赛\华为杯第二十三届论文-LaTeX模板-最终版")

# 1) 清理中间产物（保留 main.pdf 作为编译样张）
for name in ["main.aux", "main.bbl", "main.blg", "main.log", "main.out", "main.toc",
             "_pass1.log", "_pass2.log", "_pass3.log", "_bibtex.log", "main.synctex.gz"]:
    p = D / name
    if p.exists():
        p.unlink()
        print("removed:", name)

# 2) 重算清单：模板源文件（不含编译产物 PDF）
manifest_path = D / "latex-project.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
old = manifest.get("template_files", {})

sources = ["main.tex", "huawei-cup.cls", "README.md", "gmcm.bst", "reference.bib"] + \
          sorted(str(p.relative_to(D)).replace("\\", "/") for p in (D / "figures").glob("*") if p.is_file())

new_files = {}
for rel in sources:
    p = D / rel
    if not p.exists():
        print("MISSING:", rel)
        continue
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    new_files[rel] = digest
    flag = "changed" if old.get(rel) and old[rel] != digest else ("new" if rel not in old else "same")
    print(f"  {flag:8s} {rel}  {digest[:16]}…")

manifest["template_files"] = new_files
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("\nwritten:", manifest_path)
print("files in manifest:", len(new_files))
