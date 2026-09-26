# -*- coding: utf-8 -*-
"""核对：官方附件3模板 + 附件2格式规范 的届次身份，并打印格式规范要点。"""
import re
import zipfile
from pathlib import Path

DIR = Path(r"E:\读研\26届数学建模比赛\往年优秀资料\官方模板_第二十三届")


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    return " ".join(re.sub("<[^>]+>", " ", xml).split())


def doc_strings(path: Path) -> str:
    raw = path.read_bytes()
    return raw.decode("utf-16-le", "ignore") + "\n" + raw.decode("cp936", "ignore")


for name in ["附件3_第二十三届论文模板.doc", "附件2_第二十三届论文格式规范.docx"]:
    p = DIR / name
    if not p.exists():
        print("[缺失]", p)
        continue
    txt = docx_text(p) if p.suffix.lower() == ".docx" else doc_strings(p)
    print("=" * 100)
    print(f"{name}   ({p.stat().st_size/1024:.0f} KB)")
    for kw in ["第二十三届", "第二十二届", "2026", "参赛队号", "队员姓名", "承诺书", "摘要", "页眉", "logo", "LOGO"]:
        print(f"   {kw:8s}: {'命中' if kw in txt else '—'}")

# 打印《论文格式规范》正文（去掉页眉页脚噪声）
p = DIR / "附件2_第二十三届论文格式规范.docx"
if p.exists():
    t = docx_text(p)
    print("\n" + "=" * 100)
    print("《第二十三届论文格式规范》正文前 2500 字：\n")
    print(t[:2500])
