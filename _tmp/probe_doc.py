# -*- coding: utf-8 -*-
"""检查本机候选论文模板的真实身份：届次 / 年份 / 是否含官方要素。"""
import re
import zipfile
from pathlib import Path

BASE = Path(r"E:\读研\26届数学建模比赛\往年优秀资料")
CANDIDATES = [
    BASE / "“华为杯”第二十三届中国研究生数学建模竞赛论文模板 (1)(1).doc",
    BASE / "word与Latex模板" / "第二十三届研赛论文Word标准模板.docx",
    BASE / "word与Latex模板" / "附件3：“华为杯”第二十二届中国研究生数学建模竞赛论文模板.doc",
]
KEYS = ["第二十三届", "第二十二届", "二十一届", "2026", "2025", "论文格式规范", "承诺书", "参赛队号", "编号专用页", "页眉"]


def docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    return " ".join(re.sub("<[^>]+>", " ", xml).split())


def doc_text(path: Path) -> str:
    raw = path.read_bytes()
    # Word 97-2003 正文多为 UTF-16LE（中文）或 CP936 压缩存储，这里两路都试
    a = raw.decode("utf-16-le", "ignore")
    b = raw.decode("cp936", "ignore")
    return a + "\n" + b


for p in CANDIDATES:
    if not p.exists():
        print(f"[缺失] {p}")
        continue
    print("=" * 96)
    print(f"文件: {p.name}  ({p.stat().st_size / 1024:.0f} KB)")
    try:
        txt = docx_text(p) if p.suffix.lower() == ".docx" else doc_text(p)
    except Exception as exc:  # noqa: BLE001
        print("  解析失败:", exc)
        continue
    for kw in KEYS:
        print(f"  {kw:8s}: {'命中' if kw in txt else '—'}")
    m = re.search(r"第[一二三四五六七八九十百]+届", txt)
    if m:
        seg = txt[max(0, m.start() - 60): m.start() + 120]
        print("  届次上下文:", " ".join(seg.split()))
