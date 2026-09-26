# -*- coding: utf-8 -*-
"""把官方附件2（格式规范）与附件4（AI 工具规定）转成可读 UTF-8 Markdown，供后续引用。"""
import re
import zipfile
from pathlib import Path

DIR = Path(r"E:\读研\26届数学建模比赛\往年优秀资料\官方模板_第二十三届")
OUT = DIR
JOBS = [
    ("附件2_第二十三届论文格式规范.docx", "附件2_格式规范_全文.md", "“华为杯”第二十三届中国研究生数学建模竞赛 论文格式规范"),
    ("附件4_第二十三届人工智能工具及输出使用规定.docx", "附件4_AI工具规定_全文.md", "“华为杯”第二十三届中国研究生数学建模竞赛 人工智能工具及输出使用规定"),
]


def paragraphs(path: Path) -> list[str]:
    """按段落抽取文本，保留段落边界（比一整行拼接可读）。"""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    out = []
    for para in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S):
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para, re.S))
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()
        if text:
            out.append(text)
    return out


for src_name, dst_name, title in JOBS:
    src = DIR / src_name
    if not src.exists():
        print("[缺失]", src)
        continue
    paras = paragraphs(src)
    body = "\n\n".join(paras)
    header = (
        f"# {title}\n\n"
        f"> 来源：官方开赛公告附件，2026-09-23 由 CPIPC 平台直链下载（{src_name}，{src.stat().st_size/1024:.0f} KB）。\n"
        f"> 本文件为脚本抽取的纯文本副本，仅便于检索；排版以原始 docx 为准。\n\n---\n\n"
    )
    (OUT / dst_name).write_text(header + body + "\n", encoding="utf-8")
    print(f"[OK] {dst_name}  段落 {len(paras)}，字符 {len(body)}")
    print("    前 3 段预览：")
    for p in paras[:3]:
        print("     ·", p[:100])
