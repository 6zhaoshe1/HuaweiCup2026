# -*- coding: utf-8 -*-
"""官方附件3（.doc）文本骨架 -> UTF-8 文件（避免控制台 GBK 报错）。"""
import re
from pathlib import Path

SRC = Path(r"E:\读研\26届数学建模比赛\官方模板_第二十三届\附件3_第二十三届论文模板.doc")
OUT = Path(r"E:\读研\26届数学建模比赛\_tmp\official_doc3_structure.txt")
raw = SRC.read_bytes()


def clean(t: str) -> str:
    t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "\n", t)
    t = re.sub(r"[\ufff0-\uffff]", "", t)
    return t


lines_out = []
for tag, t in (("UTF16LE", clean(raw.decode("utf-16-le", "ignore"))),
               ("CP936", clean(raw.decode("cp936", "ignore")))):
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if len(ln) >= 2 and re.search(r"[\u4e00-\u9fff]", ln)]
    lines_out.append("=" * 100)
    lines_out.append(f"### {tag}: 含中文的行数 {len(lines)}")
    seen, shown = set(), 0
    for ln in lines:
        if ln[:40] in seen:
            continue
        seen.add(ln[:40])
        lines_out.append("   " + ln[:150])
        shown += 1
        if shown >= 80:
            break

# 关键结构词命中情况
text_all = raw.decode("utf-16-le", "ignore") + raw.decode("cp936", "ignore")
lines_out.append("")
lines_out.append("=" * 100)
lines_out.append("### 关键结构词命中（官方附件3）")
for kw in ["目录", "承诺", "编号专用页", "摘要", "关键词", "参考文献", "附录", "学校", "参赛队号", "队员姓名", "题目", "页眉", "页脚"]:
    lines_out.append(f"   {kw:8s}: {'命中' if kw in text_all else '—'}")

OUT.write_text("\n".join(lines_out) + "\n", encoding="utf-8")
print("written:", OUT)
