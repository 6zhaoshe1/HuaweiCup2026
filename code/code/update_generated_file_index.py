# -*- coding: utf-8 -*-
"""Rebuild the shared generated-file index without touching model interfaces.

AI assistance: OpenAI Codex, 2026-09-26.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def purpose(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".py", ".ipynb"}:
        return "可复现分析/验证/绘图源代码"
    if suffix in {".pdf", ".png", ".svg"} and ("fig" in name or "paper_main" in name):
        return "由真实结果生成的问题图表"
    if suffix in {".csv", ".json", ".gz", ".parquet"}:
        return "机器可读数据、模型结果、审计或接口工件"
    if suffix == ".md":
        return "建模、结果、风险、复现或项目状态文档"
    if suffix == ".log":
        return "正式脚本运行日志"
    if suffix in {".tex", ".bib", ".cls"}:
        return "论文排版源文件"
    return "项目辅助产物"


files: list[Path] = []
for folder_name in ["code", "results", "figures", "reports", "paper"]:
    folder = ROOT / folder_name
    if not folder.exists():
        continue
    for path in folder.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            files.append(path)

lines = [
    "# Codex 生成文件总表",
    "",
    f"> 自动生成于 {date.today().isoformat()}。范围为正式项目目录 `code/`、`results/`、`figures/`、`reports/`、`paper/`；不扫描 DeepSeek、队友或老师教程工作区。表中相对路径可直接用于交接与复现。",
    "",
    "| 相对路径 | 大小（字节） | 用途 |",
    "|---|---:|---|",
]
for path in sorted(files, key=lambda x: x.relative_to(ROOT).as_posix().lower()):
    rel = path.relative_to(ROOT).as_posix().replace("|", "\\|")
    lines.append(f"| `{rel}` | {path.stat().st_size} | {purpose(path)} |")

(ROOT / "reports" / "00_codex_generated_files.md").write_text(
    "\n".join(lines) + "\n", encoding="utf-8"
)
print(f"indexed_files={len(files)}")
