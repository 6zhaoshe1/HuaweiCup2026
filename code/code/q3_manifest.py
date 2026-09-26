#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI assistance: OpenAI Codex, OpenAI, 2026-09-25; team review required.
"""Freeze the Q3 artifact ledger after every producer and log has finished."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    candidates = [
        ROOT / "code" / "q3_modeling.py",
        ROOT / "code" / "q3_independent_verify.py",
        ROOT / "code" / "q3_report.py",
        ROOT / "code" / "q3_manifest.py",
        ROOT / "code" / "outputs" / "q3_modeling.log",
        ROOT / "code" / "outputs" / "q3_independent_verify.log",
        ROOT / "code" / "outputs" / "q3_report.log",
        ROOT / "reports" / "18_q3_independent_design.md",
        ROOT / "reports" / "19_q3_cross_audit.md",
        ROOT / "reports" / "20_q3_modeling_report.md",
        ROOT / "reports" / "21_q3_review_disposition.md",
    ]
    candidates += sorted(path for path in RES.glob("q3_*") if path.name != "q3_artifact_manifest.csv")
    candidates += sorted((ROOT / "figures").glob("q3_*"))

    rows = []
    for path in candidates:
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("code/"):
            purpose = "reproducible source or execution log"
        elif relative.startswith("reports/"):
            purpose = "design, cross-audit, or generated conclusion report"
        elif relative.startswith("figures/"):
            purpose = "data-derived evidence figure"
        else:
            purpose = "machine-readable Q3 result or cross-question interface"
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "purpose": purpose,
            }
        )

    manifest = pd.DataFrame(rows).drop_duplicates("relative_path").sort_values("relative_path")
    manifest.to_csv(RES / "q3_artifact_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"q3_manifest_rows={len(manifest)}")


if __name__ == "__main__":
    main()
