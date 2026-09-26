#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题一关键结果独立复算，不导入主脚本的评价函数。

AI 辅助信息：OpenAI Codex（GPT-5 系列），OpenAI，2026-09-23。
仅读取主脚本已冻结的机器结果，使用 NumPy/SciPy 从指标定义重新计算。
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def rho(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
    return float(np.corrcoef(rx, ry)[0, 1])


def direct_metrics(part: pd.DataFrame) -> dict:
    per_domain = []
    for _, g in part.groupby("domain"):
        y, p = g.observed_loss.to_numpy(float), g.predicted_loss.to_numpy(float)
        err = p - y
        denom = np.sum((y - np.mean(y)) ** 2)
        per_domain.append({
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "r2": float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else np.nan,
            "spearman": rho(y, p),
            "bias": float(np.mean(err)),
        })
    return {
        "macro_mae": float(np.mean([r["mae"] for r in per_domain])),
        "macro_rmse": float(np.mean([r["rmse"] for r in per_domain])),
        "macro_r2": float(np.mean([r["r2"] for r in per_domain])),
        "macro_spearman": float(np.mean([r["spearman"] for r in per_domain])),
        "micro_rmse": float(np.sqrt(np.mean((part.predicted_loss.to_numpy(float) - part.observed_loss.to_numpy(float)) ** 2))),
    }


def main() -> None:
    pred = pd.read_csv(RESULTS / "q1_predictions.csv.gz")
    reported = pd.read_csv(RESULTS / "q1_model_metrics.csv")
    summary = json.loads((RESULTS / "q1_summary.json").read_text(encoding="utf-8"))
    chosen = summary["model"]["selected_model"]
    checks = {}
    max_diff = 0.0
    for dataset, model in (("A6_A7_1m", chosen), ("A10_A11_1b", chosen), ("A10_A11_1b", f"{chosen}_logN_offset")):
        part = pred[(pred.dataset == dataset) & (pred.model == model)]
        calc = direct_metrics(part)
        rep = reported[(reported.dataset == dataset) & (reported.model == model) & (reported.domain == "macro")].iloc[0]
        diffs = {
            "mae": abs(calc["macro_mae"] - float(rep.mae)),
            "rmse": abs(calc["macro_rmse"] - float(rep.rmse)),
            "r2": abs(calc["macro_r2"] - float(rep.r2)),
            "spearman": abs(calc["macro_spearman"] - float(rep.spearman)),
        }
        max_diff = max(max_diff, *diffs.values())
        checks[f"{dataset}|{model}"] = {"calculated": calc, "reported_differences": diffs}

    score = pd.read_csv(RESULTS / "q1_quality_scores.csv.gz", usecols=["record_id", "dataset", "q_definition_equal", "q_definition_robust"])
    score_bounds_ok = bool(score[["q_definition_equal", "q_definition_robust"]].ge(0).all().all() and score[["q_definition_equal", "q_definition_robust"]].le(1).all().all())
    manifest = pd.read_csv(RESULTS / "split_manifest.csv.gz", dtype=str)
    duplicate_consistent = manifest.groupby("group_id")["fold"].nunique().max() == 1
    a6 = set(manifest.loc[manifest.source == "A6_A7", "group_id"])
    a8 = set(manifest.loc[manifest.source == "A8_A9", "group_id"])
    a4 = set(manifest.loc[manifest.source == "A4_A5", "group_id"])
    a12 = set(manifest.loc[manifest.source == "A12_A13", "group_id"])

    missing_evidence = []
    for path in sorted(RESULTS.glob("q1_*.csv")) + sorted(RESULTS.glob("q1_*.csv.gz")):
        cols = pd.read_csv(path, nrows=0).columns
        if "evidence_level" not in cols:
            missing_evidence.append(path.name)

    result = {
        "status": "PASS" if max_diff < 1e-10 and score_bounds_ok and duplicate_consistent and not missing_evidence else "FAIL",
        "max_metric_abs_difference": max_diff,
        "metric_tolerance": 1e-10,
        "quality_score_bounds_ok": score_bounds_ok,
        "quality_rows": int(len(score)),
        "quality_unique_ids": int(score.record_id.nunique()),
        "same_group_same_fold": bool(duplicate_consistent),
        "A6_A8_same_composition_groups": bool(a6 == a8),
        "A12_groups_subset_A4": bool(a12.issubset(a4)),
        "missing_evidence_level_files": missing_evidence,
        "metric_checks": checks,
    }
    (RESULTS / "q1_independent_verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
