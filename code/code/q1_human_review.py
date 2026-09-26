#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare and score the two-rater blinded Q1 text review.

AI 辅助信息：OpenAI Codex（GPT-5 系列），OpenAI，2026-09-24。
本脚本只管理盲审表、复算一致性和描述性表面效度；不把人工小样本当作训练效果真值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TMP = ROOT / "_tmp"
SEED = 42
RATING_COLUMNS = [
    "overall_quality_1to5", "content_value_1to5", "language_quality_1to5",
    "cleanliness_1to5", "reasoning_professional_1to5",
]


def review_id(record_id: object, domain: object, content_sha256: object) -> str:
    raw = f"q1-review-v1|{record_id}|{domain}|{content_sha256}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:12]


def prepare() -> None:
    sample = pd.read_csv(RESULTS / "q1_manual_text_audit_sample.csv")
    join_cols = ["record_id", "domain", "content_sha256"]
    scores = pd.read_csv(RESULTS / "q1_quality_scores.csv.gz",
                         usecols=join_cols + ["q_definition_equal", "q_definition_robust"])
    if sample[join_cols].duplicated().any() or scores[join_cols].duplicated().any():
        raise RuntimeError("composite record key is not unique; blind review cannot be built safely")
    key = sample.drop(columns=["q_definition_robust"], errors="ignore").merge(
        scores, on=join_cols, how="left", validate="one_to_one")
    if key[["q_definition_equal", "q_definition_robust"]].isna().any().any():
        raise RuntimeError("quality-score lookup failed for one or more blind-review records")
    key.insert(0, "review_id", [review_id(r.record_id, r.domain, r.content_sha256) for r in key.itertuples()])
    key.to_csv(TMP / "q1_human_review_key.csv", index=False, encoding="utf-8-sig")

    public = key[["review_id", "content_excerpt"]].copy()
    for column in RATING_COLUMNS:
        public[column] = ""
    public["measurement_error_yes_no"] = ""
    public["reviewer_note"] = ""
    for rater, seed in [(1, SEED + 101), (2, SEED + 202)]:
        form = public.sample(frac=1, random_state=seed).reset_index(drop=True)
        form.to_csv(RESULTS / f"q1_human_review_rater{rater}.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"prepared_records": len(key),
                      "rater_files": ["results/q1_human_review_rater1.csv",
                                      "results/q1_human_review_rater2.csv"],
                      "blinding_key": "_tmp/q1_human_review_key.csv"}, ensure_ascii=False, indent=2))


def _load_completed(path: Path, rater: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"review_id", *RATING_COLUMNS, "measurement_error_yes_no", "reviewer_note"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path.name} missing columns: {sorted(missing)}")
    if df.review_id.duplicated().any():
        raise RuntimeError(f"{path.name} has duplicate review_id")
    for column in RATING_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any() or not values.between(1, 5).all():
            raise RuntimeError(f"{path.name}: {column} must be fully filled with integers 1--5")
        if not np.allclose(values, np.round(values)):
            raise RuntimeError(f"{path.name}: {column} must contain integers")
        df[column] = values.astype(int)
    normalized = df.measurement_error_yes_no.astype(str).str.strip().str.lower()
    allowed = {"yes", "no", "y", "n", "是", "否", "1", "0"}
    if not normalized.isin(allowed).all():
        raise RuntimeError(f"{path.name}: measurement_error_yes_no must be yes/no or 是/否")
    df["measurement_error"] = normalized.isin({"yes", "y", "是", "1"})
    return df.rename(columns={c: f"{c}_{rater}" for c in RATING_COLUMNS + ["measurement_error", "reviewer_note"]})


def _bootstrap_kappa(a: np.ndarray, b: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(a), len(a))
        kappa = cohen_kappa_score(a[idx], b[idx], weights="quadratic")
        if np.isfinite(kappa):
            values.append(kappa)
    return tuple(np.quantile(values, [.025, .975]))


def _directional_pair_accuracy(df: pd.DataFrame, score: str) -> tuple[float, int]:
    credits = []
    for _, part in df.groupby("domain"):
        hi = part.loc[part.audit_stratum.eq("high_quality"), score].to_numpy(float)
        lo = part.loc[part.audit_stratum.eq("low_quality"), score].to_numpy(float)
        for h in hi:
            for l in lo:
                credits.append(1.0 if h > l else (0.5 if h == l else 0.0))
    return (float(np.mean(credits)), len(credits)) if credits else (np.nan, 0)


def score() -> None:
    key = pd.read_csv(TMP / "q1_human_review_key.csv")
    r1 = _load_completed(RESULTS / "q1_human_review_rater1.csv", "r1")
    r2 = _load_completed(RESULTS / "q1_human_review_rater2.csv", "r2")
    merged = key.merge(r1.drop(columns=["content_excerpt"]), on="review_id", validate="one_to_one")
    merged = merged.merge(r2.drop(columns=["content_excerpt"]), on="review_id", validate="one_to_one")
    if len(merged) != len(key):
        raise RuntimeError("completed rater files do not cover the exact blinded sample")

    metric_rows = []
    for column in RATING_COLUMNS:
        a = merged[f"{column}_r1"].to_numpy(int); b = merged[f"{column}_r2"].to_numpy(int)
        ci_low, ci_high = _bootstrap_kappa(a, b)
        metric_rows.append({
            "dimension": column, "n": len(a),
            "cohen_kappa_unweighted": cohen_kappa_score(a, b),
            "cohen_kappa_quadratic": cohen_kappa_score(a, b, weights="quadratic"),
            "kappa_quadratic_boot_ci_low": ci_low, "kappa_quadratic_boot_ci_high": ci_high,
            "exact_agreement": np.mean(a == b), "within_one_agreement": np.mean(np.abs(a-b) <= 1),
            "evidence_level": "stratified_blinded_human_review",
        })
        merged[f"{column}_consensus"] = (a + b) / 2

    human = "overall_quality_1to5_consensus"
    directional, n_pairs = _directional_pair_accuracy(merged, human)
    quality_extremes = merged.audit_stratum.isin(["high_quality", "low_quality"])
    y = merged.loc[quality_extremes, "audit_stratum"].eq("high_quality").astype(int)
    auc = roc_auc_score(y, merged.loc[quality_extremes, human])
    kappa_main = next(r for r in metric_rows if r["dimension"] == "overall_quality_1to5")["cohen_kappa_quadratic"]
    accepted = bool(kappa_main >= .4 and directional >= .70)

    comparison = []
    for q_name in ["q_definition_equal", "q_definition_robust"]:
        comparison.append({"quality_definition": q_name,
                           "spearman_with_human_consensus": spearmanr(merged[q_name], merged[human]).statistic,
                           "directional_pair_accuracy_on_huber_selected_extremes": _directional_pair_accuracy(merged, q_name)[0],
                           "selection_warning": "high/low strata were selected by robust Huber Q; comparison is exploratory",
                           "evidence_level": "stratified_blinded_human_review"})

    decision = {
        "accepted_descriptive_face_validity": accepted,
        "acceptance_rule": "overall quadratic Cohen kappa >= 0.4 and high-vs-low directional pair accuracy >= 0.70",
        "overall_quadratic_cohen_kappa": kappa_main,
        "high_vs_low_directional_pair_accuracy": directional,
        "high_vs_low_pair_count": n_pairs,
        "high_vs_low_human_auc": auc,
        "claim_limit": "descriptive face validity only; not true quality, causal effect, or training-effect validation",
        "sample_limit": "84 stratified non-random records; high/low quality strata selected by robust Huber Q",
        "fallback_if_failed": "retain four facets; use equal-weight Q as main definition and Huber Q only as robustness comparison",
    }
    pd.DataFrame(metric_rows).to_csv(RESULTS / "q1_human_review_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(comparison).to_csv(RESULTS / "q1_human_review_q_comparison.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(RESULTS / "q1_human_review_consensus.csv", index=False, encoding="utf-8-sig")
    (RESULTS / "q1_human_review_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "score"])
    args = parser.parse_args()
    prepare() if args.command == "prepare" else score()


if __name__ == "__main__":
    main()
