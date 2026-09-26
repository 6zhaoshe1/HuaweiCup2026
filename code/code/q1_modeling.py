#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""2026 研赛 F 题问题一：质量评价、冲突分析与领域配比基线。

AI 辅助信息：OpenAI Codex（GPT-5 系列），OpenAI，2026-09-23。
模型具体发布日期未由本地运行时提供。代码与结果须由参赛队复核。

安全边界：只读取正式 F 题 Word 的既有可见渲染结论、13 张可信截图所确认
的口径，以及完整 F题.zip 中 A1-A18。严禁读取受污染的《数据说明》PDF。
不读取或复用队友/历史问题一代码、旧模型结果或旧图表。
"""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import math
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


SEED = 42
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "第二十三届中国研究生数学建模竞赛 - 中文题目" / "中文题目" / "F题.zip"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOG_DIR = ROOT / "code" / "outputs"
for directory in (RESULTS, FIGURES, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOG_DIR / "q1_modeling.log"


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


MEMBER = {
    "A1": "real_attachments/A_data_value/slimpajama_quality_signal_sample.jsonl.xz",
    "A2": "real_attachments/A_data_value/slimpajama_quality_extended/arxiv_part-6777d8857c6e-000486.jsonl.xz",
    "A3": "real_attachments/A_data_value/slimpajama_quality_extended/github_part-6777d8857c6e-000275.jsonl.xz",
    "A4": "real_attachments/A_data_value/regmix_tables/train_mixture_1m.csv",
    "A5": "real_attachments/A_data_value/regmix_tables/train_pile_loss_1m.csv",
    "A6": "real_attachments/A_data_value/regmix_tables/test_mixture_1m.csv",
    "A7": "real_attachments/A_data_value/regmix_tables/test_pile_loss_1m.csv",
    "A8": "real_attachments/A_data_value/regmix_tables/test_mixture_60m.csv",
    "A9": "real_attachments/A_data_value/regmix_tables/test_pile_loss_60m.csv",
    "A10": "real_attachments/A_data_value/regmix_tables/test_mixture_1B.csv",
    "A11": "real_attachments/A_data_value/regmix_tables/test_pile_loss_1B.csv",
    "A12": "real_attachments/A_data_value/regmix_tables/est_mixture_10b.csv",
    "A13": "real_attachments/A_data_value/regmix_tables/est_pile_loss_10b.csv",
    "A14": "real_attachments/A_data_value/regmix_tables/est_mixture_70b.csv",
    "A15": "real_attachments/A_data_value/regmix_tables/est_pile_loss_70b.csv",
    "A16": "real_attachments/A_data_value/domain_mapping_guide.csv",
    "A17": "real_attachments/A_data_value/regmix_domain_summary.csv",
    "A18": "real_attachments/A_data_value/regmix_domain_sample.jsonl.xz",
}

SEMANTIC_RAW = [
    "fineweb_edu", "fluency_margin", "cleanliness_expected", "readability_expected",
    "reasoning_expected", "professionalism_expected", "qurater_style",
    "qurater_expertise", "qurater_facts", "qurater_education", "ad_margin",
]
RPS_POSITIVE = ["rps_terminal", "rps_unique", "rps_entropy"]
RPS_TYPICAL = ["rps_no_alpha", "rps_numeric", "rps_uppercase"]
RPS_REVERSE = ["rps_top2", "rps_top3"]
DSIR_COLS = ["dsir_books", "dsir_wiki", "dsir_math"]


def safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def expected_level(values) -> float:
    if not isinstance(values, list) or len(values) != 6:
        return np.nan
    x = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(x)):
        return np.nan
    x = x - np.max(x)
    p = np.exp(x)
    p = p / p.sum()
    return float(np.dot(np.arange(6, dtype=float), p))


def margin(values, positive_index: int, negative_index: int) -> float:
    if not isinstance(values, list) or len(values) != 2:
        return np.nan
    a, b = safe_float(values[positive_index]), safe_float(values[negative_index])
    return a - b if np.isfinite(a) and np.isfinite(b) else np.nan


def quality_row(obj: dict, dataset: str, inferred_domain: str | None) -> dict:
    domain = obj.get("_source_domain") or inferred_domain or "unknown"
    qurater = obj.get("qurater")
    q = [safe_float(v) for v in qurater] if isinstance(qurater, list) and len(qurater) == 4 else [np.nan] * 4
    content = obj.get("content", "")
    return {
        "record_id": str(obj.get("id")),
        "dataset": dataset,
        "domain": str(domain),
        "source_path": str(obj.get("_source_path", obj.get("sub_path", ""))),
        "content_sha256": hashlib.sha256(str(content).encode("utf-8", errors="replace")).hexdigest(),
        "fineweb_edu": safe_float(obj.get("fineweb_edu", [np.nan])[0] if isinstance(obj.get("fineweb_edu"), list) and obj.get("fineweb_edu") else np.nan),
        "fluency_margin": margin(obj.get("fluency_en"), 1, 0),
        "cleanliness_expected": expected_level(obj.get("modernbert_cleanliness")),
        "readability_expected": expected_level(obj.get("modernbert_readability")),
        "reasoning_expected": expected_level(obj.get("modernbert_reasoning")),
        "professionalism_expected": expected_level(obj.get("modernbert_professionalism")),
        "qurater_style": q[0], "qurater_expertise": q[1], "qurater_facts": q[2], "qurater_education": q[3],
        "ad_margin": margin(obj.get("ad_en"), 1, 0),
        "dsir_books": safe_float(obj.get("dsir_books")), "dsir_wiki": safe_float(obj.get("dsir_wiki")),
        "dsir_math": safe_float(obj.get("dsir_math")),
        "word_count": safe_float(obj.get("rps_doc_word_count")),
        "num_sentences": safe_float(obj.get("rps_doc_num_sentences")),
        "mean_word_length": safe_float(obj.get("rps_doc_mean_word_length")),
        "rps_terminal": safe_float(obj.get("rps_lines_ending_with_terminal_punctution_mark")),
        "rps_unique": safe_float(obj.get("rps_doc_frac_unique_words")),
        "rps_entropy": safe_float(obj.get("rps_doc_unigram_entropy")),
        "rps_no_alpha": safe_float(obj.get("rps_doc_frac_no_alph_words")),
        "rps_numeric": safe_float(obj.get("rps_lines_numerical_chars_fraction")),
        "rps_uppercase": safe_float(obj.get("rps_lines_uppercase_letter_fraction")),
        "rps_top2": safe_float(obj.get("rps_doc_frac_chars_top_2gram")),
        "rps_top3": safe_float(obj.get("rps_doc_frac_chars_top_3gram")),
    }


def iter_xz_json(zf: zipfile.ZipFile, name: str):
    with zf.open(name, "r") as raw, lzma.LZMAFile(raw, "rb") as xz, io.TextIOWrapper(xz, encoding="utf-8") as text:
        for line in text:
            if line.strip():
                yield json.loads(line)


def ecdf_apply(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    vals = np.asarray(values, dtype=float)
    out = np.full(vals.shape, np.nan, dtype=float)
    mask = np.isfinite(vals)
    if ref.size:
        out[mask] = (np.searchsorted(ref, vals[mask], side="right") + 0.5) / (ref.size + 1.0)
        out[mask] = np.clip(out[mask], 1.0 / (ref.size + 1.0), ref.size / (ref.size + 1.0))
    return out


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return 0.0, 1.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) * 1.4826
    return med, max(mad, 1e-8)


def huber_row_mean(matrix: np.ndarray, c: float = 1.5) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    med = np.nanmedian(x, axis=1)
    absdev = np.abs(x - med[:, None])
    scale = 1.4826 * np.nanmedian(absdev, axis=1)
    scale = np.where(scale > 1e-8, scale, 1.0)
    u = absdev / scale[:, None]
    w = np.where(np.isfinite(x), np.minimum(1.0, c / np.maximum(u, 1e-12)), 0.0)
    denom = w.sum(axis=1)
    return np.divide(np.nansum(w * x, axis=1), denom, out=np.full(x.shape[0], np.nan), where=denom > 0)


def fit_length_bands(df: pd.DataFrame) -> dict[str, list[float]]:
    out = {}
    ref = df[df["dataset"] == "A1"]
    for domain, part in ref.groupby("domain"):
        out[domain] = np.quantile(np.log1p(part["word_count"].clip(lower=0)), [0.2, 0.4, 0.6, 0.8]).tolist()
    return out


def assign_length_bands(df: pd.DataFrame, cuts: dict[str, list[float]]) -> np.ndarray:
    global_cuts = np.quantile(np.log1p(df.loc[df["dataset"] == "A1", "word_count"].clip(lower=0)), [0.2, 0.4, 0.6, 0.8])
    result = np.zeros(len(df), dtype=int)
    logs = np.log1p(df["word_count"].clip(lower=0).to_numpy(float))
    domains = df["domain"].astype(str).to_numpy()
    for domain in np.unique(domains):
        mask = domains == domain
        result[mask] = np.searchsorted(np.asarray(cuts.get(domain, global_cuts), dtype=float), logs[mask], side="right")
    return result


def grouped_ecdf(df: pd.DataFrame, raw_col: str, group_cols: list[str]) -> np.ndarray:
    out = np.full(len(df), np.nan)
    ref = df[df["dataset"] == "A1"]
    global_ref = ref[raw_col].to_numpy(float)
    domain_refs = {k: g[raw_col].to_numpy(float) for k, g in ref.groupby("domain")}
    group_refs = {k: g[raw_col].to_numpy(float) for k, g in ref.groupby(group_cols)}
    for key, idx in df.groupby(group_cols).groups.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        r = group_refs.get(key)
        if r is None or np.isfinite(r).sum() < 50:
            r = domain_refs.get(key_tuple[0], global_ref)
        out[np.asarray(idx, dtype=int)] = ecdf_apply(r, df.loc[idx, raw_col].to_numpy(float))
    return out


def bootstrap_mean_ci(values: np.ndarray, weights: np.ndarray | None = None, n_boot: int = 200) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x)
    x = x[mask]
    w = None if weights is None else np.asarray(weights, dtype=float)[mask]
    if x.size < 2:
        return np.nan, np.nan
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, x.size, x.size)
        estimates[b] = np.mean(x[idx]) if w is None else np.average(x[idx], weights=w[idx])
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def read_quality_data(zf: zipfile.ZipFile) -> pd.DataFrame:
    frames = []
    for dataset, domain in (("A1", None), ("A2", "arxiv"), ("A3", "github")):
        log(f"streaming {dataset}")
        rows = [quality_row(obj, dataset, domain) for obj in iter_xz_json(zf, MEMBER[dataset])]
        frames.append(pd.DataFrame.from_records(rows))
        log(f"{dataset}: {len(rows):,} records")
    df = pd.concat(frames, ignore_index=True)
    df["nonfinite_semantic_count"] = df[SEMANTIC_RAW].isna().sum(axis=1)
    return df


def score_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    ref_mask = df["dataset"] == "A1"
    for col in SEMANTIC_RAW:
        df[f"s_{col}"] = ecdf_apply(df.loc[ref_mask, col].to_numpy(float), df[col].to_numpy(float))
        missing = df[f"s_{col}"].isna()
        if missing.any():
            med = df.loc[ref_mask].groupby("domain")[f"s_{col}"].median()
            df.loc[missing, f"s_{col}"] = df.loc[missing, "domain"].map(med).fillna(0.5)

    cuts = fit_length_bands(df)
    df["length_band"] = assign_length_bands(df, cuts)
    for col in RPS_POSITIVE + RPS_TYPICAL + RPS_REVERSE:
        u = grouped_ecdf(df, col, ["domain", "length_band"])
        if col in RPS_POSITIVE:
            score = u
        elif col in RPS_REVERSE:
            score = 1.0 - u
        else:
            score = 1.0 - 2.0 * np.abs(u - 0.5)
        df[f"s_{col}"] = np.clip(score, 0.0, 1.0)

    facets = {
        "content_value": ["s_fineweb_edu", "s_qurater_facts", "s_qurater_education"],
        "language_quality": ["s_fluency_margin", "s_readability_expected", "s_qurater_style"] + [f"s_{c}" for c in RPS_POSITIVE],
        "cleanliness": ["s_cleanliness_expected", "s_ad_margin"] + [f"s_{c}" for c in RPS_TYPICAL + RPS_REVERSE],
        "reasoning_professional": ["s_reasoning_expected", "s_professionalism_expected", "s_qurater_expertise"],
    }
    for facet, cols in facets.items():
        mat = df[cols].to_numpy(float)
        df[f"facet_{facet}_equal"] = np.nanmean(mat, axis=1)
        df[f"facet_{facet}_robust"] = huber_row_mean(mat)
    equal_cols = [f"facet_{k}_equal" for k in facets]
    robust_cols = [f"facet_{k}_robust" for k in facets]
    df["q_definition_equal"] = df[equal_cols].mean(axis=1)
    df["q_definition_robust"] = df[robust_cols].mean(axis=1)
    semantic_cols = [f"s_{c}" for c in SEMANTIC_RAW]
    morphology_cols = [f"s_{c}" for c in RPS_POSITIVE + RPS_TYPICAL + RPS_REVERSE]
    df["semantic_family_score"] = df[semantic_cols].mean(axis=1)
    df["morphology_family_score"] = df[morphology_cols].mean(axis=1)
    df["family_gap"] = df["semantic_family_score"] - df["morphology_family_score"]

    ref = df[ref_mask]
    stats = {}
    for key, part in ref.groupby(["domain", "length_band"]):
        stats[key] = robust_location_scale(part["family_gap"].to_numpy(float))
    global_stat = robust_location_scale(ref["family_gap"].to_numpy(float))
    z = np.empty(len(df))
    for i, row in enumerate(df[["domain", "length_band", "family_gap"]].itertuples(index=False)):
        med, scale = stats.get((row.domain, row.length_band), global_stat)
        z[i] = abs(row.family_gap - med) / scale
    df["domain_length_conflict_z"] = z
    df["conflict_strength"] = 0.5 * np.abs(df["family_gap"]) + 0.5 * np.clip(z / 6.0, 0.0, 1.0)
    thresholds = {str(q): float(df.loc[ref_mask, "conflict_strength"].quantile(q)) for q in (0.95, 0.975, 0.99)}
    df["high_conflict"] = df["conflict_strength"] >= thresholds["0.975"]

    structural_z = np.zeros((len(df), 3))
    structural_raw = [np.log1p(df["word_count"].clip(lower=0)), np.log1p(df["num_sentences"].clip(lower=0)), df["mean_word_length"]]
    for j, values in enumerate(structural_raw):
        ref_values = pd.Series(values[ref_mask.to_numpy()], index=df.index[ref_mask])
        domain_stats = {k: robust_location_scale(ref_values.loc[g.index].to_numpy()) for k, g in ref.groupby("domain")}
        for domain, idx in df.groupby("domain").groups.items():
            med, scale = domain_stats.get(domain, robust_location_scale(ref_values.to_numpy()))
            structural_z[np.asarray(idx, dtype=int), j] = np.abs(np.asarray(values)[np.asarray(idx, dtype=int)] - med) / scale
    df["structural_outlier"] = np.nanmax(structural_z, axis=1) > 6.0
    df["measurement_anomaly"] = (df["nonfinite_semantic_count"] > 0) | df["structural_outlier"]

    dsir_rows = []
    for col in DSIR_COLS:
        df[f"{col}_per_word"] = df[col] / df["word_count"].clip(lower=1)
        df[f"{col}_len_resid"] = np.nan
        for domain, idx in df.groupby("domain").groups.items():
            idx = np.asarray(idx, dtype=int)
            train_idx = df.index[ref_mask & (df["domain"] == domain)].to_numpy()
            if train_idx.size < 30:
                train_idx = df.index[ref_mask].to_numpy()
            xtr = np.log1p(df.loc[train_idx, "word_count"].to_numpy(float))
            ytr = df.loc[train_idx, f"{col}_per_word"].to_numpy(float)
            ok = np.isfinite(xtr) & np.isfinite(ytr)
            coef = np.polyfit(xtr[ok], ytr[ok], 1)
            x = np.log1p(df.loc[idx, "word_count"].to_numpy(float))
            df.loc[idx, f"{col}_len_resid"] = df.loc[idx, f"{col}_per_word"].to_numpy(float) - np.polyval(coef, x)
        for (dataset, domain), part in df.groupby(["dataset", "domain"]):
            lx = np.log1p(part["word_count"].to_numpy(float))
            dsir_rows.append({
                "dataset": dataset, "domain": domain, "metric": col,
                "corr_raw_log_length": float(np.corrcoef(part[col], lx)[0, 1]),
                "corr_per_word_log_length": float(np.corrcoef(part[f"{col}_per_word"], lx)[0, 1]),
                "corr_residual_log_length": float(np.corrcoef(part[f"{col}_len_resid"], lx)[0, 1]),
                "evidence_level": "real_observational",
            })
    dsir_df = pd.DataFrame(dsir_rows)

    manifest_rows = []
    uses = {
        "fineweb_edu": ("内容价值", "A1 ECDF，正向", "q"), "fluency_en": ("语言质量", "正类-负类 logit margin 后 A1 ECDF", "q"),
        "modernbert_cleanliness": ("清洁度", "softmax 期望等级后 A1 ECDF；未校准", "q"),
        "modernbert_readability": ("语言质量", "softmax 期望等级后 A1 ECDF；未校准", "q"),
        "modernbert_reasoning": ("推理专业", "softmax 期望等级后 A1 ECDF；未校准", "q"),
        "modernbert_professionalism": ("推理专业", "softmax 期望等级后 A1 ECDF；未校准", "q"),
        "qurater": ("四评价头", "四位置分别 A1 ECDF，不作 argmax", "q"), "ad_en": ("清洁度", "无广告-含广告 logit margin 后 A1 ECDF", "q"),
        "dsir_books/wiki/math": ("目标域相似度", "先除以词数，再在域内对 ln(word_count) 残差化；残差为主口径", "auxiliary"),
        "rps_doc_word_count": ("结构", "仅长度带和异常检测", "structure"), "rps_doc_num_sentences": ("结构", "仅异常检测", "structure"),
        "rps_doc_unigram_entropy": ("语言形态", "域内长度带 ECDF，温和正向", "q"), "rps_doc_frac_unique_words": ("语言形态", "域内长度带 ECDF，温和正向", "q"),
        "rps_doc_frac_no_alph_words": ("域条件形态", "域内长度带典型性", "q"), "rps_doc_frac_chars_top_2gram": ("重复", "域内长度带反向 ECDF", "q"),
        "rps_doc_frac_chars_top_3gram": ("重复", "域内长度带反向 ECDF", "q"), "rps_lines_uppercase_letter_fraction": ("域条件形态", "域内长度带典型性", "q"),
        "rps_lines_ending_with_terminal_punctution_mark": ("语言形态", "域内长度带 ECDF，温和正向", "q"),
        "rps_lines_numerical_chars_fraction": ("域条件形态", "域内长度带典型性", "q"), "rps_doc_mean_word_length": ("结构", "仅异常检测", "structure"),
    }
    for order, (field, (family, transform, role)) in enumerate(uses.items(), start=1):
        manifest_rows.append({"position": order, "field": field, "family": family, "transform": transform, "role": role, "evidence_level": "metadata"})
    return df, pd.DataFrame(manifest_rows), dsir_df, {"length_band_cuts": cuts, "conflict_thresholds": thresholds, "facets": facets}


def quality_aggregates(df: pd.DataFrame, thresholds: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    priority = df["dataset"].map({"A1": 0, "A2": 1, "A3": 1}).fillna(0)
    pooled = df.assign(_priority=priority).sort_values("_priority").drop_duplicates("record_id", keep="last").drop(columns="_priority")
    pooled = pooled.assign(dataset="pooled_unique")
    work = pd.concat([df, pooled], ignore_index=True)
    rows = []
    facet_cols = ["facet_content_value_robust", "facet_language_quality_robust", "facet_cleanliness_robust", "facet_reasoning_professional_robust"]
    for (dataset, domain), part in work.groupby(["dataset", "domain"]):
        q = part["q_definition_robust"].to_numpy(float)
        weights = part["word_count"].clip(lower=1).to_numpy(float)
        lo, hi = bootstrap_mean_ci(q)
        row = {
            "dataset": dataset, "domain": domain, "n": len(part),
            "q_equal_mean": part["q_definition_equal"].mean(), "q_robust_mean": np.mean(q),
            "q_robust_median": np.median(q), "q_robust_std": np.std(q, ddof=1),
            "q_p05": np.quantile(q, 0.05), "q_p25": np.quantile(q, 0.25), "q_p75": np.quantile(q, 0.75), "q_p95": np.quantile(q, 0.95),
            "q_document_mean_ci_low": lo, "q_document_mean_ci_high": hi,
            "q_length_weighted_mean": np.average(q, weights=weights),
            "high_conflict_rate": part["high_conflict"].mean(), "measurement_anomaly_rate": part["measurement_anomaly"].mean(),
            "evidence_level": "real_observational",
        }
        row.update({c: part[c].mean() for c in facet_cols})
        rows.append(row)
    agg = pd.DataFrame(rows)

    comparisons = []
    for domain, ext in (("arxiv", "A2"), ("github", "A3")):
        sample = df[(df.dataset == "A1") & (df.domain == domain)]
        full = df[df.dataset == ext]
        ext_only = full[~full.record_id.isin(sample.record_id)]
        for target_name, target in (("full", full), ("extension_only", ext_only)):
            for score in ("q_definition_equal", "q_definition_robust", "conflict_strength"):
                a, b = sample[score].to_numpy(float), target[score].to_numpy(float)
                comparisons.append({
                    "domain": domain, "comparison": f"A1_sample_vs_{target_name}", "score": score,
                    "n_sample": len(a), "n_target": len(b), "mean_sample": np.mean(a), "mean_target": np.mean(b),
                    "mean_difference_target_minus_sample": np.mean(b) - np.mean(a),
                    "ks_statistic": ks_2samp(a, b).statistic, "wasserstein_distance": wasserstein_distance(a, b),
                    "evidence_level": "real_observational",
                })
    comparison_df = pd.DataFrame(comparisons)

    pareto_base = agg[agg.dataset == "pooled_unique"].copy()
    fcols = facet_cols
    dominated = []
    X = pareto_base[fcols].to_numpy(float)
    for i in range(len(X)):
        dominated.append(bool(np.any(np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1))))
    pareto_base["pareto_nondominated"] = ~np.asarray(dominated)
    sensitivity_rows, resolution_rows = [], []
    for (dataset, domain), part in work.groupby(["dataset", "domain"]):
        for quantile, threshold in thresholds.items():
            sensitivity_rows.append({"dataset": dataset, "domain": domain, "reference_quantile": float(quantile),
                                     "threshold": threshold, "conflict_rate": float((part.conflict_strength >= threshold).mean()),
                                     "n": len(part), "evidence_level": "real_observational"})
        delta = np.abs(part.q_definition_robust - part.q_definition_equal)
        resolution_rows.append({
            "dataset": dataset, "domain": domain, "n": len(part),
            "spearman_equal_vs_robust": spearmanr(part.q_definition_equal, part.q_definition_robust).statistic,
            "mean_abs_score_change_all": delta.mean(),
            "mean_abs_score_change_high_conflict": delta[part.high_conflict].mean() if part.high_conflict.any() else np.nan,
            "mean_abs_score_change_measurement_anomaly": delta[part.measurement_anomaly].mean() if part.measurement_anomaly.any() else np.nan,
            "evidence_level": "real_observational",
        })
    return agg, comparison_df, pareto_base, pd.DataFrame(sensitivity_rows), pd.DataFrame(resolution_rows)


def make_manual_audit_sample(zf: zipfile.ZipFile, df: pd.DataFrame) -> pd.DataFrame:
    a1 = df[df.dataset == "A1"].copy()
    selected = []
    for domain, part in a1.groupby("domain"):
        strata = {
            "high_quality": part.nlargest(3, "q_definition_robust"),
            "low_quality": part.nsmallest(3, "q_definition_robust"),
            "high_conflict": part.nlargest(3, "conflict_strength"),
            "low_conflict": part.nsmallest(3, "conflict_strength"),
        }
        for stratum, picks in strata.items():
            tmp = picks.copy(); tmp["audit_stratum"] = stratum; selected.append(tmp)
    chosen = pd.concat(selected, ignore_index=True).drop_duplicates("record_id")
    wanted = set(chosen.record_id)
    excerpts = {}
    for obj in iter_xz_json(zf, MEMBER["A1"]):
        rid = str(obj.get("id"))
        if rid in wanted:
            content = str(obj.get("content", "")).replace("\x00", "")
            excerpts[rid] = content[:2000]
    cols = ["record_id", "domain", "audit_stratum", "q_definition_robust", "conflict_strength", "measurement_anomaly", "content_sha256"]
    out = chosen[cols].copy()
    out["content_excerpt"] = out.record_id.map(excerpts)
    out["human_label"] = ""
    out["human_note"] = ""
    out["evidence_level"] = "real_observational"
    return out


def csv_member(zf: zipfile.ZipFile, logical_id: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(zf.read(MEMBER[logical_id])))


def validate_mixture_tables(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str], list[str]]:
    pairs = [("A4", "A5"), ("A6", "A7"), ("A8", "A9"), ("A10", "A11"), ("A12", "A13"), ("A14", "A15")]
    mixture_cols = [c for c in tables["A4"].columns if c.startswith("train_the_pile_")]
    loss_cols = [c for c in tables["A5"].columns if c.endswith("_val_loss")]
    if len(mixture_cols) != 17 or len(loss_cols) != 13:
        raise RuntimeError(f"Expected 17 mixture and 13 loss columns, got {len(mixture_cols)} and {len(loss_cols)}")
    rows = []
    for mix_id, loss_id in pairs:
        mix, loss = tables[mix_id], tables[loss_id]
        same_keys = mix["index"].is_unique and loss["index"].is_unique and set(mix["index"]) == set(loss["index"])
        if not same_keys:
            raise RuntimeError(f"Association key mismatch: {mix_id}/{loss_id}")
        sums = mix[mixture_cols].sum(axis=1)
        rows.append({
            "mixture_id": mix_id, "loss_id": loss_id, "rows": len(mix), "key": "index", "one_to_one": True,
            "max_abs_simplex_error": float(np.max(np.abs(sums - 1.0))),
            "negative_value_count": int((mix[mixture_cols] < 0).sum().sum()),
            "missing_mixture": int(mix[mixture_cols].isna().sum().sum()), "missing_loss": int(loss[loss_cols].isna().sum().sum()),
            "evidence_level": "extrapolated" if mix_id in ("A12", "A14") else "real_holdout",
        })
    return pd.DataFrame(rows), mixture_cols, loss_cols


def join_pair(tables: dict[str, pd.DataFrame], mix_id: str, loss_id: str, mixture_cols: list[str], loss_cols: list[str]):
    joined = tables[mix_id].merge(tables[loss_id], on="index", how="inner", validate="one_to_one")
    if len(joined) != len(tables[mix_id]):
        raise RuntimeError(f"Join row loss for {mix_id}/{loss_id}")
    return joined["index"].to_numpy(), joined[mixture_cols].to_numpy(float), joined[loss_cols].to_numpy(float)


def zero_replace(P: np.ndarray, delta: float) -> np.ndarray:
    P = np.asarray(P, dtype=float).copy()
    if np.any(P < 0):
        raise ValueError("negative composition")
    out = np.empty_like(P)
    for i, row in enumerate(P):
        zero = row <= 0
        k = int(zero.sum())
        if k * delta >= 1:
            raise ValueError("delta too large")
        positive = ~zero
        out[i, zero] = delta
        out[i, positive] = row[positive] / row[positive].sum() * (1.0 - k * delta)
    return out


ILR_BASIS = helmert(17, full=False)


def ilr_features(P: np.ndarray, delta: float, degree: int = 1) -> tuple[np.ndarray, list[str]]:
    comp = zero_replace(P, delta)
    X = np.log(comp) @ ILR_BASIS.T
    names = [f"ilr_{i+1}" for i in range(X.shape[1])]
    if degree == 2:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X = poly.fit_transform(X)
        names = list(poly.get_feature_names_out(names))
    return X, names


def fit_ridge(X: np.ndarray, Y: np.ndarray, alpha: float):
    sx, sy = StandardScaler(), StandardScaler()
    Xs, Ys = sx.fit_transform(X), sy.fit_transform(Y)
    model = Ridge(alpha=alpha).fit(Xs, Ys)
    return sx, sy, model


def predict_ridge(fitted, X: np.ndarray) -> np.ndarray:
    sx, sy, model = fitted
    return sy.inverse_transform(model.predict(sx.transform(X)))


def metrics_rows(y: np.ndarray, pred: np.ndarray, loss_cols: list[str], dataset: str, model: str, evidence: str) -> list[dict]:
    rows = []
    for j, col in enumerate(loss_cols):
        rho = spearmanr(y[:, j], pred[:, j]).statistic
        rows.append({
            "dataset": dataset, "model": model, "domain": col.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
            "n": len(y), "mae": mean_absolute_error(y[:, j], pred[:, j]),
            "rmse": mean_squared_error(y[:, j], pred[:, j]) ** 0.5,
            "r2": r2_score(y[:, j], pred[:, j]), "spearman": rho,
            "bias_pred_minus_true": float(np.mean(pred[:, j] - y[:, j])), "evidence_level": evidence,
        })
    rows.append({
        "dataset": dataset, "model": model, "domain": "macro",
        "n": len(y), "mae": np.mean([r["mae"] for r in rows]), "rmse": np.mean([r["rmse"] for r in rows]),
        "r2": np.mean([r["r2"] for r in rows]), "spearman": np.nanmean([r["spearman"] for r in rows]),
        "bias_pred_minus_true": float(np.mean(pred - y)), "evidence_level": evidence,
    })
    return rows


def cross_validate(P: np.ndarray, Y: np.ndarray, folds: np.ndarray, degree: int, deltas: list[float], alphas: list[float], extra: np.ndarray | None = None):
    rows, best = [], None
    for delta in deltas:
        base, names = ilr_features(P, delta, degree)
        X = base if extra is None else np.column_stack([base, extra])
        for alpha in alphas:
            oof = np.full_like(Y, np.nan)
            for fold in sorted(np.unique(folds)):
                tr, va = folds != fold, folds == fold
                fitted = fit_ridge(X[tr], Y[tr], alpha)
                oof[va] = predict_ridge(fitted, X[va])
            rmse = mean_squared_error(Y, oof) ** 0.5
            rho = np.nanmedian([spearmanr(Y[:, j], oof[:, j]).statistic for j in range(Y.shape[1])])
            record = {"degree": degree, "delta": delta, "alpha": alpha, "cv_rmse": rmse, "cv_spearman_median": rho, "evidence_level": "real_observational"}
            rows.append(record)
            if best is None or (rmse, -rho) < (best[0], -best[1]):
                best = (rmse, rho, delta, alpha, oof, names)
    return pd.DataFrame(rows), best


def qmix_features(P: np.ndarray, mixture_cols: list[str], mapping: pd.DataFrame, domain_quality: dict[str, float]) -> tuple[np.ndarray, pd.DataFrame]:
    col_domains = [c.removeprefix("train_the_pile_") for c in mixture_cols]
    map_dict = mapping.set_index("mixture_domain").to_dict("index")
    q = np.full(len(col_domains), np.nan)
    reliable = np.zeros(len(col_domains), dtype=bool)
    rows = []
    for j, domain in enumerate(col_domains):
        item = map_dict.get(domain, {})
        qdomain, mtype = item.get("quality_domain", "(none)"), item.get("mapping_type", "missing")
        if qdomain in domain_quality and mtype in ("direct", "near_direct"):
            q[j], reliable[j] = domain_quality[qdomain], True
        rows.append({"mixture_domain": domain, "quality_domain": qdomain, "mapping_type": mtype,
                     "domain_q": q[j], "reliable_for_qmix": reliable[j], "evidence_level": "metadata"})
    coverage = P[:, reliable].sum(axis=1)
    contribution = P[:, reliable] @ q[reliable]
    mapped_mean = np.divide(contribution, coverage, out=np.full(len(P), np.nan), where=coverage > 0)
    fill = np.nanmedian(mapped_mean)
    mapped_mean = np.where(np.isfinite(mapped_mean), mapped_mean, fill)
    return np.column_stack([coverage, mapped_mean]), pd.DataFrame(rows)


def run_mixture_models(tables: dict[str, pd.DataFrame], mixture_cols: list[str], loss_cols: list[str], mapping: pd.DataFrame, domain_quality: dict[str, float]):
    ids4, P4, Y4 = join_pair(tables, "A4", "A5", mixture_cols, loss_cols)
    folds = np.full(len(ids4), -1, dtype=int)
    for fold, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(P4)):
        folds[va] = fold
    mean_oof = np.full_like(Y4, np.nan)
    for fold in sorted(np.unique(folds)):
        tr, va = folds != fold, folds == fold
        mean_oof[va] = np.mean(Y4[tr], axis=0)
    mean_rmse = mean_squared_error(Y4, mean_oof) ** 0.5
    deltas, alphas = [1e-6, 1e-4, 5e-4, 1e-3], list(np.logspace(-4, 4, 9))
    cv1, best1 = cross_validate(P4, Y4, folds, 1, deltas, alphas)
    improvement = (mean_rmse - best1[0]) / mean_rmse
    baseline_failed = improvement < 0.10 or best1[1] < 0.50
    cv_frames = [cv1.assign(model="ilr_ridge")]
    chosen_degree, chosen = 1, best1
    if baseline_failed:
        log(f"ILR baseline failure gate triggered: improvement={improvement:.3f}, median rho={best1[1]:.3f}")
        cv2, best2 = cross_validate(P4, Y4, folds, 2, deltas, alphas)
        cv_frames.append(cv2.assign(model="quadratic_ilr_ridge"))
        if best2[0] <= 0.97 * best1[0]:
            chosen_degree, chosen = 2, best2
    model_name = "quadratic_ilr_ridge" if chosen_degree == 2 else "ilr_ridge"
    best_rmse, best_rho, delta, alpha, oof, feature_names = chosen
    X4, feature_names = ilr_features(P4, delta, chosen_degree)
    fitted = fit_ridge(X4, Y4, alpha)

    qextra4, mapping_audit = qmix_features(P4, mixture_cols, mapping, domain_quality)
    qmix_summary_rows = [{"dataset": "A4_A5_train", "mapped_coverage_min": qextra4[:, 0].min(),
                          "mapped_coverage_median": np.median(qextra4[:, 0]), "mapped_coverage_max": qextra4[:, 0].max(),
                          "mapped_q_mean": np.mean(qextra4[:, 1]), "evidence_level": "real_observational"}]
    cvq, bestq = cross_validate(P4, Y4, folds, chosen_degree, [delta], alphas, extra=qextra4)
    Xq4 = np.column_stack([X4, qextra4])
    fitted_q = fit_ridge(Xq4, Y4, bestq[3])
    cv_frames.append(cvq.assign(model=f"{model_name}_plus_qmix"))

    metric_rows = metrics_rows(Y4, oof, loss_cols, "A4_A5_oof", model_name, "real_observational")
    predictions, dataset_cache = [], {}
    specs = [
        ("A6_A7_1m", "A6", "A7", 1.0, "real_holdout"),
        ("A8_A9_60m", "A8", "A9", 60.0, "real_holdout"),
        ("A10_A11_1b", "A10", "A11", 1000.0, "real_holdout"),
        ("A12_A13_10b", "A12", "A13", 10000.0, "extrapolated"),
        ("A14_A15_70b", "A14", "A15", 70000.0, "extrapolated"),
    ]
    for label, mid, lid, nratio, evidence in specs:
        ids, P, Y = join_pair(tables, mid, lid, mixture_cols, loss_cols)
        X, _ = ilr_features(P, delta, chosen_degree)
        pred = predict_ridge(fitted, X)
        qextra, _ = qmix_features(P, mixture_cols, mapping, domain_quality)
        qmix_summary_rows.append({"dataset": label, "mapped_coverage_min": qextra[:, 0].min(),
                                  "mapped_coverage_median": np.median(qextra[:, 0]), "mapped_coverage_max": qextra[:, 0].max(),
                                  "mapped_q_mean": np.mean(qextra[:, 1]), "evidence_level": evidence})
        predq = predict_ridge(fitted_q, np.column_stack([X, qextra]))
        metric_rows += metrics_rows(Y, pred, loss_cols, label, model_name, evidence)
        metric_rows += metrics_rows(Y, predq, loss_cols, label, f"{model_name}_plus_qmix", evidence)
        dataset_cache[label] = (ids, P, Y, pred, predq, nratio, evidence)
        for i, rid in enumerate(ids):
            for j, col in enumerate(loss_cols):
                predictions.append({"dataset": label, "record_id": int(rid), "domain": col.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                                    "observed_loss": Y[i, j], "predicted_loss": pred[i, j], "model": model_name, "evidence_level": evidence})

    # One explicit continuous-scale correction. 1m/60m paired sets calibrate; 1B is untouched test.
    y1 = dataset_cache["A6_A7_1m"][2]
    y60 = dataset_cache["A8_A9_60m"][2]
    if not np.allclose(dataset_cache["A6_A7_1m"][1], dataset_cache["A8_A9_60m"][1]):
        raise RuntimeError("A6/A8 mixtures are not paired; scale correction invalid")
    scale_slope = np.median((y60 - y1) / np.log(60.0), axis=0)
    scale_rows = []
    for j, col in enumerate(loss_cols):
        diffs = (y60[:, j] - y1[:, j]) / np.log(60.0)
        scale_rows.append({"domain": col.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                           "logN_slope": scale_slope[j], "slope_iqr": np.quantile(diffs, 0.75) - np.quantile(diffs, 0.25),
                           "calibration_sets": "A6_A7+A8_A9", "evidence_level": "real_observational"})
    for label in ("A10_A11_1b", "A12_A13_10b", "A14_A15_70b"):
        ids, P, Y, pred, _, nratio, evidence = dataset_cache[label]
        corrected = pred + np.log(nratio) * scale_slope[None, :]
        metric_rows += metrics_rows(Y, corrected, loss_cols, label, f"{model_name}_logN_offset", evidence)
        for i, rid in enumerate(ids):
            for j, col in enumerate(loss_cols):
                predictions.append({"dataset": label, "record_id": int(rid), "domain": col.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                                    "observed_loss": Y[i, j], "predicted_loss": corrected[i, j], "model": f"{model_name}_logN_offset", "evidence_level": evidence})

    # Bootstrap the untouched 1B test by recipe rows.
    ids, P, Y, pred, _, _, _ = dataset_cache["A10_A11_1b"]
    corrected = pred + np.log(1000.0) * scale_slope[None, :]
    boot_rows = []
    for model_label, yp in ((model_name, pred), (f"{model_name}_logN_offset", corrected)):
        vals = []
        for _ in range(1000):
            idx = RNG.integers(0, len(Y), len(Y))
            rmse = mean_squared_error(Y[idx], yp[idx]) ** 0.5
            rho = np.nanmean([spearmanr(Y[idx, j], yp[idx, j]).statistic for j in range(Y.shape[1])])
            vals.append((rmse, rho))
        vals = np.asarray(vals)
        for metric, j in (("rmse_micro", 0), ("spearman_macro", 1)):
            boot_rows.append({"dataset": "A10_A11_1b", "model": model_label, "metric": metric,
                              "estimate": (mean_squared_error(Y, yp) ** 0.5) if j == 0 else np.nanmean([spearmanr(Y[:, k], yp[:, k]).statistic for k in range(Y.shape[1])]),
                              "ci_low": np.quantile(vals[:, j], 0.025), "ci_high": np.quantile(vals[:, j], 0.975),
                              "bootstrap_unit": "recipe_record", "n_boot": 1000, "evidence_level": "real_holdout"})

    # Proportional replacement scenario at all A4 recipes; association, not causality.
    base_pred = predict_ridge(fitted, X4)
    effects, effects_by_initial, donor_effects, pair_effects = [], [], [], []
    eps = 0.01
    for j, col in enumerate(mixture_cols):
        feasible = P4[:, j] <= 1.0 - eps
        Pm = P4[feasible].copy()
        old = Pm[:, j].copy()
        factor = (1.0 - old - eps) / np.maximum(1.0 - old, 1e-12)
        Pm *= factor[:, None]
        Pm[:, j] = old + eps
        Xm, _ = ilr_features(Pm, delta, chosen_degree)
        delta_loss = predict_ridge(fitted, Xm) - base_pred[feasible]
        for h, lcol in enumerate(loss_cols):
            vals = delta_loss[:, h]
            boots = [np.mean(vals[RNG.integers(0, len(vals), len(vals))]) for _ in range(300)]
            effects.append({"increased_domain": col.removeprefix("train_the_pile_"),
                            "validation_domain": lcol.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                            "increase_share": eps, "replacement_rule": "reduce_all_other_domains_proportionally",
                            "mean_delta_loss": np.mean(vals), "median_delta_loss": np.median(vals),
                            "ci_low": np.quantile(boots, 0.025), "ci_high": np.quantile(boots, 0.975),
                            "n_recipes": len(vals), "evidence_level": "model_scenario", "claim_scope": "noncausal_model_association"})
            shares = P4[feasible, j]
            cuts = np.unique(np.quantile(shares, [1/3, 2/3]))
            bands = np.digitize(shares, cuts, right=True)
            for band in np.unique(bands):
                chosen_band = bands == band
                effects_by_initial.append({"increased_domain": col.removeprefix("train_the_pile_"),
                                           "validation_domain": lcol.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                                           "initial_share_band": int(band), "initial_share_min": shares[chosen_band].min(),
                                           "initial_share_max": shares[chosen_band].max(), "mean_delta_loss": vals[chosen_band].mean(),
                                           "n_recipes": int(chosen_band.sum()), "evidence_level": "model_scenario",
                                           "claim_scope": "noncausal_model_association"})

    # Alternative path: take the full 1 percentage point only from Pile-CC.
    donor_j = mixture_cols.index("train_the_pile_pile_cc")
    for j, col in enumerate(mixture_cols):
        if j == donor_j:
            continue
        feasible = (P4[:, donor_j] >= eps) & (P4[:, j] <= 1.0 - eps)
        if feasible.sum() < 10:
            continue
        Pm = P4[feasible].copy(); Pm[:, j] += eps; Pm[:, donor_j] -= eps
        Xm, _ = ilr_features(Pm, delta, chosen_degree)
        dloss = predict_ridge(fitted, Xm) - base_pred[feasible]
        for h, lcol in enumerate(loss_cols):
            donor_effects.append({"increased_domain": col.removeprefix("train_the_pile_"), "donor_domain": "pile_cc",
                                  "validation_domain": lcol.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                                  "increase_share": eps, "mean_delta_loss": dloss[:, h].mean(), "median_delta_loss": np.median(dloss[:, h]),
                                  "n_recipes": int(feasible.sum()), "evidence_level": "model_scenario",
                                  "claim_scope": "noncausal_model_association"})

    # Pair interaction: add 0.5 percentage point to each member and reduce all other domains proportionally.
    half = eps / 2.0
    for j in range(len(mixture_cols)):
        for k in range(j + 1, len(mixture_cols)):
            feasible = (P4[:, j] + P4[:, k] <= 1.0 - eps)
            baseP = P4[feasible]
            if len(baseP) < 10:
                continue
            joint = baseP.copy(); remainder = 1.0 - joint[:, j] - joint[:, k]
            factor = (remainder - eps) / np.maximum(remainder, 1e-12)
            other = np.ones(17, dtype=bool); other[[j, k]] = False
            joint[:, other] *= factor[:, None]; joint[:, j] += half; joint[:, k] += half
            def single_change(target: int) -> np.ndarray:
                out = baseP.copy(); old = out[:, target].copy()
                out *= ((1.0 - old - half) / np.maximum(1.0 - old, 1e-12))[:, None]
                out[:, target] = old + half
                return out
            joint_pred = predict_ridge(fitted, ilr_features(joint, delta, chosen_degree)[0])
            pj_pred = predict_ridge(fitted, ilr_features(single_change(j), delta, chosen_degree)[0])
            pk_pred = predict_ridge(fitted, ilr_features(single_change(k), delta, chosen_degree)[0])
            base_sub = base_pred[feasible]
            interaction = (joint_pred - base_sub) - (pj_pred - base_sub) - (pk_pred - base_sub)
            for h, lcol in enumerate(loss_cols):
                vals = interaction[:, h]
                pair_effects.append({"domain_a": mixture_cols[j].removeprefix("train_the_pile_"),
                                     "domain_b": mixture_cols[k].removeprefix("train_the_pile_"),
                                     "validation_domain": lcol.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                                     "joint_total_increase": eps, "mean_interaction": vals.mean(), "median_interaction": np.median(vals),
                                     "p05_interaction": np.quantile(vals, 0.05), "p95_interaction": np.quantile(vals, 0.95),
                                     "n_recipes": len(vals), "evidence_level": "model_scenario",
                                     "claim_scope": "noncausal_model_association"})

    # Parameter table in transformed feature space.
    sx, sy, ridge_model = fitted
    coef = ridge_model.coef_ * sy.scale_[:, None] / sx.scale_[None, :]
    params = []
    for h, lcol in enumerate(loss_cols):
        for j, name in enumerate(feature_names):
            params.append({"output_domain": lcol.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                           "feature": name, "coefficient": coef[h, j], "model": model_name, "evidence_level": "metadata"})

    model_summary = {
        "mean_baseline_cv_rmse": mean_rmse, "ilr_best_cv_rmse": best1[0], "ilr_best_cv_spearman_median": best1[1],
        "ilr_improvement_over_mean": improvement, "baseline_failure_gate_triggered": bool(baseline_failed),
        "selected_model": model_name, "selected_degree": chosen_degree, "selected_delta": delta, "selected_alpha": alpha,
        "selected_cv_rmse": best_rmse, "selected_cv_spearman_median": best_rho,
        "qmix_cv_rmse": bestq[0], "qmix_cv_spearman_median": bestq[1],
        "qmix_is_deterministic_from_composition": True,
    }
    return {
        "cv": pd.concat(cv_frames, ignore_index=True), "metrics": pd.DataFrame(metric_rows),
        "predictions": pd.DataFrame(predictions), "scale": pd.DataFrame(scale_rows),
        "bootstrap": pd.DataFrame(boot_rows), "effects": pd.DataFrame(effects),
        "parameters": pd.DataFrame(params), "mapping": mapping_audit, "qmix_summary": pd.DataFrame(qmix_summary_rows),
        "effects_by_initial": pd.DataFrame(effects_by_initial), "donor_effects": pd.DataFrame(donor_effects),
        "pair_effects": pd.DataFrame(pair_effects), "summary": model_summary,
        "folds": folds, "train_ids": ids4, "train_P": P4, "cache": dataset_cache,
    }


def make_split_manifest(quality: pd.DataFrame, model_result: dict) -> pd.DataFrame:
    rows = []
    for row in quality[["record_id", "dataset", "domain"]].itertuples(index=False):
        fold = int(hashlib.sha256(row.record_id.encode()).hexdigest()[:8], 16) % 5
        role = "reference_transform" if row.dataset == "A1" else "extension_validation"
        rows.append({"record_id": row.record_id, "group_id": row.record_id, "time": "", "source": row.dataset,
                     "role": role, "fold": fold, "evidence_level": "real_observational"})
    train_folds = dict(zip(model_result["train_ids"], model_result["folds"]))
    def mixture_group_id(row: np.ndarray) -> str:
        stable = ",".join(f"{float(x):.9f}" for x in row)
        return "mixture:" + hashlib.sha256(stable.encode()).hexdigest()[:20]
    train_group_fold = {
        mixture_group_id(composition): int(fold)
        for composition, fold in zip(model_result["train_P"], model_result["folds"])
    }
    specs = [("A4_A5", "train_cv", "real_observational"), ("A6_A7", "scale_calibration_1m", "real_holdout"),
             ("A8_A9", "scale_calibration_60m", "real_holdout"), ("A10_A11", "final_test_1b", "real_holdout"),
             ("A12_A13", "extrapolation_10b", "extrapolated"), ("A14_A15", "extrapolation_70b", "extrapolated")]
    for source, role, evidence in specs:
        if source == "A4_A5":
            ids = model_result["train_ids"]
            P = model_result["train_P"]
        else:
            key = {"A6_A7": "A6_A7_1m", "A8_A9": "A8_A9_60m", "A10_A11": "A10_A11_1b", "A12_A13": "A12_A13_10b", "A14_A15": "A14_A15_70b"}[source]
            ids = model_result["cache"][key][0]
            P = model_result["cache"][key][1]
        for rid, composition in zip(ids, P):
            group_id = mixture_group_id(composition)
            fold = train_group_fold.get(group_id, train_folds.get(rid, "test") if source == "A4_A5" else "test")
            rows.append({"record_id": f"{source}:{int(rid)}", "group_id": group_id, "time": "",
                         "source": source, "role": role, "fold": fold, "evidence_level": evidence})
    return pd.DataFrame(rows)


def write_figures(quality: pd.DataFrame, aggregates: pd.DataFrame, comparisons: pd.DataFrame, model_result: dict) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT / "_模板" / "scripts"))
    from mpl_cn import CYCLE, plt, save_fig

    contracts = []
    pooled_ids = quality.assign(_priority=quality.dataset.map({"A1": 0, "A2": 1, "A3": 1})).sort_values("_priority").drop_duplicates("record_id", keep="last")
    order = sorted(pooled_ids.domain.unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    data = [pooled_ids.loc[pooled_ids.domain == d, "q_definition_robust"].to_numpy() for d in order]
    ax.boxplot(data, tick_labels=order, showfliers=False, patch_artist=True,
               boxprops={"facecolor": CYCLE[0], "alpha": 0.55})
    ax.set_ylabel("定义性稳健质量分")
    ax.tick_params(axis="x", rotation=35)
    path = FIGURES / "q1_quality_by_domain.pdf"; save_fig(fig, str(path), also_png=True); plt.close(fig)
    contracts.append({"figure": path.name, "source": "results/q1_quality_scores.csv.gz", "transform": "按ID去重后按域箱线图", "claim": "各域定义性质量分分布不同", "evidence_level": "real_observational"})

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    for ax, domain, ext in zip(axes, ("arxiv", "github"), ("A2", "A3")):
        a = quality[(quality.dataset == "A1") & (quality.domain == domain)].q_definition_robust
        b = quality[quality.dataset == ext].q_definition_robust
        ax.hist(a, bins=35, density=True, alpha=0.55, label="A1样本")
        ax.hist(b, bins=35, density=True, alpha=0.45, label="扩展集")
        ax.set_xlabel("定义性稳健质量分"); ax.set_ylabel("密度"); ax.legend(); ax.text(0.03, 0.94, domain, transform=ax.transAxes, va="top")
    path = FIGURES / "q1_sample_extension_comparison.pdf"; save_fig(fig, str(path), also_png=True); plt.close(fig)
    contracts.append({"figure": path.name, "source": "results/q1_quality_scores.csv.gz", "transform": "A1与A2/A3同域密度对照", "claim": "抽样与扩展集的差异不能只靠均值判断", "evidence_level": "real_observational"})

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    hb = ax.hexbin(quality.semantic_family_score, quality.morphology_family_score, gridsize=55, bins="log", mincnt=1, cmap="Blues")
    ax.plot([0, 1], [0, 1], "--", color="#777777", lw=1); ax.set_xlabel("内容语义家族分"); ax.set_ylabel("形态统计家族分")
    fig.colorbar(hb, ax=ax, label="log10 文档数")
    path = FIGURES / "q1_family_conflict_hexbin.pdf"; save_fig(fig, str(path), also_png=True); plt.close(fig)
    contracts.append({"figure": path.name, "source": "results/q1_quality_scores.csv.gz", "transform": "全量样本二维计数", "claim": "语义与形态评价存在结构性分歧", "evidence_level": "real_observational"})

    pred = model_result["predictions"]
    sub = pred[(pred.dataset == "A6_A7_1m") & (pred.model == model_result["summary"]["selected_model"])]
    fig, ax = plt.subplots(figsize=(5.0, 4.2)); ax.scatter(sub.observed_loss, sub.predicted_loss, s=8, alpha=0.35)
    lo = min(sub.observed_loss.min(), sub.predicted_loss.min()); hi = max(sub.observed_loss.max(), sub.predicted_loss.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#777777"); ax.set_xlabel("实际 Loss"); ax.set_ylabel("预测 Loss")
    path = FIGURES / "q1_same_scale_prediction.pdf"; save_fig(fig, str(path), also_png=True); plt.close(fig)
    contracts.append({"figure": path.name, "source": "results/q1_predictions.csv.gz", "transform": "A6/A7全域预测散点", "claim": "冻结配比模型的同尺度预测误差", "evidence_level": "real_holdout"})

    metrics = model_result["metrics"]
    macro = metrics[(metrics.domain == "macro") & metrics.dataset.isin(["A6_A7_1m", "A8_A9_60m", "A10_A11_1b"])]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5))
    for model, part in macro.groupby("model"):
        x = np.arange(len(part)); axes[0].plot(part.dataset, part.rmse, marker="o", label=model); axes[1].plot(part.dataset, part.spearman, marker="o", label=model)
    axes[0].set_ylabel("逐域 RMSE 的宏平均"); axes[1].set_ylabel("逐域 Spearman 的宏平均")
    for ax in axes: ax.tick_params(axis="x", rotation=25)
    axes[1].legend(fontsize=7)
    path = FIGURES / "q1_cross_scale_metrics.pdf"; save_fig(fig, str(path), also_png=True); plt.close(fig)
    contracts.append({"figure": path.name, "source": "results/q1_model_metrics.csv", "transform": "跨尺度宏平均", "claim": "区分绝对Loss漂移与排序保持", "evidence_level": "real_holdout"})

    eff = model_result["effects"].pivot(index="increased_domain", columns="validation_domain", values="mean_delta_loss")
    fig, ax = plt.subplots(figsize=(8.0, 6.0)); im = ax.imshow(eff.to_numpy(), aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(eff.columns)), eff.columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(eff.index)), eff.index, fontsize=7); fig.colorbar(im, ax=ax, label="增加1个百分点后的预测 Loss 变化")
    path = FIGURES / "q1_domain_substitution_effects.pdf"; save_fig(fig, str(path), also_png=True); plt.close(fig)
    contracts.append({"figure": path.name, "source": "results/q1_domain_effects.csv", "transform": "其余域按比例缩减的模型内情景", "claim": "领域替代关联具有目标域差异", "evidence_level": "model_scenario"})
    return pd.DataFrame(contracts)


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    log("Q1 workflow started")
    if not ZIP_PATH.exists():
        raise FileNotFoundError(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        missing = [name for name in MEMBER.values() if name not in zf.namelist()]
        if missing:
            raise RuntimeError(f"Missing A attachments: {missing}")
        quality = read_quality_data(zf)
        quality, indicator_manifest, dsir_sensitivity, quality_config = score_quality(quality)
        aggregates, sample_comparison, pareto, conflict_sensitivity, conflict_resolution = quality_aggregates(
            quality, quality_config["conflict_thresholds"]
        )
        audit_sample = make_manual_audit_sample(zf, quality)
        tables = {key: csv_member(zf, key) for key in MEMBER if key not in ("A1", "A2", "A3", "A18")}
        pair_audit, mixture_cols, loss_cols = validate_mixture_tables(tables)
        mapping = tables["A16"]

    domain_quality = aggregates[aggregates.dataset == "pooled_unique"].set_index("domain")["q_robust_mean"].to_dict()
    model_result = run_mixture_models(tables, mixture_cols, loss_cols, mapping, domain_quality)
    split_manifest = make_split_manifest(quality, model_result)

    score_cols = ["record_id", "dataset", "domain", "source_path", "content_sha256", "word_count", "num_sentences", "mean_word_length",
                  "length_band", "nonfinite_semantic_count", "structural_outlier", "measurement_anomaly",
                  "semantic_family_score", "morphology_family_score", "family_gap", "domain_length_conflict_z", "conflict_strength", "high_conflict",
                  "q_definition_equal", "q_definition_robust"]
    score_cols += [c for c in quality.columns if c.startswith("facet_")]
    score_cols += [f"{c}_per_word" for c in DSIR_COLS] + [f"{c}_len_resid" for c in DSIR_COLS]
    quality[score_cols].assign(evidence_level="real_observational").to_csv(RESULTS / "q1_quality_scores.csv.gz", index=False, compression="gzip")
    indicator_manifest.to_csv(RESULTS / "q1_indicator_transform_manifest.csv", index=False)
    dsir_sensitivity.to_csv(RESULTS / "q1_dsir_length_sensitivity.csv", index=False)
    aggregates.to_csv(RESULTS / "q1_domain_quality_summary.csv", index=False)
    sample_comparison.to_csv(RESULTS / "q1_sample_extension_comparison.csv", index=False)
    pareto.to_csv(RESULTS / "q1_domain_pareto_view.csv", index=False)
    conflict_sensitivity.to_csv(RESULTS / "q1_conflict_threshold_sensitivity.csv", index=False)
    conflict_resolution.to_csv(RESULTS / "q1_conflict_resolution_evaluation.csv", index=False)
    audit_sample.to_csv(RESULTS / "q1_manual_text_audit_sample.csv", index=False, encoding="utf-8-sig")
    pair_audit.to_csv(RESULTS / "q1_data_quality_audit.csv", index=False)
    split_manifest.to_csv(RESULTS / "split_manifest.csv.gz", index=False, compression="gzip")
    model_result["cv"].to_csv(RESULTS / "q1_model_cv.csv", index=False)
    model_result["metrics"].to_csv(RESULTS / "q1_model_metrics.csv", index=False)
    model_result["predictions"].to_csv(RESULTS / "q1_predictions.csv.gz", index=False, compression="gzip")
    model_result["scale"].to_csv(RESULTS / "q1_scale_correction.csv", index=False)
    model_result["bootstrap"].to_csv(RESULTS / "q1_1b_bootstrap.csv", index=False)
    model_result["effects"].to_csv(RESULTS / "q1_domain_effects.csv", index=False)
    model_result["parameters"].to_csv(RESULTS / "q1_model_parameters.csv", index=False)
    model_result["mapping"].to_csv(RESULTS / "q1_domain_mapping_audit.csv", index=False)
    model_result["qmix_summary"].to_csv(RESULTS / "q1_qmix_feature_summary.csv", index=False)
    model_result["effects_by_initial"].to_csv(RESULTS / "q1_domain_effects_by_initial_share.csv", index=False)
    model_result["donor_effects"].to_csv(RESULTS / "q1_domain_effects_pilecc_donor.csv", index=False)
    model_result["pair_effects"].to_csv(RESULTS / "q1_pair_interactions.csv", index=False)
    figure_contract = write_figures(quality, aggregates, sample_comparison, model_result)
    figure_contract.to_csv(RESULTS / "q1_figure_contract.csv", index=False)

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED,
        "source_zip_sha256": hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest(),
        "quality_records": {k: int(v) for k, v in quality.dataset.value_counts().to_dict().items()},
        "quality_unique_ids": int(quality.record_id.nunique()),
        "nonfinite_semantic_records": int((quality.nonfinite_semantic_count > 0).sum()),
        "conflict_thresholds": quality_config["conflict_thresholds"],
        "model": model_result["summary"],
        "manual_audit_status": "sample_generated_human_labels_pending",
        "q_interpretation": "definition_based_relative_score_not_independent_ground_truth",
        "q2_interface": "four_facets_and_source_standardized_effects; scalar Q calibration deferred",
    }
    (RESULTS / "q1_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Q1 workflow complete; selected model={model_result['summary']['selected_model']}")


if __name__ == "__main__":
    main()
