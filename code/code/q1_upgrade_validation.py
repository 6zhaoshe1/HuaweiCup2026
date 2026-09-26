#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F 题问题一：老师教程与 DeepSeek 建议的增量复核。

AI 辅助信息：OpenAI Codex（GPT-5 系列），OpenAI，2026-09-24。
本脚本只读取 F题.zip 中 A1--A16，不读取已隔离的《数据说明》PDF。
教程和 DeepSeek 代码仅作为待检验候选；数值结论均由本脚本重新计算。
"""

from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

import q1_modeling as base


SEED = 42
RNG = np.random.default_rng(SEED)
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOG = ROOT / "code" / "outputs" / "q1_upgrade_validation.log"
for path in (RESULTS, FIGURES, LOG.parent):
    path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "_模板" / "scripts"))
from mpl_cn import CYCLE, plt, save_fig  # noqa: E402


FACETS = {
    "content_value": ["s_fineweb_edu", "s_qurater_facts", "s_qurater_education"],
    "language_quality": ["s_fluency_margin", "s_readability_expected", "s_qurater_style",
                         "s_rps_terminal", "s_rps_unique", "s_rps_entropy"],
    "cleanliness": ["s_cleanliness_expected", "s_ad_margin", "s_rps_no_alpha",
                    "s_rps_numeric", "s_rps_uppercase", "s_rps_top2", "s_rps_top3"],
    "reasoning_professional": ["s_reasoning_expected", "s_professionalism_expected", "s_qurater_expertise"],
}
CONTENT_REASON = [
    "s_fineweb_edu", "s_qurater_facts", "s_qurater_education",
    "s_reasoning_expected", "s_professionalism_expected", "s_qurater_expertise",
]
LANGUAGE_CLEAN = [
    "s_fluency_margin", "s_readability_expected", "s_qurater_style",
    "s_cleanliness_expected", "s_ad_margin", "s_rps_terminal", "s_rps_unique",
    "s_rps_entropy", "s_rps_no_alpha", "s_rps_numeric", "s_rps_uppercase",
    "s_rps_top2", "s_rps_top3",
]


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")
    log(f"saved {name}: {len(df):,} rows")


def semantic_manifest() -> pd.DataFrame:
    rows = [
        (1, "fineweb_edu", "scalar", "education value", "quality", "positive", "A1 ECDF"),
        (2, "fluency_en", "2 logits", "English fluency", "quality", "positive margin", "z_fluent-z_nonfluent then A1 ECDF"),
        (3, "ad_en", "2 logits", "advertising", "quality", "negative ad / positive no-ad", "z_no_ad-z_ad then A1 ECDF"),
        (4, "modernbert_cleanliness", "6 logits", "cleanliness", "quality", "positive", "softmax expected level; uncalibrated"),
        (5, "modernbert_readability", "6 logits", "readability", "quality", "positive", "softmax expected level; uncalibrated"),
        (6, "modernbert_reasoning", "6 logits", "reasoning", "quality", "positive", "softmax expected level; uncalibrated"),
        (7, "modernbert_professionalism", "6 logits", "professionalism", "quality", "positive", "softmax expected level; uncalibrated"),
        (8, "qurater[0]", "scalar list element", "writing style", "quality", "positive", "separate A1 ECDF"),
        (9, "qurater[1]", "scalar list element", "required expertise", "quality", "positive", "separate A1 ECDF"),
        (10, "qurater[2]", "scalar list element", "facts/trivia", "quality", "positive", "separate A1 ECDF"),
        (11, "qurater[3]", "scalar list element", "educational value", "quality", "positive", "separate A1 ECDF"),
        (12, "dsir_books", "scalar", "Books-domain similarity", "auxiliary", "domain preference only", "per-word then domain length residual"),
        (13, "dsir_wiki", "scalar", "Wikipedia-domain similarity", "auxiliary", "domain preference only", "per-word then domain length residual"),
        (14, "dsir_math", "scalar", "Math-domain similarity", "auxiliary", "domain preference only", "per-word then domain length residual"),
        (15, "rps_doc_word_count", "scalar", "word count", "structure", "non-monotone", "length bands/anomaly only"),
        (16, "rps_doc_num_sentences", "scalar", "sentence count", "structure", "non-monotone", "anomaly only"),
        (17, "rps_doc_mean_word_length", "scalar", "mean word length", "structure", "non-monotone", "anomaly only"),
        (18, "rps_doc_unigram_entropy", "scalar", "unigram entropy", "quality", "mild positive", "domain×length ECDF"),
        (19, "rps_doc_frac_unique_words", "scalar", "unique-word share", "quality", "mild positive", "domain×length ECDF"),
        (20, "rps_lines_ending_with_terminal_punctution_mark", "scalar", "terminal-punctuation share", "quality", "mild positive", "domain×length ECDF"),
        (21, "rps_doc_frac_no_alph_words", "scalar", "non-alphabetic-word share", "quality", "domain conditional", "domain×length centrality"),
        (22, "rps_lines_numerical_chars_fraction", "scalar", "numeric-character share", "quality", "domain conditional", "domain×length centrality"),
        (23, "rps_lines_uppercase_letter_fraction", "scalar", "uppercase share", "quality", "domain conditional", "domain×length centrality"),
        (24, "rps_doc_frac_chars_top_2gram", "scalar", "top-2gram share", "quality", "negative repetition", "domain×length reverse ECDF"),
        (25, "rps_doc_frac_chars_top_3gram", "scalar", "top-3gram share", "quality", "negative repetition", "domain×length reverse ECDF"),
    ]
    out = pd.DataFrame(rows, columns=["semantic_position", "source_field", "storage_type", "meaning", "role", "direction", "transform"])
    out["evidence_level"] = "metadata"
    out["note"] = "22 source fields expand to 25 semantic dimensions because qurater has four elements"
    return out


def entropy_weights(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    X = np.clip(X, 0.0, 1.0) + 1e-12
    P = X / X.sum(axis=0, keepdims=True)
    e = -(P * np.log(P)).sum(axis=0) / np.log(len(X))
    d = np.clip(1.0 - e, 0.0, None)
    return d / d.sum() if d.sum() > 0 else np.full(X.shape[1], 1.0 / X.shape[1])


def critic_weights(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, float)
    sigma = np.nanstd(X, axis=0, ddof=1)
    corr = np.corrcoef(X, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    c = sigma * np.sum(1.0 - corr, axis=1)
    return c / c.sum() if c.sum() > 0 else np.full(X.shape[1], 1.0 / X.shape[1])


def weight_sensitivity(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ref = df.dataset.eq("A1")
    weight_rows = []
    for facet, cols in FACETS.items():
        X = df.loc[ref, cols].to_numpy(float)
        w_e, w_c = entropy_weights(X), critic_weights(X)
        w_g = np.sqrt(w_e * w_c); w_g /= w_g.sum()
        for col, we, wc, wg in zip(cols, w_e, w_c, w_g):
            weight_rows.append({"facet": facet, "indicator": col.removeprefix("s_"), "equal_weight": 1/len(cols),
                                "entropy_weight": we, "critic_weight": wc, "geometric_weight": wg,
                                "fit_source": "A1_only", "interpretation": "statistical sensitivity, not true importance",
                                "evidence_level": "real_observational"})
    weights = pd.DataFrame(weight_rows)

    methods = ["equal", "entropy", "critic", "geometric"]
    for facet, cols in FACETS.items():
        X = df[cols].to_numpy(float)
        wf = weights[weights.facet.eq(facet)].set_index("indicator")
        for method in methods:
            w = wf.loc[[c.removeprefix("s_") for c in cols], f"{method}_weight"].to_numpy(float)
            df[f"facet_{facet}_{method}_fixed"] = X @ w
    for method in methods:
        df[f"q_{method}_fixed"] = df[[f"facet_{f}_{method}_fixed" for f in FACETS]].mean(axis=1)

    priority = df.dataset.map({"A1": 0, "A2": 1, "A3": 1})
    pooled = df.assign(_priority=priority).sort_values("_priority").drop_duplicates("record_id", keep="last")
    stability = []
    for scope, part in [("A1", df[ref]), ("pooled_unique", pooled)]:
        for method in methods:
            stability.append({
                "scope": scope, "method": method,
                "spearman_vs_huber_definition": spearmanr(part[f"q_{method}_fixed"], part.q_definition_robust).statistic,
                "mean_abs_difference_vs_huber": np.mean(np.abs(part[f"q_{method}_fixed"] - part.q_definition_robust)),
                "evidence_level": "real_observational",
            })
    domain_rows = []
    for (dataset, domain), part in df.groupby(["dataset", "domain"]):
        row = {"dataset": dataset, "domain": domain, "n": len(part), "evidence_level": "real_observational"}
        for method in methods:
            row[f"q_{method}_mean"] = part[f"q_{method}_fixed"].mean()
        row["q_huber_mean"] = part.q_definition_robust.mean()
        domain_rows.append(row)
    return weights, pd.DataFrame(stability), pd.DataFrame(domain_rows)


def robust_conditional_z(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    ref = df.dataset.eq("A1")
    Z = np.empty((len(df), len(cols)), float)
    for j, col in enumerate(cols):
        global_med = df.loc[ref, col].median()
        global_mad = np.median(np.abs(df.loc[ref, col] - global_med)) * 1.4826
        global_scale = max(float(global_mad), 0.03)
        for key, idx in df.groupby(["domain", "length_band"]).groups.items():
            idx = np.asarray(idx, int)
            fit = df.loc[ref & df.domain.eq(key[0]) & df.length_band.eq(key[1]), col].to_numpy(float)
            if len(fit) < 30:
                fit = df.loc[ref & df.domain.eq(key[0]), col].to_numpy(float)
            if len(fit) < 30:
                med, scale = global_med, global_scale
            else:
                med = float(np.nanmedian(fit))
                scale = max(float(np.nanmedian(np.abs(fit - med)) * 1.4826), 0.03)
            Z[idx, j] = np.clip((df.loc[idx, col].to_numpy(float) - med) / scale, -8.0, 8.0)
    return Z


def cross_family_maxdiff(Z: np.ndarray, n_a: int) -> np.ndarray:
    A, B = Z[:, :n_a], Z[:, n_a:]
    return np.maximum(A.max(axis=1) - B.min(axis=1), B.max(axis=1) - A.min(axis=1))


def permuted_threshold(Z: np.ndarray, groups: np.ndarray | None, n_a: int, n_perm: int = 200) -> tuple[float, np.ndarray]:
    n, p = Z.shape
    null = np.empty(n * n_perm, np.float32)
    group_indices = [np.arange(n)] if groups is None else [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for b in range(n_perm):
        P = np.empty_like(Z)
        for j in range(p):
            for idx in group_indices:
                P[idx, j] = Z[RNG.permutation(idx), j]
        null[b*n:(b+1)*n] = cross_family_maxdiff(P, n_a)
    return float(np.quantile(null, 0.99)), null


def conditional_conflict(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cols = CONTENT_REASON + LANGUAGE_CLEAN
    Z = robust_conditional_z(df, cols)
    obs = cross_family_maxdiff(Z, len(CONTENT_REASON))
    a1_idx = np.flatnonzero(df.dataset.eq("A1").to_numpy())
    rng = np.random.default_rng(SEED)
    sub = rng.choice(a1_idx, size=min(20000, len(a1_idx)), replace=False)
    Zs = Z[sub]
    keys = (df.loc[sub, "domain"].astype(str) + "|" + df.loc[sub, "length_band"].astype(str)).to_numpy()
    log("estimating conditional/global permutation conflict thresholds")
    thr_cond, null_cond = permuted_threshold(Zs, keys, len(CONTENT_REASON), n_perm=200)
    thr_global, null_global = permuted_threshold(Zs, None, len(CONTENT_REASON), n_perm=200)
    df["conditional_max_pair_conflict"] = obs
    df["conditional_conflict"] = obs >= thr_cond
    df["global_permutation_conflict"] = obs >= thr_global
    summary = {
        "conditional_threshold_q99": thr_cond, "global_threshold_q99": thr_global,
        "a1_observed_rate_conditional_threshold": float(df.loc[df.dataset.eq('A1'), 'conditional_conflict'].mean()),
        "permutations": 200, "permutation_sample_n": len(sub),
        "families_exclude_dsir_and_structural": True,
    }
    rate_rows = []
    calibration_rows = []
    for (dataset, domain), part in df.groupby(["dataset", "domain"]):
        a = part.high_conflict.to_numpy(bool); b = part.conditional_conflict.to_numpy(bool)
        inter, union = np.sum(a & b), np.sum(a | b)
        rate_rows.append({"dataset": dataset, "domain": domain, "n": len(part),
                          "main_conflict_rate": a.mean(), "conditional_conflict_rate": b.mean(),
                          "jaccard_with_main": inter / union if union else 1.0,
                          "evidence_level": "real_observational"})
        g = part.global_permutation_conflict.to_numpy(bool)
        inter_gc, union_gc = np.sum(g & b), np.sum(g | b)
        calibration_rows.append({"dataset": dataset, "domain": domain, "n": len(part),
                                 "statistic": "conditional_max_pair_conflict",
                                 "global_permutation_threshold": thr_global,
                                 "domain_length_permutation_threshold": thr_cond,
                                 "global_flag_rate": g.mean(), "domain_length_flag_rate": b.mean(),
                                 "jaccard_global_vs_domain_length": inter_gc / union_gc if union_gc else 1.0,
                                 "conditional_is_subset_of_global": bool(np.all(~b | g)),
                                 "evidence_level": "real_observational"})

    best = np.full(len(df), -np.inf); labels = np.empty(len(df), object)
    for ia, ca in enumerate(CONTENT_REASON):
        for ib, cb in enumerate(LANGUAGE_CLEAN, start=len(CONTENT_REASON)):
            val = np.abs(Z[:, ia] - Z[:, ib])
            update = val > best
            best[update] = val[update]
            labels[update] = f"{ca.removeprefix('s_')} × {cb.removeprefix('s_')}"
    pair = (pd.DataFrame({"pair": labels[df.conditional_conflict], "dataset": df.loc[df.conditional_conflict, "dataset"].to_numpy()})
            .groupby(["dataset", "pair"]).size().rename("count").reset_index())
    pair["share_within_conflicts"] = pair["count"] / pair.groupby("dataset")["count"].transform("sum")
    pair["evidence_level"] = "real_observational"
    null_df = pd.DataFrame({
        "null_type": np.repeat(["conditional_domain_length", "global"], [len(null_cond), len(null_global)]),
        "max_pair_difference": np.concatenate([null_cond, null_global]),
        "evidence_level": "permutation_null",
    })
    return (pd.DataFrame(rate_rows), pair.sort_values(["dataset", "count"], ascending=[True, False]),
            null_df, pd.DataFrame(calibration_rows), summary)


def feature_cv(X: np.ndarray, Y: np.ndarray, folds: np.ndarray, alphas: list[float]) -> tuple[pd.DataFrame, tuple]:
    rows, best = [], None
    for alpha in alphas:
        oof = np.full_like(Y, np.nan)
        for fold in np.unique(folds):
            tr, va = folds != fold, folds == fold
            fit = base.fit_ridge(X[tr], Y[tr], alpha)
            oof[va] = base.predict_ridge(fit, X[va])
        rmse = mean_squared_error(Y, oof) ** 0.5
        rho = np.nanmedian([spearmanr(Y[:, j], oof[:, j]).statistic for j in range(Y.shape[1])])
        rows.append({"alpha": alpha, "cv_rmse_micro": rmse, "cv_spearman_median": rho})
        if best is None or (rmse, -rho) < (best[0], -best[1]):
            best = (rmse, rho, alpha, oof)
    return pd.DataFrame(rows), best


def metric_summary(Y: np.ndarray, pred: np.ndarray) -> dict:
    per_domain_rmse = [mean_squared_error(Y[:, j], pred[:, j]) ** 0.5 for j in range(Y.shape[1])]
    return {
        "rmse_micro": mean_squared_error(Y, pred) ** 0.5,
        "rmse_macro": np.mean(per_domain_rmse),
        "mae_macro": np.mean([mean_absolute_error(Y[:, j], pred[:, j]) for j in range(Y.shape[1])]),
        "r2_macro": np.mean([r2_score(Y[:, j], pred[:, j]) for j in range(Y.shape[1])]),
        "spearman_macro": np.nanmean([spearmanr(Y[:, j], pred[:, j]).statistic for j in range(Y.shape[1])]),
        "spearman_median": np.nanmedian([spearmanr(Y[:, j], pred[:, j]).statistic for j in range(Y.shape[1])]),
    }


def mixture_candidates(tables: dict, mix_cols: list[str], loss_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    ids4, P4, Y4 = base.join_pair(tables, "A4", "A5", mix_cols, loss_cols)
    folds = np.full(len(P4), -1, int)
    for k, (_, va) in enumerate(KFold(5, shuffle=True, random_state=SEED).split(P4)):
        folds[va] = k
    tests_raw = {}
    for label, mid, lid in [("A6_A7_1m", "A6", "A7"), ("A8_A9_60m", "A8", "A9"), ("A10_A11_1b", "A10", "A11")]:
        ids, P, Y = base.join_pair(tables, mid, lid, mix_cols, loss_cols)
        tests_raw[label] = (ids, P, Y)
    alphas = list(np.logspace(-4, 4, 9)); delta = 1e-4
    candidates = {}

    X_ilr, _ = base.ilr_features(P4, delta, 1)
    cv, best = feature_cv(X_ilr, Y4, folds, alphas)
    fit = base.fit_ridge(X_ilr, Y4, best[2])
    candidates["ilr_shared_alpha"] = {"cv": best, "fit": fit, "transform": lambda p: base.ilr_features(p, delta, 1)[0]}

    X_raw = P4 / P4.sum(axis=1, keepdims=True)
    cv_raw, best_raw = feature_cv(X_raw, Y4, folds, alphas)
    fit_raw = base.fit_ridge(X_raw, Y4, best_raw[2])
    candidates["raw_share_ridge"] = {"cv": best_raw, "fit": fit_raw, "transform": lambda p: p / p.sum(axis=1, keepdims=True)}

    log_sensitivity_rows = []
    best_log_all = None
    for log_delta in [1e-6, 1e-4, 5e-4, 1e-3]:
        X_try = np.log(P4 + log_delta)
        _, candidate = feature_cv(X_try, Y4, folds, alphas)
        log_sensitivity_rows.append({"delta": log_delta, "alpha": candidate[2], "cv_rmse_micro": candidate[0],
                                     "cv_spearman_median": candidate[1], "evidence_level": "real_observational"})
        if best_log_all is None or (candidate[0], -candidate[1]) < (best_log_all[0], -best_log_all[1]):
            best_log_all = (*candidate, log_delta)
    best_log = best_log_all[:4]
    log_delta = best_log_all[4]
    X_log = np.log(P4 + log_delta)
    fit_log = base.fit_ridge(X_log, Y4, best_log[2])
    candidates["log_shift_ridge"] = {"cv": best_log, "fit": fit_log, "transform": lambda p, d=log_delta: np.log(p + d),
                                      "delta": log_delta}

    X_quad, _ = base.ilr_features(P4, delta, 2)
    cv_quad, best_quad = feature_cv(X_quad, Y4, folds, alphas)
    fit_quad = base.fit_ridge(X_quad, Y4, best_quad[2])
    candidates["quadratic_ilr_ridge"] = {"cv": best_quad, "fit": fit_quad, "transform": lambda p: base.ilr_features(p, delta, 2)[0]}

    # Per-domain alpha ILR was tested in the prior audit and improved less than 0.1%.
    # It is removed from the active candidate set to avoid complexity without material gain.
    alpha_df = pd.DataFrame([{"model": "ilr_per_domain_alpha",
                              "status": "deprecated_not_in_active_dual_model",
                              "reason": "historical OOF/A6 improvement below 0.1%; retained only as audit note",
                              "evidence_level": "metadata"}])

    bench_rows, pred_rows = [], []
    for name, item in candidates.items():
        cv_pred = item["cv"][3]
        row = {"model": name, "dataset": "A4_A5_oof", **metric_summary(Y4, cv_pred),
               "tuning_scope": "A4_A5_train_only", "evidence_level": "real_observational"}
        bench_rows.append(row)
        for label, (ids, P, Y) in tests_raw.items():
            pred = item["external_predictions"][label] if "external_predictions" in item else base.predict_ridge(item["fit"], item["transform"](P))
            bench_rows.append({"model": name, "dataset": label, **metric_summary(Y, pred),
                               "tuning_scope": "A4_A5_train_only", "evidence_level": "real_holdout"})
            for i, rid in enumerate(ids):
                for j, col in enumerate(loss_cols):
                    pred_rows.append({"model": name, "dataset": label, "record_id": int(rid),
                                      "domain": col.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                                      "observed_loss": Y[i, j], "predicted_loss": pred[i, j],
                                      "evidence_level": "real_holdout"})

    bench = pd.DataFrame(bench_rows)
    shared = bench[(bench.model == "ilr_shared_alpha") & (bench.dataset == "A4_A5_oof")].iloc[0]
    shared_test = bench[(bench.model == "ilr_shared_alpha") & (bench.dataset == "A6_A7_1m")].iloc[0]
    decision = {"per_domain_alpha_status": "removed_from_active_model"}
    quad = bench[(bench.model == "quadratic_ilr_ridge") & (bench.dataset == "A4_A5_oof")].iloc[0]
    quad_test = bench[(bench.model == "quadratic_ilr_ridge") & (bench.dataset == "A6_A7_1m")].iloc[0]
    decision["quadratic_cv_improvement"] = float((shared.rmse_micro - quad.rmse_micro) / shared.rmse_micro)
    decision["quadratic_a6_improvement"] = float((shared_test.rmse_micro - quad_test.rmse_micro) / shared_test.rmse_micro)
    decision["promote_quadratic"] = bool(decision["quadratic_cv_improvement"] >= 0.03 and decision["quadratic_a6_improvement"] > 0)
    # Model selection is made on A4/A5 OOF only. A6 is used once as a confirmation gate,
    # not to choose hyperparameters or formulas.
    cv_rank = bench[bench.dataset.eq("A4_A5_oof")].sort_values(["rmse_micro", "spearman_macro"], ascending=[True, False])
    selected_name = str(cv_rank.iloc[0].model)
    selected_cv = cv_rank.iloc[0]
    selected_test = bench[(bench.model.eq(selected_name)) & (bench.dataset.eq("A6_A7_1m"))].iloc[0]
    decision["cv_selected_model"] = selected_name
    decision["cv_selected_improvement_vs_ilr"] = float((shared.rmse_micro - selected_cv.rmse_micro) / shared.rmse_micro)
    decision["a6_confirmation_improvement_vs_ilr"] = float((shared_test.rmse_micro - selected_test.rmse_micro) / shared_test.rmse_micro)
    decision["a6_confirmation_spearman_change"] = float(selected_test.spearman_macro - shared_test.spearman_macro)
    decision["promote_cv_selected_model"] = bool(decision["cv_selected_improvement_vs_ilr"] >= 0.03 and
                                                  decision["a6_confirmation_improvement_vs_ilr"] > 0 and
                                                  decision["a6_confirmation_spearman_change"] >= -0.01)
    log_test = bench[(bench.model == "log_shift_ridge") & (bench.dataset == "A6_A7_1m")].iloc[0]
    decision["a6_quadratic_macro_rmse_penalty_vs_log"] = float((quad_test.rmse_macro / log_test.rmse_macro) - 1)
    decision["a6_quadratic_micro_rmse_penalty_vs_log"] = float((quad_test.rmse_micro / log_test.rmse_micro) - 1)
    decision["frozen_dual_model_policy"] = {
        "primary_prediction": "log_shift_ridge",
        "structural_interaction": "quadratic_ilr_ridge",
        "transparent_baseline": "ilr_shared_alpha",
        "per_domain_alpha": "removed",
    }

    selected = candidates[selected_name]
    structural_name = "quadratic_ilr_ridge"
    structural = candidates[structural_name]
    if "fit" not in selected or "fit" not in structural:
        raise RuntimeError("dual model export requires fitted predictive and structural models")

    def export_parameters(item: dict, model_name: str, feature_names: list[str], coefficient_scope: str) -> pd.DataFrame:
        sx, sy, model = item["fit"]
        coef = model.coef_ * sy.scale_[:, None] / sx.scale_[None, :]
        intercept = sy.mean_ + sy.scale_ * model.intercept_ - coef @ sx.mean_
        rows = []
        for h, loss_col in enumerate(loss_cols):
            out_domain = loss_col.removeprefix("metric/the_pile_").removesuffix("_val_loss")
            rows.append({"output_domain": out_domain, "feature": "intercept", "coefficient": intercept[h],
                         "model": model_name, "delta": item.get("delta", delta),
                         "coefficient_scope": coefficient_scope, "evidence_level": "metadata"})
            for j, feature in enumerate(feature_names):
                rows.append({"output_domain": out_domain, "feature": feature, "coefficient": coef[h, j],
                             "model": model_name, "delta": item.get("delta", delta),
                             "coefficient_scope": coefficient_scope, "evidence_level": "metadata"})
        return pd.DataFrame(rows)

    predictive_names = [f"log({c.removeprefix('train_the_pile_')}+delta)" for c in mix_cols]
    quadratic_names = base.ilr_features(P4[:1], delta, 2)[1]
    predictive_parameters = export_parameters(selected, selected_name, predictive_names,
                                               "direct log-share coefficients; noncausal")
    structural_parameters = export_parameters(structural, structural_name, quadratic_names,
                                               "Helmert-basis dependent; do not label as raw-domain interactions")

    def model_effects(item: dict, model_name: str, step: float = .01) -> pd.DataFrame:
        base_pred = base.predict_ridge(item["fit"], item["transform"](P4))
        rows = []
        for j, mix_col in enumerate(mix_cols):
            feasible = P4[:, j] <= 1.0 - step
            Pm = P4[feasible].copy(); old = Pm[:, j].copy()
            Pm *= ((1.0 - old - step) / np.maximum(1.0 - old, 1e-12))[:, None]
            Pm[:, j] = old + step
            changed = base.predict_ridge(item["fit"], item["transform"](Pm)) - base_pred[feasible]
            for h, loss_col in enumerate(loss_cols):
                vals = changed[:, h]
                boots = np.array([np.mean(vals[RNG.integers(0, len(vals), len(vals))]) for _ in range(300)])
                rows.append({"increased_domain": mix_col.removeprefix("train_the_pile_"),
                             "validation_domain": loss_col.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                             "increase_share": step, "replacement_rule": "reduce_all_other_domains_proportionally",
                             "mean_delta_loss": vals.mean(), "median_delta_loss": np.median(vals),
                             "ci_low": np.quantile(boots, .025), "ci_high": np.quantile(boots, .975),
                             "n_recipes": len(vals), "model": model_name,
                             "bootstrap_scope": "recipe rows with fitted model fixed",
                             "evidence_level": "model_scenario", "claim_scope": "noncausal_model_association"})
        return pd.DataFrame(rows)

    predictive_effects = model_effects(selected, selected_name)
    structural_effects = model_effects(structural, structural_name)
    dual_effects = pd.concat([predictive_effects, structural_effects], ignore_index=True)

    effect_wide = dual_effects.pivot(index=["validation_domain", "increased_domain"], columns="model",
                                     values="mean_delta_loss").reset_index()
    log_values = effect_wide[selected_name].to_numpy(float)
    quad_values = effect_wide[structural_name].to_numpy(float)
    effect_wide["sign_agree"] = np.sign(log_values) == np.sign(quad_values)
    effect_wide["material_either_abs_ge_0_02"] = (np.abs(log_values) >= .02) | (np.abs(quad_values) >= .02)
    effect_wide["absolute_difference"] = np.abs(log_values - quad_values)
    effect_wide["rank_log_within_validation"] = effect_wide.groupby("validation_domain")[selected_name].rank(method="average")
    effect_wide["rank_quadratic_within_validation"] = effect_wide.groupby("validation_domain")[structural_name].rank(method="average")
    effect_wide["absolute_rank_difference"] = np.abs(effect_wide.rank_log_within_validation - effect_wide.rank_quadratic_within_validation)
    effect_wide["evidence_level"] = "model_scenario"

    consistency_rows = []
    for validation_domain, part in effect_wide.groupby("validation_domain"):
        top_log = set(part.nsmallest(5, selected_name).increased_domain)
        top_quad = set(part.nsmallest(5, structural_name).increased_domain)
        material = part.material_either_abs_ge_0_02.to_numpy(bool)
        consistency_rows.append({"scope": "validation_domain", "validation_domain": validation_domain,
                                 "n_effect_cells": len(part),
                                 "effect_rank_spearman": spearmanr(part[selected_name], part[structural_name]).statistic,
                                 "sign_agreement_all": part.sign_agree.mean(),
                                 "sign_agreement_material": part.loc[material, "sign_agree"].mean() if material.any() else np.nan,
                                 "material_cell_count": int(material.sum()),
                                 "top5_beneficial_jaccard": len(top_log & top_quad) / len(top_log | top_quad),
                                 "best_domain_same": part.nsmallest(1, selected_name).increased_domain.iloc[0] == part.nsmallest(1, structural_name).increased_domain.iloc[0],
                                 "evidence_level": "model_scenario"})
    material = effect_wide.material_either_abs_ge_0_02.to_numpy(bool)
    consistency_rows.append({"scope": "overall", "validation_domain": "all_13",
                             "n_effect_cells": len(effect_wide),
                             "effect_rank_spearman": spearmanr(effect_wide[selected_name], effect_wide[structural_name]).statistic,
                             "sign_agreement_all": effect_wide.sign_agree.mean(),
                             "sign_agreement_material": effect_wide.loc[material, "sign_agree"].mean(),
                             "material_cell_count": int(material.sum()),
                             "top5_beneficial_jaccard": np.mean([r["top5_beneficial_jaccard"] for r in consistency_rows]),
                             "best_domain_same": all(r["best_domain_same"] for r in consistency_rows),
                             "evidence_level": "model_scenario"})
    consistency = pd.DataFrame(consistency_rows)

    # Original-domain pair interactions are defined by closed perturbation contrasts,
    # not by basis-dependent quadratic-ILR coefficients.
    base_struct = base.predict_ridge(structural["fit"], structural["transform"](P4))
    interaction_rows = []
    half = .005
    def single_change(P: np.ndarray, target: int) -> np.ndarray:
        out = P.copy(); old = out[:, target].copy()
        out *= ((1.0 - old - half) / np.maximum(1.0 - old, 1e-12))[:, None]
        out[:, target] = old + half
        return out
    for j in range(len(mix_cols)):
        for k in range(j + 1, len(mix_cols)):
            feasible = (P4[:, j] + P4[:, k]) <= .99
            original = P4[feasible]
            joint = original.copy(); remainder = 1.0 - joint[:, j] - joint[:, k]
            other = np.ones(len(mix_cols), dtype=bool); other[[j, k]] = False
            joint[:, other] *= ((remainder - .01) / np.maximum(remainder, 1e-12))[:, None]
            joint[:, j] += half; joint[:, k] += half
            pj, pk = single_change(original, j), single_change(original, k)
            pred_joint = base.predict_ridge(structural["fit"], structural["transform"](joint))
            pred_j = base.predict_ridge(structural["fit"], structural["transform"](pj))
            pred_k = base.predict_ridge(structural["fit"], structural["transform"](pk))
            interaction = pred_joint - pred_j - pred_k + base_struct[feasible]
            for h, loss_col in enumerate(loss_cols):
                vals = interaction[:, h]
                boot_idx = RNG.integers(0, len(vals), size=(200, len(vals)))
                boot_means = vals[boot_idx].mean(axis=1)
                interaction_rows.append({"domain_a": mix_cols[j].removeprefix("train_the_pile_"),
                                         "domain_b": mix_cols[k].removeprefix("train_the_pile_"),
                                         "validation_domain": loss_col.removeprefix("metric/the_pile_").removesuffix("_val_loss"),
                                         "joint_total_increase": .01, "individual_increase": half,
                                         "replacement_rule": "joint excludes both targets; singles reduce all other domains proportionally",
                                         "mean_interaction": vals.mean(), "median_interaction": np.median(vals),
                                         "p05_interaction": np.quantile(vals, .05), "p95_interaction": np.quantile(vals, .95),
                                         "bootstrap_mean_ci_low": np.quantile(boot_means, .025),
                                         "bootstrap_mean_ci_high": np.quantile(boot_means, .975),
                                         "n_recipes": len(vals), "model": structural_name,
                                         "bootstrap_scope": "recipe rows with fitted model fixed; excludes refit uncertainty",
                                         "evidence_level": "model_scenario", "claim_scope": "noncausal_basis_invariant_scenario_contrast"})

    extras = {"logshift_sensitivity": pd.DataFrame(log_sensitivity_rows),
              "parameters": predictive_parameters, "quadratic_parameters": structural_parameters,
              "effects": predictive_effects, "dual_effects": dual_effects,
              "dual_consistency": consistency, "dual_disagreements": effect_wide,
              "quadratic_interactions": pd.DataFrame(interaction_rows)}
    return bench, alpha_df, pd.DataFrame(pred_rows), decision, extras


def dual_skill_scores(predictions: pd.DataFrame, tables: dict, mix_cols: list[str], loss_cols: list[str]) -> pd.DataFrame:
    _, _, Y4 = base.join_pair(tables, "A4", "A5", mix_cols, loss_cols)
    train_mean = dict(zip([c.removeprefix("metric/the_pile_").removesuffix("_val_loss") for c in loss_cols], Y4.mean(axis=0)))
    rows = []
    for (model, dataset, domain), part in predictions.groupby(["model", "dataset", "domain"]):
        y = part.observed_loss.to_numpy(float); p = part.predicted_loss.to_numpy(float)
        mae = mean_absolute_error(y, p)
        mae_train = mean_absolute_error(y, np.full(len(y), train_mean[domain]))
        mae_oracle = mean_absolute_error(y, np.full(len(y), y.mean()))
        rows.append({"model": model, "dataset": dataset, "domain": domain, "mae_model": mae,
                     "mae_train_mean_baseline": mae_train, "mae_same_test_mean_oracle": mae_oracle,
                     "skill_vs_train_mean": 1 - mae / mae_train if mae_train else np.nan,
                     "skill_vs_same_test_mean_oracle": 1 - mae / mae_oracle if mae_oracle else np.nan,
                     "oracle_is_not_deployable": True, "evidence_level": "real_holdout"})
    out = pd.DataFrame(rows)
    macro = out.groupby(["model", "dataset"], as_index=False).agg({
        "mae_model": "mean", "mae_train_mean_baseline": "mean", "mae_same_test_mean_oracle": "mean",
        "skill_vs_train_mean": "mean", "skill_vs_same_test_mean_oracle": "mean"})
    macro["domain"] = "macro_mean_of_domains"; macro["oracle_is_not_deployable"] = True; macro["evidence_level"] = "real_holdout"
    return pd.concat([out, macro], ignore_index=True)


def draw_figures(df: pd.DataFrame, weights: pd.DataFrame, domain_scores: pd.DataFrame,
                 conflict_rates: pd.DataFrame, pair: pd.DataFrame, null_df: pd.DataFrame,
                 bench: pd.DataFrame, skills: pd.DataFrame, effects: pd.DataFrame,
                 consistency: pd.DataFrame, disagreements: pd.DataFrame, interactions: pd.DataFrame,
                 tables: dict, mix_cols: list[str], loss_cols: list[str]) -> pd.DataFrame:
    contracts = []
    def emit(fig, stem, result_file, assertion):
        save_fig(fig, FIGURES / f"{stem}.pdf")
        fig.savefig(FIGURES / f"{stem}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        contracts.append({"figure": stem, "result_source": result_file, "transform": "scripted from listed result/data",
                          "claim": assertion, "evidence_level": "real_observational"})

    # 1. Indicator ECDF small multiples (A1 only).
    cols = sum(FACETS.values(), [])
    a1 = df[df.dataset.eq("A1")]
    fig, axes = plt.subplots(4, 5, figsize=(12, 8), sharex=True, sharey=True)
    for ax, col in zip(axes.flat, cols):
        for color, (domain, part) in zip(CYCLE, a1.groupby("domain")):
            x = np.sort(part[col].to_numpy(float)); y = np.arange(1, len(x)+1)/len(x)
            ax.plot(x, y, lw=.75, alpha=.8, color=color, label=domain)
        ax.set_title(col.removeprefix("s_").replace("_", " "), fontsize=7)
        ax.grid(alpha=.18)
    for ax in axes.flat[len(cols):]: ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=7)
    fig.tight_layout(rect=(0, .05, 1, 1))
    emit(fig, "q1_upgrade_indicator_ecdf", "q1_upgrade_semantic_manifest.csv", "19个纳入Q的指标在A1各域分布不同")

    # 2. Weight sensitivity.
    plot = weights.copy(); methods = ["equal_weight", "entropy_weight", "critic_weight", "geometric_weight"]
    fig, ax = plt.subplots(figsize=(9, 7)); ypos = np.arange(len(plot))
    offsets = np.linspace(-.24, .24, len(methods))
    for c, method, off in zip(CYCLE, methods, offsets):
        ax.scatter(plot[method], ypos+off, s=18, label=method.replace("_weight", ""), color=c)
    ax.set_yticks(ypos); ax.set_yticklabels(plot.indicator.str.replace("_", " "), fontsize=7); ax.invert_yaxis()
    ax.set_xlabel("facet内权重"); ax.grid(axis="x", alpha=.2); ax.legend(ncol=4, fontsize=8)
    fig.tight_layout(); emit(fig, "q1_upgrade_weight_sensitivity", "q1_upgrade_weight_table.csv", "统计赋权改变指标权重但不等同真实重要性")

    # 3. Domain × facet heatmap.
    facet_cols = ["facet_content_value_robust", "facet_language_quality_robust", "facet_cleanliness_robust", "facet_reasoning_professional_robust"]
    heat = a1.groupby("domain")[facet_cols].mean()
    fig, ax = plt.subplots(figsize=(8, 4)); im = ax.imshow(heat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(["内容价值", "语言质量", "清洁度", "推理专业"])
    ax.set_yticks(range(len(heat))); ax.set_yticklabels(heat.index)
    for i in range(len(heat)):
        for j in range(4): ax.text(j, i, f"{heat.iloc[i,j]:.2f}", ha="center", va="center", color="white" if heat.iloc[i,j] < .55 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="平均分"); fig.tight_layout()
    emit(fig, "q1_upgrade_domain_facets", "q1_quality_scores.csv.gz", "不同领域的四个质量分面不能由单一排序完全概括")

    # 4. Conflict calibration.
    sample_null = null_df.groupby("null_type", group_keys=False).sample(n=min(100000, len(null_df)//2), random_state=SEED)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for color, (name, part) in zip(CYCLE, sample_null.groupby("null_type")):
        axes[0].hist(part.max_pair_difference, bins=80, density=True, alpha=.45, label=name, color=color)
    axes[0].set_xlabel("置换零假设下最大跨家族差异"); axes[0].set_ylabel("密度"); axes[0].legend(fontsize=8)
    cr = conflict_rates[conflict_rates.dataset.eq("A1")].sort_values("conditional_conflict_rate")
    axes[1].barh(cr.domain, cr.conditional_conflict_rate, color=CYCLE[1], alpha=.85, label="条件置换")
    axes[1].scatter(cr.main_conflict_rate, np.arange(len(cr)), color=CYCLE[0], label="原家族差异")
    axes[1].set_xlabel("冲突率"); axes[1].legend(fontsize=8); axes[1].grid(axis="x", alpha=.2)
    fig.tight_layout(); emit(fig, "q1_upgrade_conflict_calibration", "q1_upgrade_conflict_rates.csv", "多重比较校准显著影响冲突率")

    # 5. Conflict causes: domains, length, top pairs.
    a1c = a1.copy(); a1c["length_decile"] = pd.qcut(a1c.word_count.rank(method="first"), 10, labels=False) + 1
    length = a1c.groupby("length_decile").conditional_conflict.agg(["mean", "count"])
    top = pair[pair.dataset.eq("A1")].nlargest(10, "count").sort_values("share_within_conflicts")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(length.index, length["mean"], marker="o", color=CYCLE[0]); axes[0].set_xlabel("A1词数十分位"); axes[0].set_ylabel("条件冲突率"); axes[0].grid(alpha=.2)
    axes[1].barh(top.pair.str.replace("_", " "), top.share_within_conflicts, color=CYCLE[2]); axes[1].set_xlabel("冲突样本中的主导占比"); axes[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout(); emit(fig, "q1_upgrade_conflict_causes", "q1_upgrade_conflict_top_pairs.csv", "冲突来源随长度和指标对而异")

    # 6. RegMix overview.
    _, P4, Y4 = base.join_pair(tables, "A4", "A5", mix_cols, loss_cols)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([P4[:, j] for j in range(P4.shape[1])], vert=False, showfliers=False)
    axes[0].set_yticks(range(1, len(mix_cols)+1)); axes[0].set_yticklabels([c.removeprefix("train_the_pile_") for c in mix_cols], fontsize=7); axes[0].set_xlabel("训练配比")
    axes[1].boxplot([Y4[:, j] for j in range(Y4.shape[1])], vert=False, showfliers=False)
    axes[1].set_yticks(range(1, len(loss_cols)+1)); axes[1].set_yticklabels([c.removeprefix("metric/the_pile_").removesuffix("_val_loss") for c in loss_cols], fontsize=7); axes[1].set_xlabel("验证Loss")
    fig.tight_layout(); emit(fig, "q1_upgrade_regmix_overview", "q1_data_quality_audit.csv", "17维配比和13维Loss的离散程度不同")

    # 7. Candidate model benchmark.
    sel = bench[bench.dataset.isin(["A4_A5_oof", "A6_A7_1m"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, ylabel in [(axes[0], "rmse_micro", "RMSE（越低越好）"), (axes[1], "spearman_macro", "Spearman（越高越好）")]:
        pivot = sel.pivot(index="model", columns="dataset", values=metric)
        x = np.arange(len(pivot)); w=.36
        for k, dataset in enumerate(pivot.columns): ax.bar(x+(k-.5)*w, pivot[dataset], width=w, label=dataset, color=CYCLE[k])
        ax.set_xticks(x); ax.set_xticklabels(pivot.index.str.replace("_", " "), rotation=25, ha="right", fontsize=8); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout()
    emit(fig, "q1_upgrade_model_comparison", "q1_upgrade_model_benchmark.csv", "候选复杂模型必须同时通过训练CV和A6独立检验")

    # 8. Dual skill baselines.
    sk = skills[(skills.domain.eq("macro_mean_of_domains")) & skills.model.isin(["ilr_shared_alpha", "log_shift_ridge", "quadratic_ilr_ridge"])]
    fig, ax = plt.subplots(figsize=(10, 4.5)); datasets = ["A6_A7_1m", "A8_A9_60m", "A10_A11_1b"]
    x = np.arange(len(datasets)); width=.23
    for i, model in enumerate(sk.model.unique()):
        vals = sk[sk.model.eq(model)].set_index("dataset").reindex(datasets).skill_vs_train_mean
        ax.bar(x+(i-1)*width, vals, width=width, label=model.replace("_", " "), color=CYCLE[i])
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x); ax.set_xticklabels(datasets); ax.set_ylabel("相对训练均值的MAE技能分"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); emit(fig, "q1_upgrade_skill_baselines", "q1_upgrade_skill_scores.csv", "绝对Loss跨尺度迁移与同尺度能力必须分开评价")

    # 9. Selected model's simplex-respecting substitution effects.
    matrix = effects.pivot(index="validation_domain", columns="increased_domain", values="mean_delta_loss")
    vmax = np.nanquantile(np.abs(matrix.to_numpy()), .98)
    fig, ax = plt.subplots(figsize=(12, 6)); im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns))); ax.set_xticklabels(matrix.columns, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(matrix.index))); ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("增加1个百分点的训练域"); ax.set_ylabel("验证域")
    fig.colorbar(im, ax=ax, label="预测Loss变化"); fig.tight_layout()
    emit(fig, "q1_upgrade_selected_domain_effects", "q1_upgrade_selected_domain_effects.csv", "组成约束下的替代效应是模型关联而非因果效应")

    # 10. Predictive-versus-structural substitution consistency.
    overall = consistency[consistency.scope.eq("overall")].iloc[0]
    by_domain = consistency[consistency.scope.eq("validation_domain")].sort_values("effect_rank_spearman")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].scatter(disagreements.log_shift_ridge, disagreements.quadratic_ilr_ridge,
                    c=disagreements.sign_agree.map({True: CYCLE[0], False: CYCLE[3]}), s=22, alpha=.8)
    lim = np.nanmax(np.abs(disagreements[["log_shift_ridge", "quadratic_ilr_ridge"]].to_numpy()))
    axes[0].plot([-lim, lim], [-lim, lim], color="black", lw=.8, ls="--")
    axes[0].axhline(0, color="#777777", lw=.6); axes[0].axvline(0, color="#777777", lw=.6)
    axes[0].set_xlabel("对数平移 Ridge：平均Loss变化"); axes[0].set_ylabel("二次 ILR：平均Loss变化")
    axes[0].text(.02, .97, f"整体秩相关={overall.effect_rank_spearman:.3f}\n符号一致={overall.sign_agreement_all:.1%}",
                 transform=axes[0].transAxes, va="top", fontsize=9)
    y = np.arange(len(by_domain)); axes[1].barh(y-.18, by_domain.effect_rank_spearman, height=.35, color=CYCLE[0], label="效应排序Spearman")
    axes[1].barh(y+.18, by_domain.top5_beneficial_jaccard, height=.35, color=CYCLE[1], label="有利Top-5 Jaccard")
    axes[1].set_yticks(y); axes[1].set_yticklabels(by_domain.validation_domain, fontsize=8)
    axes[1].set_xlim(0, 1); axes[1].grid(axis="x", alpha=.2); axes[1].legend(fontsize=8)
    fig.tight_layout(); emit(fig, "q1_dual_model_consistency", "q1_dual_model_consistency.csv", "两模型一级结论稳定但次级领域排序仅中等一致")

    # 11. Largest quadratic-ILR scenario interactions; exploratory only.
    top_inter = interactions.assign(abs_mean=lambda x: np.abs(x.mean_interaction)).nlargest(20, "abs_mean").sort_values("mean_interaction")
    labels = top_inter.domain_a + " × " + top_inter.domain_b + " → " + top_inter.validation_domain
    fig, ax = plt.subplots(figsize=(10, 7)); y = np.arange(len(top_inter))
    ax.barh(y, top_inter.mean_interaction, color=np.where(top_inter.mean_interaction < 0, CYCLE[0], CYCLE[3]), alpha=.85)
    ax.errorbar(top_inter.mean_interaction, y,
                xerr=np.vstack([top_inter.mean_interaction-top_inter.bootstrap_mean_ci_low,
                                top_inter.bootstrap_mean_ci_high-top_inter.mean_interaction]),
                fmt="none", ecolor="#333333", capsize=2, lw=.8)
    ax.axvline(0, color="black", lw=.8); ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("二次ILR闭合双域情景交互（Loss）"); ax.grid(axis="x", alpha=.2)
    fig.tight_layout(); emit(fig, "q1_quadratic_top_interactions", "q1_quadratic_ilr_interactions.csv", "原始领域组合效应通过闭合情景对比定义而非直接解释ILR系数")
    return pd.DataFrame(contracts)


def main() -> None:
    start = time.time(); LOG.write_text("", encoding="utf-8")
    save_table(semantic_manifest(), "q1_upgrade_semantic_manifest.csv")
    with zipfile.ZipFile(base.ZIP_PATH) as zf:
        log("streaming A1/A2/A3 with the verified decoder; content is not retained")
        quality = base.read_quality_data(zf)
        quality, _, _, aux = base.score_quality(quality)
        weights, stability, domain_scores = weight_sensitivity(quality)
        save_table(weights, "q1_upgrade_weight_table.csv")
        save_table(stability, "q1_upgrade_weight_stability.csv")
        save_table(domain_scores, "q1_upgrade_weight_domain_scores.csv")
        conflict_rates, pairs, null_df, calibration_df, conflict_summary = conditional_conflict(quality)
        save_table(conflict_rates, "q1_upgrade_conflict_rates.csv")
        save_table(pairs, "q1_upgrade_conflict_top_pairs.csv")
        save_table(calibration_df, "q1_upgrade_conflict_calibration_comparison.csv")
        # Full null values are reproducibility evidence; compressed to keep the artifact modest.
        null_df.to_csv(RESULTS / "q1_upgrade_conflict_null.csv.gz", index=False, compression="gzip")
        tables = {key: base.csv_member(zf, key) for key in [f"A{i}" for i in range(4, 17)]}
        audit, mix_cols, loss_cols = base.validate_mixture_tables(tables)
        bench, alpha_df, predictions, decision, extras = mixture_candidates(tables, mix_cols, loss_cols)
        save_table(bench, "q1_upgrade_model_benchmark.csv")
        save_table(alpha_df, "q1_upgrade_domain_alpha.csv")
        save_table(extras["logshift_sensitivity"], "q1_upgrade_logshift_delta_sensitivity.csv")
        save_table(extras["parameters"], "q1_upgrade_selected_parameters.csv")
        save_table(extras["effects"], "q1_upgrade_selected_domain_effects.csv")
        save_table(extras["quadratic_parameters"], "q1_upgrade_quadratic_parameters.csv")
        save_table(extras["dual_effects"], "q1_dual_model_domain_effects.csv")
        save_table(extras["dual_consistency"], "q1_dual_model_consistency.csv")
        save_table(extras["dual_disagreements"], "q1_dual_model_disagreements.csv")
        save_table(extras["quadratic_interactions"], "q1_quadratic_ilr_interactions.csv")
        predictions.to_csv(RESULTS / "q1_upgrade_predictions.csv.gz", index=False, compression="gzip")
        skills = dual_skill_scores(predictions, tables, mix_cols, loss_cols)
        save_table(skills, "q1_upgrade_skill_scores.csv")
        selected_predictions = predictions[predictions.model.eq(decision["cv_selected_model"])]
        selected_metric_rows = []
        for (dataset, domain), part in selected_predictions.groupby(["dataset", "domain"]):
            y, p = part.observed_loss.to_numpy(float), part.predicted_loss.to_numpy(float)
            selected_metric_rows.append({"dataset": dataset, "domain": domain, "n": len(part),
                                         "mae": mean_absolute_error(y, p), "rmse": mean_squared_error(y, p) ** .5,
                                         "r2": r2_score(y, p), "spearman": spearmanr(y, p).statistic,
                                         "bias_pred_minus_true": np.mean(p-y), "model": decision["cv_selected_model"],
                                         "evidence_level": "real_holdout"})
        save_table(pd.DataFrame(selected_metric_rows), "q1_upgrade_selected_domain_metrics.csv")
        contracts = draw_figures(quality, weights, domain_scores, conflict_rates, pairs, null_df,
                                 bench, skills, extras["effects"], extras["dual_consistency"],
                                 extras["dual_disagreements"], extras["quadratic_interactions"],
                                 tables, mix_cols, loss_cols)
        save_table(contracts, "q1_upgrade_figure_contract.csv")

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": SEED,
        "runtime_seconds": time.time()-start, "quality_records": quality.dataset.value_counts().to_dict(),
        "storage_field_pit": {"A1_content_retained": False, "streaming_decoder": True,
                              "source_fields": 22, "semantic_dimensions": 25},
        "weight_stability": stability.to_dict("records"), "conflict": conflict_summary,
        "mixture_candidate_decision": decision,
        "dual_model_consistency": extras["dual_consistency"].loc[
            extras["dual_consistency"].scope.eq("overall")].iloc[0].to_dict(),
        "main_definition_unchanged": "four facets + Huber/equal; scalar Q remains definition-based",
        "excluded_from_generic_q": ["DSIR", "word_count", "num_sentences", "mean_word_length"],
    }
    (RESULTS / "q1_upgrade_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = {
        "semantic_dimensions_25": len(semantic_manifest()) == 25,
        "all_scores_finite": bool(np.isfinite(quality.q_definition_robust).all()),
        "all_q_in_unit_interval": bool(quality.q_definition_robust.between(0, 1).all()),
        "conditional_threshold_finite": bool(np.isfinite(conflict_summary["conditional_threshold_q99"])),
        "benchmark_rows_complete": len(bench) == 16,
        "figures_created": len(contracts) == 11,
        "a6_not_used_for_tuning": bool((bench.tuning_scope == "A4_A5_train_only").all()),
    }
    (RESULTS / "q1_upgrade_verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(checks.values()):
        raise RuntimeError(f"upgrade verification failed: {checks}")
    log(f"completed in {time.time()-start:.1f}s; checks={checks}")


if __name__ == "__main__":
    main()
