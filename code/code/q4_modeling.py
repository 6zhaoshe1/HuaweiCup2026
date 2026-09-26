# -*- coding: utf-8 -*-
"""Question 4 formal models: frontier, decomposition, bridge and forecast.

AI assistance: OpenAI Codex, 2026-09-25.

The script uses the frozen cohort contract from q4_prepare.py.  All model
selection is based on rolling-origin or leave-one-size-out evaluation.  A
conditional time coefficient is called a non-scale progress proxy, never a
causal technology effect.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression, QuantileRegressor, TheilSenRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
C_DIR = (
    ROOT
    / "第二十三届中国研究生数学建模竞赛 - 中文题目"
    / "中文题目"
    / "F题"
    / "real_attachments"
    / "C_efficiency_evolution"
)
RESULTS = ROOT / "results"
RNG_SEED = 42
ORIGIN = pd.Timestamp("2024-06-08")
Q = 0.90
EPS_SCORE = 0.05

MODEL_SPECS = [
    "scale_linear",
    "scale_quadratic",
    "scale_time_linear",
    "scale_time_interaction",
    "gradient_boosting",
]
SCALE_ONLY = {"scale_linear", "scale_quadratic"}
TIME_MODELS = {"scale_time_linear", "scale_time_interaction", "gradient_boosting"}


def score_to_logit(score: np.ndarray | pd.Series) -> np.ndarray:
    value = np.clip(np.asarray(score, dtype=float), EPS_SCORE, 100.0 - EPS_SCORE)
    return np.log(value / (100.0 - value))


def logit_to_score(value: np.ndarray | float) -> np.ndarray:
    z = np.clip(np.asarray(value, dtype=float), -30, 30)
    return 100.0 / (1.0 + np.exp(-z))


def pinball(y: np.ndarray, pred: np.ndarray, q: float = Q) -> float:
    err = np.asarray(y) - np.asarray(pred)
    return float(np.mean(np.maximum(q * err, (q - 1.0) * err)))


def weighted_pinball(
    y: np.ndarray, pred: np.ndarray, weights: np.ndarray, q: float = Q
) -> float:
    err = np.asarray(y) - np.asarray(pred)
    loss = np.maximum(q * err, (q - 1.0) * err)
    return float(np.average(loss, weights=np.asarray(weights, dtype=float)))


def feature_frame(
    df: pd.DataFrame, spec: str, ln_center: float, time_origin: pd.Timestamp = ORIGIN
) -> pd.DataFrame:
    ln_n = np.log(pd.to_numeric(df["params_B"], errors="coerce").to_numpy(float)) - ln_center
    t = (pd.to_datetime(df["submission_date"]) - time_origin).dt.days.to_numpy(float) / 365.25
    is_chat = df["stratum"].eq("chat_finetuned").to_numpy(float)
    values: dict[str, np.ndarray] = {"lnN": ln_n}
    if spec in {"scale_quadratic", "scale_time_interaction"}:
        values["lnN2"] = ln_n**2
    if spec in {"scale_time_linear", "scale_time_interaction", "gradient_boosting"}:
        values["time_year"] = t
    values["is_chat"] = is_chat
    if spec == "scale_time_interaction":
        values["lnN_time"] = ln_n * t
    return pd.DataFrame(values, index=df.index)


def fit_quantile(train: pd.DataFrame, spec: str):
    ln_center = float(np.log(train["params_B"].astype(float)).mean())
    x = feature_frame(train, spec, ln_center)
    y = train["score_logit"].to_numpy(float)
    family_counts = train["family_id"].value_counts()
    weights = train["family_id"].map(lambda value: 1.0 / family_counts[value]).to_numpy(float)
    weights = weights / np.mean(weights)
    if spec == "gradient_boosting":
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=Q,
            n_estimators=160,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=12,
            random_state=RNG_SEED,
        )
    else:
        model = QuantileRegressor(quantile=Q, alpha=0.0, solver="highs", fit_intercept=True)
    model.fit(x, y, sample_weight=weights)
    return model, ln_center, list(x.columns)


def predict_quantile(model, df: pd.DataFrame, spec: str, ln_center: float) -> np.ndarray:
    return np.asarray(model.predict(feature_frame(df, spec, ln_center)), dtype=float)


def rolling_quantile_experiment(entity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    cutoffs = pd.to_datetime(
        [
            "2024-08-01",
            "2024-09-01",
            "2024-10-01",
            "2024-11-01",
            "2024-12-01",
            "2025-01-01",
            "2025-02-01",
        ]
    )
    horizon_days = 60
    rows: list[dict] = []
    prediction_rows: list[dict] = []
    final_models: dict = {}
    coef_rows: list[dict] = []

    scopes = {"strict": "strict_open", "wide": "wide_open"}
    strata = {
        "core_combined": ["pretrained", "chat_finetuned"],
        "pretrained": ["pretrained"],
        "chat_finetuned": ["chat_finetuned"],
        "derived": ["derived"],
    }
    for scope, scope_col in scopes.items():
        for stratum, allowed in strata.items():
            data = entity[
                entity[scope_col]
                & entity["valid_core_fields"]
                & entity["stratum"].isin(allowed)
            ].copy()
            data = data.sort_values("submission_date")
            if len(data) < 35:
                continue
            for cutoff in cutoffs:
                train = data[data["submission_date"].lt(cutoff)].copy()
                test_end = min(cutoff + pd.Timedelta(days=horizon_days), data["submission_date"].max())
                test = data[
                    data["submission_date"].ge(cutoff)
                    & data["submission_date"].le(test_end)
                ].copy()
                if len(train) < 30 or len(test) < 8:
                    continue
                for spec in MODEL_SPECS:
                    try:
                        model, center, features = fit_quantile(train, spec)
                        pred = predict_quantile(model, test, spec, center)
                    except Exception as exc:
                        rows.append(
                            {
                                "scope": scope,
                                "stratum": stratum,
                                "cutoff": cutoff.date(),
                                "test_end": test_end.date(),
                                "model": spec,
                                "n_train": len(train),
                                "n_test": len(test),
                                "status": f"failed:{type(exc).__name__}",
                            }
                        )
                        continue
                    truth = test["score_logit"].to_numpy(float)
                    test_family_counts = test["family_id"].value_counts()
                    test_weights = test["family_id"].map(
                        lambda value: 1.0 / test_family_counts[value]
                    ).to_numpy(float)
                    rows.append(
                        {
                            "scope": scope,
                            "stratum": stratum,
                            "cutoff": cutoff.date(),
                            "test_end": test_end.date(),
                            "model": spec,
                            "n_train": len(train),
                            "n_test": len(test),
                            "status": "ok",
                            "pinball_logit": pinball(truth, pred),
                            "pinball_logit_familyweighted": weighted_pinball(
                                truth, pred, test_weights
                            ),
                            "coverage_q90": float(np.mean(truth <= pred)),
                            "mae_score": mean_absolute_error(test["score_equal"], logit_to_score(pred)),
                            "spearman_score": float(spearmanr(test["score_equal"], logit_to_score(pred)).statistic),
                        }
                    )
                    score_pred = logit_to_score(pred)
                    for (_, record), pred_logit, pred_score, test_weight in zip(
                        test.iterrows(), pred, score_pred, test_weights
                    ):
                        prediction_rows.append(
                            {
                                "scope": scope,
                                "stratum": stratum,
                                "cutoff": cutoff.date(),
                                "test_end": test_end.date(),
                                "model": spec,
                                "entity_id": record["entity_id"],
                                "family_id": record["family_id"],
                                "submission_date": record["submission_date"].date(),
                                "truth_logit": float(record["score_logit"]),
                                "prediction_logit": float(pred_logit),
                                "truth_score": float(record["score_equal"]),
                                "prediction_score": float(pred_score),
                                "family_weight": float(test_weight),
                            }
                        )

            # Final fit for every candidate so coefficients and scenario models remain traceable.
            for spec in MODEL_SPECS:
                try:
                    model, center, features = fit_quantile(data, spec)
                    final_models[(scope, stratum, spec)] = (model, center, features, data)
                    if hasattr(model, "coef_"):
                        for feature, value in zip(features, np.asarray(model.coef_).ravel()):
                            coef_rows.append(
                                {
                                    "scope": scope,
                                    "stratum": stratum,
                                    "model": spec,
                                    "term": feature,
                                    "coefficient": float(value),
                                    "intercept": float(model.intercept_),
                                    "lnN_center": center,
                                    "n": len(data),
                                }
                            )
                except Exception:
                    pass

    backtest = pd.DataFrame(rows)
    backtest.to_csv(RESULTS / "q4_quantile_backtest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv(
        RESULTS / "q4_quantile_backtest_predictions.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    ok = backtest[backtest["status"].eq("ok")].copy()
    comparison = (
        ok.groupby(["scope", "stratum", "model"], as_index=False)
        .agg(
            folds=("cutoff", "nunique"),
            test_rows=("n_test", "sum"),
            pinball_mean=("pinball_logit", "mean"),
            pinball_familyweighted_mean=("pinball_logit_familyweighted", "mean"),
            pinball_median=("pinball_logit", "median"),
            coverage_mean=("coverage_q90", "mean"),
            mae_score_mean=("mae_score", "mean"),
            spearman_mean=("spearman_score", "mean"),
        )
    )
    selections = []
    for (scope, stratum), group in comparison.groupby(["scope", "stratum"]):
        scale = group[group["model"].isin(SCALE_ONLY)].sort_values(
            "pinball_familyweighted_mean"
        )
        timed = group[group["model"].isin(TIME_MODELS)].sort_values(
            "pinball_familyweighted_mean"
        )
        if scale.empty or timed.empty:
            continue
        best_scale = scale.iloc[0]
        best_time = timed.iloc[0]
        improvement = (
            best_scale["pinball_familyweighted_mean"]
            - best_time["pinball_familyweighted_mean"]
        ) / best_scale["pinball_familyweighted_mean"]
        time_pass = bool(improvement >= 0.05)
        selected = str(best_time["model"] if time_pass else best_scale["model"])
        selections.append(
            {
                "scope": scope,
                "stratum": stratum,
                "best_scale_model": best_scale["model"],
                "best_scale_pinball": best_scale["pinball_familyweighted_mean"],
                "best_time_model": best_time["model"],
                "best_time_pinball": best_time["pinball_familyweighted_mean"],
                "relative_time_improvement": improvement,
                "time_model_pass_5pct_gate": time_pass,
                "selected_model": selected,
                "selection_rule": "time candidate only if rolling pinball improves >=5%; otherwise best scale-only",
            }
        )
    selected_df = pd.DataFrame(selections)
    comparison.to_csv(RESULTS / "q4_quantile_model_comparison.csv", index=False, encoding="utf-8-sig")
    selected_df.to_csv(RESULTS / "q4_quantile_selected_models.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coef_rows).to_csv(
        RESULTS / "q4_quantile_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    return backtest, comparison, selected_df, final_models


def task_level_effects() -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.api as sm

    tasks = pd.read_csv(RESULTS / "q4_c8_task_long.csv.gz", parse_dates=["submission_date"])
    tasks = tasks[
        tasks["strict_open"].astype(bool)
        & tasks["stratum"].isin(["pretrained", "chat_finetuned"])
        & tasks["params_B"].gt(0)
        & tasks["score_raw"].notna()
    ].copy()
    signature_counts = tasks.groupby("eval_signature")["entity_id"].nunique().sort_values(ascending=False)
    dominant = str(signature_counts.index[0])
    rows = []
    for analysis_scope, data in {
        "dominant_signature": tasks[tasks["eval_signature"].eq(dominant)],
        "all_complete_signatures": tasks[
            tasks.groupby("eval_signature")["task"].transform("nunique").ge(39)
        ],
    }.items():
        for task, part in data.groupby("task"):
            if len(part) < 25 or part["family_id"].nunique() < 12:
                continue
            y = score_to_logit(part["score_raw"].to_numpy(float) * 100.0)
            x = pd.DataFrame(
                {
                    "lnN": np.log(part["params_B"].to_numpy(float)),
                    "time_year": (part["submission_date"] - ORIGIN).dt.days.to_numpy(float) / 365.25,
                    "is_chat": part["stratum"].eq("chat_finetuned").to_numpy(float),
                },
                index=part.index,
            )
            if analysis_scope == "all_complete_signatures":
                dummies = pd.get_dummies(part["eval_signature"], prefix="sig", drop_first=True, dtype=float)
                dummies.index = part.index
                x = pd.concat([x, dummies], axis=1)
            x = sm.add_constant(x.astype(float), has_constant="add")
            try:
                fit = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": part["family_id"]})
            except Exception:
                continue
            rows.append(
                {
                    "analysis_scope": analysis_scope,
                    "dominant_signature": dominant,
                    "task": task,
                    "task_family": part["task_family"].iloc[0],
                    "n": len(part),
                    "families": part["family_id"].nunique(),
                    "time_effect_logit_per_year": float(fit.params["time_year"]),
                    "time_ci_low": float(fit.conf_int().loc["time_year", 0]),
                    "time_ci_high": float(fit.conf_int().loc["time_year", 1]),
                    "time_pvalue": float(fit.pvalues["time_year"]),
                    "r2": float(fit.rsquared),
                    "evidence": "conditional association; not causal technology effect",
                }
            )
    effects = pd.DataFrame(rows)
    # Control the family of 39 task-wise tests within each analysis scope.
    effects["time_qvalue_bh"] = np.nan
    for _, idx in effects.groupby("analysis_scope").groups.items():
        idx = list(idx)
        p = effects.loc[idx, "time_pvalue"].to_numpy(float)
        order = np.argsort(p)
        ranked = p[order]
        adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        restored = np.empty_like(adjusted)
        restored[order] = np.clip(adjusted, 0.0, 1.0)
        effects.loc[idx, "time_qvalue_bh"] = restored
    effects["significant_positive_bh"] = (
        effects["time_effect_logit_per_year"].gt(0)
        & effects["time_qvalue_bh"].lt(0.05)
    )
    summary = (
        effects.groupby(["analysis_scope", "task_family"], as_index=False)
        .agg(
            tasks=("task", "nunique"),
            median_time_effect=("time_effect_logit_per_year", "median"),
            positive_share=("time_effect_logit_per_year", lambda s: float(np.mean(s > 0))),
            significant_positive_share=("significant_positive_bh", "mean"),
            median_r2=("r2", "median"),
        )
    )
    effects.to_csv(RESULTS / "q4_c8_task_time_effects.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS / "q4_c8_family_time_summary.csv", index=False, encoding="utf-8-sig")
    return effects, summary


def decompose_progress(
    entity: pd.DataFrame, selections: pd.DataFrame, final_models: dict
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(RNG_SEED)
    for stratum in ["pretrained", "chat_finetuned"]:
        data = entity[
            entity["strict_open"]
            & entity["valid_core_fields"]
            & entity["stratum"].eq(stratum)
        ].copy()
        if data.empty:
            continue
        start_cut = pd.Timestamp("2024-07-31")
        start_pool = data[data["submission_date"].le(start_cut)]
        if start_pool.empty:
            continue
        start = start_pool.loc[start_pool["score_equal"].idxmax()]
        end = data.loc[data["score_equal"].idxmax()]
        # Use the interpretable linear scale+time model for decomposition.
        spec = "scale_time_linear"
        model, center, features, _ = final_models[("strict", stratum, spec)]

        def f(ln_n: float, date: pd.Timestamp) -> float:
            temp = pd.DataFrame(
                {
                    "params_B": [math.exp(ln_n)],
                    "submission_date": [date],
                    "stratum": [stratum],
                }
            )
            return float(model.predict(feature_frame(temp, spec, center))[0])

        x0, x1 = math.log(float(start["params_B"])), math.log(float(end["params_B"]))
        t0, t1 = pd.Timestamp(start["submission_date"]), pd.Timestamp(end["submission_date"])
        f00, f10, f01, f11 = f(x0, t0), f(x1, t0), f(x0, t1), f(x1, t1)
        phi_scale = 0.5 * ((f10 - f00) + (f11 - f01))
        phi_time = 0.5 * ((f01 - f00) + (f11 - f10))
        observed = float(end["score_logit"] - start["score_logit"])
        residual = observed - phi_scale - phi_time

        selection = selections[
            selections["scope"].eq("strict") & selections["stratum"].eq(stratum)
        ]
        time_gate = bool(selection["time_model_pass_5pct_gate"].iloc[0]) if len(selection) else False

        # Family bootstrap refits the same interpretable decomposition model.
        boot = []
        family_ids = data["family_id"].unique()
        for _ in range(200):
            sampled = rng.choice(family_ids, size=len(family_ids), replace=True)
            pieces = []
            for j, family in enumerate(sampled):
                piece = data[data["family_id"].eq(family)].copy()
                piece["family_id"] = piece["family_id"].astype(str) + f"__boot{j}"
                pieces.append(piece)
            bdata = pd.concat(pieces, ignore_index=True)
            try:
                bm, bc, _ = fit_quantile(bdata, spec)
                def bf(ln_n: float, date: pd.Timestamp) -> float:
                    temp = pd.DataFrame({"params_B": [math.exp(ln_n)], "submission_date": [date], "stratum": [stratum]})
                    return float(bm.predict(feature_frame(temp, spec, bc))[0])
                b00, b10, b01, b11 = bf(x0, t0), bf(x1, t0), bf(x0, t1), bf(x1, t1)
                boot.append((0.5 * ((b10-b00)+(b11-b01)), 0.5 * ((b01-b00)+(b11-b10))))
            except Exception:
                continue
        b = np.asarray(boot, dtype=float)
        rows.append(
            {
                "stratum": stratum,
                "start_model": start["model_name"],
                "start_date": t0.date(),
                "start_params_B": start["params_B"],
                "start_score": start["score_equal"],
                "end_model": end["model_name"],
                "end_date": t1.date(),
                "end_params_B": end["params_B"],
                "end_score": end["score_equal"],
                "observed_delta_logit": observed,
                "scale_component_logit": phi_scale,
                "time_proxy_component_logit": phi_time,
                "unexplained_residual_logit": residual,
                "scale_share_of_observed": phi_scale / observed if observed != 0 else np.nan,
                "time_proxy_share_of_observed": phi_time / observed if observed != 0 else np.nan,
                "unexplained_share_of_observed": residual / observed if observed != 0 else np.nan,
                "scale_ci_low": float(np.quantile(b[:, 0], 0.05)) if len(b) else np.nan,
                "scale_ci_high": float(np.quantile(b[:, 0], 0.95)) if len(b) else np.nan,
                "time_ci_low": float(np.quantile(b[:, 1], 0.05)) if len(b) else np.nan,
                "time_ci_high": float(np.quantile(b[:, 1], 0.95)) if len(b) else np.nan,
                "bootstrap_success": len(b),
                "time_model_passed_rolling_gate": time_gate,
                "evidence": "descriptive Shapley decomposition; time term is non-scale progress proxy",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "q4_progress_decomposition.csv", index=False, encoding="utf-8-sig")
    return out


def bridge_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv(C_DIR / "loss_benchmark_bridge_expanded.csv")
    targets = ["LB_Average", "LB_IFEval", "LB_BBH", "LB_MATH", "LB_GPQA", "LB_MUSR", "LB_MMLU_PRO"]
    high = data[data["Loss_Comparability"].astype(str).str.startswith("High")].copy()
    medium = data[data["Loss_Comparability"].astype(str).str.startswith("Medium")].copy()
    rows, pred_rows = [], []

    def fit_predict(kind: str, xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray) -> np.ndarray:
        if kind == "constant":
            return np.repeat(np.mean(ytr), len(xte))
        if kind == "linear":
            return LinearRegression().fit(xtr[:, None], ytr).predict(xte[:, None])
        if kind == "isotonic":
            return IsotonicRegression(increasing=False, out_of_bounds="clip").fit(xtr, ytr).predict(xte)
        if kind == "logistic":
            def fun(x, upper, slope, center):
                return upper / (1.0 + np.exp(slope * (x - center)))
            pars, _ = curve_fit(
                fun, xtr, ytr,
                p0=[max(60.0, float(np.max(ytr))), 2.0, float(np.median(xtr))],
                bounds=([0.0, 0.0, float(np.min(xtr))-2], [100.0, 20.0, float(np.max(xtr))+2]),
                maxfev=20000,
            )
            return fun(xte, *pars)
        raise ValueError(kind)

    for target in targets:
        x = high["Val_Loss"].to_numpy(float)
        y = high[target].to_numpy(float)
        for kind in ["constant", "linear", "isotonic", "logistic"]:
            preds = np.full(len(high), np.nan)
            for i in range(len(high)):
                keep = np.arange(len(high)) != i
                try:
                    preds[i] = fit_predict(kind, x[keep], y[keep], x[[i]])[0]
                except Exception:
                    pass
            mask = np.isfinite(preds)
            rows.append(
                {
                    "target": target,
                    "model": kind,
                    "n_high": int(mask.sum()),
                    "loo_mae": mean_absolute_error(y[mask], preds[mask]) if mask.any() else np.nan,
                    "loo_rmse": math.sqrt(mean_squared_error(y[mask], preds[mask])) if mask.any() else np.nan,
                    "loo_spearman": float(spearmanr(y[mask], preds[mask]).statistic) if mask.sum() >= 3 else np.nan,
                }
            )
            for i, pred in enumerate(preds):
                pred_rows.append(
                    {
                        "target": target,
                        "model": kind,
                        "row": int(high.index[i]),
                        "loss": x[i],
                        "truth": y[i],
                        "prediction": pred,
                        "evidence_level": "real_holdout_high_comparability",
                    }
                )

    metrics = pd.DataFrame(rows)
    decisions = []
    for target, group in metrics.groupby("target"):
        base = group[group["model"].eq("constant")].iloc[0]
        candidates = group[~group["model"].eq("constant")].sort_values("loo_rmse")
        best = candidates.iloc[0]
        improvement = (base["loo_rmse"] - best["loo_rmse"]) / base["loo_rmse"] if base["loo_rmse"] else np.nan
        accept = bool(np.isfinite(improvement) and improvement >= 0.05 and best["loo_spearman"] > 0)
        decisions.append(
            {
                "target": target,
                "constant_rmse": base["loo_rmse"],
                "best_nonconstant": best["model"],
                "best_nonconstant_rmse": best["loo_rmse"],
                "relative_rmse_improvement": improvement,
                "accepted_for_bridge": accept,
                "decision": "local_monotone_bridge" if accept else "not_identified_beyond_constant",
                "support_loss_min": float(high["Val_Loss"].min()),
                "support_loss_max": float(high["Val_Loss"].max()),
                "n_high": len(high),
                "n_medium_sensitivity": len(medium),
            }
        )
    decisions_df = pd.DataFrame(decisions)
    metrics.to_csv(RESULTS / "q4_bridge_loo_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(RESULTS / "q4_bridge_loo_predictions.csv", index=False, encoding="utf-8-sig")
    decisions_df.to_csv(RESULTS / "q4_bridge_decisions.csv", index=False, encoding="utf-8-sig")
    return metrics, pd.DataFrame(pred_rows), decisions_df


def compute_growth_model() -> tuple[pd.DataFrame, dict]:
    c4 = pd.read_csv(C_DIR / "epoch_all_ai_models.csv")
    c4["date"] = pd.to_datetime(c4["Publication date"], errors="coerce")
    c4["compute"] = pd.to_numeric(c4["Training compute (FLOP)"], errors="coerce")
    c4["parameters"] = pd.to_numeric(c4["Parameters"], errors="coerce")
    cutoff = pd.Timestamp("2025-03-13")
    data = c4[
        c4["date"].between("2018-01-01", cutoff)
        & c4["Domain"].fillna("").str.contains("Language", case=False)
        & c4["Model accessibility"].eq("Open weights (unrestricted)")
        & c4["compute"].gt(0)
        & c4["parameters"].gt(0)
    ].copy()
    data["time_year"] = (data["date"] - pd.Timestamp("2018-01-01")).dt.days / 365.25
    data["log10_compute"] = np.log10(data["compute"])
    data["log10_params"] = np.log10(data["parameters"])
    qmodel = QuantileRegressor(quantile=0.9, alpha=0.0, solver="highs").fit(
        data[["time_year"]], data["log10_compute"]
    )
    relation = TheilSenRegressor(random_state=RNG_SEED).fit(
        data[["log10_compute"]], data["log10_params"]
    )
    yearly = (
        data.assign(year=data["date"].dt.year)
        .groupby("year", as_index=False)
        .agg(n=("Model", "size"), compute_q90=("log10_compute", lambda s: float(s.quantile(0.9))),
             compute_max=("log10_compute", "max"), params_q90=("log10_params", lambda s: float(s.quantile(0.9))))
    )
    yearly.to_csv(RESULTS / "q4_compute_growth_history.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n": len(data),
        "date_min": str(data["date"].min().date()),
        "date_max": str(data["date"].max().date()),
        "q90_log10_compute_slope_per_year": float(qmodel.coef_[0]),
        "q90_log10_compute_intercept": float(qmodel.intercept_),
        "log10_params_per_log10_compute": float(relation.coef_[0]),
        "parameter_relation_intercept": float(relation.intercept_),
        "evidence": "C4 observational/partly estimated compute; scenario calibration only",
    }
    (RESULTS / "q4_compute_growth_model.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return data, summary


def simulate_record(
    train: pd.DataFrame,
    model_spec: str,
    horizon_days: int,
    n_sim: int,
    param_log_shift_per_year: float,
    time_effect_enabled: bool,
    rng: np.random.Generator,
    arrival_rate_multiplier: float = 1.0,
) -> np.ndarray:
    spec = model_spec
    if not time_effect_enabled and spec in TIME_MODELS:
        spec = "scale_quadratic"
    model, center, _ = fit_quantile(train, spec)
    train_pred = predict_quantile(model, train, spec, center)
    residuals = train["score_logit"].to_numpy(float) - train_pred
    residual_family_counts = train["family_id"].value_counts()
    residual_weights = train["family_id"].map(
        lambda value: 1.0 / residual_family_counts[value]
    ).to_numpy(float)
    residual_weights = residual_weights / residual_weights.sum()
    current = float(train["score_logit"].max())
    recent_cut = train["submission_date"].max() - pd.Timedelta(days=120)
    recent = train[train["submission_date"].ge(recent_cut)].copy()
    if len(recent) < 20:
        recent = train.copy()
    observed_days = max((train["submission_date"].max() - train["submission_date"].min()).days, 30)
    rate = len(train) / observed_days
    out = np.full(n_sim, current)
    for s in range(n_sim):
        n_new = int(rng.poisson(rate * horizon_days * arrival_rate_multiplier))
        if n_new <= 0:
            continue
        recent_family_counts = recent["family_id"].value_counts()
        recent_weights = recent["family_id"].map(
            lambda value: 1.0 / recent_family_counts[value]
        )
        sampled = recent.sample(
            n_new,
            replace=True,
            weights=recent_weights,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).copy()
        future_offsets = rng.uniform(1, horizon_days, size=n_new)
        sampled["submission_date"] = train["submission_date"].max() + pd.to_timedelta(future_offsets, unit="D")
        shift = param_log_shift_per_year * future_offsets / 365.25
        sampled["params_B"] = np.exp(np.log(sampled["params_B"].to_numpy(float)) + shift)
        pred = predict_quantile(model, sampled, spec, center)
        draw = pred + rng.choice(
            residuals, size=n_new, replace=True, p=residual_weights
        )
        out[s] = max(current, float(np.max(draw)))
    return logit_to_score(out)


def record_backtest_and_forecast(
    entity: pd.DataFrame, selections: pd.DataFrame, compute_summary: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = entity[
        entity["strict_open"]
        & entity["valid_core_fields"]
        & entity["stratum"].isin(["pretrained", "chat_finetuned"])
    ].sort_values("submission_date").copy()
    chosen_row = selections[
        selections["scope"].eq("strict") & selections["stratum"].eq("core_combined")
    ].iloc[0]
    chosen_scale = str(chosen_row["best_scale_model"])
    chosen_time = str(chosen_row["best_time_model"])
    time_gate = bool(chosen_row["time_model_pass_5pct_gate"])
    cutoffs = pd.to_datetime(
        [
            "2024-07-01",
            "2024-07-15",
            "2024-08-01",
            "2024-08-15",
            "2024-09-01",
            "2024-09-15",
            "2024-10-01",
            "2024-12-01",
            "2025-02-01",
        ]
    )
    horizons = [30, 60]
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for cutoff in cutoffs:
        train = data[data["submission_date"].lt(cutoff)].copy()
        if len(train) < 50:
            continue
        current = float(train["score_equal"].max())
        record_points = train.sort_values("submission_date").copy()
        record_points["cummax"] = record_points["score_equal"].cummax()
        record_points = record_points[record_points["cummax"].diff().fillna(1).gt(0)]
        xdays = (record_points["submission_date"] - record_points["submission_date"].min()).dt.days.to_numpy(float)
        yz = score_to_logit(record_points["cummax"])
        trend = TheilSenRegressor(random_state=RNG_SEED).fit(xdays[:, None], yz) if len(record_points) >= 3 else None
        # Training-window parameter trend is used only for backtest scale simulation.
        t = (train["submission_date"] - train["submission_date"].min()).dt.days.to_numpy(float) / 365.25
        param_slope = float(LinearRegression().fit(t[:, None], np.log(train["params_B"])).coef_[0]) if len(np.unique(t)) > 1 else 0.0
        for horizon in horizons:
            end = min(cutoff + pd.Timedelta(days=horizon), data["submission_date"].max())
            truth_pool = data[data["submission_date"].lt(end + pd.Timedelta(days=1))]
            truth = float(truth_pool["score_equal"].max())
            rows.append({"cutoff": cutoff.date(), "horizon_days": horizon, "model": "persistence", "prediction": current,
                         "low90": current, "high90": current, "truth": truth})
            if trend is not None:
                future_x = (end - record_points["submission_date"].min()).days
                pred = max(current, float(logit_to_score(trend.predict([[future_x]])[0])))
                rows.append({"cutoff": cutoff.date(), "horizon_days": horizon, "model": "record_theilsen", "prediction": pred,
                             "low90": np.nan, "high90": np.nan, "truth": truth})
            for label, spec, enabled in [
                ("record_process_scale", chosen_scale, False),
                ("record_process_time_candidate", chosen_time, True),
            ]:
                sims = simulate_record(train, spec, horizon, 500, param_slope, enabled, rng)
                rows.append({"cutoff": cutoff.date(), "horizon_days": horizon, "model": label,
                             "prediction": float(np.median(sims)), "low90": float(np.quantile(sims, .05)),
                             "high90": float(np.quantile(sims, .95)), "truth": truth})
    backtest = pd.DataFrame(rows)
    backtest["abs_error"] = (backtest["prediction"] - backtest["truth"]).abs()
    # Compare truth with the persistence value for the same forecast case.
    persistence_by_case = (
        backtest[backtest["model"].eq("persistence")]
        .set_index(["cutoff", "horizon_days"])["prediction"]
    )
    backtest["persistence_value"] = [
        persistence_by_case.loc[(row.cutoff, row.horizon_days)] for row in backtest.itertuples()
    ]
    backtest["record_updated"] = backtest["truth"].gt(backtest["persistence_value"] + 1e-12)
    backtest["covered90"] = (
        backtest["low90"].notna() & backtest["truth"].between(backtest["low90"], backtest["high90"])
    )
    alpha = 0.10
    backtest["interval_width90"] = backtest["high90"] - backtest["low90"]
    backtest["interval_score90"] = backtest["interval_width90"]
    below = backtest["truth"].lt(backtest["low90"])
    above = backtest["truth"].gt(backtest["high90"])
    backtest.loc[below, "interval_score90"] += (2.0 / alpha) * (
        backtest.loc[below, "low90"] - backtest.loc[below, "truth"]
    )
    backtest.loc[above, "interval_score90"] += (2.0 / alpha) * (
        backtest.loc[above, "truth"] - backtest.loc[above, "high90"]
    )
    metrics = (
        backtest.groupby("model", as_index=False)
        .agg(cases=("truth", "size"), mae=("abs_error", "mean"), median_abs_error=("abs_error", "median"),
             update_cases=("record_updated", "sum"),
             interval_cases=("low90", lambda s: int(s.notna().sum())), coverage90=("covered90", "mean"),
             mean_interval_width90=("interval_width90", "mean"),
             mean_interval_score90=("interval_score90", "mean"))
        .sort_values(["mae", "model"])
    )
    update_mae = (
        backtest[backtest["record_updated"]]
        .groupby("model")["abs_error"].mean()
        .rename("mae_record_update_cases")
    )
    metrics = metrics.merge(update_mae, on="model", how="left")
    metrics["selected_main"] = False
    metrics.loc[metrics.index[0], "selected_main"] = True

    forecast_rows = []
    compute_slope = float(compute_summary["q90_log10_compute_slope_per_year"])
    elasticity = float(compute_summary["log10_params_per_log10_compute"])
    current = float(data["score_equal"].max())
    scenario_factors = {
        "historical_trend": 1.0,
        "moderate_slowdown": 0.5,
        "strong_slowdown": 0.25,
        "zero_scale_growth": 0.0,
    }
    for scenario, factor in scenario_factors.items():
        # log10 C growth -> natural-log N growth.
        param_shift_per_year = math.log(10.0) * compute_slope * elasticity * factor
        for horizon in [365, 730]:
            for label, spec, enabled, evidence in [
                (
                    "record_process_scale",
                    chosen_scale,
                    False,
                    "conditional_scenario_extrapolation",
                ),
                (
                    "record_process_time_candidate",
                    chosen_time,
                    True,
                    "rejected_time_candidate_sensitivity"
                    if not time_gate
                    else "conditional_scenario_extrapolation",
                ),
            ]:
                sims = simulate_record(
                    data,
                    spec,
                    horizon,
                    2000,
                    param_shift_per_year,
                    enabled,
                    rng,
                )
                forecast_rows.append(
                    {
                        "scenario": scenario,
                        "horizon_months": 12 if horizon == 365 else 24,
                        "forecast_date": (data["submission_date"].max() + pd.Timedelta(days=horizon)).date(),
                        "model": label,
                        "conditional_time_effect_enabled": bool(enabled),
                        "passed_rolling_selection_gate": bool((not enabled) or time_gate),
                        "current_frontier": current,
                        "p05": float(np.quantile(sims, .05)),
                        "p50": float(np.quantile(sims, .50)),
                        "p95": float(np.quantile(sims, .95)),
                        "mean": float(np.mean(sims)),
                        "probability_frontier_update": float(np.mean(sims > current + 1e-10)),
                        "compute_growth_factor": factor,
                        "logN_shift_per_year": param_shift_per_year,
                        "evidence_level": evidence,
                    }
                )
            forecast_rows.append(
                {
                    "scenario": scenario,
                    "horizon_months": 12 if horizon == 365 else 24,
                    "forecast_date": (data["submission_date"].max() + pd.Timedelta(days=horizon)).date(),
                    "model": "persistence",
                    "conditional_time_effect_enabled": False,
                    "passed_rolling_selection_gate": True,
                    "current_frontier": current,
                    "p05": current,
                    "p50": current,
                    "p95": current,
                    "mean": current,
                    "probability_frontier_update": 0.0,
                    "compute_growth_factor": factor,
                    "logN_shift_per_year": param_shift_per_year,
                    "evidence_level": "baseline",
                }
            )
    forecast = pd.DataFrame(forecast_rows)
    backtest.to_csv(RESULTS / "q4_record_backtest.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(RESULTS / "q4_record_model_comparison.csv", index=False, encoding="utf-8-sig")
    forecast.to_csv(RESULTS / "q4_frontier_forecast.csv", index=False, encoding="utf-8-sig")

    history = data[["submission_date", "model_name", "stratum", "params_B", "score_equal", "family_id"]].copy()
    history = history.sort_values("submission_date")
    history["cumulative_frontier"] = history["score_equal"].cummax()
    history["is_record_update"] = history["cumulative_frontier"].diff().fillna(1).gt(0)
    history.to_csv(RESULTS / "q4_frontier_history.csv", index=False, encoding="utf-8-sig")
    return backtest, metrics, forecast


def main() -> None:
    entity = pd.read_csv(RESULTS / "q4_entity_cohort.csv.gz", parse_dates=["submission_date"])
    backtest, comparison, selections, final_models = rolling_quantile_experiment(entity)
    task_effects, task_summary = task_level_effects()
    decomposition = decompose_progress(entity, selections, final_models)
    bridge_metrics, bridge_preds, bridge_decisions = bridge_experiment()
    _, compute_summary = compute_growth_model()
    record_bt, record_metrics, forecast = record_backtest_and_forecast(entity, selections, compute_summary)

    output = {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "status": "q4_core_models_complete",
        "prediction_origin": "2025-03-13",
        "prediction_end_12m": "2026-03-13",
        "prediction_end_24m": "2027-03-13",
        "quantile_selected": selections.to_dict(orient="records"),
        "record_model_comparison": record_metrics.to_dict(orient="records"),
        "bridge_decisions": bridge_decisions.to_dict(orient="records"),
        "decomposition": decomposition.to_dict(orient="records"),
        "task_effect_rows": int(len(task_effects)),
        "forecast_rows": int(len(forecast)),
        "limitations": [
            "Leaderboard submission history spans only 2024-06-08 to 2025-03-13",
            "12/24 month forecasts are conditional scenario extrapolations beyond the observed horizon",
            "C4 compute is observational and partly estimated",
            "conditional time effects are non-scale progress proxies, not causal technology effects",
            "Q2/Q3 outputs are model scenarios and enter only through a separately validated bridge",
        ],
    }
    (RESULTS / "q4_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({
        "status": output["status"],
        "quantile_selections": len(selections),
        "task_effects": len(task_effects),
        "record_best": record_metrics.iloc[0]["model"],
        "bridge_accepted": int(bridge_decisions["accepted_for_bridge"].sum()),
        "forecast_rows": len(forecast),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
