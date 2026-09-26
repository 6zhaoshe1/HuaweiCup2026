# -*- coding: utf-8 -*-
"""Question 4 final targeted validation and closure experiments.

AI assistance: OpenAI Codex, 2026-09-26.

This script does not overwrite the frozen Question 1--3 interfaces or the
original Question 4 artifacts.  It addresses three identified risks:

1. nested, cutoff-local model selection for record-process backtests;
2. record arrivals versus independent family arrivals and repeated tail draws;
3. interpretation of endpoint Shapley components versus endpoint residuals.
"""

from __future__ import annotations

import json
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor, TheilSenRegressor

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))

from q4_modeling import (  # noqa: E402
    fit_quantile,
    logit_to_score,
    predict_quantile,
    score_to_logit,
    weighted_pinball,
)

SEED = 20260926
Q = 0.90
CORE_STRATA = ["pretrained", "chat_finetuned"]
C4 = (
    ROOT
    / "第二十三届中国研究生数学建模竞赛 - 中文题目"
    / "中文题目"
    / "F题"
    / "real_attachments"
    / "C_efficiency_evolution"
    / "epoch_all_ai_models.csv"
)


def load_core() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "q4_entity_cohort.csv.gz", parse_dates=["submission_date"])
    return d[
        d["strict_open"].astype(bool)
        & d["valid_core_fields"].astype(bool)
        & d["stratum"].isin(CORE_STRATA)
    ].sort_values("submission_date").copy()


def family_weights(d: pd.DataFrame) -> np.ndarray:
    counts = d["family_id"].value_counts()
    return d["family_id"].map(lambda x: 1.0 / counts[x]).to_numpy(float)


def nested_select(train: pd.DataFrame) -> tuple[str, list[dict]]:
    """Select a scale-only specification using only data available in train."""
    specs = ["scale_linear", "scale_quadratic"]
    start = train["submission_date"].min().normalize() + pd.Timedelta(days=45)
    stop = train["submission_date"].max().normalize() - pd.Timedelta(days=30)
    cutoffs = list(pd.date_range(start, stop, freq="30D")) if start <= stop else []
    rows: list[dict] = []
    for cutoff in cutoffs:
        tr = train[train["submission_date"].lt(cutoff)].copy()
        te = train[
            train["submission_date"].between(
                cutoff, cutoff + pd.Timedelta(days=29), inclusive="both"
            )
        ].copy()
        if len(tr) < 50 or len(te) < 8 or tr["family_id"].nunique() < 30:
            continue
        w = family_weights(te)
        for spec in specs:
            model, center, _ = fit_quantile(tr, spec)
            pred = predict_quantile(model, te, spec, center)
            rows.append(
                {
                    "inner_cutoff": cutoff.date(),
                    "model": spec,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "pinball_familyweighted": weighted_pinball(
                        te["score_logit"].to_numpy(float), pred, w, Q
                    ),
                }
            )
    if not rows:
        return "scale_linear", rows
    scores = pd.DataFrame(rows).groupby("model")["pinball_familyweighted"].mean()
    linear = float(scores.get("scale_linear", np.inf))
    quadratic = float(scores.get("scale_quadratic", np.inf))
    # Complexity is admitted only for a visible, pre-fixed 2% inner-fold gain.
    chosen = "scale_quadratic" if quadratic <= 0.98 * linear else "scale_linear"
    return chosen, rows


def arrival_parameters(train: pd.DataFrame, unit: str, recent_days: int | None) -> dict:
    d = train.copy()
    if unit == "family":
        d = d.sort_values("submission_date").drop_duplicates("family_id", keep="first")
    max_date = train["submission_date"].max()
    if recent_days is not None:
        recent = d[d["submission_date"].ge(max_date - pd.Timedelta(days=recent_days))]
        if len(recent) >= 20:
            d = recent
    days = max((d["submission_date"].max() - d["submission_date"].min()).days + 1, 30)
    rate_day = len(d) / days
    weekly = (
        d.set_index("submission_date")
        .resample("7D")
        .size()
        .astype(float)
    )
    mean_w = float(weekly.mean()) if len(weekly) else rate_day * 7
    var_w = float(weekly.var(ddof=1)) if len(weekly) > 1 else mean_w
    if var_w > mean_w + 1e-12 and mean_w > 0:
        nb_size_week = mean_w**2 / (var_w - mean_w)
    else:
        nb_size_week = math.inf
    return {
        "rate_day": rate_day,
        "weekly_mean": mean_w,
        "weekly_variance": var_w,
        "fano": var_w / mean_w if mean_w > 0 else np.nan,
        "nb_size_week": nb_size_week,
        "n_events": len(d),
        "window_days": days,
    }


def draw_count(rng: np.random.Generator, mean_count: float, nb_size_week: float, horizon: int) -> int:
    if not np.isfinite(nb_size_week):
        return int(rng.poisson(mean_count))
    size = max(nb_size_week * horizon / 7.0, 1e-6)
    p = size / (size + mean_count)
    return int(rng.negative_binomial(size, p))


def simulate(
    train: pd.DataFrame,
    spec: str,
    horizon: int,
    n_sim: int,
    logn_shift_year: float,
    variant: str,
    rng: np.random.Generator,
    arrival_multiplier: float = 1.0,
) -> np.ndarray:
    model, center, _ = fit_quantile(train, spec)
    train = train.copy()
    train["residual"] = train["score_logit"].to_numpy(float) - predict_quantile(
        model, train, spec, center
    )
    current = float(train["score_logit"].max())

    if variant == "legacy_record_iid":
        pool = train[train["submission_date"].ge(train["submission_date"].max() - pd.Timedelta(days=120))]
        if len(pool) < 20:
            pool = train
        params = arrival_parameters(train, "record", None)
        residual_pool = train["residual"].to_numpy(float)
        rw = family_weights(train)
        rw = rw / rw.sum()
        recent_family_counts = pool["family_id"].value_counts()
        pool_weights = pool["family_id"].map(lambda x: 1.0 / recent_family_counts[x])
        cap = np.inf
        use_nb = False
    else:
        first = train.sort_values("submission_date").drop_duplicates("family_id", keep="first")
        pool = first[first["submission_date"].ge(train["submission_date"].max() - pd.Timedelta(days=180))]
        if len(pool) < 20:
            pool = first
        params = arrival_parameters(train, "family", 180)
        residual_by_family = train.groupby("family_id")["residual"].median()
        residual_pool = residual_by_family.to_numpy(float)
        rw = None
        pool_weights = None
        cap = float(np.quantile(residual_pool, 0.95)) if variant == "family_nb_tail95" else np.inf
        use_nb = variant == "family_nb_tail95"

    mean_count = params["rate_day"] * horizon * arrival_multiplier
    out = np.full(n_sim, current, dtype=float)
    for s in range(n_sim):
        n_new = (
            draw_count(rng, mean_count, params["nb_size_week"], horizon)
            if use_nb
            else int(rng.poisson(mean_count))
        )
        if n_new <= 0:
            continue
        sampled = pool.sample(
            n_new,
            replace=True,
            weights=pool_weights,
            random_state=int(rng.integers(0, 2**31 - 1)),
        ).copy()
        offsets = rng.uniform(1, horizon, n_new)
        sampled["submission_date"] = train["submission_date"].max() + pd.to_timedelta(offsets, unit="D")
        sampled["params_B"] = np.exp(
            np.log(sampled["params_B"].to_numpy(float))
            + logn_shift_year * offsets / 365.25
        )
        pred = predict_quantile(model, sampled, spec, center)
        res = rng.choice(residual_pool, size=n_new, replace=True, p=rw)
        res = np.minimum(res, cap)
        out[s] = max(current, float(np.max(pred + res)))
    return logit_to_score(out)


def arrival_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for unit in ["record", "family"]:
        for window in [None, 90, 120, 180]:
            p = arrival_parameters(data, unit, window)
            rows.append({"unit": unit, "recent_days": "full" if window is None else window, **p})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "q4_final_arrival_diagnostics.csv", index=False, encoding="utf-8-sig")
    return out


def backtest(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cutoffs = pd.to_datetime(
        [
            "2024-07-01", "2024-07-15", "2024-08-01", "2024-08-15",
            "2024-09-01", "2024-09-15", "2024-10-01", "2024-12-01", "2025-02-01",
        ]
    )
    variants = ["legacy_record_iid", "family_poisson_block", "family_nb_tail95"]
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    selection_rows: list[dict] = []
    inner_rows: list[dict] = []
    for cutoff in cutoffs:
        train = data[data["submission_date"].lt(cutoff)].copy()
        if len(train) < 50:
            continue
        spec, inner = nested_select(train)
        for row in inner:
            inner_rows.append({"outer_cutoff": cutoff.date(), **row})
        inner_folds = len({str(x["inner_cutoff"]) for x in inner})
        selection_rows.append(
            {
                "outer_cutoff": cutoff.date(),
                "selected_model": spec,
                "inner_folds": inner_folds,
                "selection_evidence": "nested_rolling" if inner_folds else "predeclared_linear_fallback",
            }
        )
        t = (train["submission_date"] - train["submission_date"].min()).dt.days.to_numpy(float) / 365.25
        param_slope = float(LinearRegression().fit(t[:, None], np.log(train["params_B"])).coef_[0])
        current = float(train["score_equal"].max())
        for horizon in [30, 60]:
            requested_end = cutoff + pd.Timedelta(days=horizon)
            end = min(requested_end, data["submission_date"].max())
            actual_horizon = int((end - cutoff).days)
            complete_window = bool(requested_end <= data["submission_date"].max())
            truth = float(data[data["submission_date"].le(end)]["score_equal"].max())
            case = {
                "cutoff": cutoff.date(), "horizon_days": horizon,
                "requested_end": requested_end.date(), "observed_end": end.date(),
                "actual_horizon_days": actual_horizon, "complete_window": complete_window,
                "truth": truth, "persistence_value": current,
                "record_updated": truth > current + 1e-12,
                "selected_scale_model": spec, "inner_folds": inner_folds,
            }
            rows.append({**case, "model": "persistence", "prediction": current, "low90": current, "high90": current})
            for variant in variants:
                sims = simulate(train, spec, horizon, 700, param_slope, variant, rng)
                rows.append(
                    {
                        **case,
                        "model": variant,
                        "prediction": float(np.median(sims)),
                        "low90": float(np.quantile(sims, 0.05)),
                        "high90": float(np.quantile(sims, 0.95)),
                    }
                )
    out = pd.DataFrame(rows)
    out["abs_error"] = (out["prediction"] - out["truth"]).abs()
    out["covered90"] = out["truth"].between(out["low90"], out["high90"])
    out["interval_width90"] = out["high90"] - out["low90"]
    alpha = 0.10
    out["interval_score90"] = out["interval_width90"]
    below, above = out["truth"] < out["low90"], out["truth"] > out["high90"]
    out.loc[below, "interval_score90"] += 2 / alpha * (out.loc[below, "low90"] - out.loc[below, "truth"])
    out.loc[above, "interval_score90"] += 2 / alpha * (out.loc[above, "truth"] - out.loc[above, "high90"])
    out["disjoint_60d_case"] = out["horizon_days"].eq(60) & pd.to_datetime(out["cutoff"]).isin(
        pd.to_datetime(["2024-07-01", "2024-09-01", "2024-12-01", "2025-02-01"])
    )
    metrics = []
    for model, part in out.groupby("model"):
        complete = part[part["complete_window"]].copy()
        complete_60_cutoffs = set(
            out[out["complete_window"] & out["horizon_days"].eq(60)]["cutoff"].astype(str)
        )
        subsets = [
            ("all_including_incomplete", part),
            ("formal_complete_pooled", complete),
            ("formal_complete_30d", complete[complete["horizon_days"].eq(30)]),
            ("formal_complete_60d", complete[complete["horizon_days"].eq(60)]),
            ("common_cutoffs_30d", complete[complete["horizon_days"].eq(30) & complete["cutoff"].astype(str).isin(complete_60_cutoffs)]),
            ("common_cutoffs_60d", complete[complete["horizon_days"].eq(60) & complete["cutoff"].astype(str).isin(complete_60_cutoffs)]),
            ("disjoint_60d", part[part["disjoint_60d_case"] & part["complete_window"]]),
        ]
        for subset, use in subsets:
            if use.empty:
                continue
            metrics.append(
                {
                    "model": model, "evaluation_subset": subset, "cases": len(use),
                    "mae": float(use["abs_error"].mean()),
                    "mae_update_cases": float(use.loc[use["record_updated"], "abs_error"].mean()),
                    "coverage90": float(use["covered90"].mean()),
                    "mean_interval_width90": float(use["interval_width90"].mean()),
                    "mean_interval_score90": float(use["interval_score90"].mean()),
                }
            )
    selection = pd.DataFrame(selection_rows)
    out.to_csv(RESULTS / "q4_final_record_backtest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metrics).to_csv(RESULTS / "q4_final_record_metrics.csv", index=False, encoding="utf-8-sig")
    selection.to_csv(RESULTS / "q4_final_nested_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inner_rows).to_csv(
        RESULTS / "q4_final_nested_inner_scores.csv", index=False, encoding="utf-8-sig"
    )
    return out, pd.DataFrame(metrics), selection


def cutoff_cluster_bootstrap(bt: pd.DataFrame, n_boot: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample outer cutoff clusters, preserving the paired 30/60-day rows."""
    rng = np.random.default_rng(SEED + 11)
    bt = bt[bt["complete_window"]].copy()
    cutoffs = np.array(sorted(bt["cutoff"].astype(str).unique()))
    models = ["legacy_record_iid", "family_poisson_block", "family_nb_tail95"]
    draws = []
    for b in range(n_boot):
        sampled = rng.choice(cutoffs, size=len(cutoffs), replace=True)
        pieces = [bt[bt["cutoff"].astype(str).eq(c)] for c in sampled]
        d = pd.concat(pieces, ignore_index=True)
        base = float(d[d["model"].eq("persistence")]["abs_error"].mean())
        for model in models:
            mae = float(d[d["model"].eq(model)]["abs_error"].mean())
            draws.append({"bootstrap":b,"model":model,"mae_gain_vs_persistence":base-mae})
    out = pd.DataFrame(draws)
    summary = out.groupby("model",as_index=False).agg(
        gain_p05=("mae_gain_vs_persistence",lambda s:float(s.quantile(.05))),
        gain_p50=("mae_gain_vs_persistence","median"),
        gain_p95=("mae_gain_vs_persistence",lambda s:float(s.quantile(.95))),
        probability_gain_positive=("mae_gain_vs_persistence",lambda s:float(np.mean(s>0))),
    )
    out.to_csv(RESULTS/"q4_final_cutoff_cluster_bootstrap.csv.gz",index=False,compression="gzip")
    summary.to_csv(RESULTS/"q4_final_cutoff_cluster_bootstrap_summary.csv",index=False,encoding="utf-8-sig")
    return out, summary


def compute_mapping_bootstrap(n_boot: int = 200) -> tuple[pd.DataFrame, dict]:
    c4 = pd.read_csv(C4)
    c4["date"] = pd.to_datetime(c4["Publication date"], errors="coerce")
    c4["compute"] = pd.to_numeric(c4["Training compute (FLOP)"], errors="coerce")
    c4["parameters"] = pd.to_numeric(c4["Parameters"], errors="coerce")
    d = c4[
        c4["date"].between("2018-01-01", "2025-03-13")
        & c4["Domain"].fillna("").str.contains("Language", case=False)
        & c4["Model accessibility"].eq("Open weights (unrestricted)")
        & c4["compute"].gt(0) & c4["parameters"].gt(0)
    ].copy()
    d["time_year"] = (d["date"] - pd.Timestamp("2018-01-01")).dt.days / 365.25
    d["log10_compute"] = np.log10(d["compute"])
    d["log10_params"] = np.log10(d["parameters"])
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for b in range(n_boot):
        s = d.iloc[rng.integers(0, len(d), len(d))]
        try:
            q = QuantileRegressor(quantile=.9, alpha=0, solver="highs").fit(s[["time_year"]], s["log10_compute"])
            ts = TheilSenRegressor(random_state=SEED + b).fit(s[["log10_compute"]], s["log10_params"])
            rows.append({"bootstrap": b, "compute_slope": q.coef_[0], "parameter_elasticity": ts.coef_[0],
                         "logN_shift_per_year": math.log(10) * q.coef_[0] * ts.coef_[0]})
        except Exception:
            continue
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "q4_final_compute_mapping_bootstrap.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_source": len(d), "bootstrap_success": len(out),
        "compute_slope_p05_p50_p95": [float(x) for x in out["compute_slope"].quantile([.05,.5,.95])],
        "elasticity_p05_p50_p95": [float(x) for x in out["parameter_elasticity"].quantile([.05,.5,.95])],
        "logN_shift_p05_p50_p95": [float(x) for x in out["logN_shift_per_year"].quantile([.05,.5,.95])],
        "evidence": "row bootstrap of observational/partly estimated C4; scenario uncertainty, not causal confidence interval",
    }
    return out, summary


def extreme_search(data: pd.DataFrame, spec: str) -> pd.DataFrame:
    model, center, _ = fit_quantile(data, spec)
    d = data.copy()
    d["residual"] = d["score_logit"] - predict_quantile(model, d, spec, center)
    family_res = d.groupby("family_id")["residual"].median().to_numpy(float)
    record_res = d["residual"].to_numpy(float)
    cap = np.quantile(family_res, .95)
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for n in [1, 5, 10, 25, 50, 100, 250, 500]:
        for mechanism, pool, upper in [
            ("record_iid", record_res, np.inf),
            ("family_block", family_res, np.inf),
            ("family_block_tail95", family_res, cap),
        ]:
            maxima = np.max(np.minimum(rng.choice(pool, size=(4000, n), replace=True), upper), axis=1)
            rows.append({"n_candidates": n, "residual_mechanism": mechanism,
                         "max_residual_p50": np.quantile(maxima,.5), "max_residual_p95": np.quantile(maxima,.95)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "q4_final_extreme_search.csv", index=False, encoding="utf-8-sig")
    return out


def forecast(data: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 3)
    base_shift = float(mapping["logN_shift_per_year"].median())
    factors = {"historical_trend": 1.0, "moderate_slowdown": .5, "strong_slowdown": .25, "zero_scale_growth": 0.0}
    rows = []
    current = float(data["score_equal"].max())
    for scenario, factor in factors.items():
        for horizon, months in [(365,12),(730,24)]:
            for variant in ["legacy_record_iid", "family_poisson_block", "family_nb_tail95"]:
                # 1,000 paths are sufficient for the scenario contrast here; the
                # independent verifier repeats the main rows with a second seed.
                sims = simulate(data, "scale_quadratic", horizon, 1000, base_shift*factor, variant, rng)
                rows.append({"scenario":scenario,"compute_growth_factor":factor,"horizon_months":months,
                             "forecast_date":(data["submission_date"].max()+pd.Timedelta(days=horizon)).date(),
                             "model":variant,"current_frontier":current,
                             "p05":np.quantile(sims,.05),"p50":np.quantile(sims,.5),"p95":np.quantile(sims,.95),
                             "probability_frontier_update":np.mean(sims>current+1e-10),
                             "evidence_level":"conditional_scenario_extrapolation"})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "q4_final_frontier_forecast.csv", index=False, encoding="utf-8-sig")
    return out


def frozen_forecast(data: pd.DataFrame) -> pd.DataFrame:
    """Single paper-facing forecast using the frozen point mapping and 5,000 paths."""
    mapping = json.loads((RESULTS/"q4_compute_growth_model.json").read_text(encoding="utf-8"))
    base_shift = math.log(10.0) * float(mapping["q90_log10_compute_slope_per_year"]) * float(mapping["log10_params_per_log10_compute"])
    factors={"historical_trend":1.0,"moderate_slowdown":.5,"strong_slowdown":.25,"zero_scale_growth":0.0}
    rng=np.random.default_rng(SEED+300)
    current=float(data["score_equal"].max())
    rows=[]
    for scenario,factor in factors.items():
        for horizon,months in [(365,12),(730,24)]:
            sims=simulate(data,"scale_quadratic",horizon,5000,base_shift*factor,"legacy_record_iid",rng)
            rows.append({"scenario":scenario,"compute_growth_factor":factor,"horizon_months":months,
                         "forecast_date":(data["submission_date"].max()+pd.Timedelta(days=horizon)).date(),
                         "model":"frozen_family_balanced_record_process","paths":5000,
                         "current_frontier":current,"p05":np.quantile(sims,.05),"p50":np.quantile(sims,.5),"p95":np.quantile(sims,.95),
                         "probability_frontier_update":np.mean(sims>current+1e-10),
                         "logN_shift_per_year":base_shift*factor,
                         "evidence_level":"conditional_scenario_extrapolation"})
    out=pd.DataFrame(rows)
    out.to_csv(RESULTS/"q4_frozen_frontier_forecast.csv",index=False,encoding="utf-8-sig")
    return out


def mapping_forecast_sensitivity(data: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Propagate the C4 mapping bootstrap quantiles without claiming calibration."""
    rng = np.random.default_rng(SEED + 30)
    shifts = mapping["logN_shift_per_year"].quantile([.05, .50, .95])
    rows = []
    for qlabel, shift in zip(["p05", "p50", "p95"], shifts):
        for horizon, months in [(365, 12), (730, 24)]:
            sims = simulate(
                data, "scale_quadratic", horizon, 600, float(shift),
                "legacy_record_iid", rng,
            )
            rows.append(
                {
                    "mapping_quantile": qlabel,
                    "logN_shift_per_year": float(shift),
                    "horizon_months": months,
                    "p05": float(np.quantile(sims, .05)),
                    "p50": float(np.quantile(sims, .50)),
                    "p95": float(np.quantile(sims, .95)),
                    "evidence": "conditional propagation of C4 bootstrap mapping uncertainty",
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(
        RESULTS / "q4_final_compute_mapping_forecast_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return out


def monte_carlo_stability(data: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Repeat the retained forecast with independent seeds to expose MC noise."""
    shift = float(mapping["logN_shift_per_year"].median())
    rows = []
    for rep in range(8):
        rng = np.random.default_rng(SEED + 100 + rep)
        for horizon, months in [(365,12),(730,24)]:
            sims=simulate(data,"scale_quadratic",horizon,400,shift,"legacy_record_iid",rng)
            rows.append({"replicate":rep,"horizon_months":months,"paths":400,
                         "p05":np.quantile(sims,.05),"p50":np.quantile(sims,.5),"p95":np.quantile(sims,.95)})
    out=pd.DataFrame(rows)
    out.to_csv(RESULTS/"q4_final_monte_carlo_stability.csv",index=False,encoding="utf-8-sig")
    return out


def decomposition_sensitivity(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rolling = pd.read_csv(RESULTS / "q4_quantile_selected_models.csv")
    for stratum in CORE_STRATA:
        d = data[data["stratum"].eq(stratum)].copy()
        gate = bool(rolling[(rolling["scope"].eq("strict")) & (rolling["stratum"].eq(stratum))]["time_model_pass_5pct_gate"].iloc[0])
        for start_cut in pd.to_datetime(["2024-07-15", "2024-07-31", "2024-08-15"]):
            before = d[d["submission_date"].le(start_cut)]
            if before.empty:
                continue
            start = before.loc[before["score_equal"].idxmax()]
            end = d.loc[d["score_equal"].idxmax()]
            for spec in ["scale_time_linear", "scale_time_interaction"]:
                model, center, _ = fit_quantile(d, spec)
                def f(n: float, t: pd.Timestamp) -> float:
                    tmp = pd.DataFrame({"params_B":[n],"submission_date":[t],"stratum":[stratum]})
                    return float(predict_quantile(model,tmp,spec,center)[0])
                n0,n1=float(start["params_B"]),float(end["params_B"]); t0,t1=start["submission_date"],end["submission_date"]
                scale=.5*((f(n1,t0)-f(n0,t0))+(f(n1,t1)-f(n0,t1)))
                time=.5*((f(n0,t1)-f(n0,t0))+(f(n1,t1)-f(n1,t0)))
                y0,y1=float(start["score_logit"]),float(end["score_logit"])
                r0=y0-f(n0,t0); r1=y1-f(n1,t1); observed=y1-y0
                rows.append({"stratum":stratum,"start_cutoff":start_cut.date(),"model":spec,
                             "start_model":start["model_name"],"end_model":end["model_name"],
                             "observed_delta_logit":observed,"conditional_q90_delta_logit":scale+time,
                             "scale_component_logit":scale,"time_proxy_component_logit":time,
                             "start_endpoint_residual_logit":r0,"end_endpoint_residual_logit":r1,
                             "endpoint_residual_change_logit":r1-r0,
                             "identity_error":observed-scale-time-(r1-r0),
                             "time_model_passed_rolling_gate":gate,
                             "evidence":"descriptive endpoint identity; time proxy not causal"})
    out=pd.DataFrame(rows)
    out.to_csv(RESULTS/"q4_final_decomposition_sensitivity.csv",index=False,encoding="utf-8-sig")
    return out


def write_closure_interface() -> dict:
    """Freeze the final paper-facing claims from verified result artifacts."""
    original = pd.read_csv(RESULTS / "q4_frontier_forecast.csv")
    original = original[original["model"].eq("record_process_scale")]
    frozen = pd.read_csv(RESULTS / "q4_frozen_frontier_forecast.csv")
    metrics = pd.read_csv(RESULTS / "q4_final_record_metrics.csv")
    clusters = pd.read_csv(RESULTS / "q4_final_cutoff_cluster_bootstrap_summary.csv")
    dec = pd.read_csv(RESULTS / "q4_progress_decomposition.csv")
    interface = {
        "as_of": "2025-03-13",
        "retained_model": "family-balanced random record process with scale-only q90 model",
        "model_change": "none; corrected historical selection audit retains the original model family",
        "formal_forecast": frozen.to_dict("records"),
        "superseded_forecast_for_comparison": original[["scenario","horizon_months","p05","p50","p95","current_frontier"]].to_dict("records"),
        "corrected_backtest": metrics.to_dict("records"),
        "cutoff_cluster_bootstrap": clusters.to_dict("records"),
        "contribution_decomposition": dec[["stratum","scale_component_logit","time_proxy_component_logit","unexplained_residual_logit","time_model_passed_rolling_gate"]].to_dict("records"),
        "forecast_interval_label": "conditional scenario interval; not a calibrated 12/24-month probability guarantee",
        "contribution_label": "descriptive scale component, conditional-time proxy, and endpoint residual; not causal technology shares",
        "c8_decision": "retain; independent BH recomputation passed and no positive task survives 5% BH",
        "bridge_decision": "retain local High-comparability evidence only; no general Loss-to-Benchmark map",
    }
    (RESULTS / "q4_final_closure_interface.json").write_text(
        json.dumps(interface, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return interface


def write_final_manifest() -> pd.DataFrame:
    paths = []
    for folder in [ROOT/"code", ROOT/"results", ROOT/"figures"]:
        paths.extend(
            p for p in folder.glob("q4_final_*")
            if p.is_file() and p.name != "q4_final_artifact_manifest.csv"
        )
    report = ROOT/"reports"/"25_q4_final_optimization_report.md"
    if report.exists():
        paths.append(report)
    rows=[]
    for path in sorted(set(paths)):
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"relative_path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,"sha256":digest})
    out=pd.DataFrame(rows)
    out.to_csv(RESULTS/"q4_final_artifact_manifest.csv",index=False,encoding="utf-8-sig")
    return out


def main() -> None:
    data = load_core()
    arrival = arrival_diagnostics(data)
    bt, metrics, selection = backtest(data)
    _, cluster_summary = cutoff_cluster_bootstrap(bt)
    mapping, mapping_summary = compute_mapping_bootstrap()
    extreme = extreme_search(data, "scale_quadratic")
    fc = forecast(data, mapping)
    frozen_fc = frozen_forecast(data)
    mapping_fc = mapping_forecast_sensitivity(data, mapping)
    mc = monte_carlo_stability(data, mapping)
    dec = decomposition_sensitivity(data)

    all_metrics = metrics[metrics["evaluation_subset"].eq("formal_complete_pooled")].set_index("model")
    robust = all_metrics.loc["family_nb_tail95"]
    legacy = all_metrics.loc["legacy_record_iid"]
    persistence = all_metrics.loc["persistence"]
    robust_supported = bool(
        robust["mae"] <= legacy["mae"] * 1.05
        and robust["mae"] < persistence["mae"]
        and robust["mean_interval_score90"] <= legacy["mean_interval_score90"] * 1.10
    )
    chosen = "family_nb_tail95" if robust_supported else "legacy_record_iid"
    summary = {
        "data_cutoff": str(data["submission_date"].max().date()),
        "n_core_records": len(data), "n_families": int(data["family_id"].nunique()),
        "confirmed_defect": "original record backtest reused a model specification selected across later rolling windows",
        "overlap_warning": "18 cases are overlapping forecast windows; disjoint_60d subset is reported separately",
        "final_conditional_model": chosen,
        "selection_rule": "robust model only if MAE <= legacy*1.05, beats persistence, and interval score <= legacy*1.10",
        "robust_supported": robust_supported,
        "compute_mapping_uncertainty": mapping_summary,
        "artifact_rows": {"arrival":len(arrival),"backtest":len(bt),"selection":len(selection),"cluster_bootstrap_summary":len(cluster_summary),"extreme":len(extreme),"forecast_sensitivity":len(fc),"frozen_forecast":len(frozen_fc),"mapping_forecast":len(mapping_fc),"monte_carlo_stability":len(mc),"decomposition":len(dec)},
    }
    (RESULTS / "q4_final_closure_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    write_closure_interface()
    write_final_manifest()
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
