# -*- coding: utf-8 -*-
"""Independent numerical checks for Question 4 formal artifacts.

AI assistance: OpenAI Codex, 2026-09-25.
This verifier recomputes claims from saved row-level outputs and does not
import any function from q4_modeling.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "q4_independent_verification.csv"
TOL = 1e-9


def add(rows: list[dict], name: str, passed: bool, evidence: str) -> None:
    rows.append(
        {
            "check": name,
            "status": "PASS" if bool(passed) else "FAIL",
            "evidence": evidence,
        }
    )


def main() -> None:
    rows: list[dict] = []

    pred = pd.read_csv(
        R / "q4_quantile_backtest_predictions.csv.gz",
        parse_dates=["cutoff", "test_end", "submission_date"],
    )
    bt = pd.read_csv(R / "q4_quantile_backtest.csv", parse_dates=["cutoff", "test_end"])
    err = pred["truth_logit"] - pred["prediction_logit"]
    pred["pinball"] = np.maximum(0.9 * err, -0.1 * err)
    rebuilt = (
        pred.groupby(["scope", "stratum", "cutoff", "model"], as_index=False)
        .apply(
            lambda d: pd.Series(
                {
                    "pinball": float(d["pinball"].mean()),
                    "pinball_fw": float(
                        np.average(d["pinball"], weights=d["family_weight"])
                    ),
                    "coverage": float(
                        np.mean(d["truth_logit"] <= d["prediction_logit"])
                    ),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    joined = bt[bt["status"].eq("ok")].merge(
        rebuilt,
        on=["scope", "stratum", "cutoff", "model"],
        how="left",
        validate="one_to_one",
    )
    delta = max(
        (joined["pinball_logit"] - joined["pinball"]).abs().max(),
        (
            joined["pinball_logit_familyweighted"] - joined["pinball_fw"]
        ).abs().max(),
        (joined["coverage_q90"] - joined["coverage"]).abs().max(),
    )
    add(rows, "q90_metrics_recomputed", delta < TOL, f"max_abs_diff={delta:.3e}")
    leakage = (pred["submission_date"] < pred["cutoff"]).sum() + (
        pred["submission_date"] > pred["test_end"]
    ).sum()
    add(rows, "rolling_windows_no_date_leakage", leakage == 0, f"violations={leakage}")
    family_weight_sum = pred.groupby(
        ["scope", "stratum", "cutoff", "model", "family_id"]
    )["family_weight"].sum()
    fw_delta = float((family_weight_sum - 1.0).abs().max())
    add(rows, "test_family_equal_weighting", fw_delta < TOL, f"max_family_sum_diff={fw_delta:.3e}")

    dec = pd.read_csv(R / "q4_progress_decomposition.csv")
    identity = (
        dec["scale_component_logit"]
        + dec["time_proxy_component_logit"]
        + dec["unexplained_residual_logit"]
        - dec["observed_delta_logit"]
    ).abs().max()
    add(rows, "shapley_residual_identity", identity < TOL, f"max_abs_diff={identity:.3e}")

    bridge_pred = pd.read_csv(R / "q4_bridge_loo_predictions.csv")
    bridge_metrics = pd.read_csv(R / "q4_bridge_loo_metrics.csv")
    rebuilt_bridge = (
        bridge_pred.assign(sq=lambda d: (d["truth"] - d["prediction"]) ** 2)
        .groupby(["target", "model"], as_index=False)["sq"]
        .mean()
    )
    rebuilt_bridge["rmse_rebuilt"] = np.sqrt(rebuilt_bridge["sq"])
    bj = bridge_metrics.merge(rebuilt_bridge, on=["target", "model"], validate="one_to_one")
    bridge_delta = float((bj["loo_rmse"] - bj["rmse_rebuilt"]).abs().max())
    add(rows, "bridge_loo_rmse_recomputed", bridge_delta < TOL, f"max_abs_diff={bridge_delta:.3e}")

    record = pd.read_csv(R / "q4_record_backtest.csv")
    record_metrics = pd.read_csv(R / "q4_record_model_comparison.csv")
    rebuilt_record = record.groupby("model", as_index=False).agg(
        mae_rebuilt=("abs_error", "mean"),
        updates_rebuilt=("record_updated", "sum"),
    )
    rj = record_metrics.merge(rebuilt_record, on="model", validate="one_to_one")
    record_delta = float((rj["mae"] - rj["mae_rebuilt"]).abs().max())
    update_delta = int((rj["update_cases"] - rj["updates_rebuilt"]).abs().max())
    add(
        rows,
        "record_backtest_recomputed",
        record_delta < TOL and update_delta == 0,
        f"mae_diff={record_delta:.3e}; update_count_diff={update_delta}",
    )
    add(
        rows,
        "record_backtest_contains_updates",
        int(record["record_updated"].sum()) > 0,
        f"row_model_update_flags={int(record['record_updated'].sum())}; unique_cases={record.loc[record['record_updated'], ['cutoff','horizon_days']].drop_duplicates().shape[0]}",
    )

    forecast = pd.read_csv(R / "q4_frontier_forecast.csv")
    bounds_ok = bool(
        (
            (forecast["p05"] <= forecast["p50"])
            & (forecast["p50"] <= forecast["p95"])
            & (forecast["p05"] >= forecast["current_frontier"] - TOL)
        ).all()
    )
    add(rows, "forecast_bounds_and_cumulative_monotonicity", bounds_ok, f"rows={len(forecast)}")
    rejected = forecast[forecast["model"].eq("record_process_time_candidate")]
    rejected_ok = bool(
        (~rejected["passed_rolling_selection_gate"].astype(bool)).all()
        and rejected["evidence_level"].eq("rejected_time_candidate_sensitivity").all()
    )
    add(rows, "rejected_time_candidate_is_flagged", rejected_ok, f"rows={len(rejected)}")

    reconstruction = pd.read_csv(R / "q4_c8_reconstruction_metrics.csv")
    internal_reconstruction = reconstruction[
        reconstruction["comparison"].eq("leaf_vs_json_group")
    ]
    max_reconstruction = float(internal_reconstruction["max_abs_error"].max())
    add(
        rows,
        "c8_weighted_aggregation_reconstruction",
        max_reconstruction < 1e-12,
        f"max_abs_error={max_reconstruction:.3e}",
    )
    task_effects = pd.read_csv(R / "q4_c8_task_time_effects.csv")
    q_ok = bool(task_effects["time_qvalue_bh"].between(0, 1).all())
    sign_ok = bool(
        (~task_effects["significant_positive_bh"].astype(bool)
         | task_effects["time_effect_logit_per_year"].gt(0)).all()
    )
    add(rows, "c8_bh_qvalues_and_signs", q_ok and sign_ok, f"rows={len(task_effects)}")

    score_check = pd.read_csv(R / "q4_score_aggregation_check.csv").iloc[0]
    add(
        rows,
        "six_score_average_matches_reported_average",
        float(score_check["max_abs_difference"]) < TOL,
        f"n={int(score_check['n'])}; max_abs_diff={float(score_check['max_abs_difference']):.3e}",
    )

    arrival = pd.read_csv(R / "q4_arrival_rate_sensitivity.csv")
    monotone_compute = True
    for _, part in arrival.groupby(["arrival_rate_factor", "horizon_months"]):
        monotone_compute &= bool(
            np.all(np.diff(part.sort_values("compute_growth_factor")["p50"]) >= -0.10)
        )
    monotone_arrival = True
    for _, part in arrival.groupby(["compute_growth_factor", "horizon_months"]):
        monotone_arrival &= bool(
            np.all(np.diff(part.sort_values("arrival_rate_factor")["p50"]) >= -0.10)
        )
    add(
        rows,
        "arrival_compute_scenario_monotonicity",
        monotone_compute and monotone_arrival,
        f"rows={len(arrival)}; tolerance=0.10_score_points",
    )

    time_axis = pd.read_csv(R / "q4_time_axis_sensitivity.csv")
    axis_ok = bool(
        len(time_axis) == 6
        and time_axis["bootstrap_success"].ge(95).all()
        and time_axis["time_coef_logit_per_year"].notna().all()
    )
    add(rows, "time_axis_sensitivity_complete", axis_ok, f"rows={len(time_axis)}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")
    summary = {
        "checks": len(out),
        "passed": int(out["status"].eq("PASS").sum()),
        "failed": int(out["status"].eq("FAIL").sum()),
        "status": "PASS" if out["status"].eq("PASS").all() else "FAIL",
    }
    (R / "q4_independent_verification.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
