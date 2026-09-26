# -*- coding: utf-8 -*-
"""Question 4 time-axis and arrival-process sensitivity experiments.

AI assistance: OpenAI Codex, 2026-09-25.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

import q4_modeling as q4

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
SEED = 314159


def fit_axis(data: pd.DataFrame, date_col: str, rng: np.random.Generator) -> dict:
    date = pd.to_datetime(data[date_col], errors="coerce", utc=True).dt.tz_localize(None)
    keep = date.notna()
    d = data.loc[keep].copy()
    date = date.loc[keep]
    origin = date.min()
    ln_n = np.log(d["params_B"].to_numpy(float))
    x = pd.DataFrame(
        {
            "lnN": ln_n - ln_n.mean(),
            "lnN2": (ln_n - ln_n.mean()) ** 2,
            "time_year": (date - origin).dt.days.to_numpy(float) / 365.25,
            "is_chat": d["stratum"].eq("chat_finetuned").to_numpy(float),
        }
    )
    y = d["score_logit"].to_numpy(float)
    counts = d["family_id"].value_counts()
    w = d["family_id"].map(lambda z: 1.0 / counts[z]).to_numpy(float)
    model = QuantileRegressor(quantile=.9, alpha=0.0, solver="highs").fit(x, y, sample_weight=w)
    coef = float(model.coef_[2])
    boots = []
    families = d["family_id"].unique()
    for _ in range(100):
        sampled = rng.choice(families, len(families), replace=True)
        pieces = []
        for j, fam in enumerate(sampled):
            p = d[d["family_id"].eq(fam)].copy()
            p["family_id"] = p["family_id"].astype(str) + f"__b{j}"
            pieces.append(p)
        b = pd.concat(pieces, ignore_index=True)
        bd = pd.to_datetime(b[date_col], errors="coerce", utc=True).dt.tz_localize(None)
        bln = np.log(b["params_B"].to_numpy(float))
        bx = pd.DataFrame(
            {
                "lnN": bln - bln.mean(),
                "lnN2": (bln - bln.mean()) ** 2,
                "time_year": (bd - origin).dt.days.to_numpy(float) / 365.25,
                "is_chat": b["stratum"].eq("chat_finetuned").to_numpy(float),
            }
        )
        by = b["score_logit"].to_numpy(float)
        bc = b["family_id"].value_counts()
        bw = b["family_id"].map(lambda z: 1.0 / bc[z]).to_numpy(float)
        try:
            bm = QuantileRegressor(quantile=.9, alpha=0.0, solver="highs").fit(bx, by, sample_weight=bw)
            boots.append(float(bm.coef_[2]))
        except Exception:
            continue
    return {
        "n": len(d),
        "families": d["family_id"].nunique(),
        "date_min": origin.date(),
        "date_max": date.max().date(),
        "time_coef_logit_per_year": coef,
        "ci05": float(np.quantile(boots, .05)),
        "ci95": float(np.quantile(boots, .95)),
        "bootstrap_success": len(boots),
    }


def time_axis_sensitivity(entity: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    base = entity[
        entity["strict_open"].astype(bool)
        & entity["valid_core_fields"].astype(bool)
        & entity["stratum"].isin(["pretrained", "chat_finetuned"])
    ].copy()
    rows = []
    strata = {
        "core_combined": ["pretrained", "chat_finetuned"],
        "pretrained": ["pretrained"],
        "chat_finetuned": ["chat_finetuned"],
    }
    for stratum, allowed in strata.items():
        d = base[base["stratum"].isin(allowed)].copy()
        for label, column in [("submission_date", "submission_date"), ("hub_upload_date", "Upload To Hub Date")]:
            result = fit_axis(d, column, rng)
            result.update({"stratum": stratum, "time_axis": label})
            rows.append(result)
    out = pd.DataFrame(rows)
    out.to_csv(R / "q4_time_axis_sensitivity.csv", index=False, encoding="utf-8-sig")
    return out


def arrival_sensitivity(entity: pd.DataFrame) -> pd.DataFrame:
    data = entity[
        entity["strict_open"].astype(bool)
        & entity["valid_core_fields"].astype(bool)
        & entity["stratum"].isin(["pretrained", "chat_finetuned"])
    ].sort_values("submission_date").copy()
    selections = pd.read_csv(R / "q4_quantile_selected_models.csv")
    chosen = str(
        selections[
            selections["scope"].eq("strict")
            & selections["stratum"].eq("core_combined")
        ]["best_scale_model"].iloc[0]
    )
    compute = json.loads((R / "q4_compute_growth_model.json").read_text(encoding="utf-8"))
    slope = float(compute["q90_log10_compute_slope_per_year"])
    elasticity = float(compute["log10_params_per_log10_compute"])
    rng = np.random.default_rng(SEED)
    rows = []
    for compute_factor in [0.0, 0.25, 0.5, 1.0]:
        shift = math.log(10.0) * slope * elasticity * compute_factor
        for arrival_factor in [0.5, 1.0, 1.5]:
            for days, months in [(365, 12), (730, 24)]:
                sims = q4.simulate_record(
                    data,
                    chosen,
                    days,
                    1000,
                    shift,
                    False,
                    rng,
                    arrival_rate_multiplier=arrival_factor,
                )
                rows.append(
                    {
                        "compute_growth_factor": compute_factor,
                        "arrival_rate_factor": arrival_factor,
                        "horizon_months": months,
                        "p05": float(np.quantile(sims, .05)),
                        "p50": float(np.quantile(sims, .50)),
                        "p95": float(np.quantile(sims, .95)),
                        "probability_frontier_update": float(np.mean(sims > data["score_equal"].max() + 1e-10)),
                        "evidence_level": "conditional_scenario_extrapolation",
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(R / "q4_arrival_rate_sensitivity.csv", index=False, encoding="utf-8-sig")
    return out


def score_aggregation_check(entity: pd.DataFrame) -> pd.DataFrame:
    raw = pd.to_numeric(entity["Average \u2b06\ufe0f"], errors="coerce")
    valid = raw.notna() & entity["score_equal"].notna()
    diff = entity.loc[valid, "score_equal"] - raw.loc[valid]
    out = pd.DataFrame(
        [
            {
                "n": int(valid.sum()),
                "mean_difference_equal_minus_reported": float(diff.mean()),
                "mae": float(diff.abs().mean()),
                "max_abs_difference": float(diff.abs().max()),
                "correlation": float(entity.loc[valid, "score_equal"].corr(raw.loc[valid])),
            }
        ]
    )
    out.to_csv(R / "q4_score_aggregation_check.csv", index=False, encoding="utf-8-sig")
    return out


def main() -> None:
    entity = pd.read_csv(R / "q4_entity_cohort.csv.gz", parse_dates=["submission_date"], low_memory=False)
    time_axis = time_axis_sensitivity(entity)
    arrival = arrival_sensitivity(entity)
    score = score_aggregation_check(entity)
    print(
        json.dumps(
            {
                "time_axis_rows": len(time_axis),
                "arrival_sensitivity_rows": len(arrival),
                "score_check_rows": len(score),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
