# -*- coding: utf-8 -*-
"""Independent numerical verification for the Q4 final closure artifacts.

AI assistance: OpenAI Codex, 2026-09-26.
This file deliberately does not import q4_final_validation.py or q4_modeling.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = R / "q4_final_independent_verification.csv"
TOL = 1e-9


def add(rows: list[dict], name: str, passed: bool, evidence: str) -> None:
    rows.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})


def bh_adjust(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def main() -> None:
    rows: list[dict] = []

    bt = pd.read_csv(R / "q4_final_record_backtest.csv")
    met = pd.read_csv(R / "q4_final_record_metrics.csv")
    rebuilt = []
    for model, part in bt.groupby("model"):
        complete=part[part["complete_window"].astype(bool)]
        for subset, use in [
            ("formal_complete_pooled",complete),
            ("formal_complete_30d",complete[complete["horizon_days"].eq(30)]),
            ("formal_complete_60d",complete[complete["horizon_days"].eq(60)]),
            ("disjoint_60d",part[part["disjoint_60d_case"].astype(bool)&part["complete_window"].astype(bool)]),
        ]:
            rebuilt.append({"model":model,"evaluation_subset":subset,"mae2":use["abs_error"].mean(),
                            "coverage2":use["covered90"].astype(bool).mean(),"width2":use["interval_width90"].mean(),
                            "score2":use["interval_score90"].mean()})
    j = met.merge(pd.DataFrame(rebuilt), on=["model","evaluation_subset"], validate="one_to_one")
    delta = max((j["mae"]-j["mae2"]).abs().max(), (j["coverage90"]-j["coverage2"]).abs().max(),
                (j["mean_interval_width90"]-j["width2"]).abs().max(),
                (j["mean_interval_score90"]-j["score2"]).abs().max())
    add(rows,"final_backtest_metrics_recomputed",delta<TOL,f"max_abs_diff={delta:.3e}")
    cases=bt[["cutoff","horizon_days","requested_end","observed_end","actual_horizon_days","complete_window"]].drop_duplicates()
    incomplete=cases[~cases["complete_window"].astype(bool)]
    window_ok=(len(cases)==18 and len(incomplete)==1 and int(incomplete.iloc[0]["horizon_days"])==60
               and int(incomplete.iloc[0]["actual_horizon_days"])==40)
    add(rows,"backtest_window_completeness_audited",bool(window_ok),
        f"cases={len(cases)}; incomplete={len(incomplete)}; complete30={int((cases.complete_window.astype(bool)&cases.horizon_days.eq(30)).sum())}; complete60={int((cases.complete_window.astype(bool)&cases.horizon_days.eq(60)).sum())}")

    sel = pd.read_csv(R / "q4_final_nested_selection.csv", parse_dates=["outer_cutoff"])
    inner = pd.read_csv(R / "q4_final_nested_inner_scores.csv", parse_dates=["outer_cutoff","inner_cutoff"])
    leak = int((inner["inner_cutoff"] >= inner["outer_cutoff"]).sum())
    add(rows,"nested_selection_no_future_cutoff",leak==0,f"violations={leak}; outer_cutoffs={len(sel)}")
    fallback_ok = bool(sel.loc[sel["inner_folds"].eq(0),"selected_model"].eq("scale_linear").all())
    add(rows,"nested_selection_fallback_predeclared",fallback_ok,f"fallback_cases={int(sel['inner_folds'].eq(0).sum())}")

    allm=met[met["evaluation_subset"].eq("formal_complete_pooled")].set_index("model")
    benefit=allm.loc["persistence","mae"]-allm.loc["legacy_record_iid","mae"]
    add(rows,"record_process_beats_persistence_after_nested_selection",benefit>0,f"mae_gain={benefit:.6f}")
    robust_ok=(allm.loc["family_nb_tail95","mae"]<=1.05*allm.loc["legacy_record_iid","mae"] and
               allm.loc["family_nb_tail95","mae"]<allm.loc["persistence","mae"] and
               allm.loc["family_nb_tail95","mean_interval_score90"]<=1.10*allm.loc["legacy_record_iid","mean_interval_score90"])
    summary=json.loads((R/"q4_final_closure_summary.json").read_text(encoding="utf-8"))
    add(rows,"formal_model_selection_rule_recomputed",summary["robust_supported"]==robust_ok and
        summary["final_conditional_model"]==("family_nb_tail95" if robust_ok else "legacy_record_iid"),
        f"robust_supported={robust_ok}; selected={summary['final_conditional_model']}")
    cb=pd.read_csv(R/"q4_final_cutoff_cluster_bootstrap_summary.csv").set_index("model")
    add(rows,"cluster_bootstrap_gain_lower_bound_positive",cb.loc["legacy_record_iid","gain_p05"]>0,
        f"gain_p05={cb.loc['legacy_record_iid','gain_p05']:.6f}; probability_positive={cb.loc['legacy_record_iid','probability_gain_positive']:.4f}")

    arr=pd.read_csv(R/"q4_final_arrival_diagnostics.csv")
    full=arr[arr["recent_days"].astype(str).eq("full")].set_index("unit")
    add(rows,"family_arrival_rate_below_record_rate",full.loc["family","rate_day"]<full.loc["record","rate_day"],
        f"family={full.loc['family','rate_day']:.4f}; record={full.loc['record','rate_day']:.4f}")
    add(rows,"arrival_overdispersion_detected",bool((arr["fano"]>1).all()),f"min_fano={arr['fano'].min():.3f}")

    extreme=pd.read_csv(R/"q4_final_extreme_search.csv")
    piv=extreme.pivot(index="n_candidates",columns="residual_mechanism",values="max_residual_p95")
    add(rows,"tail_cap_limits_repeated_extreme_search",piv.loc[500,"family_block_tail95"]<piv.loc[500,"record_iid"],
        f"cap95={piv.loc[500,'family_block_tail95']:.6f}; iid={piv.loc[500,'record_iid']:.6f}")

    fc=pd.read_csv(R/"q4_final_frontier_forecast.csv")
    bounds=bool(((fc.p05<=fc.p50)&(fc.p50<=fc.p95)&(fc.p05>=fc.current_frontier-TOL)).all())
    add(rows,"forecast_bounds_and_cumulative_floor",bounds,f"rows={len(fc)}")
    for model, part in fc.groupby("model"):
        hist=part[part["scenario"].eq("historical_trend")].set_index("horizon_months")
        zero=part[part["scenario"].eq("zero_scale_growth")].set_index("horizon_months")
        ok=bool((hist["p50"]>=zero["p50"]-0.25).all())
        add(rows,f"compute_slowdown_direction_{model}",ok,
            f"min_historical_minus_zero={(hist['p50']-zero['p50']).min():.4f}")
    frozen=pd.read_csv(R/"q4_frozen_frontier_forecast.csv")
    frozen_bounds=bool(((frozen.p05<=frozen.p50)&(frozen.p50<=frozen.p95)&(frozen.p05>=frozen.current_frontier-TOL)).all())
    add(rows,"frozen_forecast_unique_and_ordered",frozen_bounds and len(frozen)==8 and frozen.paths.eq(5000).all(),
        f"rows={len(frozen)}; paths={sorted(frozen.paths.unique().tolist())}")
    mfs=pd.read_csv(R/"q4_final_compute_mapping_forecast_sensitivity.csv")
    map_ok=True
    for _,part in mfs.groupby("horizon_months"):
        map_ok &= bool(np.all(np.diff(part.sort_values("logN_shift_per_year")["p50"])>=-1e-12))
    add(rows,"compute_mapping_forecast_monotonicity",map_ok,f"rows={len(mfs)}")
    mc=pd.read_csv(R/"q4_final_monte_carlo_stability.csv")
    mc_range=mc.groupby("horizon_months")["p50"].agg(lambda s:float(s.max()-s.min()))
    add(rows,"monte_carlo_median_stability",bool((mc_range<1.0).all()),
        "; ".join(f"{int(h)}m_range={v:.4f}" for h,v in mc_range.items()))

    dec=pd.read_csv(R/"q4_final_decomposition_sensitivity.csv")
    ident=float(dec["identity_error"].abs().max())
    add(rows,"endpoint_decomposition_identity",ident<TOL,f"max_abs_error={ident:.3e}")
    add(rows,"decomposition_time_gate_flag_preserved",bool((~dec["time_model_passed_rolling_gate"].astype(bool)).all()),
        f"rows={len(dec)}")

    c8=pd.read_csv(R/"q4_c8_task_time_effects.csv")
    max_q_delta=0.0
    for _, part in c8.groupby("analysis_scope"):
        rebuilt_q=bh_adjust(part["time_pvalue"].to_numpy(float))
        max_q_delta=max(max_q_delta,float(np.max(np.abs(rebuilt_q-part["time_qvalue_bh"].to_numpy(float)))))
    add(rows,"c8_bh_recomputed_independently",max_q_delta<1e-10,f"max_abs_diff={max_q_delta:.3e}")

    bp=pd.read_csv(R/"q4_bridge_loo_predictions.csv")
    bm=pd.read_csv(R/"q4_bridge_loo_metrics.csv")
    rb=(bp.assign(sq=(bp.truth-bp.prediction)**2).groupby(["target","model"],as_index=False).sq.mean())
    rb["rmse2"]=np.sqrt(rb["sq"])
    bj=bm.merge(rb,on=["target","model"],validate="one_to_one")
    bd=float((bj["loo_rmse"]-bj["rmse2"]).abs().max())
    add(rows,"bridge_loo_recomputed",bd<TOL,f"max_abs_diff={bd:.3e}")

    out=pd.DataFrame(rows)
    out.to_csv(OUT,index=False,encoding="utf-8-sig")
    result={"checks":len(out),"passed":int(out.status.eq("PASS").sum()),"failed":int(out.status.eq("FAIL").sum())}
    result["status"]="PASS" if result["failed"]==0 else "FAIL"
    (R/"q4_final_independent_verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
