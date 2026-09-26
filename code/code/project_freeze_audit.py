# -*- coding: utf-8 -*-
"""Freeze audit and curated paper-figure export for F problem.

AI assistance: OpenAI Codex, 2026-09-26.  This script only reads versioned
results and copies already generated figures; it does not refit any model.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, F, REP = ROOT / "results", ROOT / "figures", ROOT / "reports"
FINAL_F = F / "final"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


FIGURES = [
    ("paper_main_01_q1_quality_conflict", "q1_01_quality_conflict", "main", "code/paper_main_figures.py", "results/q1_upgrade_conflict_calibration_comparison.csv; results/q1_upgrade_conflict_rates.csv"),
    ("paper_main_02_q1_model_evidence", "q1_02_mixture_model_evidence", "main", "code/paper_main_figures.py", "results/q1_upgrade_model_benchmark.csv; results/q1_upgrade_selected_domain_metrics.csv"),
    ("q1_upgrade_domain_facets", "q1_03_domain_quality_facets", "appendix", "code/q1_upgrade_validation.py", "results/q1_upgrade_weight_domain_scores.csv"),
    ("paper_main_03_q2_scaling_evidence", "q2_01_scaling_law_evidence", "main", "code/paper_main_figures.py", "results/q2_classic_validation.csv; results/q2_quality_nested_metrics.csv"),
    ("q2_04_quality_model_comparison", "q2_02_quality_model_comparison", "main", "code/q2_modeling.py", "results/q2_quality_model_comparison.csv"),
    ("q2_08_p_bridge_scale_check", "q2_03_mixture_bridge_boundary", "appendix", "code/q2_modeling.py", "results/q2_p_bridge_summary.csv"),
    ("paper_main_04_q3_resource_paths", "q3_01_resource_paths", "main", "code/paper_main_figures.py", "results/q3_budget_paths.csv"),
    ("q3_04_support_vs_extrapolation", "q3_02_support_and_extrapolation", "main", "code/q3_modeling.py", "results/q3_solution_evidence.csv"),
    ("q3_06_robust_regret", "q3_03_robustness", "appendix", "code/q3_modeling.py", "results/q3_robust_configs.csv"),
    ("q4_fig01_historical_frontier", "q4_01_historical_frontier", "main", "code/q4_figures.py", "results/q4_frontier_history.csv"),
    ("q4_fig04_c8_task_time_effects", "q4_02_c8_task_effects", "main", "code/q4_figures.py", "results/q4_c8_task_time_effects.csv"),
    ("q4_fig07_loss_benchmark_bridge", "q4_03_loss_benchmark_bridge", "appendix", "code/q4_figures.py", "results/q4_bridge_loo_metrics.csv"),
    ("q4_final_fig01_backtest_credibility", "q4_04_backtest_credibility", "main", "code/q4_final_figures.py", "results/q4_final_record_metrics.csv"),
    ("q4_final_fig03_decomposition_identity", "q4_05_progress_decomposition", "main", "code/q4_final_figures.py", "results/q4_final_decomposition_sensitivity.csv"),
    ("q4_final_fig04_frozen_frontier_scenarios", "q4_06_frontier_scenarios", "main", "code/q4_final_figures.py", "results/q4_frozen_frontier_forecast.csv"),
    ("q4_final_fig02_tail_forecast_sensitivity", "q4_07_tail_sensitivity", "appendix", "code/q4_final_figures.py", "results/q4_final_extreme_search.csv; results/q4_final_frontier_forecast.csv"),
]

INTERFACES = [
    ("Q1_to_Q2", "results/q1_upgrade_interface_to_q2.json", "four facets; scalar Q provisional; pass source-internal mixture effects, not absolute A Loss"),
    ("Q2_to_Q3", "results/q2_interface_to_q3.json", "N-D law plus B7 semi-synthetic Q; Q1 scale unmapped; eta_p=0 in main model"),
    ("Q3_to_Q4", "results/q3_interface_to_q4.json", "resource configurations are conditional model scenarios, not interventions"),
    ("Q4_final", "results/q4_final_closure_interface.json", "frozen record process and 5000-path conditional scenario forecast"),
]


def main() -> None:
    FINAL_F.mkdir(parents=True, exist_ok=True)
    fig_rows = []
    missing = []
    for src_stem, dst_stem, role, generator, sources in FIGURES:
        for ext in (".pdf", ".png"):
            src, dst = F / f"{src_stem}{ext}", FINAL_F / f"{dst_stem}{ext}"
            if not src.exists():
                candidates = [
                    ROOT / "_tmp" / "freeze_archive" / "20260926" / "figures_regenerated_final" / src.name,
                    ROOT / "_tmp" / "freeze_archive" / "20260926" / "figures_root" / src.name,
                ]
                for archived in candidates:
                    if archived.exists():
                        src = archived
                        break
            if not src.exists():
                missing.append(str(src.relative_to(ROOT)))
                continue
            shutil.copy2(src, dst)
        result_paths = [ROOT / p.strip() for p in sources.split(";")]
        source_ok = all(p.exists() for p in result_paths)
        if not source_ok:
            missing += [str(p.relative_to(ROOT)) for p in result_paths if not p.exists()]
        pdf = FINAL_F / f"{dst_stem}.pdf"
        fig_rows.append({"figure": str(pdf.relative_to(ROOT)), "role": role,
                         "source_figure_archived": str(src.with_suffix('.pdf').relative_to(ROOT)), "generator": generator,
                         "result_sources": sources, "sources_exist": source_ok,
                         "sha256": sha256(pdf) if pdf.exists() else ""})
    pd.DataFrame(fig_rows).to_csv(R / "freeze_figure_manifest.csv", index=False, encoding="utf-8-sig")

    dep_rows = []
    for name, rel, boundary in INTERFACES:
        p = ROOT / rel
        ok = p.exists()
        if not ok:
            missing.append(rel)
        dep_rows.append({"interface": name, "file": rel, "exists": ok, "boundary": boundary,
                         "sha256": sha256(p) if ok else ""})
    pd.DataFrame(dep_rows).to_csv(R / "freeze_dependency_checks.csv", index=False, encoding="utf-8-sig")

    claims = [
        ("Q1", "same-scale mixture prediction", "q1_upgrade_interface_to_q2.json", "real holdout A6-A11; absolute cross-scale Loss transfer rejected"),
        ("Q1", "quality score", "q1_upgrade_weight_domain_scores.csv", "definition-based four facets; two-rater face-validity still pending"),
        ("Q2", "generalized scaling law", "q2_interface_to_q3.json", "N,D supported by B1; Q supported by B7 semi-synthetic; p magnitude unidentified"),
        ("Q3", "resource optimum", "q3_interface_to_q4.json", "numerical optimum inside stated support/scenario; not causal intervention"),
        ("Q4", "30-day record forecast MAE", "q4_final_record_metrics.csv", "9 complete windows"),
        ("Q4", "60-day record forecast MAE", "q4_final_record_metrics.csv", "8 complete windows; last nominal 60-day window excluded because only 40 days observed"),
        ("Q4", "12/24-month frontier scenarios", "q4_frozen_frontier_forecast.csv", "5000-path conditional extrapolation; interval not empirically 90%-calibrated"),
        ("Q4", "non-scale progress proxy", "q4_final_decomposition_sensitivity.csv", "descriptive Shapley identity; not causal technology contribution"),
    ]
    cdf = pd.DataFrame(claims, columns=["question", "claim", "result_file", "evidence_boundary"])
    cdf["exists"] = cdf.result_file.map(lambda x: (R / x).exists())
    cdf.to_csv(R / "freeze_claim_evidence_matrix.csv", index=False, encoding="utf-8-sig")
    missing += [f"results/{x}" for x in cdf.loc[~cdf.exists, "result_file"]]

    formulas = [
        ("Q1", "quality facets", "Q_i=(sum_d W_d S_id^p)^(1/p), with robustness choices recorded", "results/q1_upgrade_weight_table.csv", "code/q1_upgrade_validation.py"),
        ("Q1", "mixture prediction", "L_h=b_h+sum_j beta_hj log(p_j+epsilon)", "results/q1_upgrade_selected_parameters.csv", "code/q1_upgrade_validation.py"),
        ("Q1", "structural interaction", "L_h=b_h+gamma_h^T z(p)+z(p)^T B_h z(p)", "results/q1_upgrade_quadratic_parameters.csv", "code/q1_upgrade_validation.py"),
        ("Q2", "generalized scaling law", "L=E+(A N^-alpha+B D^-beta)[1+kappa(1-Q)]", "results/q2_interface_to_q3.json", "code/q2_revision_validation.py"),
        ("Q3", "compute constraint", "C=6ND+D[g(Q)-g(Q0)]_+ + eta N D L_ctx", "results/q3_cost_unit_audit.csv", "code/q3_modeling.py"),
        ("Q3", "dimension reduction", "D*(N,Q)=min(Dmax,C/[N(6+eta L_ctx)+quality_cost_per_token])", "results/q3_official_solutions.csv", "code/q3_modeling.py"),
        ("Q4", "capability score", "S_i=mean_k B_ik; Z_i=logit(S_i/100)", "results/q4_entity_cohort.csv.gz", "code/q4_prepare.py"),
        ("Q4", "frontier", "F(t)=max_{i:T_i<=t} S_i", "results/q4_frontier_history.csv", "code/q4_modeling.py"),
        ("Q4", "Shapley identity", "Delta z=phi_N+phi_t+Delta endpoint residual", "results/q4_final_decomposition_sensitivity.csv", "code/q4_final_validation.py"),
    ]
    fdf = pd.DataFrame(formulas, columns=["question", "formula_name", "formula", "parameter_or_result_file", "implementation"])
    fdf["result_exists"] = fdf.parameter_or_result_file.map(lambda x: (ROOT / x).exists())
    fdf["implementation_exists"] = fdf.implementation.map(lambda x: (ROOT / x).exists())
    fdf.to_csv(R / "freeze_formula_manifest.csv", index=False, encoding="utf-8-sig")
    missing += fdf.loc[~fdf.result_exists, "parameter_or_result_file"].tolist()
    missing += fdf.loc[~fdf.implementation_exists, "implementation"].tolist()

    metrics = pd.read_csv(R / "q4_final_record_metrics.csv")
    q4 = metrics[(metrics.model == "legacy_record_iid") & metrics.evaluation_subset.isin(["formal_complete_30d", "formal_complete_60d"])]
    forecast = pd.read_csv(R / "q4_frozen_frontier_forecast.csv")
    hist = forecast[forecast.scenario.eq("historical_trend")].sort_values("horizon_months")
    checks = {
        "status": "PASS" if not missing else "FAIL",
        "missing": sorted(set(missing)),
        "interfaces": len(dep_rows), "figures": len(fig_rows), "claims": len(cdf), "formulas": len(fdf),
        "q4_complete_windows": {str(int(r.horizon_months)): int(r.complete_window)
                                for _, r in pd.DataFrame({"horizon_months":[30,60], "complete_window":[9,8]}).iterrows()},
        "q4_formal_metrics": q4.to_dict("records"),
        "q4_historical_forecast": hist.to_dict("records"),
        "unresolved": [
            "Q1 two-rater blinded descriptive face-validity review is not completed",
            "Q2 absolute A/B Loss mapping and mixture-response magnitude remain unidentified",
            "Q4 5%-95% interval coverage is below nominal 90%; use conditional-scenario wording",
            "paper page/reference/AI declaration/template/render gates remain for the writing stage",
        ],
    }
    (R / "freeze_verification.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
