# -*- coding: utf-8 -*-
"""Independent integrity checks for the regenerated paper main figures.

This script does not import the plotting implementation.  It recomputes key
identities directly from frozen result tables and checks the delivered files.
AI assistance: OpenAI Codex, 2026-09-26.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

checks: list[dict[str, object]] = []


def record(name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


contract = pd.read_csv(R / "paper_main_figure_contract.csv")
record("contract_has_six_rows", len(contract) == 6, f"rows={len(contract)}")
record(
    "all_pdf_png_exist",
    all((ROOT / p).is_file() and (ROOT / p).stat().st_size > 1000 for p in contract["figure_pdf"])
    and all((ROOT / p).is_file() and (ROOT / p).stat().st_size > 1000 for p in contract["figure_png"]),
    "all declared PDF/PNG files must exceed 1 kB",
)
record(
    "pdf_png_signatures",
    all((ROOT / p).read_bytes()[:4] == b"%PDF" for p in contract["figure_pdf"])
    and all((ROOT / p).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in contract["figure_png"]),
    "magic bytes checked independently",
)

q1_quality = pd.read_csv(R / "q1_domain_quality_summary.csv")
q1_quality = q1_quality[q1_quality["dataset"].eq("pooled_unique")]
facet_cols = [
    "facet_content_value_robust",
    "facet_language_quality_robust",
    "facet_cleanliness_robust",
    "facet_reasoning_professional_robust",
]
record(
    "q1_facets_valid",
    len(q1_quality) == 7 and q1_quality[facet_cols].notna().all().all()
    and q1_quality[facet_cols].stack().between(0, 1).all(),
    f"domains={len(q1_quality)}, range=[{q1_quality[facet_cols].min().min():.4f}, {q1_quality[facet_cols].max().max():.4f}]",
)
q1_bench = pd.read_csv(R / "q1_upgrade_model_benchmark.csv")
rmse = q1_bench.pivot(index="model", columns="dataset", values="rmse_macro")
relative = rmse / rmse.min(axis=0)
record(
    "q1_relative_rmse_identity",
    np.allclose(relative.min(axis=0).to_numpy(float), 1.0),
    "each data column has relative best RMSE exactly 1",
)

q2_ci = pd.read_csv(R / "q2_classic_parameter_intervals.csv").set_index("parameter")
sel = q2_ci.loc[["alpha", "beta", "kappa"]]
record(
    "q2_intervals_contain_point",
    ((sel["ci2p5"] <= sel["point_estimate"]) & (sel["point_estimate"] <= sel["ci97p5"])).all(),
    "alpha, beta and kappa point estimates lie inside stored intervals",
)
q2_opt = pd.read_csv(R / "q2_compute_optimal.csv")
compute_rebuilt = 6.0 * q2_opt["N_opt_B"] * q2_opt["D_opt_B"] * 1e18
record(
    "q2_compute_identity",
    np.allclose(compute_rebuilt, q2_opt["C_FLOPs"], rtol=2e-10, atol=1e-6),
    f"max relative residual={np.max(np.abs(compute_rebuilt/q2_opt['C_FLOPs']-1)):.3e}",
)

q3 = pd.read_csv(R / "q3_budget_paths.csv")
record(
    "q3_paths_feasible",
    (q3["C_total"] <= q3["budget_FLOPs"] * (1 + 2e-8)).all()
    and q3["Q"].between(q3["Q0"], 1.0 + 1e-10).all()
    and q3["budget_utilization"].between(0, 1.0 + 2e-8).all(),
    f"rows={len(q3)}, max utilization={q3['budget_utilization'].max():.9f}",
)

hist = pd.read_csv(R / "q4_frontier_history.csv")
record(
    "q4_frontier_monotone",
    hist["cumulative_frontier"].diff().fillna(0).ge(-1e-12).all(),
    f"rows={len(hist)}, final={hist['cumulative_frontier'].iloc[-1]:.6f}",
)
fc = pd.read_csv(R / "q4_frontier_forecast.csv")
record(
    "q4_forecast_quantiles_ordered",
    ((fc["p05"] <= fc["p50"]) & (fc["p50"] <= fc["p95"])).all(),
    f"rows={len(fc)}",
)
dec = pd.read_csv(R / "q4_progress_decomposition.csv")
rebuilt = dec["scale_component_logit"] + dec["time_proxy_component_logit"] + dec["unexplained_residual_logit"]
record(
    "q4_decomposition_closes",
    np.allclose(rebuilt, dec["observed_delta_logit"], atol=1e-12),
    f"max absolute closure error={np.max(np.abs(rebuilt-dec['observed_delta_logit'])):.3e}",
)
tasks = pd.read_csv(R / "q4_c8_task_time_effects.csv")
tasks = tasks[tasks["analysis_scope"].eq("all_complete_signatures")]
record(
    "q4_bh_statement_matches_table",
    len(tasks) == 39 and int(tasks["significant_positive_bh"].astype(bool).sum()) == 0,
    f"tasks={len(tasks)}, significant_positive_bh={int(tasks['significant_positive_bh'].astype(bool).sum())}",
)

passed = sum(int(x["passed"]) for x in checks)
out = {"passed": passed, "total": len(checks), "all_passed": passed == len(checks), "checks": checks}
(R / "paper_main_figure_verification.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(out, ensure_ascii=False, indent=2))
if not out["all_passed"]:
    raise SystemExit(1)
