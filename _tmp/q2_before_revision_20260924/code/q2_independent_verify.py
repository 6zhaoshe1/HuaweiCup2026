#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题二独立复核：不导入 q2_modeling.py，不复用其评价函数。

AI 辅助信息：OpenAI Codex（GPT-5 系列），OpenAI，2026-09-24。
"""
from __future__ import annotations

import json
import hashlib
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "第二十三届中国研究生数学建模竞赛 - 中文题目" / "中文题目" / "F题.zip"
RES = ROOT / "results"
PREF = "real_attachments/B_scaling_laws/"


def main() -> None:
    checks = []
    with zipfile.ZipFile(ZIP_PATH) as zf:
        b1 = pd.read_csv(zf.open(PREF + "pythia_training_log_existing.csv"))
        b6 = pd.read_csv(zf.open(PREF + "supplementary_NQ_experiment.csv"))
        b7 = pd.read_csv(zf.open(PREF + "supplementary_NQ_experiment_expanded.csv"))

    keys = ["N_params_B", "D_tokens_B", "Q_score"]
    ov = b6.merge(b7, on=keys, suffixes=("_6", "_7"))
    ok_overlap = len(ov) == len(b6) and np.allclose(ov["val_loss_6"], ov["val_loss_7"], atol=1e-12)
    checks.append({"check": "B6_exact_subset_B7", "pass": bool(ok_overlap), "value": len(ov)})

    saved = pd.read_csv(RES / "q2_classic_parameters.csv").iloc[0]
    x = np.vstack([b1["N_params_B"].to_numpy(float), b1["D_tokens_B"].to_numpy(float)])
    y = b1["val_loss"].to_numpy(float)

    def law(xv, e, aa, alpha, bb, beta):
        return e + aa * xv[0] ** (-alpha) + bb * xv[1] ** (-beta)

    popt, _ = curve_fit(law, x, y, p0=[saved.E, saved.A, saved.alpha, saved.B, saved.beta],
                        bounds=([.05, 1e-6, .02, 1e-6, .02], [y.min()*.999, 5000, 1.5, 5000, 1.5]),
                        maxfev=100000)
    rel = np.abs((popt - saved[["E","A","alpha","B","beta"]].to_numpy(float)) /
                 saved[["E","A","alpha","B","beta"]].to_numpy(float))
    checks.append({"check": "classic_parameters_independent_curve_fit", "pass": bool(rel.max() < .01),
                   "max_relative_difference": float(rel.max()), "independent_parameters": popt.tolist()})

    # Closed-form OLS cross-check of the selected reducible-linear quality coefficient.
    e, aa, alpha, bb, beta = saved[["E","A","alpha","B","beta"]].to_numpy(float)
    r = aa*b7["N_params_B"].to_numpy(float)**(-alpha) + bb*b7["D_tokens_B"].to_numpy(float)**(-beta)
    xq = r * (1-b7["Q_score"].to_numpy(float))
    residual = b7["val_loss"].to_numpy(float) - (e+r)
    k_ols = float(np.dot(xq, residual) / np.dot(xq, xq))
    with (RES / "q2_interface_to_q3.json").open(encoding="utf-8") as f:
        interface = json.load(f)
    k_saved = float(interface["quality_model"]["parameters"][0])
    checks.append({"check": "quality_coefficient_closed_form_ols", "pass": bool(abs(k_ols-k_saved) < .03),
                   "ols": k_ols, "robust_log_fit": k_saved, "absolute_difference": abs(k_ols-k_saved)})

    pred = pd.read_csv(RES / "q2_quality_oof_predictions.csv.gz")
    sub = pred[pred["model"] == interface["quality_model"]["kind"]]
    rmse = float(np.sqrt(np.mean((sub["predicted"]-sub["observed"])**2)))
    table = pd.read_csv(RES / "q2_quality_model_comparison.csv")
    recorded = float(table[(table.model == interface["quality_model"]["kind"]) &
                           (table.split == "grouped_ND_5fold")].rmse.iloc[0])
    checks.append({"check": "quality_oof_metric_recomputed", "pass": bool(abs(rmse-recorded) < 1e-12),
                   "recomputed_rmse": rmse, "recorded_rmse": recorded})

    p = pd.read_csv(RES / "q2_p_bridge_summary.csv")
    checks.append({"check": "p_bridge_finite_and_directional", "pass": bool(np.isfinite(p.select_dtypes('number')).all().all() and
                                                                             (p.spearman_p_index_vs_better_loss > 0).all()),
                   "min_spearman": float(p.spearman_p_index_vs_better_loss.min())})
    reason = interface["mixture_bridge"]["reason"].lower()
    checks.append({"check": "eta_p_unidentified_default_zero", "pass": bool(interface["mixture_bridge"]["eta_p_default"] == 0 and
                                                                               ("not identified" in reason or "not identifiable" in reason)),
                   "eta_p_default": interface["mixture_bridge"]["eta_p_default"]})

    m = pd.read_csv(RES / "q2_marginal_and_substitution.csv")
    checks.append({"check": "finite_marginal_scenarios", "pass": bool(np.isfinite(m[["dL_dN","dL_dD","dL_dQ"]]).all().all() and
                                                                        (m["equivalent_N_multiplier"] >= 1).all()),
                   "rows": len(m)})

    result = {"all_pass": all(x["pass"] for x in checks), "checks": checks,
              "note": "Independent implementation; no imports from q2_modeling.py."}
    with (RES / "q2_independent_verification.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    artifacts = []
    for base in [RES, ROOT / "figures"]:
        for path in sorted(base.glob("q2_*")):
            if path.name == "q2_artifact_manifest.csv" or not path.is_file():
                continue
            data = path.read_bytes()
            artifacts.append({"path":str(path.relative_to(ROOT)).replace("\\","/"),
                              "bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    pd.DataFrame(artifacts).to_csv(RES / "q2_artifact_manifest.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
