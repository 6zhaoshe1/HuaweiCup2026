#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题二：按模型轨迹分组 Bootstrap 与算力最优配置。

AI 辅助信息：OpenAI Codex（GPT-5 系列），OpenAI，2026-09-24。
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import sys

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "第二十三届中国研究生数学建模竞赛 - 中文题目" / "中文题目" / "F题.zip"
RES = ROOT / "results"
FIG = ROOT / "figures"
PREF = "real_attachments/B_scaling_laws/"
RNG = np.random.default_rng(42)
sys.path.insert(0, str(ROOT / "_模板" / "scripts"))
from mpl_cn import plt, save_fig  # noqa: E402


def law(x, e, a0, alpha, b0, beta):
    n, d = x
    return e + a0*n**(-alpha) + b0*d**(-beta)


def main() -> None:
    with zipfile.ZipFile(ZIP_PATH) as zf:
        b1 = pd.read_csv(zf.open(PREF + "pythia_training_log_existing.csv"))
    saved = pd.read_csv(RES / "q2_classic_parameters.csv").iloc[0]
    p0 = saved[["E","A","alpha","B","beta"]].to_numpy(float)
    groups = sorted(b1["N_params_B"].unique())
    rows = []
    for rep in range(300):
        take = RNG.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([b1[b1["N_params_B"] == g] for g in take], ignore_index=True)
        x = np.vstack([sample["N_params_B"].to_numpy(float), sample["D_tokens_B"].to_numpy(float)])
        y = sample["val_loss"].to_numpy(float)
        try:
            p, _ = curve_fit(law, x, y, p0=p0,
                             bounds=([.05,1e-6,.02,1e-6,.02], [y.min()*.999,5000,1.5,5000,1.5]),
                             maxfev=40000)
            rows.append({"bootstrap_id":rep,"success":True,**dict(zip(["E","A","alpha","B","beta"],p))})
        except Exception:
            rows.append({"bootstrap_id":rep,"success":False,"E":np.nan,"A":np.nan,"alpha":np.nan,"B":np.nan,"beta":np.nan})
    boot = pd.DataFrame(rows)
    boot.to_csv(RES / "q2_classic_trajectory_bootstrap.csv", index=False, encoding="utf-8-sig")

    good = boot[boot.success]
    intervals = []
    for col in ["E","A","alpha","B","beta"]:
        intervals.append({"parameter":col,"point_estimate":float(saved[col]),
                          "bootstrap_median":float(good[col].median()),
                          "ci2p5":float(good[col].quantile(.025)),"ci97p5":float(good[col].quantile(.975)),
                          "successful_replicates":len(good),"resampling_unit":"complete N trajectory",
                          "evidence_level":"real_observational_group_bootstrap"})
    interval_df = pd.DataFrame(intervals)
    interval_df.to_csv(RES / "q2_classic_parameter_intervals.csv", index=False, encoding="utf-8-sig")

    e, a0, alpha, b0, beta = p0
    cmin = float(b1["C_FLOPs_1e21"].min())
    cmax = float(b1["C_FLOPs_1e21"].max())
    budgets = np.geomspace(cmin, cmax, 30)
    optimum = []
    for c21 in budgets:
        k = c21 / .006  # because C/1e21 = 0.006*N_B*D_B
        nstar = ((alpha*a0)/(beta*b0) * k**beta) ** (1/(alpha+beta))
        dstar = k/nstar
        loss = law(np.array([[nstar],[dstar]]), *p0)[0]
        optimum.append({"C_FLOPs_1e21":c21,"C_FLOPs":c21*1e21,"N_opt_B":nstar,"D_opt_B":dstar,
                        "D_over_N":dstar/nstar,"predicted_loss":loss,
                        "N_compute_exponent":beta/(alpha+beta),"D_compute_exponent":alpha/(alpha+beta),
                        "within_B1_compute_range":True,"evidence_level":"model_derived_within_range"})
    optimum_df = pd.DataFrame(optimum)
    optimum_df.to_csv(RES / "q2_compute_optimal.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].hist(good["alpha"], bins=24, alpha=.75, label=r"$\alpha$")
    axes[0].hist(good["beta"], bins=24, alpha=.65, label=r"$\beta$")
    axes[0].axvline(alpha, color="#2E5A87", ls="--"); axes[0].axvline(beta, color="#D1495B", ls="--")
    axes[0].set_xlabel("指数"); axes[0].set_ylabel("Bootstrap 频数"); axes[0].legend()
    axes[1].scatter(good["alpha"], good["beta"], s=12, alpha=.4)
    axes[1].set_xlabel(r"参数指数 $\alpha$"); axes[1].set_ylabel(r"数据指数 $\beta$")
    save_fig(fig, FIG / "q2_09_parameter_uncertainty.pdf", also_png=True); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].plot(optimum_df["C_FLOPs_1e21"], optimum_df["N_opt_B"], label="N*（十亿）")
    axes[0].plot(optimum_df["C_FLOPs_1e21"], optimum_df["D_opt_B"], label="D*（十亿）")
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[0].set_xlabel(r"算力预算 $C/10^{21}$")
    axes[0].set_ylabel("最优规模"); axes[0].legend()
    axes[1].plot(optimum_df["C_FLOPs_1e21"], optimum_df["D_over_N"])
    axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel(r"算力预算 $C/10^{21}$")
    axes[1].set_ylabel(r"最优 $D^*/N^*$")
    save_fig(fig, FIG / "q2_10_compute_optimal.pdf", also_png=True); plt.close(fig)

    pd.DataFrame([
        {"figure":"q2_09_parameter_uncertainty.pdf","result_source":"q2_classic_trajectory_bootstrap.csv",
         "assertion":"按完整模型轨迹重采样评估指数不确定性"},
        {"figure":"q2_10_compute_optimal.pdf","result_source":"q2_compute_optimal.csv",
         "assertion":"在 B1 已观测算力范围内给出 N-D 最优分配曲线"},
    ]).to_csv(RES / "q2_uncertainty_figure_contract.csv", index=False, encoding="utf-8-sig")

    summary = {"bootstrap_success":int(len(good)),"bootstrap_total":300,
               "N_compute_exponent":float(beta/(alpha+beta)),
               "D_compute_exponent":float(alpha/(alpha+beta)),
               "note":"Selected Q and conditional p factors multiply the whole reducible loss, so they do not alter N:D optimum under fixed Q,p."}
    with (RES / "q2_uncertainty_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
