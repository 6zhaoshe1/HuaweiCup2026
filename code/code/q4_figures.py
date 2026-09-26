# -*- coding: utf-8 -*-
"""Generate Question 4 evidence figures from frozen result tables.

AI assistance: OpenAI Codex, 2026-09-25.
Every figure is generated from saved result/data artifacts and has a row in
q4_figure_contract.csv. No synthetic demonstration data are used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_模板" / "scripts"))
from mpl_cn import CYCLE, panel_label, plt, save_fig  # noqa: E402

R = ROOT / "results"
F = ROOT / "figures"
F.mkdir(exist_ok=True)
contracts: list[dict] = []


def save(fig, stem: str, sources: str, transform: str, assertion: str) -> None:
    path = F / f"{stem}.pdf"
    save_fig(fig, str(path), also_png=True)
    plt.close(fig)
    contracts.append(
        {
            "figure_pdf": str(path.relative_to(ROOT)),
            "figure_png": str(path.with_suffix(".png").relative_to(ROOT)),
            "source_artifacts": sources,
            "transformation": transform,
            "key_assertion": assertion,
            "generator": "code/q4_figures.py",
        }
    )


def frontier_history() -> None:
    d = pd.read_csv(R / "q4_entity_cohort.csv.gz", parse_dates=["submission_date"])
    d = d[
        d["strict_open"].astype(bool)
        & d["valid_core_fields"].astype(bool)
        & d["stratum"].isin(["pretrained", "chat_finetuned"])
    ].sort_values("submission_date")
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    labels = {"pretrained": "预训练/持续预训练", "chat_finetuned": "对话/微调"}
    for i, (key, part) in enumerate(d.groupby("stratum")):
        ax.scatter(
            part["submission_date"],
            part["score_equal"],
            s=15,
            alpha=0.38,
            color=CYCLE[i],
            label=labels[key],
        )
    cum = d["score_equal"].cummax()
    ax.step(d["submission_date"], cum, where="post", color="#1A1A1A", lw=2.2, label="累计历史前沿")
    record = d.loc[cum.diff().fillna(1).gt(0)]
    ax.scatter(record["submission_date"], record["score_equal"], s=35, color="#1A1A1A", zorder=4)
    ax.set_ylabel("六项 Benchmark 等权平均 / 分")
    ax.set_xlabel("Leaderboard 提交日期")
    ax.set_ylim(0, max(58, d["score_equal"].max() + 4))
    ax.legend(ncol=3, loc="upper left")
    save(
        fig,
        "q4_fig01_historical_frontier",
        "results/q4_entity_cohort.csv.gz",
        "严格开源、核心两层；按提交日排序并取累计最大值",
        "能力分布与累计最高能力是不同对象，前沿在2024-09后未再刷新",
    )


def quantile_comparison() -> None:
    d = pd.read_csv(R / "q4_quantile_model_comparison.csv")
    d = d[d["stratum"].eq("core_combined")].copy()
    order = ["scale_linear", "scale_quadratic", "scale_time_linear", "scale_time_interaction", "gradient_boosting"]
    names = ["规模线性", "规模二次", "规模+时间", "规模×时间", "梯度提升"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.7))
    x = np.arange(len(order))
    width = 0.36
    for j, scope in enumerate(["strict", "wide"]):
        part = d[d["scope"].eq(scope)].set_index("model").reindex(order)
        axes[0].bar(x + (j - .5) * width, part["pinball_familyweighted_mean"], width, label="严格口径" if scope == "strict" else "宽口径")
        axes[1].bar(x + (j - .5) * width, part["coverage_mean"], width, label="严格口径" if scope == "strict" else "宽口径")
    axes[0].set_ylabel("家族等权滚动 Pinball 损失（越低越好）")
    axes[1].set_ylabel("经验覆盖率")
    axes[1].axhline(.9, color="#444444", ls="--", lw=1, label="目标 0.90")
    for ax in axes:
        ax.set_xticks(x, names, rotation=25, ha="right")
    axes[0].legend()
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    save(
        fig,
        "q4_fig02_quantile_model_comparison",
        "results/q4_quantile_model_comparison.csv",
        "7个滚动窗口的家族等权Pinball均值与经验覆盖率",
        "严格主口径的时间模型未通过相对改善5%门，宽口径仅作敏感性",
    )


def quantile_rolling() -> None:
    d = pd.read_csv(R / "q4_quantile_backtest.csv", parse_dates=["cutoff"])
    d = d[d["status"].eq("ok") & d["scope"].eq("strict") & d["stratum"].eq("core_combined")]
    keep = ["scale_quadratic", "gradient_boosting"]
    names = {"scale_quadratic": "规模二次（入选）", "gradient_boosting": "最佳时间候选（未入选）"}
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    for i, model in enumerate(keep):
        p = d[d["model"].eq(model)].sort_values("cutoff")
        ax.plot(p["cutoff"], p["pinball_logit_familyweighted"], marker="o", color=CYCLE[i], label=names[model])
    ax.set_ylabel("家族等权 Pinball 损失")
    ax.set_xlabel("滚动训练截止日")
    ax.legend()
    save(
        fig,
        "q4_fig03_quantile_rolling",
        "results/q4_quantile_backtest.csv",
        "严格核心层按截止日展示最佳纯规模与最佳时间候选",
        "时间候选并非每个历史窗口都稳定优于规模基线",
    )


def task_effect_forest() -> None:
    d = pd.read_csv(R / "q4_c8_task_time_effects.csv")
    d = d[d["analysis_scope"].eq("all_complete_signatures")].sort_values("time_effect_logit_per_year")
    colors = {name: CYCLE[i % len(CYCLE)] for i, name in enumerate(sorted(d["task_family"].unique()))}
    fig, ax = plt.subplots(figsize=(7.5, 7.2))
    y = np.arange(len(d))
    ax.hlines(y, d["time_ci_low"], d["time_ci_high"], color=[colors[x] for x in d["task_family"]], alpha=.65, lw=1)
    ax.scatter(d["time_effect_logit_per_year"], y, color=[colors[x] for x in d["task_family"]], s=20)
    ax.axvline(0, color="#333333", ls="--", lw=1)
    short_tasks = (
        d["task"]
        .str.replace("leaderboard_", "", regex=False)
        .str.replace("_", " ", regex=False)
    )
    ax.set_yticks(y, short_tasks)
    ax.set_xlabel("控制规模、模型层与评测签名后的条件时间系数 / logit每年")
    ax.set_ylabel("C8 逐任务")
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=c, label=k) for k, c in colors.items()]
    ax.legend(handles=handles, ncol=3, loc="lower right")
    save(
        fig,
        "q4_fig04_c8_task_time_effects",
        "results/q4_c8_task_time_effects.csv",
        "完整评测签名层；OLS按基础模型家族聚类稳健标准误；BH校正另存表",
        "任务异质性显著，BH校正后没有任务可稳健声明为正向条件时间效应",
    )


def c8_snapshot_divergence() -> None:
    d = pd.read_csv(R / "q4_c8_reconstruction_metrics.csv")
    internal = d[d["comparison"].eq("leaf_vs_json_group")].set_index("task_family")
    snapshot = d[d["comparison"].eq("json_group_vs_c9_raw")].set_index("task_family")
    idx = list(snapshot.index)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    axes[0].barh(idx, internal.loc[idx, "max_abs_error"], color=CYCLE[2])
    axes[0].set_xlabel("C8内部加权重构最大绝对误差")
    axes[1].barh(idx, snapshot.loc[idx, "mae"], color=CYCLE[1])
    axes[1].set_xlabel("C8 JSON组得分与C9快照 MAE")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    save(
        fig,
        "q4_fig05_c8_reconstruction_snapshot",
        "results/q4_c8_reconstruction_metrics.csv",
        "区分C8内部leaf加权重构与C8-C9跨快照差异",
        "C8聚合公式可精确复原，但MATH等维度与C9并非同一快照口径",
    )


def decomposition() -> None:
    d = pd.read_csv(R / "q4_progress_decomposition.csv")
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    y = np.arange(len(d))
    parts = [
        ("scale_component_logit", "规模分量", CYCLE[0]),
        ("time_proxy_component_logit", "条件时间代理", CYCLE[2]),
        ("unexplained_residual_logit", "未解释残差", CYCLE[1]),
    ]
    pos = np.zeros(len(d)); neg = np.zeros(len(d))
    for col, label, color in parts:
        values = d[col].to_numpy(float)
        left = np.where(values >= 0, pos, neg)
        ax.barh(y, values, left=left, label=label, color=color)
        pos += np.where(values >= 0, values, 0)
        neg += np.where(values < 0, values, 0)
    ax.set_yticks(y, ["预训练层", "对话/微调层"])
    ax.set_xlabel("端点能力变化分解 / logit")
    ax.axvline(0, color="#333333", lw=1)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    save(
        fig,
        "q4_fig06_progress_decomposition",
        "results/q4_progress_decomposition.csv",
        "端点对齐两因素Shapley分解，残差显式保留",
        "非规模代理只解释约20%，绝大部分端点差异未被规模与时间线性项解释",
    )


def bridge() -> None:
    m = pd.read_csv(R / "q4_bridge_loo_metrics.csv")
    d = pd.read_csv(R / "q4_bridge_decisions.csv")
    targets = d["target"].tolist()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    base = m[m["model"].eq("constant")].set_index("target").loc[targets]
    best_values = []
    for row in d.itertuples():
        best_values.append(float(m[(m["target"].eq(row.target)) & (m["model"].eq(row.best_nonconstant))]["loo_rmse"].iloc[0]))
    x = np.arange(len(targets)); width=.36
    ax.bar(x-width/2, base["loo_rmse"], width, label="常数基准", color=CYCLE[0])
    bars = ax.bar(x+width/2, best_values, width, label="最佳非线性/线性候选", color=CYCLE[1])
    for b, accepted in zip(bars, d["accepted_for_bridge"]):
        if accepted:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+.04, "局部", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [t.replace("LB_", "") for t in targets])
    ax.set_ylabel("High可比性7点 LOO RMSE")
    ax.legend()
    save(
        fig,
        "q4_fig07_loss_benchmark_bridge",
        "results/q4_bridge_loo_metrics.csv; results/q4_bridge_decisions.csv",
        "High可比性样本逐点留一；常数基准对照",
        "综合分桥接不可识别，仅BBH/IFEval/MATH存在样本内局部映射证据",
    )


def record_backtest() -> None:
    d = pd.read_csv(R / "q4_record_backtest.csv", parse_dates=["cutoff"])
    d = d[d["model"].isin(["persistence", "record_process_scale", "record_process_time_candidate"])]
    names = {"persistence":"持久性", "record_process_scale":"随机纪录过程（主）", "record_process_time_candidate":"时间候选（未入选）"}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for i, model in enumerate(names):
        p=d[d["model"].eq(model)]
        axes[0].scatter(p["truth"], p["prediction"], s=23, alpha=.8, color=CYCLE[i], label=names[model])
    lo=min(d["truth"].min(),d["prediction"].min()); hi=max(d["truth"].max(),d["prediction"].max())
    axes[0].plot([lo,hi],[lo,hi],ls="--",color="#444444",lw=1)
    axes[0].set_xlabel("真实累计前沿"); axes[0].set_ylabel("预测累计前沿")
    metrics=pd.read_csv(R/"q4_record_model_comparison.csv").set_index("model")
    vals=[metrics.loc[m,"mae"] for m in names]
    axes[1].bar(np.arange(len(names)),vals,color=CYCLE[:len(names)])
    axes[1].set_xticks(np.arange(len(names)),[names[m] for m in names],rotation=20,ha="right")
    axes[1].set_ylabel("18个滚动情景 MAE / 分")
    axes[0].legend(fontsize=8)
    panel_label(axes[0],"(a)"); panel_label(axes[1],"(b)")
    save(
        fig,
        "q4_fig08_record_backtest",
        "results/q4_record_backtest.csv; results/q4_record_model_comparison.csv",
        "9个截止日×30/60日滚动回测；含11个真实纪录更新情景",
        "家族平衡随机纪录过程优于持久性，时间候选未获额外收益",
    )


def forecast() -> None:
    d = pd.read_csv(R / "q4_frontier_forecast.csv")
    d = d[d["model"].eq("record_process_scale")].copy()
    order=["historical_trend","moderate_slowdown","strong_slowdown","zero_scale_growth"]
    labels=["历史增长","中度放缓","强放缓","零规模增长"]
    fig, axes=plt.subplots(1,2,figsize=(8.3,3.8),sharey=True)
    for ax,h in zip(axes,[12,24]):
        p=d[d["horizon_months"].eq(h)].set_index("scenario").loc[order]
        y=np.arange(len(order))
        ax.errorbar(p["p50"],y,xerr=[p["p50"]-p["p05"],p["p95"]-p["p50"]],fmt="o",capsize=3,color=CYCLE[0])
        ax.axvline(float(p["current_frontier"].iloc[0]),ls="--",color="#444444",lw=1,label="当前前沿")
        ax.set_yticks(y,labels);ax.set_xlabel(f"{h}个月累计前沿 / 分")
        ax.legend()
    panel_label(axes[0],"(a)");panel_label(axes[1],"(b)")
    save(
        fig,
        "q4_fig09_frontier_scenarios",
        "results/q4_frontier_forecast.csv",
        "仅展示通过回测选型的规模随机纪录过程；点为中位数，线为90%情景区间",
        "算力增长放缓压低情景前沿，但零规模增长仍含新模型到达/搜索效应，不等于技术因果效应",
    )


def compute_history() -> None:
    d=pd.read_csv(R/"q4_compute_growth_history.csv")
    fig,ax=plt.subplots(figsize=(6.8,3.8))
    ax.plot(d["year"],d["compute_q90"],marker="o",label="训练算力90%分位")
    ax.plot(d["year"],d["compute_max"],marker="s",label="训练算力最大值")
    ax.set_xlabel("发布日期年份");ax.set_ylabel("log10(训练算力 / FLOPs)")
    ax2=ax.twinx();ax2.plot(d["year"],d["params_q90"],marker="^",color=CYCLE[2],label="参数量90%分位")
    ax2.set_ylabel("log10(参数量)")
    h1,l1=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels();ax.legend(h1+h2,l1+l2,loc="upper left")
    save(
        fig,
        "q4_fig10_compute_history",
        "results/q4_compute_growth_history.csv",
        "C4中开放权重（unrestricted）语言模型按年聚合",
        "算力趋势仅用于情景标定，C4含观测与估计值，不能视为实验因果输入",
    )


def arrival_heatmap() -> None:
    d = pd.read_csv(R / "q4_arrival_rate_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True)
    compute = sorted(d["compute_growth_factor"].unique())
    arrival = sorted(d["arrival_rate_factor"].unique())
    vmin, vmax = d["p50"].min(), d["p50"].max()
    image_obj = None
    for ax, horizon in zip(axes, [12, 24]):
        p = d[d["horizon_months"].eq(horizon)].pivot(
            index="arrival_rate_factor", columns="compute_growth_factor", values="p50"
        ).loc[arrival, compute]
        image_obj = ax.imshow(p.to_numpy(), aspect="auto", origin="lower", cmap="Blues", vmin=vmin, vmax=vmax)
        for i in range(len(arrival)):
            for j in range(len(compute)):
                ax.text(j, i, f"{p.iloc[i,j]:.1f}", ha="center", va="center", color="#1A1A1A", fontsize=9)
        ax.set_xticks(np.arange(len(compute)), [f"{x:.2g}" for x in compute])
        ax.set_yticks(np.arange(len(arrival)), [f"{x:.1f}" for x in arrival])
        ax.set_xlabel("历史算力增长倍数")
        ax.set_title(f"{horizon}个月", fontsize=10.5)
    axes[0].set_ylabel("未来模型到达率倍数")
    fig.colorbar(image_obj, ax=axes, label="累计前沿中位数 / 分", fraction=.035, pad=.03)
    panel_label(axes[0], "(a)"); panel_label(axes[1], "(b)")
    save(
        fig,
        "q4_fig11_arrival_compute_sensitivity",
        "results/q4_arrival_rate_sensitivity.csv",
        "4档算力增长×3档模型到达率×12/24个月蒙特卡洛中位数",
        "前沿预测同时受规模趋势与模型到达/搜索强度影响，零规模增长并不等于零更新",
    )


def main() -> None:
    frontier_history()
    quantile_comparison()
    quantile_rolling()
    task_effect_forest()
    c8_snapshot_divergence()
    decomposition()
    bridge()
    record_backtest()
    forecast()
    compute_history()
    arrival_heatmap()
    pd.DataFrame(contracts).to_csv(R / "q4_figure_contract.csv", index=False, encoding="utf-8-sig")
    print(f"generated_figures={len(contracts)}")


if __name__ == "__main__":
    main()
