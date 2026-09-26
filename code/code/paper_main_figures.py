# -*- coding: utf-8 -*-
"""Regenerate six paper-facing main figures from frozen Q1--Q4 results.

AI assistance: OpenAI Codex, 2026-09-26.
The advanced-chart library is used only as a verified layout reference.  All
plotted values are read from machine-readable project results; no preview or
synthetic data from the library are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_模板" / "scripts"))
from mpl_cn import CYCLE, panel_label, plt, save_fig  # noqa: E402

R = ROOT / "results"
F = ROOT / "figures"
F.mkdir(exist_ok=True)

mpl.rcParams.update(
    {
        "axes.grid": False,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.handlelength": 2.2,
    }
)

contracts: list[dict[str, str]] = []


def save(
    fig,
    stem: str,
    paper_role: str,
    sources: str,
    fields: str,
    transform: str,
    template: str,
    assertion: str,
) -> None:
    """Save vector/raster versions and append the evidence contract."""
    pdf = F / f"{stem}.pdf"
    save_fig(fig, str(pdf), also_png=True)
    plt.close(fig)
    contracts.append(
        {
            "figure_pdf": str(pdf.relative_to(ROOT)),
            "figure_png": str(pdf.with_suffix(".png").relative_to(ROOT)),
            "paper_role": paper_role,
            "source_artifacts": sources,
            "source_fields": fields,
            "transformation": transform,
            "library_template_reference": template,
            "template_verification_status": "已验证（2026-09-05）；仅借鉴布局，示例数据未使用",
            "key_assertion": assertion,
            "generator": "code/paper_main_figures.py",
        }
    )


def annotate_heatmap(ax, values: np.ndarray, fmt: str, cmap, norm) -> None:
    """Label heatmap cells with luminance-aware text."""
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = float(values[i, j])
            rgba = cmap(norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                j,
                i,
                format(v, fmt),
                ha="center",
                va="center",
                fontsize=8.7,
                color="#222222" if lum > 0.56 else "white",
            )


def q1_quality_conflict() -> None:
    quality = pd.read_csv(R / "q1_domain_quality_summary.csv")
    quality = quality[quality["dataset"].eq("pooled_unique")].copy()
    domains = ["arxiv", "book", "c4", "commoncrawl", "github", "stackexchange", "wikipedia"]
    quality = quality.set_index("domain").loc[domains]
    facet_cols = [
        "facet_content_value_robust",
        "facet_language_quality_robust",
        "facet_cleanliness_robust",
        "facet_reasoning_professional_robust",
    ]
    facet_names = ["内容价值", "语言质量", "清洁度", "推理与专业性"]
    mat = quality[facet_cols].to_numpy(float)

    conflict = pd.read_csv(R / "q1_upgrade_conflict_rates.csv")
    conflict["label"] = conflict["dataset"] + "·" + conflict["domain"]
    conflict = conflict.sort_values("main_conflict_rate", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.05, 1.15]})
    cmap = mpl.colormaps["cividis"]
    norm = mpl.colors.Normalize(0, 1)
    im = axes[0].imshow(mat, cmap=cmap, norm=norm, aspect="auto")
    annotate_heatmap(axes[0], mat, ".2f", cmap, norm)
    axes[0].set_xticks(range(4), facet_names, rotation=22, ha="right")
    axes[0].set_yticks(range(len(domains)), domains)
    cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.03)
    cb.set_label("稳健分面得分")
    axes[0].set_xlabel("质量分面")
    axes[0].set_ylabel("去重合并后的语料域")

    y = np.arange(len(conflict))
    main = 100 * conflict["main_conflict_rate"].to_numpy(float)
    cond = 100 * conflict["conditional_conflict_rate"].to_numpy(float)
    for yi, a, b in zip(y, main, cond):
        axes[1].plot([a, b], [yi, yi], color="#B8BEC5", lw=1.3, zorder=1)
    axes[1].scatter(main, y, color=CYCLE[0], s=38, label="主冲突规则", zorder=3)
    axes[1].scatter(cond, y, color=CYCLE[1], marker="D", s=30, label="域×长度条件置换", zorder=3)
    axes[1].set_yticks(y, conflict["label"])
    axes[1].set_xlabel("识别为冲突的样本比例 / %")
    axes[1].set_ylabel("")
    axes[1].grid(axis="x", alpha=0.25, ls="--")
    axes[1].legend(loc="lower right")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.62)
    save(
        fig,
        "paper_main_01_q1_quality_conflict",
        "问题一：质量分面与冲突定义",
        "results/q1_domain_quality_summary.csv; results/q1_upgrade_conflict_rates.csv",
        "四个facet_*_robust；main_conflict_rate；conditional_conflict_rate",
        "去重合并域的四分面均值；两种预注册冲突规则按域并列",
        "评测热力图 + 分组点区间图（已验证）",
        "质量结构具有明显领域异质性；冲突率依赖定义，条件置换不是主规则的简单子集",
    )


def q1_model_evidence() -> None:
    d = pd.read_csv(R / "q1_upgrade_model_benchmark.csv")
    models = ["raw_share_ridge", "ilr_shared_alpha", "quadratic_ilr_ridge", "log_shift_ridge"]
    datasets = ["A4_A5_oof", "A6_A7_1m", "A8_A9_60m", "A10_A11_1b"]
    model_names = ["原始份额 Ridge", "线性 ILR", "二次 ILR", "对数平移 Ridge"]
    data_names = ["训练 OOF", "1M 独立检验", "60M 跨尺度", "1B 跨尺度"]
    rank = d.pivot(index="model", columns="dataset", values="spearman_macro").loc[models, datasets]
    rmse = d.pivot(index="model", columns="dataset", values="rmse_macro").loc[models, datasets]
    relative = rmse / rmse.min(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    cmap_a = mpl.colormaps["cividis"]
    norm_a = mpl.colors.Normalize(0.70, 0.95)
    im0 = axes[0].imshow(rank.to_numpy(), cmap=cmap_a, norm=norm_a, aspect="auto")
    annotate_heatmap(axes[0], rank.to_numpy(), ".2f", cmap_a, norm_a)
    axes[0].set_xticks(range(4), data_names, rotation=24, ha="right")
    axes[0].set_yticks(range(4), model_names)
    axes[0].set_xlabel("数据与验证层级")
    axes[0].set_ylabel("候选模型")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.03)
    cb0.set_label("宏平均 Spearman")

    cmap_b = mpl.colormaps["YlOrBr"]
    vmax = max(1.45, float(relative.to_numpy().max()))
    norm_b = mpl.colors.Normalize(1.0, vmax)
    im1 = axes[1].imshow(relative.to_numpy(), cmap=cmap_b, norm=norm_b, aspect="auto")
    annotate_heatmap(axes[1], relative.to_numpy(), ".2f", cmap_b, norm_b)
    axes[1].set_xticks(range(4), data_names, rotation=24, ha="right")
    axes[1].set_yticks(range(4), model_names)
    axes[1].set_xlabel("数据与验证层级")
    axes[1].set_ylabel("")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.03)
    cb1.set_label("相对最优 RMSE（1 为本列最优）")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.46)
    save(
        fig,
        "paper_main_02_q1_model_evidence",
        "问题一：17域配比模型选型与跨尺度边界",
        "results/q1_upgrade_model_benchmark.csv",
        "model,dataset,rmse_macro,spearman_macro,evidence_level",
        "按模型×数据透视；RMSE除以每个数据集的列内最优值",
        "评测热力图（已验证）",
        "对数平移模型同尺度预测最优，二次ILR提供接近的排序能力；跨尺度绝对误差不能由问题一模型解决",
    )


def q2_scaling_evidence() -> None:
    intervals = pd.read_csv(R / "q2_classic_parameter_intervals.csv")
    intervals = intervals.set_index("parameter").loc[["alpha", "beta", "kappa"]].reset_index()
    opt = pd.read_csv(R / "q2_compute_optimal.csv").sort_values("C_FLOPs")
    inside = opt["unconstrained_in_B1_box"].astype(str).str.lower().eq("true")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.25), gridspec_kw={"width_ratios": [0.85, 1.45]})
    labels = [r"参数规模弹性 $\alpha$", r"数据规模弹性 $\beta$", r"质量惩罚系数 $\kappa$"]
    y = np.arange(3)
    point = intervals["point_estimate"].to_numpy(float)
    lo = intervals["ci2p5"].to_numpy(float)
    hi = intervals["ci97p5"].to_numpy(float)
    colors = [CYCLE[0], CYCLE[0], CYCLE[1]]
    axes[0].hlines(y, lo, hi, color=colors, lw=2)
    axes[0].scatter(point, y, color=colors, s=48, zorder=3)
    for yi, p, low, high in zip(y, point, lo, hi):
        axes[0].text(high + 0.008, yi, f"{p:.4f} [{low:.4f}, {high:.4f}]", va="center", fontsize=8.3)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("点估计与 95% 轨迹 Bootstrap 区间")
    axes[0].set_xlim(max(0, lo.min() - 0.04), hi.max() + 0.13)
    axes[0].grid(axis="x", alpha=0.25, ls="--")
    axes[0].text(
        0.02,
        0.03,
        r"$\kappa$ 来自半合成质量层" + "\n" + r"其证据等级低于 $\alpha,\beta$",
        transform=axes[0].transAxes,
        fontsize=8.4,
        color=CYCLE[1],
    )

    c = opt["C_FLOPs"].to_numpy(float)
    for col, label, color, marker in [
        ("N_opt_B", "最优参数量 $N$", CYCLE[0], "o"),
        ("D_opt_B", "最优数据量 $D$", CYCLE[2], "s"),
    ]:
        axes[1].plot(c, opt[col], color=color, lw=1.8, label=label)
        axes[1].scatter(c[inside], opt.loc[inside, col], color=color, marker=marker, s=25, zorder=3)
        axes[1].scatter(
            c[~inside],
            opt.loc[~inside, col],
            facecolors="white",
            edgecolors=color,
            marker=marker,
            s=25,
            zorder=3,
        )
    if inside.any():
        supported_min = float(opt.loc[inside, "C_FLOPs"].min())
        supported_max = float(opt.loc[inside, "C_FLOPs"].max())
        axes[1].axvspan(supported_min, supported_max, color=CYCLE[0], alpha=0.055)
        axes[1].axvline(supported_min, color="#777777", ls="--", lw=0.8)
        axes[1].axvline(supported_max, color="#777777", ls="--", lw=0.8)
        axes[1].text(
            np.sqrt(supported_min * supported_max),
            0.96,
            "B1 联合支持区",
            transform=axes[1].get_xaxis_transform(),
            va="top",
            ha="center",
            fontsize=8.2,
            color="#555555",
        )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("训练算力 $C$ / FLOPs")
    axes[1].set_ylabel("最优规模 / 十亿（B）")
    axes[1].grid(which="both", alpha=0.22, ls="--")
    axes[1].legend(loc="upper left")
    axes[1].text(
        0.98,
        0.04,
        "实心：B1 支持框内\n空心：理论外推",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.4,
    )
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.38)
    save(
        fig,
        "paper_main_03_q2_scaling_evidence",
        "问题二：广义标度律参数与算力最优结构",
        "results/q2_classic_parameter_intervals.csv; results/q2_compute_optimal.csv",
        "alpha,beta,kappa及区间；C_FLOPs,N_opt_B,D_opt_B,unconstrained_in_B1_box",
        "参数轨迹Bootstrap森林图；固定算力解析最优曲线并按B1支持框区分实心/空心",
        "森林图 + 分箱效应曲线（已验证，按本题口径改造）",
        "规模弹性识别稳定，质量系数证据较弱；固定算力最优曲线离开B1支持框后只能作理论外推",
    )


def q3_resource_paths() -> None:
    paths = pd.read_csv(R / "q3_budget_paths.csv")
    paths = paths[paths["context_tokens"].eq(4096)].copy()
    paths = paths.sort_values(["cost_family", "budget_FLOPs"])
    official = pd.read_csv(R / "q3_official_solutions.csv")
    official = official[official["context_tokens"].eq(4096)].copy()

    names = {"exponential": "指数质量成本", "power": "幂质量成本", "logarithmic": "对数质量成本"}
    colors = {k: CYCLE[i] for i, k in enumerate(names)}
    active = paths["budget_active"].astype(str).str.lower().eq("true")
    saturation_candidates = paths.loc[~active, "budget_FLOPs"]
    saturation = float(saturation_candidates.min()) if len(saturation_candidates) else np.nan

    fig, axes = plt.subplots(2, 2, figsize=(10.3, 7.0), sharex=True)
    specs = [
        ("N_B", "参数量 $N$ / B", True),
        ("D_B", "训练数据量 $D$ / B tokens", True),
        ("Q", "质量变量 $Q$", False),
        ("budget_utilization", "预算利用率", False),
    ]
    for ax, (col, ylabel, logy) in zip(axes.flat, specs):
        for key in names:
            p = paths[paths["cost_family"].eq(key)]
            ax.plot(
                p["budget_FLOPs"],
                p[col],
                color=colors[key],
                label=names[key],
                markevery=15,
                marker="o",
                ms=3,
            )
        if np.isfinite(saturation):
            ax.axvspan(saturation, paths["budget_FLOPs"].max(), color="#9A9A9A", alpha=0.10)
            ax.axvline(saturation, color="#777777", ls="--", lw=0.9)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(which="both", alpha=0.22, ls="--")
    axes[1, 0].set_xlabel("算力预算 / FLOPs")
    axes[1, 1].set_xlabel("算力预算 / FLOPs")
    axes[1, 0].axhline(1.0, color="#555555", ls=":", lw=0.9)
    axes[1, 1].set_ylim(0, 1.05)

    tiers = {
        "C_supported_boundary_scenario": ("o", "支持域边界情景"),
        "D_theoretical_extrapolation": ("X", "理论外推"),
        "I_infeasible_under_frozen_constraints": ("x", "冻结约束下不可行"),
    }
    for ax, (col, _, _) in zip(axes.flat, specs):
        for tier, (marker, _) in tiers.items():
            p = official[official["overall_evidence_tier"].eq(tier)]
            if len(p):
                ax.scatter(
                    p["budget_FLOPs"],
                    p[col],
                    marker=marker,
                    s=42,
                    facecolors="none" if marker == "o" else CYCLE[3],
                    edgecolors=CYCLE[3],
                    linewidths=1.1,
                    zorder=4,
                )
    axes[0, 0].legend(loc="upper left")
    evidence_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=CYCLE[3], label="支持域边界情景"),
        plt.Line2D([0], [0], marker="X", color="none", markerfacecolor=CYCLE[3], markeredgecolor=CYCLE[3], label="理论外推"),
    ]
    axes[0, 1].legend(handles=evidence_handles, loc="upper left", fontsize=8.1)
    if np.isfinite(saturation):
        axes[0, 1].text(
            saturation * 1.08,
            0.07,
            "变量上界饱和区\n预算剩余≠现实中无需投入",
            transform=axes[0, 1].get_xaxis_transform(),
            fontsize=8.3,
            color="#555555",
        )
    for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        panel_label(ax, label)
    fig.subplots_adjust(hspace=0.17, wspace=0.28)
    save(
        fig,
        "paper_main_04_q3_resource_paths",
        "问题三：联合资源配置路径与支持边界",
        "results/q3_budget_paths.csv; results/q3_official_solutions.csv",
        "budget_FLOPs,N_B,D_B,Q,budget_utilization,cost_family,overall_evidence_tier",
        "固定4096上下文，三类质量成本沿预算网格求解；叠加官方预算点证据等级与预算非活跃起点",
        "状态轨迹相图双面板 + 平行坐标方案筛选（已验证，改为四面板轨迹）",
        "Q较早触顶且高预算利用率下降源于变量支持上界；高预算解须与支持域解分表解释",
    )


def q4_frontier_forecast() -> None:
    history = pd.read_csv(R / "q4_frontier_history.csv", parse_dates=["submission_date"])
    history = history.sort_values("submission_date")
    forecast = pd.read_csv(R / "q4_frontier_forecast.csv", parse_dates=["forecast_date"])
    forecast = forecast[forecast["model"].eq("record_process_scale")].copy()
    labels = {
        "historical_trend": "历史算力趋势",
        "moderate_slowdown": "中度放缓",
        "strong_slowdown": "强放缓",
        "zero_scale_growth": "零规模增长",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.35), gridspec_kw={"width_ratios": [1.2, 1.0]})
    strata = {"pretrained": "预训练/持续预训练", "chat_finetuned": "对话/微调"}
    for i, key in enumerate(strata):
        p = history[history["stratum"].eq(key)]
        axes[0].scatter(p["submission_date"], p["score_equal"], s=14, alpha=0.30, color=CYCLE[i], label=strata[key])
    axes[0].step(
        history["submission_date"],
        history["cumulative_frontier"],
        where="post",
        color="#202020",
        lw=2.2,
        label="累计历史前沿",
    )
    rec = history[history["is_record_update"].astype(str).str.lower().eq("true")]
    axes[0].scatter(rec["submission_date"], rec["score_equal"], s=31, color="#202020", zorder=4)
    axes[0].set_xlabel("Leaderboard 提交日期")
    axes[0].set_ylabel("六项 Benchmark 等权平均 / 分")
    axes[0].legend(loc="upper left", ncol=2, fontsize=8.3)
    axes[0].grid(alpha=0.22, ls="--")
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].tick_params(axis="x", rotation=20)

    current_date = history["submission_date"].max()
    current = float(forecast["current_frontier"].iloc[0])
    for i, key in enumerate(labels):
        p = forecast[forecast["scenario"].eq(key)].sort_values("horizon_months")
        dates = np.array([current_date, *p["forecast_date"].tolist()], dtype="datetime64[ns]")
        med = np.array([current, *p["p50"].to_numpy(float)])
        low = np.array([current, *p["p05"].to_numpy(float)])
        high = np.array([current, *p["p95"].to_numpy(float)])
        axes[1].fill_between(dates, low, high, color=CYCLE[i], alpha=0.10)
        axes[1].plot(dates, med, marker="o", color=CYCLE[i], label=labels[key])
    axes[1].axhline(current, color="#555555", ls="--", lw=1, label="持久性基准")
    axes[1].set_xlabel("预测日期")
    axes[1].set_ylabel("累计前沿条件预测 / 分")
    axes[1].grid(alpha=0.22, ls="--")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(loc="upper left", fontsize=8.1)
    axes[1].text(
        0.98,
        0.04,
        "阴影：90% 条件区间\n全部长期结果均为条件外推",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8.3,
    )
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.28)
    save(
        fig,
        "paper_main_05_q4_frontier_forecast",
        "问题四：历史累计前沿与算力放缓情景",
        "results/q4_frontier_history.csv; results/q4_frontier_forecast.csv",
        "submission_date,score_equal,cumulative_frontier,is_record_update；scenario,p05,p50,p95",
        "历史提交按日累计最大；主随机纪录过程在四档算力情景下连接当前值与12/24个月分位数",
        "预测扇形图（已验证）",
        "累计最高能力不同于新模型条件分位数；算力放缓降低前沿增长，但长期区间较宽且不构成必然预测",
    )


def q4_non_scale_evidence() -> None:
    tasks = pd.read_csv(R / "q4_c8_task_time_effects.csv")
    tasks = tasks[tasks["analysis_scope"].eq("all_complete_signatures")].copy()
    decomp = pd.read_csv(R / "q4_progress_decomposition.csv")
    families = ["BBH", "GPQA", "IFEval", "MATH Lvl 5", "MMLU-PRO", "MUSR"]
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.45), gridspec_kw={"width_ratios": [1.15, 1.0]})
    for yi, fam in enumerate(families):
        p = tasks[tasks["task_family"].eq(fam)]["time_effect_logit_per_year"].to_numpy(float)
        jitter = rng.uniform(-0.11, 0.11, len(p))
        axes[0].scatter(p, yi + jitter, s=28, alpha=0.72, color=CYCLE[yi % len(CYCLE)], edgecolor="white", linewidth=0.35)
        axes[0].scatter(np.median(p), yi, marker="D", s=58, color="#202020", zorder=4)
        q25, q75 = np.quantile(p, [0.25, 0.75])
        axes[0].hlines(yi, q25, q75, color="#202020", lw=2.2, zorder=3)
    axes[0].axvline(0, color="#555555", ls="--", lw=1)
    axes[0].set_yticks(range(len(families)), families)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("控制规模、模型层与评测签名后的条件时间系数 / logit每年")
    axes[0].set_ylabel("C8 任务族（点为逐任务，菱形为族内中位数）")
    axes[0].grid(axis="x", alpha=0.22, ls="--")
    axes[0].text(
        0.02,
        0.04,
        "39 个任务经 BH 校正后：\n稳健正向任务数 = 0",
        transform=axes[0].transAxes,
        fontsize=8.4,
        va="bottom",
    )

    y = np.arange(len(decomp))
    parts = [
        ("scale_component_logit", "规模分量", CYCLE[0]),
        ("time_proxy_component_logit", "非规模进步代理", CYCLE[2]),
        ("unexplained_residual_logit", "未解释残差", CYCLE[1]),
    ]
    pos = np.zeros(len(decomp))
    neg = np.zeros(len(decomp))
    for col, label, color in parts:
        vals = decomp[col].to_numpy(float)
        left = np.where(vals >= 0, pos, neg)
        axes[1].barh(y, vals, left=left, color=color, label=label, height=0.55)
        pos += np.where(vals >= 0, vals, 0)
        neg += np.where(vals < 0, vals, 0)
    axes[1].axvline(0, color="#333333", lw=0.9)
    axes[1].set_yticks(y, ["预训练层", "对话/微调层"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("端点能力变化的描述性 Shapley 分解 / logit")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, fontsize=8.2)
    axes[1].grid(axis="x", alpha=0.22, ls="--")
    for yi, row in decomp.reset_index(drop=True).iterrows():
        axes[1].text(
            pos[yi] + 0.02,
            yi,
            f"时间代理 {100*row['time_proxy_share_of_observed']:.0f}%\n残差 {100*row['unexplained_share_of_observed']:.0f}%",
            va="center",
            fontsize=8.0,
        )
    axes[1].text(
        0.03,
        0.49,
        "时间模型未通过滚动预测准入门\n分解仅作描述，不作因果解释",
        transform=axes[1].transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.36)
    save(
        fig,
        "paper_main_06_q4_non_scale_evidence",
        "问题四：非规模进步代理的任务异质性与贡献边界",
        "results/q4_c8_task_time_effects.csv; results/q4_progress_decomposition.csv",
        "task_family,time_effect_logit_per_year,time_qvalue_bh；scale/time_proxy/residual components",
        "完整评测签名层逐任务点分布与族内IQR；端点两因素Shapley分解显式保留残差",
        "效应森林图 + 多尺度分解堆叠（已验证，按本题证据层改造）",
        "逐任务条件时间效应高度异质且BH后无稳健正向任务；约20%的端点变化可归于时间代理，主要变化仍未解释",
    )


def main() -> None:
    q1_quality_conflict()
    q1_model_evidence()
    q2_scaling_evidence()
    q3_resource_paths()
    q4_frontier_forecast()
    q4_non_scale_evidence()

    contract = pd.DataFrame(contracts)
    contract.to_csv(R / "paper_main_figure_contract.csv", index=False, encoding="utf-8-sig")
    summary = {
        "generated_figures": len(contract),
        "pdf_count": int(sum((ROOT / p).is_file() for p in contract["figure_pdf"])),
        "png_count": int(sum((ROOT / p).is_file() for p in contract["figure_png"])),
        "uses_synthetic_data": False,
        "style_module": "_模板/scripts/mpl_cn.py",
        "advanced_library_use": "verified layout references only",
    }
    (R / "paper_main_figure_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
