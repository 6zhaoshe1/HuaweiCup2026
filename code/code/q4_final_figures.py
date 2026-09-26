# -*- coding: utf-8 -*-
"""Evidence figures for the targeted Q4 closure validation.

AI assistance: OpenAI Codex, 2026-09-26.  All panels use saved real-result
artifacts; no synthetic demonstration data are used.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R, F = ROOT / "results", ROOT / "figures"
sys.path.insert(0, str(ROOT / "_模板" / "scripts"))
from mpl_cn import CYCLE, panel_label, plt, save_fig  # noqa: E402

contracts: list[dict] = []


def save(fig, stem: str, sources: str, transform: str, assertion: str) -> None:
    path = F / f"{stem}.pdf"
    save_fig(fig, str(path), also_png=True)
    plt.close(fig)
    contracts.append({"figure_pdf":str(path.relative_to(ROOT)),"figure_png":str(path.with_suffix('.png').relative_to(ROOT)),
                      "source_artifacts":sources,"transformation":transform,"key_assertion":assertion,
                      "generator":"code/q4_final_figures.py"})


def backtest_credibility() -> None:
    d=pd.read_csv(R/"q4_final_record_metrics.csv")
    models=["persistence","legacy_record_iid","family_poisson_block","family_nb_tail95"]
    names=["持久性","原过程","家族泊松","家族NB+尾约束"]
    fig,axes=plt.subplots(1,2,figsize=(8.7,3.8))
    x=np.arange(len(models)); width=.36
    for j,(subset,label) in enumerate([("formal_complete_30d","完整30天（9窗）"),("formal_complete_60d","完整60天（8窗）")]):
        p=d[d.evaluation_subset.eq(subset)].set_index("model").loc[models]
        axes[0].bar(x+(j-.5)*width,p.mae,width,label=label,color=CYCLE[j])
    for j,(subset,label) in enumerate([("formal_complete_30d","完整30天"),("formal_complete_60d","完整60天")]):
        p=d[d.evaluation_subset.eq(subset)].set_index("model").loc[models]
        axes[1].bar(x+(j-.5)*width,p.coverage90,width,label=label,color=CYCLE[j])
    axes[0].set_xticks(x,names,fontsize=8);axes[0].set_ylabel("累计前沿预测 MAE / 分");axes[0].legend(fontsize=8)
    axes[1].axhline(.9,ls="--",lw=1,color="#555555",label="目标0.90")
    axes[1].set_xticks(x,names,fontsize=8);axes[1].set_ylabel("90%区间经验覆盖率")
    axes[1].set_ylim(0,1.05);axes[1].legend(fontsize=7.5,loc="upper left")
    panel_label(axes[0],"(a)");panel_label(axes[1],"(b)")
    save(fig,"q4_final_fig01_backtest_credibility",
         "results/q4_final_record_metrics.csv",
         "外层截止日前重新选型；仅使用实际观测完整的30天与60天窗口并分开汇总",
         "消除选型泄漏并统一时间跨度后原纪录过程仍优于持久性；经验覆盖未达到0.90")


def tail_and_forecast() -> None:
    e=pd.read_csv(R/"q4_final_extreme_search.csv")
    f=pd.read_csv(R/"q4_final_frontier_forecast.csv")
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.8))
    labels={"record_iid":"记录级独立残差","family_block":"家族级残差","family_block_tail95":"家族级+95%上尾约束"}
    for i,(key,p) in enumerate(e.groupby("residual_mechanism")):
        p=p.sort_values("n_candidates")
        axes[0].plot(p.n_candidates,p.max_residual_p95,marker="o",ms=3,label=labels[key],color=CYCLE[i])
    axes[0].set_xscale("log");axes[0].set_xlabel("未来候选数");axes[0].set_ylabel("最大残差的95%分位 / logit");axes[0].legend(fontsize=7.5)
    hist=f[f.scenario.eq("historical_trend")]
    order=["legacy_record_iid","family_poisson_block","family_nb_tail95"]
    names=["原纪录过程","家族泊松","家族负二项+尾约束"]
    for j,h in enumerate([12,24]):
        p=hist[hist.horizon_months.eq(h)].set_index("model").loc[order]
        axes[1].errorbar(p.p50,np.arange(3)+(j-.5)*.13,xerr=[p.p50-p.p05,p.p95-p.p50],fmt="o",capsize=3,
                         label=f"{h}个月",color=CYCLE[j])
    axes[1].set_yticks(np.arange(3),names);axes[1].set_xlabel("历史算力情景累计前沿 / 分");axes[1].legend()
    panel_label(axes[0],"(a)");panel_label(axes[1],"(b)")
    save(fig,"q4_final_fig02_tail_forecast_sensitivity",
         "results/q4_final_extreme_search.csv; results/q4_final_frontier_forecast.csv",
         "左图固定残差池改变候选数；右图统一算力情景比较到达/残差机制",
         "独立重复抽样会饱和到样本最大残差，但尾约束虽压低长期上界却损害历史区间表现")


def decomposition_identity() -> None:
    d=pd.read_csv(R/"q4_final_decomposition_sensitivity.csv")
    d=d[d.model.eq("scale_time_linear")].copy()
    strata=["pretrained","chat_finetuned"]
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.8),sharey=True)
    for ax,stratum in zip(axes,strata):
        p=d[d.stratum.eq(stratum)].sort_values("start_cutoff")
        x=np.arange(len(p)); bottom=np.zeros(len(p))
        for col,label,color in [("scale_component_logit","规模分量",CYCLE[0]),
                                ("time_proxy_component_logit","条件时间代理",CYCLE[2]),
                                ("endpoint_residual_change_logit","端点偏离变化",CYCLE[1])]:
            vals=p[col].to_numpy(float)
            ax.bar(x,vals,bottom=bottom,label=label,color=color)
            bottom+=vals
        ax.scatter(x,p.observed_delta_logit,marker="_",s=170,color="#111111",label="观测端点变化",zorder=4)
        ax.set_xticks(x,p.start_cutoff.str.slice(5),rotation=0);ax.set_xlabel("起始截止日（月-日）")
        ax.set_title("预训练层" if stratum=="pretrained" else "对话/微调层",fontsize=10.5)
    axes[0].set_ylabel("端点变化 / logit")
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,ncol=4,loc="upper center",bbox_to_anchor=(.5,1.02),fontsize=8)
    panel_label(axes[0],"(a)");panel_label(axes[1],"(b)")
    save(fig,"q4_final_fig03_decomposition_identity",
         "results/q4_final_decomposition_sensitivity.csv",
         "线性规模+时间分位模型；按三种起始截止日重取历史前沿；三项恒等闭合",
         "端点偏离变化主导且对端点敏感，时间代理模型未通过滚动预测门，不能视为因果贡献")


def frozen_scenarios() -> None:
    d = pd.read_csv(R / "q4_frozen_frontier_forecast.csv")
    order = ["historical_trend", "moderate_slowdown", "strong_slowdown", "zero_scale_growth"]
    names = ["历史趋势", "中度放缓", "强放缓", "零规模增长"]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    y = np.arange(len(order))
    for j, horizon in enumerate([12, 24]):
        p = d[d.horizon_months.eq(horizon)].set_index("scenario").loc[order]
        yy = y + (j - .5) * .18
        ax.errorbar(p.p50, yy, xerr=[p.p50-p.p05, p.p95-p.p50], fmt="o", capsize=3,
                    label=f"{horizon}个月", color=CYCLE[j], zorder=3)
    ax.axvline(float(d.current_frontier.iloc[0]), color="#555555", ls="--", lw=1, label="2025-03-13前沿")
    ax.set_yticks(y, names); ax.set_xlabel("累计能力前沿 / 分")
    ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(.5, 1.01), frameon=False)
    ax.grid(axis="x", alpha=.25)
    save(fig, "q4_final_fig04_frozen_frontier_scenarios",
         "results/q4_frozen_frontier_forecast.csv",
         "冻结家族平衡纪录过程；5000路径；同一参数与到达/残差口径；点和横线为中位数与5%—95%条件情景区间",
         "算力增长放缓会稳定压低条件前沿，但零规模增长下的上升仍包含到达与极值搜索效应")


def main() -> None:
    F.mkdir(exist_ok=True)
    backtest_credibility(); tail_and_forecast(); decomposition_identity(); frozen_scenarios()
    pd.DataFrame(contracts).to_csv(R/"q4_final_figure_contract.csv",index=False,encoding="utf-8-sig")
    print(f"generated_figures={len(contracts)}")


if __name__ == "__main__":
    main()
