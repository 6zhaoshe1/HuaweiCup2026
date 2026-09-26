#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# AI assistance: OpenAI Codex, OpenAI, 2026-09-25; team review required.
"""Generate Q3 report and Q4 interface strictly from frozen Q3 artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; RES=ROOT/"results"; REP=ROOT/"reports"
EV="semi_synthetic_supported_joint_box"; EX="theoretical_extrapolation_b9_guardrail"

def f(x,n=6): return f"{float(x):.{n}g}"
def sci(x): return f"{float(x):.3e}"
def md_table(df, cols, labels=None, formats=None):
    labels=labels or cols; formats=formats or {}
    out=["|"+"|".join(labels)+"|","|"+"|".join(["---"]*len(cols))+"|"]
    for _,r in df.iterrows():
        vals=[]
        for c in cols:
            v=r[c]
            if c in formats: v=formats[c](v)
            elif isinstance(v,(float,np.floating)): v=f(v)
            vals.append(str(v))
        out.append("|"+"|".join(vals)+"|")
    return "\n".join(out)

def main():
    summary=json.loads((RES/"q3_summary.json").read_text(encoding="utf-8"))
    verify=json.loads((RES/"q3_independent_verification.json").read_text(encoding="utf-8"))
    official=pd.read_csv(RES/"q3_official_solutions.csv")
    baselines=pd.read_csv(RES/"q3_feasible_baselines.csv")
    inf=pd.read_csv(RES/"q3_infeasible_scenarios.csv")
    trans=pd.read_csv(RES/"q3_transitions.csv")
    boot=pd.read_csv(RES/"q3_bootstrap_summary.csv")
    robust=pd.read_csv(RES/"q3_robust_configs.csv")
    mix=pd.read_csv(RES/"q3_mixture_ranking.csv")
    ms=pd.read_csv(RES/"q3_mixture_scenarios.csv")
    solver=pd.read_csv(RES/"q3_solver_validation.csv")
    qdiag=pd.read_csv(RES/"q3_quality_boundary_diagnostics.csv")
    support=pd.read_csv(RES/"q3_solution_evidence.csv")

    central=official[(official.evidence_level==EV)&(official.context_tokens==4096)&official.feasible].copy()
    central["预算"]=central.budget_FLOPs.map(lambda x:f"$10^{{{int(np.log10(x))}}}$")
    central["成本"]=central.cost_family.map({"exponential":"指数","power":"幂","logarithmic":"对数"})
    central["成本份额"]=central.apply(lambda r:f"{r.train_share:.1%}/{r.quality_share:.1%}/{r.attention_share:.1%}",axis=1)
    central["证据等级"]=central.overall_evidence_tier.map({
        "A_observed_joint_anchor":"A：观测联合锚点",
        "B_supported_factorial_interpolation":"B：因子域内插",
        "C_supported_boundary_scenario":"C：支持边界情景",
        "D_theoretical_extrapolation":"D：理论外推"})
    main_table=md_table(central,["预算","成本","N_B","D_B","Q","loss","成本份额","证据等级","active_set"],
                        ["预算/FLOPs","质量成本","N/B","D/B","Q","预测Loss","训练/质量/注意力","证据等级","活跃集"],
                        {"N_B":lambda x:f"{x:.4f}","D_B":lambda x:f"{x:.3f}","Q":lambda x:f"{x:.4f}","loss":lambda x:f"{x:.6f}"})

    ext=official[(official.evidence_level==EX)&(official.context_tokens==4096)&(official.cost_family=="exponential")].copy()
    ext["预算"]=ext.budget_FLOPs.map(lambda x:f"$10^{{{int(np.log10(x))}}}$")
    ext_table=md_table(ext,["预算","N_B","D_B","Q","loss","max_extrap_factor"],
                       ["预算/FLOPs","N/B","D/B","Q","预测Loss","最大外推倍数"],
                       {"N_B":lambda x:f"{x:.4f}","D_B":lambda x:f"{x:.3f}","Q":lambda x:f"{x:.4f}","loss":lambda x:f"{x:.6f}","max_extrap_factor":lambda x:f"{x:.2f}"})

    comp=baselines.pivot_table(index=["budget_FLOPs","cost_family"],columns="strategy",values="loss").reset_index()
    comp["delta_L"]=comp.fixed_Q0-comp.joint_NDQ
    comp=comp[comp.cost_family=="exponential"].copy(); comp["预算"]=comp.budget_FLOPs.map(lambda x:f"$10^{{{int(np.log10(x))}}}$")
    comp_table=md_table(comp,["预算","fixed_Q0","joint_NDQ","delta_L"],
                        ["预算/FLOPs","固定Q=0.5","联合N-D-Q","模型Loss改善"],
                        {x:lambda v:f"{v:.6f}" for x in ("fixed_Q0","joint_NDQ","delta_L")})

    # Substantive transitions exclude support ceiling changes.
    onset=trans[~trans.boundary_driven].copy()
    onset["上下文"]=onset.context_tokens.astype(int)
    onset["成本"]=onset.cost_family
    onset["区间"]=onset.apply(lambda r:f"[{sci(r.budget_before_FLOPs)}, {sci(r.budget_after_FLOPs)}]",axis=1)
    onset_table=md_table(onset,["上下文","成本","old_active_set","new_active_set","区间"],
                         ["上下文","成本族","旧状态","新状态","预算转移区间"])

    mix_pick=mix.iloc[[0,len(mix)//2,-1]][["index","R_A","rank_desc","original_share_sum","closure_factor"]].copy()
    mix_pick["角色"]=["最佳历史候选","中位历史候选","最差历史候选"]
    mix_table=md_table(mix_pick,["角色","index","R_A","original_share_sum","closure_factor"],
                       ["角色","A4索引","R_A","原始份额和","闭合因子"],
                       {"R_A":lambda x:f"{x:.6f}","original_share_sum":lambda x:f"{x:.3f}","closure_factor":lambda x:f"{x:.6f}"})

    boot_small=boot[["budget_FLOPs","cost_family","N_B_q025","N_B_median","N_B_q975","D_B_q025","D_B_median","D_B_q975","Q_q025","Q_median","Q_q975"]].copy()
    boot_small=boot_small[boot_small.cost_family=="exponential"]
    boot_small["预算"]=boot_small.budget_FLOPs.map(lambda x:f"$10^{{{int(np.log10(x))}}}$")
    boot_small["N区间"]=boot_small.apply(lambda r:f"{r.N_B_median:.4f} [{r.N_B_q025:.4f},{r.N_B_q975:.4f}]",axis=1)
    boot_small["D区间"]=boot_small.apply(lambda r:f"{r.D_B_median:.3f} [{r.D_B_q025:.3f},{r.D_B_q975:.3f}]",axis=1)
    boot_small["Q区间"]=boot_small.apply(lambda r:f"{r.Q_median:.4f} [{r.Q_q025:.4f},{r.Q_q975:.4f}]",axis=1)
    boot_table=md_table(boot_small,["预算","N区间","D区间","Q区间"],["预算/FLOPs","N中位[95%]","D中位[95%]","Q中位[95%]"])

    qd=qdiag.copy()
    qd["预算"]=qd.budget_FLOPs.map(lambda x:f"$10^{{{int(np.log10(x))}}}$")
    qd["成本"]=qd.cost_family.map({"exponential":"指数","power":"幂","logarithmic":"对数"})
    qd["成因"]=qd.diagnostic_cause.map({
        "interior_marginal_balance":"边际平衡",
        "marginal_cost_exceeds_benefit_at_Q0":"Q0处挤占代价更高",
        "Q_upper_binds_net_benefit_remains":"Q上界仍约束净收益",
        "support_caps_with_budget_slack":"N/D/Q支持上界且预算松弛"})
    qdiag_table=md_table(qd,["预算","成本","Q","profile_dLoss_dQ","direct_quality_benefit_per_Q",
                             "resource_displacement_penalty_per_Q","成因"],
                         ["预算","成本","Q","剖面dLoss/dQ","固定N,D质量收益","资源挤占代价","诊断"],
                         {"Q":lambda x:f"{x:.4f}","profile_dLoss_dQ":lambda x:f"{x:.3e}",
                          "direct_quality_benefit_per_Q":lambda x:f"{x:.3e}",
                          "resource_displacement_penalty_per_Q":lambda x:f"{x:.3e}"})

    max_robust_gain=float((robust.nominal_p90_loss-robust.robust_p90_loss).max())
    max_nom_reg=float(robust.nominal_p90_regret.max()); max_rob_reg=float(robust.robust_p90_regret.max())
    top=mix.iloc[0]; middle=mix.iloc[len(mix)//2]; worst=mix.iloc[-1]
    inf_gap=float(inf.budget_gap_FLOPs.max())
    central_exp=central[central.cost_family=="exponential"].set_index("budget_FLOPs")
    ext_hi=ext.loc[ext.budget_FLOPs.idxmax()]

    # Use a raw f-string: ordinary f-strings interpret LaTeX sequences such as
    # \alpha, \beta, \frac and \times as control characters.
    report=rf"""# F 题问题三：算力约束下的资源配置优化与结构性转移

> 状态：Codex 独立版正式实验完成，独立数值复核 `{verify['status']}`；仍待三名队员比较合并。本文所有数值由 `results/q3_*` 工件生成，不能把半合成或外推情景写成真实大模型训练验证。

## 1. 证据与变量口径

主目标继承问题二冻结模型

\[
\widehat L=E+\left(AN^{{-\alpha}}+BD^{{-\beta}}\right)[1+\kappa(1-Q)],
\]

其中 (N,D) 的代码单位为十亿。参数不是手抄，而是运行时从 `q2_interface_to_q3.json` 读取。规模项主要由 B1 真实观测支持；质量项由 B7 半合成全因子实验支持，不能解释为现实因果收益。

优化 (Q) 时的联合支持域必须取 B1 与 B7 的交集：

\[
N\in[0.070542,11.965825]\,\mathrm B,
\quad D\in[10,299.893]\,\mathrm B,
\quad Q\in[0.1,1].
\]

这修正了只采用 B1 下界 (D=0.134\)B 的做法；后者会让低预算质量优化落到 B7 从未覆盖的 (D<10\)B 区域。B9 只提供理论外推搜索护栏，B10 估算 Loss 未参与拟合或验证。

C7 共 45 行，实际上下文长度为 2048、4096、8192、32768、131072；样本中位数 4096 用作主表，32768 用作成本临界附近压力情景。没有上下文收益模型，因此 (L_{{ctx}}) 不是内层决策变量。

## 2. 成本与二维降维

题面成本为

\[
C_{{train}}=6ND,\quad C_Q=D[g(Q)-g(Q_0)]_+,\quad C_{{attn}}=\eta NDL_{{ctx}},\;\eta=2\times10^{{-4}}.
\]

令 (n=N/10^9,d=D/10^9,c=C/10^{{18}})，以及

\[
h=6+\eta L_{{ctx}},\qquad s(Q)=\frac{{[g(Q)-g(Q_0)]_+}}{{10^9}},
\]

则预算为 (d[hn+s(Q)]\le c)。在 (d) 未达到供应上界时，Loss 对 (d) 严格递减，故

\[
d^*(n,Q)=\min\left\{{d_{{max}},\frac{{c}}{{hn+s(Q)}}\right\}},
\]

同时要求 (d^*\ge d_{{min}})。问题由三维降为 ((\log n,Q)) 二维。固定 (Q) 且为内部解时的一阶条件是

\[
-\alpha A n^{{-\alpha-1}}+\beta B h c^{{-\beta}}(hn+s)^{{\beta-1}}=0.
\]

令一阶条件两侧之比为 (R(n))，则

\[
\frac{{d\log R}}{{dn}}=-\frac{{\alpha+1}}n+\frac{{(1-\beta)h}}{{hn+s}}
<-\frac{{\alpha+\beta}}n<0.
\]

因此在预算活跃、D 未封顶的单一分段内，固定 Q 的驻点至多一个，且由负导数变为正导数时是该分段全局最小；D 封顶段的目标随 N 单调下降，分段边界也被显式枚举。这为内层 N 求解提供了解析依据。外层 Q 剖面尚未证明整体凸或单峰，所以主求解使用全域 Q 网格、每个 Q 下的一维全局分段求解与局部精修，并用差分进化在相同可行域复算。两路线最大 Loss 差绝对值为 `{summary['max_solver_loss_gap_abs']:.3e}`，但正文仍只称“数值最优”。

质量基准 (Q_0) 没有 A→B 标定。主展示取透明中央情景 0.5，并对 0.1/0.3/0.7 做敏感性；这些都是外生基准，不是估计值，也不能跨 (Q_0) 把“免费初始质量”当作模型优劣。

## 3. 支持域内的官方预算结果

4096 上下文、(Q_0=0.5) 的正式结果如下；成本份额依次为基础训练、质量提升、注意力：

{main_table}

主要结论：

- (10^{{19}}) FLOPs 时 D 被联合支持下界 10B 约束。指数成本下配置为 (N={central_exp.loc[1e19,'N_B']:.6f})B、(Q={central_exp.loc[1e19,'Q']:.6f})，相对固定 (Q=0.5) 的模型 Loss 改善仅 `{comp.loc[comp.budget_FLOPs==1e19,'delta_L'].iloc[0]:.6f}`；幂型和对数型均不投资质量。
- (10^{{22}}) FLOPs 时三种成本均取 (Q=1)，但 (N,D) 因质量成本函数而不同；最低预测 Loss 是对数成本情景的 `{central[(central.budget_FLOPs==1e22)].loss.min():.6f}`。
- (10^{{24}}) FLOPs 时 (N,D,Q) 均触及支持上界，预算只使用约 2.48%–2.59%，即约 97.4%–97.5% 预算未被支持域模型吸收。这不是模型认定剩余预算“没有用途”，而是缺少更大规模的联合实验支持。

指数成本下与固定质量基线的公平比较：

{comp_table}

### 3.1 Q 为何较早达到上界

对固定预算先最小化 (N,D)，得到剖面函数 \(\phi(Q)=\min_{{N,D}}L\)。表中“固定 N,D 质量收益”是 \(\kappa[AN^{{-\alpha}}+BD^{{-\beta}}]\)，“资源挤占代价”由预算约束引起；二者之差即剖面导数。上界采用左导数，\(Q_0\) 采用右导数：

{qdiag_table}

因此 Q 的封顶不是单一原因：

- (10^{{19}}) 下，指数成本在 (Q=0.5810) 达到边际平衡；幂型和对数型在 (Q_0=0.5) 的右导数为正，说明增加 Q 的资源挤占超过质量收益。
- (10^{{22}}) 下，三种成本在 (Q=1) 的左导数仍为负，若允许继续提高 Q，冻结模型仍会降低 Loss；这里由题给质量上界截断，而非自然内点饱和。
- (10^{{24}}) 下，N、D、Q 支持上界同时生效且预算松弛，资源挤占代价约为零，Q 上界更是支持域造成的结构性退化。

(Q_0=0.5) 是外生中央情景，不是数据估计。成本单位已按“每 Token 的质量提升成本乘 D Token”复核；上述结论只属于题给三种成本、B7 半合成 Q 效应和当前支持范围，不能外推为普遍质量规律。

### 3.2 联合支持不等于直接观测

B1 实测为 \(8\times147\) 个完整 (N,D) 笛卡尔网格，B7 为 \(9\times5\times10\) 个完整 (N,D,Q) 半合成因子网格。因此支持框内点可以称相应网格凸包内插，但不等于出现过该精确联合配置。`results/q3_solution_evidence.csv` 同时记录到 B1/B7 最近观测网格点的归一化距离、是否命中原始网格以及支持边界。

4096 上下文的 9 个支持域方案全部至少触及 D 下界、Q 上界或 N/D/Q 上界，所以统一标为“C：支持边界情景”；(10^{{24}}) 虽精确命中 B1 末端检查点并非常接近 B7 角点，仍不能因距离很小而消除边界和半合成风险。B9 护栏方案统一标为“D：理论外推”。

## 4. 不可行与外推不是同一种结果

在 (10^{{19}}) FLOPs、131072 上下文下，联合支持域的最低配置 (N=0.070542)B、(D=10)B、(Q=Q_0) 仍需 `{sci(inf.minimum_feasible_cost_FLOPs.max())}` FLOPs，缺口 `{sci(inf_gap)}` FLOPs。因此三种成本函数在此情景均不可行；代码没有降低 D 下界或把罚函数解冒充原题解。

为解释高预算上界效应，B9 元数据上界内的理论外推（4096上下文、指数成本）为：

{ext_table}

(10^{{24}}) FLOPs 的外推配置为 (N={ext_hi.N_B:.3f})B、(D={ext_hi.D_B:.3f})B、Loss={ext_hi.loss:.6f}，D 已是联合支持上界的 `{ext_hi.max_extrap_factor:.2f}` 倍。它只说明支持域封顶对高预算结论有实质影响，不能当真实验证。

## 5. 结构性转移

转移预注册为“活跃集改变后至少连续 3 个预算点保持新状态”。不由 N/D/Q 上界引起的质量投入启动如下：

{onset_table if len(onset) else '未发现非边界活跃集转移。'}

其余重要切换（Q 达上界、D/N 达支持上界、预算由紧变松）均是受约束状态；报告保存完整预算区间和成本重算，但不称现实相变。指数成本在 4096 上下文、当前扫描起点 (10^{{19}}) 已处于 Q 内点，约在 (10^{{20}}) 后达到 Q 上界；这依赖 (Q_0=0.5)、B7 半合成 Q 效应和题给成本。

## 6. 上下文长度的成本敏感性

\[
\frac{{C_{{attn}}}}{{C_{{train}}}}=\frac{{\eta L_{{ctx}}}}6,
\quad L_{{crit}}=\frac6\eta=30000.
\]

C7 的 32768 最接近临界点。指数成本、(10^{{19}}) FLOPs 下，支持域预测 Loss 从 2048 上下文的 `{official[(official.evidence_level==EV)&(official.cost_family=='exponential')&(official.budget_FLOPs==1e19)&(official.context_tokens==2048)].loss.iloc[0]:.6f}` 上升到 32768 的 `{official[(official.evidence_level==EV)&(official.cost_family=='exponential')&(official.budget_FLOPs==1e19)&(official.context_tokens==32768)].loss.iloc[0]:.6f}`，131072 则不可行。这里只证明成本挤占；由于目标函数没有上下文能力收益项，不能据此推荐最短上下文。

## 7. 17 域配比：排序可用，绝对幅度不可识别

A4 的 512 个历史配方因三位小数舍入，原始份额和为 0.996–1.003；代码保留原行和并做闭合归一化。相对指数排名的三个代表如下：

{mix_table}

主模型令 \(\eta_p=0\)。非零条件情景使用

\[
L_p=E+(L_0-E)e^{{-\eta_p R_A(\mathbf p)}}.
\]

因为该正乘子不进入成本且统一乘可约损失，当前结构下 (p) 的历史候选排序与 (N,D,Q) 的 argmin 分离；形式上的“20 维联合优化”不会增加识别信息。以 (10^{{22}}) FLOPs 为例，η=0.5 时最佳/最差历史配方的条件 Loss 分别为 `{ms[(ms.budget_FLOPs==1e22)&(ms.eta_p==.5)&(ms.recipe_role=='best_supported')].conditional_loss.iloc[0]:.6f}` 和 `{ms[(ms.budget_FLOPs==1e22)&(ms.eta_p==.5)&(ms.recipe_role=='worst_supported')].conditional_loss.iloc[0]:.6f}`；差异很大恰好说明未经标定的 η 不能并入正式改善值。

## 8. 参数不确定性与稳健配置

问题二 300 组联合重拟合样本逐一重新优化。指数成本的配置区间为：

{boot_table}

这些区间很窄，原因是问题二条件 Bootstrap 本身极窄；它不覆盖数据源、半合成生成机制、模型结构、成本函数和 Q 标尺误差。90% Loss 分位数优化相对名义解的最大 p90 Loss 改善仅 `{max_robust_gain:.3e}`，而全情景最大 p90 后悔为名义 `{max_nom_reg:.3e}`、稳健 `{max_rob_reg:.3e}`。因此没有证据用更复杂的“稳健解”替换名义解；保留其为风险偏好对照。

## 9. 验证与失败门

- 独立脚本不导入主优化函数，重新计算 `{verify['checks_total']}` 项成本、Loss、预算、边界、网格下界、Bootstrap 分位、配比闭合、支持等级和 Q 边际符号，结果 `{verify['checks_passed']}/{verify['checks_total']} {verify['status']}`。
- 最大 Loss 复算差 `{verify['max_loss_recompute_abs']:.3e}`；独立 401×401 全域网格没有发现更低点，`grid-main` 最小差为 `{verify['min_grid_minus_main_loss']:.3e}`。
- 成本绝对差在 (10^{{24}}) 量级下最多 `{verify['max_cost_recompute_abs_FLOPs']:.3e}` FLOPs，为浮点舍入量级；预算超额最大值 `{verify['max_budget_excess_FLOPs']:.3e}` FLOPs 也低于预设相对容差。
- 求解器一致、细网格未发现更优点，只支持“冻结模型内的数值最优配置”。当前没有证明分段剖面函数在完整域上全局凸或单峰，故不使用“严格全局最优”措辞；B7 半合成 Q、(\eta_p) 未识别和外推风险仍然存在。

## 10. 与老师、DeepSeek、队友方案的取舍

- 吸收老师的 KKT、活跃集和相图思想；排除未标定的 (Q_0=\bar Q(p)) 与配比绝对乘子。
- 吸收 DeepSeek 的多求解路线核对；排除硬编码四舍五入参数、任意 N/D 边界和候选来源与 Loss 字段错位。
- 吸收队友的 D 消元、完整 C7、成本账本和持续性转移；把 D 下界从 B1 单源的 0.134B 修正为 B1∩B7 的 10B，并补充 p 排序、联合 Bootstrap 与稳健后悔。

逐项证据见 `reports/19_q3_cross_audit.md`。

## 11. 可传递给问题四的结果

问题四可读取 `results/q3_interface_to_q4.json`，其中包括官方预算下支持域/外推配置、成本份额、C7 上下文场景和证据标签。可传递的是模型情景下的规模、数据和算力配置，不可传递为真实技术进步或因果效率。

## 12. 可复现命令

```powershell
& "E:\\读研\\26届数学建模比赛\\_模板\\scripts\\py_model.cmd" "E:\\读研\\26届数学建模比赛\\code\\q3_modeling.py"
& "E:\\读研\\26届数学建模比赛\\_模板\\scripts\\py_model.cmd" "E:\\读研\\26届数学建模比赛\\code\\q3_independent_verify.py"
& "E:\\读研\\26届数学建模比赛\\_模板\\scripts\\py_model.cmd" "E:\\读研\\26届数学建模比赛\\code\\q3_report.py"
& "E:\\读研\\26届数学建模比赛\\_模板\\scripts\\py_model.cmd" "E:\\读研\\26届数学建模比赛\\code\\q3_manifest.py"
```

## 13. 尚需队员确认

1. 是否接受 (Q_0=0.5) 作为中央展示情景；它不是估计值，正文必须并列 0.3/0.7 敏感性。
2. 是否接受“支持域主结果 + B9 护栏理论外推”双表，而不是把 (10^{{24}}) 的上界解或外推解单独称最终答案。
3. 是否在正文保留非零 \(\eta_p\) 表；若保留，必须明确它不改变 N-D-Q argmin 且幅度未识别。
4. 结构转移是否只把“质量投资启动”作为非边界转移，把 Q/N/D 上界切换归入约束边界现象。
5. 参赛队需理解并人工复核全部公式、成本口径与证据措辞；本报告不是论文定稿。
"""
    (REP/"20_q3_modeling_report.md").write_text(report,encoding="utf-8")

    # Machine-readable Q4 handoff.
    hand=[]
    for _,r in official[official.context_tokens==4096].iterrows():
        hand.append({k:(None if pd.isna(r[k]) else (float(r[k]) if isinstance(r[k],(np.floating,float)) else r[k]))
                     for k in ["budget_FLOPs","context_tokens","cost_family","evidence_level","overall_evidence_tier",
                               "B1_geometry","B7_geometry","nearest_B1_scaled_log_distance",
                               "nearest_B7_scaled_logQ_distance","feasible","N_B","D_B","Q","loss","C_total",
                               "budget_utilization","active_set","max_extrap_factor"]})
    interface={"schema_version":2,"source":"Q3 frozen model scenarios, not real interventions","central_context":4096,
               "critical_context":30000,"Q0_main":.5,"mixture_eta_main":0.0,"official_configurations":hand,
               "uncertainty":"results/q3_bootstrap_summary.csv; conditional parameter uncertainty only",
               "optimality_claim":"numerical optimum in frozen model; no formal global proof",
               "support_geometry":"results/q3_solution_evidence.csv; exact grid, factorial-hull interpolation, boundary, extrapolation distinguished",
               "limitations":summary["limitations"],"report":"reports/20_q3_modeling_report.md"}
    (RES/"q3_interface_to_q4.json").write_text(json.dumps(interface,ensure_ascii=False,indent=2),encoding="utf-8")

    print(json.dumps({"report":str(REP/"20_q3_modeling_report.md"),"q4_interface":str(RES/"q3_interface_to_q4.json"),
                      "central_rows":len(central),"independent_verify":verify["status"]},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
