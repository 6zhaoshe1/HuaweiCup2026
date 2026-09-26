#!/usr/bin/env python3
# AI assistance: OpenAI Codex, OpenAI, 2026-09-24; team review required.
"""Build revised reports from versioned result artifacts; no manually copied experiment numbers."""
import json
import hashlib
import platform
import sys
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';REPORT=ROOT/'reports'


def table(df,columns=None):
    df=df if columns is None else df[columns]
    def fmt(v):
        if isinstance(v,(float,np.floating)):return f'{v:.7g}' if np.isfinite(v) else '不适用'
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+''.join(
        '| '+' | '.join(fmt(v) for v in row)+' |\n' for row in df.itertuples(index=False,name=None))


def main():
    read=lambda name:pd.read_csv(RES/name)
    load=lambda name:json.loads((RES/name).read_text(encoding='utf-8'))
    meta=load('q2_interface_to_q3.json');classic=read('q2_classic_validation.csv')
    qcv=read('q2_quality_model_comparison.csv');nested=read('q2_quality_nested_metrics.csv')
    folds=read('q2_quality_nested_folds.csv');source=read('q2_source_heldout_metrics.csv')
    ci=read('q2_classic_parameter_intervals.csv');marg=read('q2_marginal_and_substitution.csv')
    opt=read('q2_compute_optimal.csv');unc=load('q2_uncertainty_summary.json')
    verify=load('q2_independent_verification.json');pb=read('q2_p_bridge_summary.csv')
    timestamp=datetime.now().isoformat(timespec='seconds')
    # Final schema normalization also supports existing v1 calibration artifacts.
    source_detail=read('q2_source_calibration_predictions.csv.gz')
    source_detail['evidence_level']=source_detail.dataset.map({'B2':'semi_synthetic','B4':'real_observational',
                                                             'B5':'real_observational','B10':'extrapolated'})
    if source_detail.evidence_level.isna().any():
        raise ValueError('Unclassified source in calibration artifact')
    source_detail.to_csv(RES/'q2_source_calibration_predictions.csv.gz',index=False,compression='gzip')
    # Freeze reproducibility metadata without hashing the multi-GB original archive.
    with zipfile.ZipFile(ROOT/'第二十三届中国研究生数学建模竞赛 - 中文题目/中文题目/F题.zip') as z:
        inputs=[]
        for info in z.infolist():
            if info.filename.startswith('real_attachments/B_scaling_laws/') and info.filename.endswith('.csv'):
                data=z.read(info)
                inputs.append({'zip_member':info.filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                 'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,
                 'sklearn':sklearn.__version__,'seed':42,'B_input_members':inputs,'evidence_level':'metadata'}
    (RES/'q2_reproducibility_environment.json').write_text(json.dumps(environment,ensure_ascii=False,indent=2),encoding='utf-8')
    meta['schema_version']=2
    meta['callable']='code/q2_predict.py:Q2Predictor.predict'
    meta['quality_model']['evidence']='B7 semi-synthetic; outer5/inner4 ND retrospective selection validation; unknown shared generator dependence'
    meta['uncertainty']={'artifact':'results/q2_joint_bootstrap.csv','kind':'conditional_parameter_interval_not_observation_prediction_interval',
                         'scope':unc['scope']}
    (RES/'q2_interface_to_q3.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    summary_main=load('q2_summary.json')
    summary_main['revision_results']={'metrics':'results/q2_quality_nested_metrics.csv',
                                      'source_holdout':'results/q2_source_heldout_metrics.csv',
                                      'summary':'results/q2_revision_summary.json',
                                      'note':'original CV scores are selection comparisons; revised interpretation overrides v1'}
    (RES/'q2_summary.json').write_text(json.dumps(summary_main,ensure_ascii=False,indent=2),encoding='utf-8')
    disposition=[
        ('P0','主模型与 p 条件模型分离','采纳','main 不接受 p/非零 eta；p_scenario 要求显式 eta；主模型不能优化 p','code/q2_predict.py'),
        ('P0','Q1 与 B7 Q 标尺','采纳并收紧','拒绝 Q1 标尺输入；没有锚点就不标定；Q=1 仅是 B 模型参考约定','results/q2_interface_to_q3.json'),
        ('P0','B1 极低误差说明合成或平滑','不支持该推断','可见说明标为真实，manifest 标为 retained；生成处理链未知，不从残差判来源','results/q2_evidence_matrix.csv'),
        ('P0','B7/B8 同键冲突说明特定生成机制','部分采纳','冲突可复核，具体生成器未知；只证明不能作为一致版本拼接','results/q2_overlap_audit.csv'),
        ('P1','同估计器 Bootstrap','采纳','完整 N 轨迹、同对数 Huber/边界/12 起点，条件重拟合 Q，300 次','results/q2_joint_bootstrap.csv'),
        ('P1','参数相关性/边界/失败记录','采纳','记录条件数、收敛、边界命中、独特组数；不以窄区间证明真实精度','results/q2_joint_parameter_correlations.csv'),
        ('P1','只用目标源全量校准不算留出','采纳','B4 留族，B5 留来源及留族；对照未校准与训练均值；B5 增益不稳定','results/q2_source_heldout_metrics.csv'),
        ('P1','B7 同 CV 选模又报分','采纳','外 5 内 4 折 ND 分组，外层涵盖六候选选择，原 CV 降为比较表','results/q2_quality_nested_metrics.csv'),
        ('P1','B1 两个外推任务 RMSE 平均选模','采纳','留规模为主，后段为次；事后修订，不能称事前注册','reports/16_q2_revision_contract.md'),
        ('P1','B1 run_id 当训练运行','纠正潜在误用','逐行唯一，不是独立轨迹；B2 的 run_id 才对应七条轨迹','results/q2_independent_unit_audit.csv'),
        ('P1','等效 N 超大求根上界','采纳','解析判有限/极限/不可达，独立恒等式验证，标注外推','results/q2_marginal_and_substitution.csv'),
        ('P1','算力在范围内就算插值','采纳纠正','分别核验 N,D；并给观测矩形约束解，仍非联合支持证明','results/q2_compute_optimal.csv'),
        ('P1','OLS 接近 Huber 作为验收','采纳纠正','手写 Huber 目标重拟合/梯度、指标和约束独立复算；OLS 只作对照','results/q2_independent_verification.json'),
        ('P1','联合 Bootstrap 是完整不确定性','限定采纳','假定 B1/B7 重抽样独立；未知共享生成机制、源差异和重新选模均未覆盖','results/q2_uncertainty_summary.json'),
        ('P2','Codex 没有领域组合效应，必须换 Lasso','不采纳必要性论断','Q1 已有二次 ILR 原始域闭合组合情景；未比较 Lasso，不能说它更好或更差','results/q1_quadratic_ilr_interactions.csv'),
        ('P2','B6→B7 新 Q 档属于新独立实验','纠正','共享 ND 的半合成 Q 插值检查，不是外源泛化','results/q2_split_manifest.csv'),
        ('P2','图6过宽，图7 ND 乘积混淆','采纳','等 Loss 限于 B1/B7 交集矩形；替代按 D 分面、N 横轴并画联合区间与外推标记','figures/q2_07_quality_substitution.pdf'),
        ('P2','统一划分、逐层残差、证据分层','采纳','统一 manifest，B1 按规模、B7 按 N/D/Q 输出残差统计','results/q2_residual_strata.csv'),
        ('P2','不同 eta 的任意网格代表估计区间','不采纳','0/.25/.5 仅情景偏好；无统计覆盖保证，不把 p 最优当实证建议','code/q2_predict.py'),
        ('P1','新增发现：多起点选择目标错位','主动修复','Huber 求解改按 Huber cost 选择起点，不再按普通 SSE','code/q2_modeling.py'),
    ]
    disp=pd.DataFrame(disposition,columns=['priority','feedback','decision','reason','evidence_path']).assign(evidence_level='metadata')
    disp.to_csv(RES/'q2_review_disposition.csv',index=False,encoding='utf-8-sig')
    compatible='''# Q2 来源与质量标尺兼容性

- 证据：`reports/00_verified_data_description.md`（截图 3–6、9–11），ZIP 内 `source_manifest.json`，原始表全量读取。
- B1 被可见说明列为真实来源。manifest 为 EleutherAI Pythia / retained from current contest data；原始日志到比赛表的平滑或重建步骤未披露。低残差不证明合成。
- A 是多验证领域的 RegMix Loss，B1 是 Pythia 轨迹 `val_loss`。具体验证集合、tokenizer 版本、token 归一化、聚合权重的一致性未建立。字段都叫 Loss 不构成合并依据。
- B4/B5 的族/文献分组已用于留出；这检验数字形状的迁移，不补足物理测量口径。不能将校准系数当跨 tokenizer 的通用单位换算。
- B7 是半合成 Q_score；Q1 综合质量分与之没有同对象双标尺锚点，不能以同在 [0,1] 或同向就认定相同。
- B1 不带 Q，基线的 Q=1 是参考规范化，不证明真实训练语料质量为 1。B6/B7 生成来源提到 Pythia，因此两模块不被视作独立实验证据。
- B6 为 B7 子集；B7/B8 同键冲突，禁止池化。B3 插值、B10 估算只作辅助。
- 若要真正识别 p 或标定 Q：需要相同 tokenizer/验证集/训练条件下的联合 (N,D,Q,p) 变化或可靠匹配锚点，之后另设组外验证。
'''
    (RES/'q2_source_compatibility.md').write_text(compatible,encoding='utf-8')
    formula=r'''
\[
R=AN^{-\alpha}+BD^{-\beta},\quad g(Q)=1+\kappa(1-Q),\quad L_{main}=E+Rg(Q).
\]
\[
L_{scenario}=E+Rg(Q)\exp[-\eta_p R_A(\mathbf p)].
\]

N、D 单位是十亿；Q 只能使用 B7_Q_score。主模式没有可识别的 p 响应；条件模式的 eta 是用户假设，不是数据估计。Q1 分面或综合分不得直接代入。

正式估计为带边界、多起点的对数 Huber：残差 `log(pred)-log(y)`，转折尺度 0.01，12 起点；多起点以同一 Huber cost 选择。经典参数边界依次为 `[0.05,1e-4,0.02,1e-4,0.02]` 到 `[0.999 min(y),5000,1.5,5000,1.5]`。质量线性系数边界 `[0,20]`。数值边界是求解约束，不是科学先验置信区间。

配比指数：由问题一对数平移模型预测 13 域 Loss，各域用 A4 预测的均值/标准差标准化，均值聚合后再按 A4 聚合分的标准差缩放并取负。保存全部参数，不把“平均 p 的预测”误作均值预测。A 源描述性斜率不等于 B 源 eta。
'''
    report=f'''# F 题问题二：修订后的模型、证据与接口

生成时间：{timestamp}。数字由 `code/q2_revision_report.py` 从结果工件生成。
本报告替代旧版问题二结果解读；历史版本保存在 `_tmp/q2_before_revision_20260924/`。不是参赛论文定稿。

## 1. 结论

保留简单的可约项线性质量修正，没有为追求复杂度更换核心结构。新增外层验证覆盖了选模流程；跨来源校准并非普遍成功。不能宣称完整四变量规律已获真实实验验证。
{formula}
## 2. 数据与可信度

`results/q2_evidence_matrix.csv` 覆盖 B1–B12 的数量、范围、性质和来源。B1 run_id 逐行唯一，按完整 N 轨迹切分；B2 按运行及早晚 D 切分；B7 按 ND 切分所有 Q 档。B6 不与 B7 重复拟合；B3、B8、B10 不作为新的真实独立验证。
{table(read('q2_data_inventory.csv'),['dataset','rows','evidence_level'])}
Q1→B7 的标尺映射、A/B Loss 绝对兼容性均未确认；来源证据详见 `results/q2_source_compatibility.md`。不读取原 PDF 隐藏层。

## 3. 经典律与不确定性

留一规模是本轮主任务，后段 Token 为次验证；随机行只诊断。该规则是阅过旧结果后的修订，不称预注册或全新盲测。
{table(classic,['model','split','n','rmse','r2','spearman','bias'])}
正式参数及同估计器重抽样区间：
{table(ci,['parameter','point_estimate','ci2p5','ci97p5','successful_replicates'])}
完整轨迹重拟合后，每次重抽 B7 ND 组并条件重拟合 Q。成功 {unc['bootstrap_success']}/{unc['bootstrap_total']}，边界命中 {unc['active_bounds']}。
这些是固定模型下的**条件参数区间**，不是未来观测预测区间。仅八条 B1 轨迹，区间可能过窄；假设两表重抽样扰动独立，未覆盖未知共同生成依赖、模型选择、架构/tokenizer、残差、p 或 Q 标尺误差。

## 4. 质量模型选择与外层评价

保留原同划分候选比较，但不把用来选模的 CV 分数视作额外独立证据：
{table(qcv[qcv.split=='grouped_ND_5fold'],['model','rmse','r2','spearman'])}
新增外 5 内 4 折 ND 分组，内层只访问外层训练部分。五个外层折的选择记录：
{table(folds,['outer_fold','selected','n','rmse','r2'])}
整个选模流程的外层表现：
{table(nested,['protocol','n','mae','rmse','r2','spearman','bias'])}
与原选中模型的五折值相同，是因为所有外层都选中了同一结构，且外层划分相同，不是另一份独立数据再次验证成功。评价仍限于 B7 半合成数据和固定 B1 模块，未知生成机制可能带来结构乐观性。
B6 到 B7 新 Q 档是共享 ND 的质量插值；B8 仍迁移失败：
{table(qcv[qcv.split.str.startswith('B8_')],['split','n','rmse','r2','spearman','evidence_level'])}
同键冲突证明不能混合版本，不能由此断言具体生成算法。

## 5. 跨来源校准的真实检验

目标源全量仿射拟合只作样本内诊断。新增留出结果与未校准、训练均值比较：
{table(source,['source','grouping','model','n','rmse','r2','spearman','bias'])}
B4 校准保留了目标组外预测价值；但 B4 含 Pythia，固定 B1 模块已使用该模型族，不能称整个流程从未见过该族。上表另列从校准和评价中同时剔除 Pythia 的敏感性结果。B5 留来源与未校准很接近、留族未稳定改善。因此保留来源分层，不将两条回归式相加，不宣称一个仿射变换就解决所有 Loss 口径。

## 6. 配比与跨问接口

{table(pb,['scale','n','spearman_p_index_vs_better_loss','eta_A_within_source'])}
这张表的 A 源 eta 与相应标准化 Loss 在同一来源拟合，只是描述性斜率；排序不是因果效应，A6/A8/A10 已在问题一使用，不称全新盲测。

`code/q2_predict.py` 提供 `Q2Predictor.predict` 与 `predict_loss`。主模式只收 N,D,Q；条件模式需要 `mode='p_scenario'`、显式 eta、17 维单纯形 p。顺序见 JSON 的 domain_names；不静默归一化非法输入。未声明 B7 标尺、错误单位、越界且未显式允许外推都会拒绝。
接口返回 Loss、解析偏导、支持域标记、条件参数区间和限制警告；不对 p 宣称联合支持，不把 eta 网格当置信区间。示例：

```python
from q2_predict import predict_loss
out = predict_loss(1.0, 100.0, 0.7, q_scale='B7_Q_score')
```

## 7. 边际效应与可行性

''' + r'''
\[
L_N=-\alpha AN^{-\alpha-1}g(Q),\quad L_D=-\beta BD^{-\beta-1}g(Q),\quad L_Q=-\kappa R.
\]
取 `s=[R g(Q+ΔQ)/g(Q)-BD^{-β}]/A`，则 s>0 时 `N_eq=s^(-1/α)`；s=0 只能 N→∞ 达到，s<0 不可达。不再用十万十亿参数的任意根区间。工程上限未给定，故未评价实际工程可行性。

固定算力 `C=6·10^18·N_B·D_B`，令 `K=C/(6·10^18)`，无约束解为
\[
N^*=[\alpha A K^\beta/(\beta B)]^{1/(\alpha+\beta)},\quad D^*=K/N^*.
\]
矩形约束下，N 的可行区间为 `[max(Nmin,K/Dmax),min(Nmax,K/Dmin)]`，将唯一无约束极小点投影到该区间后恢复 D。
''' + f'''
共 {len(marg)} 个等效情景，其中 {unc['extrapolated_equivalent_scenarios']} 个解超出 B1 或 B7 的 N/D 支持矩形；不能再称全部“范围内情景”。有限倍率范围 {marg.equivalent_N_multiplier.min():.5f}–{marg.equivalent_N_multiplier.max():.5f}，包括外推解，不是可靠工程推荐区间。
30 个经典算力基准点中，有 {unc['unsupported_unconstrained_optima']} 个无约束最优点超出 B1 的 N 或 D 范围。已补矩形约束解并校验预算残差。矩形内不等于完整联合验证。
固定 Q,p 的乘子不会改变经典 N:D 最优比例；这里只是问题三参考基准，未含数据/质量/注意力成本，不能代替正式问题三优化。

## 8. 复核、图表与复现

独立数值实现复核：{sum(x['pass'] for x in verify['checks'])}/{len(verify['checks'])} 项通过。性质为作者的不同实现数值复算，不是外部独立科学审稿。检查 Huber 目标、导数、450 条预测真值/指标、划分隔离、解析替代、预算和输入门禁；移除原 OLS 系数差任意容差。
图 q2_01–q2_12 均来自实算，有 PDF/PNG；新增跨源留出和嵌套残差，修订等 Loss 范围、替代分面/区间、约束最优曲线。图—结果契约见三份 `q2_*figure_contract.csv`。

在项目根按序运行：

```powershell
& './_模板/scripts/py_model.cmd' code/q2_modeling.py
& './_模板/scripts/py_model.cmd' code/q2_revision_validation.py
& './_模板/scripts/py_model.cmd' code/q2_uncertainty.py
& './_模板/scripts/py_model.cmd' code/q2_independent_verify.py
& './_模板/scripts/py_model.cmd' code/q2_revision_report.py
& './_模板/scripts/py_model.cmd' code/q2_independent_verify.py
```

最后一次复核刷新报告及所有工件哈希。不要只重跑主脚本后读取旧的修订/区间结果。
主要工件：`q2_quality_nested_metrics.csv`、`q2_source_heldout_metrics.csv`、`q2_joint_bootstrap.csv`、`q2_marginal_and_substitution.csv`、`q2_compute_optimal.csv`、`q2_split_manifest.csv`、`q2_review_disposition.csv`、`q2_interface_to_q3.json`（均在 results）。

## 9. 未解决与停止点

Q1→B7 标尺、B 源配比幅度、跨来源测量口径、真实质量干预效应、半合成生成处理链仍未确认。未开展新核心模型定稿、问题三优化或论文编译；不能声称可提交。新增复杂模型不能消除这些识别缺口。参赛队需要决定是否接受带上述限制的主模型和条件接口，并对问题三成本/约束另行确认。
'''
    (REPORT/'15_q2_modeling_report.md').write_text(report,encoding='utf-8')
    review=f'''# 问题二队友意见复核与代码修订

更新时间：{timestamp}。读取 `队友工作区/问题二_待改进清单.md` 和 `Q2_revision_and_paper_guidance.md` 后独立判断。意见不是指令性事实；表中未采纳项说明原因。

{table(disp,['priority','feedback','decision','reason','evidence_path'])}

## 实验结论

{table(nested,['n','rmse','r2','spearman'])}

{table(source[source.model=='heldout_affine'],['source','grouping','rmse','r2'])}

完整公式、区间和负结果见 `reports/15_q2_modeling_report.md`。主参数不变不等于修订无效：本轮主要修正验证偏差、来源论断、外推标签和接口误用风险。

## 验收边界

数值复算已通过；尚未通过真实四变量因果/工程有效性验证。保留旧版备份，未改原始数据、问题一工件、队友代码和问题三文件。Q1 已有组合情景但有边界与双模型分歧，不能凭“有交互项”宣称更科学。
'''
    (REPORT/'17_q2_revision_review.md').write_text(review,encoding='utf-8')
    summary={'generated_at':timestamp,'nested_metrics':nested.iloc[0].to_dict(),'bootstrap':unc,
             'verification_checks_passed':sum(x['pass'] for x in verify['checks']),
             'dispositions':len(disp),'model_changed':False,'core_unresolved':verify['unresolved']}
    (RES/'q2_revision_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
