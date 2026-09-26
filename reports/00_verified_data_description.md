# F 题可信数据说明（由 13 张可见截图重建）

## 1. 核验边界

13 张 `数据说明1.png`—`数据说明13.png` 已全部逐页核验。未读取原始《数据说明》PDF 或其文字层。本说明只重建截图肉眼可见内容，并用完整 `F题.zip` 的实际文件补充存在性、行列数和可解析性。

## 2. 截图逐页覆盖

| 页 | 可见内容 |
|---:|---|
| 1 | 资料性质、编号规则、A1–A2 开始 |
| 2 | A3–A16 文件对照 |
| 3 | A17–A18、B1–B11 |
| 4 | B12、B 组说明、C1–C10 |
| 5 | 读取规则、单位换算、强制使用清单 |
| 6 | 跨附件字段兼容与连接键 |
| 7–9 | A 组来源、规模、字段、22 个质量信号和 17 域说明 |
| 9–11 | B 组来源、文件结构、单位与标度律数据汇总 |
| 11–13 | C 组来源、规模、字段、用途和 C8 完整性说明 |

核验状态：**13/13 完成**。

## 3. 数据性质词典

- 真实：直接观测或公开数据；
- 半合成：真实数据校准后叠加噪声；
- 插值：检查点插值生成；
- 子集：取自其他表的训练集子集，不是独立实验；
- 外推/估算：模型外推或文献估算，不是直接观测；
- 混合：来源和径不完全统一，必须分层使用；
- 参考/说明：用于字段或映射解释，不能替代主数据。

## 4. 编号与文件对照

### 附件 A：数据质量与领域配比

| 编号 | 文件 | 截图定义 | 性质 |
|---|---|---|---|
| A1 | `slimpajama_quality_signal_sample.jsonl.xz` | 51,230 条、27 字段；22 个质量指标和内容/来源辅助字段 | 真实 |
| A2 | `slimpajama_quality_extended/arxiv_*.jsonl.xz` | 17,523 条、24 字段；arxiv 扩展质量信号 | 真实 |
| A3 | `slimpajama_quality_extended/github_*.jsonl.xz` | 203,752 条、24 字段；github 扩展质量信号 | 真实 |
| A4/A5 | `train_mixture_1m.csv` / `train_pile_loss_1m.csv` | 512 组 1M 训练配比与 13 域 Loss | 真实训练 |
| A6/A7 | `test_mixture_1m.csv` / `test_pile_loss_1m.csv` | 256 组 1M 检验配比与 Loss | 真实检验 |
| A8/A9 | `test_mixture_60m.csv` / `test_pile_loss_60m.csv` | 256 组 60M 检验配比与 Loss | 真实检验 |
| A10/A11 | `test_mixture_1B.csv` / `test_pile_loss_1B.csv` | 64 组 1B 检验配比与 Loss | 真实检验 |
| A12/A13 | `est_mixture_10b.csv` / `est_pile_loss_10b.csv` | 63 组 10B 配比子集与外推 Loss | 子集 / 外推 |
| A14/A15 | `est_mixture_70b.csv` / `est_pile_loss_70b.csv` | 63 组 70B 配比子集与外推 Loss | 子集 / 外推 |
| A16 | `domain_mapping_guide.csv` | 17 个配方域到质量域的参考映射 | 参考 |
| A17 | `regmix_domain_summary.csv` | 域摘要 | 真实 |
| A18 | `regmix_domain_sample.jsonl.xz` | 138,034 条域原始文本样例 | 真实、可选辅助 |

A1 的 22 个质量信号中，14 个是标量，8 个是多维列表；多维列表必须先压缩为有明确含义的标量再建模。所有指标处理后必须统一为“越高越好”。A1 覆盖 7 个质量域；17 个配方域通过 A16 参考映射连接，但映射规则仍需队伍论证。

### 附件 B：标度律数据

| 编号 | 文件 | 截图定义 | 性质 |
|---|---|---|---|
| B1 | `pythia_training_log_existing.csv` | 1,176 行 Pythia 训练日志 | 真实、主拟合 |
| B2 | `cerebras_training_log.csv` | 1,029 行 Cerebras 轨迹 | 半合成、族外验证 |
| B3 | `training_trajectories/*.csv` | 8 条轨迹，每条 500 行 | 插值、轨迹验证 |
| B4 | `scaling_baseline.csv` | 57 个跨族收敛点 | 真实 |
| B5 | `published_scaling_data.csv` | 44 个已发表标度律基准点 | 真实/文献 |
| B6–B8 | `supplementary_NQ_experiment*.csv` | 360、450、1,704 个 N–D–Q 点 | 半合成；B8 含外推 |
| B9 | `supplementary_large_models.csv` | 132 个百亿参数以上模型元数据 | 真实 |
| B10 | `supplementary_large_baseline.csv` | 128 个大模型 Loss | 估算 |
| B11 | `open_model_family_metadata.csv` | 18 行模型族元数据 | 真实、辅助 |
| B12 | `pythia_checkpoint_index.csv` | 1,386 行检查点索引 | 真实、辅助 |

单位：`N_params_B` 和 `D_tokens_B` 均以十亿为单位；代入 `C=6ND` 时恢复到实际数量级。`C_FLOPs_1e21` 的单位为 `10^21 FLOPs`。

### 附件 C：能力评测与技术演进

| 编号 | 文件 | 截图定义 | 性质 |
|---|---|---|---|
| C1 | `leaderboard_cleaned.csv` | 4,576 行、12 列，6 维 Benchmark | 真实、核心 |
| C2 | `leaderboard_enhanced.csv` | 4,576 行、15 列，增加 Epoch AI 字段 | 真实；可替代 C1 |
| C3 | `leaderboard_extended_timeseries.csv` | 4,599 行、11 列，2019–2025 时序 | 混合，含历史模拟 |
| C4 | `epoch_all_ai_models.csv` | 3,523 行、57 列模型元数据 | 真实、宏观辅助 |
| C5 | `loss_benchmark_bridge.csv` | 43 行桥接 | 混合来源 |
| C6 | `loss_benchmark_bridge_expanded.csv` | 75 行扩展桥接 | 混合来源、主用候选 |
| C7 | `model_architecture_metadata.csv` | 45 行、7 列架构元数据 | 真实，问题三/四 |
| C8 | `detailed_results/` | 1,863 个目录、1,958 个逐任务 JSON，约 0.23 GB | 真实；4 个截断 |
| C9 | `data/*.parquet` | 4,576 行、36 列；与 C1 核心六维等价并含原始附加列 | 真实、可选 |
| C10 | `pythia_*_eval_details/` | 7 个评测说明目录 | 说明 |

C8 全量解析确认 1,954 个 JSON 有效、4 个截断；1,860 个目录至少有一个有效 JSON。问题四必须做至少一项逐任务聚合，不能只使用 C1/C2 汇总表。

## 5. 读取与使用规则

- XZ/JSONL 必须流式读取，不应整体解压进内存；本轮已全量执行。
- A4–A15 配比与 Loss 按 `index` 成对连接；配比是 17 维，Loss 是 13 域。
- A6–A11 是检验集；A12–A15 是外推/子集，只支持外推稳健性讨论。
- B6–B8 必须标注半合成，B10 必须标注估算。
- C1、C2、C9 是同一核心记录的不同版本，不能重复计样本。
- C5/C6 的 Loss 来源和可比性等级必须进入建模权重、分层或敏感性分析。
- 实际文件与说明冲突时以实际附件为准，但语义变化须登记，不能静默更改。
