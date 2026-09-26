# Q2 来源与质量标尺兼容性

- 证据：`reports/00_verified_data_description.md`（截图 3–6、9–11），ZIP 内 `source_manifest.json`，原始表全量读取。
- B1 被可见说明列为真实来源。manifest 为 EleutherAI Pythia / retained from current contest data；原始日志到比赛表的平滑或重建步骤未披露。低残差不证明合成。
- A 是多验证领域的 RegMix Loss，B1 是 Pythia 轨迹 `val_loss`。具体验证集合、tokenizer 版本、token 归一化、聚合权重的一致性未建立。字段都叫 Loss 不构成合并依据。
- B4/B5 的族/文献分组已用于留出；这检验数字形状的迁移，不补足物理测量口径。不能将校准系数当跨 tokenizer 的通用单位换算。
- B7 是半合成 Q_score；Q1 综合质量分与之没有同对象双标尺锚点，不能以同在 [0,1] 或同向就认定相同。
- B1 不带 Q，基线的 Q=1 是参考规范化，不证明真实训练语料质量为 1。B6/B7 生成来源提到 Pythia，因此两模块不被视作独立实验证据。
- B6 为 B7 子集；B7/B8 同键冲突，禁止池化。B3 插值、B10 估算只作辅助。
- 若要真正识别 p 或标定 Q：需要相同 tokenizer/验证集/训练条件下的联合 (N,D,Q,p) 变化或可靠匹配锚点，之后另设组外验证。
