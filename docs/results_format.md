# 正式五折结果与公开边界

正式运行必须带 `protocol=resnet9-v1`、`purpose=formal`、实验编号、fold、实际配置、清单／权重哈希和源码指纹。`completed` 表示实际运行成功；fixture、smoke、资源测速和历史模型不进入本轮成绩表。

| 产物 | 含义 |
|---|---|
| `run.json`、`config.json` | 状态、来源、输入、选模规则和环境 |
| `history.json` | 每轮损失与指标、样本顺序指纹、时间和显存 |
| `best.pt`、`last.pt` | 最佳模型与最后完整 epoch 的恢复状态，留在集群 |
| `predictions.csv` | 最佳 epoch 的 sample_id＋九类概率 |
| `evaluation/` | 每折指标、每类统计及混淆矩阵 |
| `summary/cv_summary.json` | 五折均值、样本标准差、pooled OOF、预定 refit 轮数 |
| `summary/oof_predictions.csv` | 每个开发病例恰好一次的折外概率 |

指标存储为 0～1；文档百分比为乘以 100 后显示。Macro F1 固定平均九类，无 support 类别仍保留；argmax 同分取类别顺序中第一类。样本标准差不是置信区间，OOF 指标不是五折均值。

`reports/experiments/20261029_core/` 下 `SUMMARY.md`、`summary_metrics.csv`、`fold_metrics.csv`、`public_summary.json` 和 `summary_comparison.png` 是可公开聚合结果。C 还可按交接清单领取 pooled OOF 的每类指标和混淆矩阵，不包含逐病例 ID 或个人部署路径。

原始 run／环境／日志、逐病例预测和标签、完整 `cv_summary.json`、原始 `metrics.json` 留在 private，其中可能包含集群路径；不能整目录复制到 public。来源 SHA-256 可公开。

`reports/experiments/experiment_status.csv` 覆盖 E00～E09 各五折。E00～E02 completed，E03～E09 planned 且成绩为空。`collect_results.py` 从完成折的概率重新计算指标；`summarize_cv.py` 拒绝缺折、混合版本、smoke 或 OOF 覆盖错误。

29 日另核对 30 个 checkpoint 的轮数、优化器步数和冻结参数，并重载 15 个 best checkpoint 复算验证概率，结果一致。完整证据保存在 private preparation 记录。CV 是开发选模估计；E02 均值高于 E00 不表示完成了显著性检验。refit 和 holdout 尚未执行。组员人工复核状态与自动验收分开记录。
