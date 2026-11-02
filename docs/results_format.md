# 正式实验结果与公开边界

当前真实实验为 E00～E10 共 55 折 completed，各组 OOF 恰好覆盖 3,968 例。
E10 主模型／E07 对照 refit 已完成，固定 holdout 各 1,037 例也已评估。
这里的 completed 描述已验收实验，不表示 C 单独分支已通过 A／B／C 集成检查。

## 公开结果入口

| 路径 | 内容 |
|---|---|
| `reports/experiments/experiment_status.csv` | 55 行正式运行状态，缺失运行不会由收集脚本伪造为完成 |
| `reports/experiments/20261031_e10/summary_metrics.csv` | 全部十一组五折均值、样本标准差、pooled OOF、条件 refit 轮数 |
| 同目录 `fold_metrics.csv` | 55 折实际 epoch、早停原因、训练／验证数量和指标 |
| 同目录 `public_summary.json`、`SUMMARY.md`、`cv_comparison.png` | 当前公开聚合说明、训练来源指纹及比较图 |
| 同目录 `e10_vs_e07_folds.csv`、`e10_vs_e07_per_class.csv` | 逐折与 pooled OOF 逐类对照 |
| `reports/experiments/20261030_expansion/E03/summary/pooled_oof/` 至 `E09` | 每组逐类表与原始／归一化混淆矩阵 CSV／PNG |
| `reports/experiments/20261031_e10/E10/summary/pooled_oof/` | E10 同类聚合表图 |
| `reports/holdout/20261031_holdout_e10_e07_v1/` | 双模型最终总体／逐类指标及混淆矩阵，只公开聚合内容 |

[29 日核心结果](../reports/experiments/20261029_core/SUMMARY.md) 保留原历史快照。
最新十一组表已包含 E00～E09，不把十组旧总表再拼入造成重复；旧快照中 holdout 未执行
是当时状态，当前状态以 [CV 摘要](../reports/experiments/20261031_e10/SUMMARY.md)
和 [最终评估](../reports/holdout/20261031_holdout_e10_e07_v1/SUMMARY.md) 为准。

## 指标解释

分数保存为 0～1，百分比展示乘以 100。Macro F1 固定平均九类，无 support 类别仍保留；
argmax 同分取固定类别顺序中第一类。样本标准差不是置信区间，pooled OOF 不是五折均值。
CV 验证折参与选 epoch，属于开发选模估计；在线训练诊断不是最终测试成绩。
只有 E10／E07 实际 refit；其他配置的预定轮数不表示已训练。

## 本地完整运行产物

| 产物 | 用途及公开处理 |
|---|---|
| `run.json`、`config.json`、`history.json`、`metadata.json` | 完整配置、输入、环境及训练记录留在内部 |
| `best.pt`、`last.pt` | 权重与恢复状态不公开 |
| `predictions.csv`、`predictions_aligned.csv`、`oof_predictions.csv` | 新逐病例概率／标签不公开 |
| `summary/cv_summary.json` | 原始完整五折汇总留在内部；公开版仅提取允许的聚合字段 |
| `evaluation/metrics.json` | 只发布经来源路径脱敏的最终 holdout 聚合版，保留数值／策略／来源哈希 |
| `per_class_metrics.csv`、`confusion_matrix*.csv/png` | 允许公开的聚合结果 |

`public_summary.json` 是当前状态的公开导出，不冒充训练时的原始快照。
已脱敏 metrics JSON 与原始 JSON 不同字节；来源哈希仍标识内部保存的输入。
源码指纹属于实际训练快照，不能替换成较晚 public 提交的哈希。

收集与汇总真实运行需要本地清单和逐病例产物。仅有公开表图不能重新计算逐病例概率，
也不能据此声称独立完整复现已经完成。组员人工复核和候选集成检查分别记录在
[贡献说明](contributions.md)。
