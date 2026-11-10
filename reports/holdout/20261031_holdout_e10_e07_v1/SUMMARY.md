# E10／E07 固定 holdout 最终结果

2026-10-31（America/New_York），两个最终 refit 模型在 H100 MIG 1g.10gb 上完成
同一固定 MILK holdout 1,037 例的评估，总用时 52 秒。采用固定九类 argmax，
E10 主模型／E07 对照在评估前已经确定。

| 模型 | refit 轮数 | Macro F1 | Accuracy | 平衡准确率（Macro Recall） | 正确数 |
|---|---:|---:|---:|---:|---:|
| E10（主模型） | 13 | 56.62% | 70.49% | 54.64% | 731/1037 |
| E07（对照） | 14 | 53.28% | 70.01% | 53.09% | 726/1037 |

E10 的 Macro F1 比 E07 高 3.33 个百分点，Accuracy 高 0.48 个百分点，多正确 5 例。这是同一 holdout 上的描述性差值，不是统计显著性结论；不据此重新训练、改配置或调换主模型／对照。

这些是最终 refit 模型的 holdout 指标，与 refit 训练日志里的约 86% 在线训练 F1 分开报告。原五折 CV 是开发选模估计（E10 均值 57.00%、E07 均值 54.70%）；此次未重跑 CV 或训练。

## 逐类 F1

| 类别 | Holdout 支持数 | E10 F1 | E07 F1 | 差值（百分点） |
|---|---:|---:|---:|---:|
| AKIEC | 61 | 27.12% | 38.46% | -11.34 |
| BCC | 504 | 83.27% | 82.70% | +0.57 |
| BKL | 109 | 43.22% | 36.11% | +7.10 |
| DF | 10 | 48.00% | 38.71% | +9.29 |
| INF | 10 | 30.77% | 26.67% | +4.10 |
| MEL | 90 | 60.24% | 56.80% | +3.44 |
| NV | 149 | 76.92% | 77.55% | -0.63 |
| SCCKA | 95 | 60.00% | 61.00% | -1.00 |
| VASC | 9 | 80.00% | 61.54% | +18.46 |

E10 在 6 类 F1 更高，AKIEC、NV、SCCKA 更低，不能称为所有类别均改善。DF／INF 各只有 10 例，VASC 只有 9 例，小类别分数对个别预测较敏感。固定 holdout 有历史使用记录，不能称整个项目从未使用过的外部盲测；没有患者 ID，不宣称患者级隔离。

## Refit 与评估方法

E10 的五折最佳 epoch 为 9、51、21、10、13，中位数 13；E07 为 48、14、40、9、14，
中位数 14。两者从 ImageNet V2 独立初始化，在 3,968 个 MILK 开发病例＋20,279 张 ISIC
训练图上按固定轮数训练；无验证早停，不从 CV checkpoint 续训。类别权重按完整训练侧
重新计算，E10 元数据也只在完整训练侧重新拟合。

四个 refit checkpoint 内容及加载核对通过。最终评估每个模型对 holdout 推理一次，
复用其已拟合元数据，不做 TTA、概率校准、融合或阈值调整。九类平均包含全部固定类别，
undefined precision／recall／F1 记为 0；相同概率取类别顺序中第一类。

## 公开表图与来源

- [总体对照](summary.csv) 包含精确分数、角色、样本数、轮数及实际 checkpoint SHA-256。
- [逐类对照](per_class_comparison.csv) 保留全部九类，包括 E10 分数更低的类别。
- E10：[指标](E10/evaluation/metrics.json)、[逐类表](E10/evaluation/per_class_metrics.csv)、[计数图](E10/evaluation/confusion_matrix.png)、[归一化图](E10/evaluation/confusion_matrix_normalized.png)。
- E07：[指标](E07/evaluation/metrics.json)、[逐类表](E07/evaluation/per_class_metrics.csv)、[计数图](E07/evaluation/confusion_matrix.png)、[归一化图](E07/evaluation/confusion_matrix_normalized.png)。
- 两模型每种混淆矩阵也提供同名 CSV，行是真实类别，列是预测类别。

结果回收时，20 个原始产物的大小与 SHA-256 全部一致，已核对每组 1,037 个唯一病例的
完整覆盖、概率与保存的九类指标。整数计数精确一致；跨 Python 浮点复算最大差异
`1.1102230246251565e-16`，在既有 `1e-12` 绝对容差内。公开交付不再次推理。

本目录只保留聚合结果。两份公开 metrics JSON 移除了来源路径，数值与原始结果相同，
但文件哈希已改变；其中来源 SHA 仍标识未公开的原始标签／概率文件。原图、权重、
逐病例概率、对齐标签、完整日志和内部运行配置均未发布。

B 的 [refit 公开摘要](../../refit/20261031_e10_e07_v1/README.md) 已纳入整合版本；
A／B／C 的交付和集成检查见 [贡献记录](../../../docs/contributions.md)。
