# ResNet 九类开发集五折协议

当前入口为 `src/experiment.py`、`src/cross_validate.py`、`src/infer.py` 和 `scripts/summarize_cv.py`。仅在 **3,968 例 MILK 开发集**分五折，固定 holdout **1,037 例**不参与分折或选模。旧 train＋val 合并、最后一轮选模属于历史协议，不用于本轮；历史说明由内部项目保留。

## 输入与独立训练

`DATA/manifests/folds/fold_0/`～`fold_4/` 为冻结清单，使用 `StratifiedGroupKFold(5, shuffle=True, random_state=42)`。病灶／相似图组不跨训练与验证边界，每个开发病例在每个实验中恰好验证一次。没有患者 ID，不能称患者级隔离。

E00／E01／E02 使用相同 MILK 病例。ISIC 只进入相应对照的训练侧。各折独立创建模型、优化器及训练侧拟合统计，不继承上一折 checkpoint。

`src.experiment` 运行单折，`src.cross_validate` 顺序运行五折，通用命令见 [训练说明](training.md)。集群资源申请和并行调度由集成人在内部脚本维护。正式训练只在分配的 Nibi GPU 上执行，资源见 [计算计划](compute_plan.md)。

## 选模与汇总

每折最多 100 epoch，按验证 Macro F1 严格改善保存 best，同分保留较早 epoch，连续 15 轮无改善停止。规则见 [模型选择](model_selection.md)。每折保存来源记录、历史、best／last checkpoint、最佳预测和统一评估。实际训练来源由源码与清单哈希标识，不能用较晚的交付提交号替代。

完整五折后在仓库根目录汇总：

```bash
python scripts/summarize_cv.py --runs RESULTS_ROOT/E00 --output RESULTS_ROOT/E00/summary
```

汇总拒绝 smoke、缺折、非 completed、混合配置／源码／清单及 OOF 缺失或重复。按 ID 与九类列名重算各折指标，报告均值和样本标准差；合并折外概率后另算 pooled OOF。验证折用于选 epoch，因此属于开发选模估计，不是独立最终测试。

## 已完成的 CV 与最终评估

E00～E10 共 55 折全部完成，各组 OOF 均覆盖同一 3,968 例。110 个 CV checkpoint
已审计，55 个 best checkpoint 完整验证折重推理最大概率绝对差为 0。
见 [十一组结果](../reports/experiments/20261031_e10/SUMMARY.md) 与 [结果格式](results_format.md)。

小组在 CV 后按五折平均九类 Macro F1 选择 E10 主模型，并固定 E07 为对照；
E10 是查看原十组结果后追加的配置，不能称为训练前注册的跨配置选择。
各自最佳 epoch 中位数为 13／14，两组已在全部 3,968 MILK＋20,279 ISIC 上独立 refit，
重新拟合训练侧统计，不使用验证早停。两模型随后完成同一 1,037 例固定 holdout 的评估，
详见 [最终结果](../reports/holdout/20261031_holdout_e10_e07_v1/SUMMARY.md)。

E10 对 E07 只改变元数据开关。元数据统计／词表只用各折训练侧拟合，refit 时用完整训练侧
重新拟合，holdout 复用已保存的编码器。原图和逐病例记录不随本次公开交付。

本 C 分支发布聚合结果，B 的 E10 配置与注册、A 的最终数据说明仍待整合；测试留到合并时
执行，不把真实实验完成等同于本分支通过集成检查。见 [分工状态](contributions.md)。
