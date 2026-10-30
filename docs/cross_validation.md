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

## 29 日实际状态

E00／E01／E02 共 15 折全部成功，三组 OOF 各覆盖 3,968 例。见 [结果摘要](../reports/experiments/20261029_core/SUMMARY.md) 和 [结果格式](results_format.md)。后续以五折最佳 epoch 中位数 refit 完整开发集，再评估固定 holdout；这两步尚未执行。
