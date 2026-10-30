# 新正式 ResNet 实验

2026-10-29，E00／E01／E02 各五折已完成。`experiment_status.csv` 覆盖 E00～E09 共 50 折：15 折 completed，E03～E09 的 35 折 planned 且成绩为空。meeting1 历史成绩、人工短检查和资源测速不计入正式结果。

本次 [核心摘要](20261029_core/SUMMARY.md) 包含五折均值／样本标准差、pooled OOF、实际轮数、硬件与来源指纹；三组 OOF 均覆盖相同的 3,968 例。公开文件只有聚合表、每类指标、混淆矩阵和统计图，不包含逐病例预测或标签。完整 checkpoint 和日志保存在训练环境，原始运行记录与完整审计证据保留在内部项目。

`scripts/collect_results.py` 从真实完成折收集指标，`scripts/summarize_cv.py` 检查五折一致性与 OOF 覆盖。只有具备相应运行产物和本地清单才能重新汇总，不能用聚合表代替逐病例输入。

重绘聚合比较图，不需要原图或逐病例预测：

```bash
python tools/plot_cv_summary.py --summary reports/experiments/20261029_core/public_summary.json --output reports/experiments/20261029_core/summary_comparison.png
```

这些是开发选模结果，不是独立最终测试。refit 与固定 holdout 尚未执行；预定 refit 轮数不表示已训练。A 的 29 日数据验收、B 的训练修复已在本版本整合，组员人工复核另行记录，见 [分工说明](../../docs/contributions.md) 和 [结果格式](../../docs/results_format.md)。
