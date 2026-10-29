# 新正式 ResNet 实验

本目录不混入 meeting1 历史成绩或人工短检查结果。`experiment_status.csv` 覆盖 E00～E09 各五折；未训练时状态为 planned，指标空白。

训练期间完整 checkpoint／日志保存在训练环境，不提交 public。只有真实完成的折才从 `scripts/collect_results.py` 收集指标；完整五折由 `scripts/summarize_cv.py` 校验后汇总。历史模型档案不在本次公开交付中，其成绩不能填写到本轮表中。A／B 的依赖已整合，实际汇总仍需对应的真实运行产物和本地清单。
