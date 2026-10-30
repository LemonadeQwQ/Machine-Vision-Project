# 模型选择与固定 holdout 边界

这些规则在本轮正式训练前确定，不根据最终成绩更改。

- 各折从相同 ResNet-50 ImageNet V2 权重独立创建模型和 AdamW，不使用上一折权重；seed 为 42。
- 每折最多 100 epoch。只有折验证九类 Macro F1 严格大于历史最佳才更新 `best.pt`，同分保留较早 epoch。
- 连续 15 轮未提升则停止；每轮保存 `last.pt`。恢复须保持同一折、配置、数据和源码指纹，只恢复最后完整 epoch。
- 最佳 epoch 的概率用于报告五折均值、样本标准差及 pooled OOF。验证折参与选 epoch，CV 属于开发选模估计。

| 实验 | 五折最佳 epoch（fold 0～4） | 预定 refit epoch |
|---|---|---:|
| E00 | 92、10、20、39、39 | 39 |
| E01 | 43、32、24、49、11 | 32 |
| E02 | 38、27、18、16、20 | 20 |

29 日全部 15 折完成：E00 fold 0 达到 100 轮，其余 14 折早停。检查点、历史和停止边界均已核对；15 个最佳模型重推理原验证折，概率与保存结果一致。

refit 轮数取五折最佳 epoch 中位数，在完整开发集重新初始化并训练。对应扩充实验才加入 ISIC。完成 refit 后才允许固定 holdout 推理，不得用 holdout 调参或选轮数。

上表只给出按既定规则算出的轮数，**本次未执行 refit 或 holdout 评估**。`src/infer.py` 拒绝用普通折 checkpoint 或未完成的 refit checkpoint 预测固定 holdout。

指标见 [结果摘要](../reports/experiments/20261029_core/SUMMARY.md)，数据边界见 [数据协议](data_protocol.md)。
