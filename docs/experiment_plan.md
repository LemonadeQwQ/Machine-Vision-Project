# ResNet 九类实验计划

更新日期：2026-10-28。内部数据准备和离线验收已完成；public 已整合 A／B／C 的数据、训练、评估和测试代码。真实数据 GPU 短检、正式训练和模型效果评估均未开始。PanDerm、MONET、Gate 的记录是历史实验，不能填入本表。

## 28 日的完成边界

28 日要求统一九类数据、E00／E01 单模态训练入口和评估接口，跑通人工输入的前向、反向、检查点保存／恢复和预测读回。此阶段不要求真实 GPU 训练或完整五折。五折、双模态、对照实验和 refit 的接口已提前实现；接口存在不代表这些实验已经运行。

内部完整版本的本地离线验收为 107 项测试和 3 项初始化检查通过，50 个实验／折输入检查通过；这不是 public 分支的测试数量。恢复回归覆盖 0／4 workers，人工 CPU 运行的模型、AdamW 状态、样本顺序和预测与连续运行一致。public 整合检查、远程 CI、Nibi 新版本检查及真实 GPU 短检须分别记录实际状态。

## 输入与比较人群

- 类别固定为 AKIEC、BCC、BKL、DF、INF、MEL、NV、SCCKA、VASC；映射见 `configs/schema/label_mapping.csv`。
- MILK 开发集 3,968 个病例，固定 holdout 1,037 个；当前这两部分均有皮肤镜和临床图，E00／E01／E02 使用同一比较人群。
- ISIC 扩充池 20,279 张，只能进入对应实验的训练侧。ISIC 无临床图，双模态实验用可用性 mask，缺失分支不编码伪图。
- 开发集按病灶／相似图组分为五折。与 holdout 相似组关联的 182 个开发病例已保守排除；ISIC 的相关组与冲突组也已隔离。候选不等于人工确认重复，原图未删除。
- 原始表没有患者 ID，不能宣称完成患者级隔离。固定 holdout 继承历史成员，不是整个项目从未使用过的外部盲测。

## 实验矩阵

所有实验配置已由 B 交付并整合在 `configs/experiments/`；下表全部为 implemented / training planned，不能把接口实现当成训练完成。

| 编号 | 问题与配置 | 对照及主要变化 | 数据与优先级 |
|---|---|---|---|
| E00 | `E00_derm.yaml`，皮肤镜 ResNet-50 | 基准；普通 CE、无元数据、开启增强 | MILK；核心，五折 |
| E01 | `E01_clinical.yaml`，临床 ResNet-50 | 对 E00 只改变输入模态 | MILK；核心，五折 |
| E02 | `E02_pair.yaml`，两个独立 ResNet 特征拼接 | 与单模态比较双图融合；参数量也随双骨干增加 | MILK；核心，五折 |
| E03 | `E03_derm_weighted.yaml` | 对 E00 只增加逆频率加权 CE | MILK；后续优先对照 |
| E04 | `E04_derm_isic.yaml` | 对 E00 只扩充 ISIC 训练样本 | MILK + ISIC；后续优先对照 |
| E05 | `E05_derm_isic_weighted.yaml` | 对 E03 增加 ISIC，或对 E04 增加加权 | MILK + ISIC；后续组合对照 |
| E06 | `E06_pair_weighted.yaml` | 对 E02 只增加加权 CE | MILK；后续对照 |
| E07 | `E07_pair_isic_weighted.yaml` | 对 E06 只增加 ISIC；覆盖缺失临床图 | MILK + ISIC；后续对照 |
| E08 | `E08_pair_metadata.yaml` | 对 E02 只增加训练折拟合的元数据 | MILK；后续对照 |
| E09 | `E09_pair_aug_off.yaml` | 对 E02 只关闭几何增强 | MILK；后续对照 |

本轮核心工作是 E00、E01、E02 各五折，共 15 个独立训练任务。E03～E09 不作为启动核心任务的前置阶段；历史目录也不是必须依次重跑的阶段。

## 固定训练与选模规则

每折从 ImageNet V2 独立初始化模型和优化器，配置种子为 42。训练 layer4 和新分类头，其余骨干参数冻结，全部骨干 BatchNorm 运行统计固定。输入 224×224，保持比例补边；训练双图共享几何增强，验证不做随机增强。

AdamW 骨干学习率 1e-5、头部 1e-4、weight decay 1e-4；每折最多 100 epoch。以折验证 Macro F1 选最佳模型，同分保留较早 epoch，连续 15 轮不提升则停止。核心实验采用普通 CE。加权实验仅根据该折训练样本计算 `(1/n_c) / mean(1/n_c)`。

当前 batch 32、4 workers、BF16、channels_last、TF32 是待实测的候选运行设置。Nibi 上先验证显存、吞吐、缺失模态和恢复流程；必要调整须在正式 CV 前记录并冻结，不能在五折中途混用设置。

## 指标、CV 与最终 refit

主要指标为固定九类 Macro F1；同时导出 Accuracy、每类 precision／recall／F1／support 和混淆矩阵。预测 CSV 按 `sample_id` 与类别名对齐。没有 support 的类别也保留在九类平均中。

五折分别报告最佳 epoch 及指标，再报告五折平均值、样本标准差和 pooled OOF；后者由每个开发病例恰好一次的折外预测计算。CV 用于开发选模，不能作为独立最终测试。

最终 refit 的 epoch 数为五折最佳 epoch 的中位数，在完整开发集训练；对应扩充实验才加入 ISIC。完成 refit 后才评估固定 holdout，不能用 holdout 调参或选轮数。真实结果只在完成相应阶段后填入结果表；人工 smoke 始终单独标注。

## 后续顺序

用户授权后部署 Nibi → 新版本离线检查 → E00／E01／E02／E07／E08 真实 GPU 短检与恢复验证 → 冻结资源设置 → 启动核心 15 折 → 五折汇总 → 按规则 refit → 固定 holdout 最终评估。参见 [训练说明](training.md) 和 [数据协议](data_protocol.md)。本次 public 合并不执行集群操作。
