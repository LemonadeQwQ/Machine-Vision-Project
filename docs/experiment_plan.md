# 九类实验计划

更新日期：2026-11-10。E11～E13 的 C 公开交付已准备，A／B 依赖待整合；本次不发布新实验成绩。
下列已完成记录属于 E00～E10：共 55 折、110 个 checkpoint 审计及 55 个最佳模型
完整验证折重推理均完成，最大概率绝对差为 0。E10 主模型／E07 对照分别完成 13／14 轮
refit，并完成同一 1,037 例 holdout。见 [十一组 CV](../reports/experiments/20261031_e10/SUMMARY.md)
和 [最终评估](../reports/holdout/20261031_holdout_e10_e07_v1/SUMMARY.md)。

11 月 9 日公开基线已整合 A 的最终数据说明、B 的 E10 配置／注册及 refit 公开摘要，以及 C 的结果、
汇总／绘图、协议测试和公共文档。原提交与作者保留；集成检查见 [贡献说明](contributions.md)。
既有真实训练和最终推理不因本次合并而重跑。
PanDerm、MONET、Gate 的记录是历史实验，不填入本轮成绩。

## 输入与比较人群

- 类别固定为 AKIEC、BCC、BKL、DF、INF、MEL、NV、SCCKA、VASC；映射见 `configs/schema/label_mapping.csv`。
- MILK 开发集 3,968 个病例，固定 holdout 1,037 个；当前这两部分均有皮肤镜和临床图，E00／E01／E02 使用同一比较人群。
- ISIC 扩充池 20,279 张，只能进入对应实验的训练侧。ISIC 无临床图，双模态实验用可用性 mask，缺失分支不编码伪图。
- 开发集按病灶／相似图组分为五折。与 holdout 相似组关联的 182 个开发病例已保守排除；ISIC 的相关组与冲突组也已隔离。候选不等于人工确认重复，原图未删除。
- 原始表没有患者 ID，不能宣称完成患者级隔离。固定 holdout 继承历史成员，不是整个项目从未使用过的外部盲测。

## 实验矩阵

E00～E10 的配置和训练入口均已整合。下表描述实际完成的实验设计，全部五折均为 completed。

| 编号 | 问题与配置 | 对照及主要变化 | 数据与优先级 |
|---|---|---|---|
| E00 | `E00_derm.yaml`，皮肤镜 ResNet-50 | 基准；普通 CE、无元数据、开启增强 | MILK；核心，五折 |
| E01 | `E01_clinical.yaml`，临床 ResNet-50 | 对 E00 只改变输入模态 | MILK；核心，五折 |
| E02 | `E02_pair.yaml`，两个独立 ResNet 特征拼接 | 与单模态比较双图融合；参数量也随双骨干增加 | MILK；核心，五折 |
| E03 | `E03_derm_weighted.yaml` | 对 E00 只增加逆频率加权 CE | MILK；五折已完成 |
| E04 | `E04_derm_isic.yaml` | 对 E00 只扩充 ISIC 训练样本 | MILK + ISIC；五折已完成 |
| E05 | `E05_derm_isic_weighted.yaml` | 对 E03 增加 ISIC，或对 E04 增加加权 | MILK + ISIC；五折已完成 |
| E06 | `E06_pair_weighted.yaml` | 对 E02 只增加加权 CE | MILK；五折已完成 |
| E07 | `E07_pair_isic_weighted.yaml` | 对 E06 只增加 ISIC；覆盖缺失临床图 | MILK + ISIC；五折已完成 |
| E08 | `E08_pair_metadata.yaml` | 对 E02 只增加训练折拟合的元数据 | MILK；五折已完成 |
| E09 | `E09_pair_aug_off.yaml` | 对 E02 只关闭几何增强 | MILK；五折已完成 |
| E10 | `E10_pair_isic_weighted_metadata.yaml` | 对 E07 只增加元数据；对 E08 同时增加 ISIC 和加权 | MILK + ISIC；五折已完成 |

E00～E02 是核心 15 折，E03～E09 是扩展 35 折，E10 是后追加的 5 折。E10 在查看此前十组结果后加入，不宣称训练前注册；历史目录不是必须依次重跑的阶段。

## E11～E13：三个独立增量，公开整合待完成

每组以 E10 的双图、ISIC、加权 CE、训练折拟合元数据和增强为共同对照，只增加下表一项。
沿用九类、3,968 例开发集、1,037 例固定 holdout、20,279 张训练侧 ISIC 与原五折成员。

| 编号 | B 待交付配置 | 相对 E10 的唯一结构增量 | public 状态 |
|---|---|---|---|
| E11 | `E11_pair_gate.yaml` | 两个 ResNet 原始特征增加中性初始化的模态 Gate | C 检查与说明已准备；待 A／B |
| E12 | `E12_pair_panderm.yaml` | 皮肤镜骨干替换为 PanDerm_Base，保留临床 ResNet | C 检查与说明已准备；待 A／B |
| E13 | `E13_pair_monet.yaml` | 保留双 ResNet，追加冻结 MONET 的四概念投影 | C 检查与说明已准备；待 A／B |

新实验各需五个独立折。收集器扩为 E00～E13 共 70 个计划位置，不代表已训练 70 折。
不把历史 PanDerm／MONET／Gate 数字写作这三个新配置的结果；本分支没有新增正式 CV 指标。
完整模型定义、依赖文件、组件指纹及检查方法见 [公开实现契约](e11_e13_implementation.md)。
E14 尚未注册；不得把 Gate、PanDerm、MONET 自动组合后沿用 E11／E12／E13 的身份。

新组仍以完整五折的未四舍五入平均九类 Macro F1 作开发比较，同时报告样本标准差、
pooled OOF、参数量和计算成本。元数据词表只在各折训练侧拟合，允许折间维度不同。
现有 E10／E07 的主模型／对照身份和 holdout 记录保持历史事实，不因新计划改变。
本分支不提供 E11～E13 refit／holdout 结果；不得用 holdout 调整结构、超参数或选轮数。

## E00～E10 已执行的固定训练与选模规则

每折从 ImageNet V2 独立初始化模型和优化器，配置种子为 42。训练 layer4 和新分类头，其余骨干参数冻结，全部骨干 BatchNorm 运行统计固定。输入 224×224，保持比例补边；训练双图共享几何增强，验证不做随机增强。

AdamW 骨干学习率 1e-5、头部 1e-4、weight decay 1e-4；每折最多 100 epoch。以折验证 Macro F1 选最佳模型，同分保留较早 epoch，连续 15 轮不提升则停止。核心实验采用普通 CE。加权实验仅根据该折训练样本计算 `(1/n_c) / mean(1/n_c)`。

本次实际训练使用 batch 32、4 workers、BF16、channels_last、TF32、H100 MIG 1g.10gb，最多两折并发；cuDNN 为 benchmark=False、deterministic=True。这些设置已在 Nibi 真实输入上检查并用于本轮正式实验，不能在同一五折中途混用配置。可公开的设置、源码／清单／预训练权重哈希和验收摘要见 [public_summary.json](../reports/experiments/20261031_e10/public_summary.json)。B 的 cuDNN 修复已接入；本段描述实际训练快照，公开版特有适配不改变原始来源记录。

## 指标、CV 与最终 refit

主要指标为固定九类 Macro F1；同时导出 Accuracy、每类 precision／recall／F1／support 和混淆矩阵。预测 CSV 按 `sample_id` 与类别名对齐。没有 support 的类别也保留在九类平均中。

五折分别报告最佳 epoch 及指标，再报告五折平均值、样本标准差和 pooled OOF；后者由每个开发病例恰好一次的折外预测计算。CV 用于开发选模，不能作为独立最终测试。

最终 refit 的 epoch 数为五折最佳 epoch 的中位数，在完整开发集训练；对应扩充实验才加入 ISIC。完成 refit 后才评估固定 holdout，不能用 holdout 调参或选轮数。真实结果只在完成相应阶段后填入结果表；人工 smoke 始终单独标注。

## 已确定的选择与后续交付

小组按五折平均九类 Macro F1 选择 E10 为主模型，并保留 E07 为对照，角色在最终评估前确定。
两组独立 refit 时各使用 24,247 条训练样本，按完整训练侧重新计算类别权重，E10 重新拟合
元数据编码器；holdout 推理复用训练侧统计。当前 holdout Macro F1 为 56.62%／53.28%，
Accuracy 为 70.49%／70.01%，完整报告两者，不按 holdout 再调参或换角色。

E00～E10 的 A／B／C 代码与聚合结果已整合。组员实际复核、方法／报告与完整复现资料、
发布说明及最终标签；这些不能因为实验结束而标记为全部完成。
[训练说明](training.md)、[选模说明](model_selection.md) 已更新为当前状态，
本轮进度见 [贡献说明](contributions.md)。
