# 新 ResNet 协议的数据来源

本页记录本轮实际输入。官方 URL、大小和 SHA-256 以 [source_files.json](../configs/data/source_files.json) 为准，九类映射及理由见 [label_mapping.csv](../configs/schema/label_mapping.csv)。

| 来源 | 使用的官方资料 | 本轮角色 |
|---|---|---|
| MILK10k Training | Input ZIP、GroundTruth、Metadata、Supplement | 九类病例及配对皮肤镜／临床图；开发集和固定 holdout |
| ISIC 2019 Training | Input ZIP、GroundTruth、Metadata | 仅对应扩充实验的训练侧皮肤镜图片 |

MILK 原开发成员九类过滤后为 4,150 例，再排除与 holdout 相似组相关的 182 例，得到 **3,968** 例开发集。固定 holdout **1,037** 例保持不变。ISIC 经映射、MILK 相似组隔离、冲突组和精确重复处理后为 **20,279** 张。排除顺序见 [数据协议](data_protocol.md)。

E00／E01／E02 只使用 MILK，相同病例用于三种模态比较。ISIC 不进入折验证；E04／E05／E07 才使用该训练池。缺失临床图使用 mask，不复制皮肤镜充当临床图。官方 Test 没有参与核心训练或指标计算。

分组依据病灶 ID、图像 SHA-256 和保守相似候选连通组。官方表没有患者 ID，无法宣称患者级隔离；候选不等于人工确认重复。原图未删除。固定 holdout 继承历史成员，不是整个项目从未使用过的外部盲测。

29 日实际验收：50 项配置／折输入检查通过；三组正式五折各产生 3,968 例 OOF，每个开发病例在各实验中恰好验证一次。聚合证据见 [数据验收记录](../reports/data_audit/20261029_cv_acceptance.json)。原图、逐病例新清单和相似性明细不随 public 交付，使用者自行按官方来源准备并核对清单哈希。
