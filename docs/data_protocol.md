# ResNet 九类数据协议 v1

## 来源和标签

官方来源：[下载页](https://challenge.isic-archive.com/data/)、[MILK 类别定义](https://challenge.isic-archive.com/landing/milk10k/)、[ISIC 2019 类别定义](https://challenge.isic-archive.com/landing/2019/)。

九类顺序固定为 `AKIEC,BCC,BKL,DF,INF,MEL,NV,SCCKA,VASC`。MILK 排除 BEN_OTH／MAL_OTH，这是课程既定九类子任务，不是官方十一类完整任务。ISIC 的 AK 并入更宽的 AKIEC，SCC 并入 SCCKA；BKL／VASC 作为相应更宽目标类别的子集。两个来源的诊断细分并不完全相同，具体映射、理由和来源逐项记录在 `configs/schema/label_mapping.csv`。UNK 排除，绝不当作 INF；未知标签阻断准备。

数据单位：MILK 每病灶一行，关联两张真实图片；ISIC 每图一行，已知 lesion ID 用于分组，允许同病灶多张不同照片。sample_id 带来源前缀，图片路径相对统一数据根目录。不使用旧 L 编号，也不猜原始编号与旧编号的映射。

元数据仅 age、sex、site、skin tone；诊断、病理确认、厚度、来源标识和 MONET 概念分数不进入本轮模型。年龄按训练折均值／标准差编码；类别词表仅训练折拟合；未知／缺失独立编码，肤色保留官方 0～5 编码。ISIC 缺临床图和肤色，使用明确 mask，不生成伪临床图。原始表无 patient ID，分组依据病灶和图片相似性，不能宣称患者级隔离。

## 划分与保守去重

保留 `DATA/PROCESSED_DATA/train_split.csv`／`val_split.csv` 原始成员关系，先过滤为九类：开发 4,150、holdout 1,037。固定 holdout 标签不参与调参，只有其图片相似关系用于泄漏隔离。

对两套训练图片共 35,811 张生成 64 位 pHash（32×32 灰度 DCT），查询 8 种旋转／镜像的最小 Hamming 距离 ≤4；共 9,253 对候选。结合原始文件 SHA-256、MILK 配对关系、ISIC 已知病灶 ID 形成连通组。这些是保守候选，不是人工确认的重复。裁剪、视角差异和未知患者关系仍可能遗漏；相似但不同病例也可能被合组。

本协议采用以下保守的数据隔离规则：

- 固定 holdout 不移动。与它同组的开发病例排除 182，留下开发 3,968。
- ISIC 中与任一 MILK 图片同组的图片排除 4,449；组内标签冲突排除 566；剩余完全相同图片只保留字典序第一张，排除 37。以上按该顺序应用，原因互斥。
- ISIC 最终 20,279 张。原始 50 个精确重复组中的两组冲突也被隔离；566 是包含相似／病灶分组后的数目，不能称全部都是确证错标。
- 不删除任何原图；完整排除表、类别计数、输入来源哈希在 `DATA/manifests/`；相似性明细缓存位于被忽略的 `DATA/cache/`。

只在开发集上以 `StratifiedGroupKFold(5, shuffle=True, random_state=42)` 分组分层；编号 0～4。逐折训练／验证无同组交集，OOF 覆盖每个开发病例恰好一次。每折所有九类均须存在，否则报错。ISIC 不进入验证或 holdout，与 MILK 的潜在重复已在全局保守隔离。

## 重建和版本

在仓库根目录用 CPU 环境运行：

```powershell
.\.venv\Scripts\python.exe -m src.data.deduplicate
.\.venv\Scripts\python.exe -m src.data.prepare
```

前者逐图解码并缓存相似性；后者生成新清单。输入必须为下载审计对应的官方文件。`version.json` 记录文件哈希；同目录重建如有任何差异会拒绝覆盖，要求新输出目录与复核。跨系统重建也须比较哈希，不能静默替换。

public 只包含 `class_counts.csv`、`summary.json` 和 `version.json` 三份汇总／哈希材料。逐病例清单、排除明细及 `folds/` 需在具有官方数据和审计输入的本地环境生成，未随此仓库公开。版本文件列出这些未公开清单的哈希是预期行为；不能创建空文件代替。离线 CI 使用人工输入，不要求真实图片或逐病例清单。

官方测试集全部与本轮训练隔离。固定 holdout 继承历史成员，不能称“整个项目从未看过的外部盲测”。本轮指标固定九类 argmax Macro F1，和官方 MILK 十一类逐类别阈值评分不是同一个评分任务。
