# ResNet 九类训练协议

E00～E10 共 55 折已完成，各组 OOF 覆盖同一 3,968 例。E10 主模型／E07 对照分别完成
13／14 轮 refit 和同一 1,037 例固定 holdout 评估。见 [CV 汇总](../reports/experiments/20261031_e10/SUMMARY.md)、
[refit 记录](../reports/refit/20261031_e10_e07_v1/README.md) 和 [最终结果](../reports/holdout/20261031_holdout_e10_e07_v1/SUMMARY.md)。
以下是复现入口，已有实验不需要重跑；复现时使用新的输出目录。

## 模型和实验

| 配置 | 模态 | 加权 CE | ISIC 扩充 | 元数据 | 几何增强 |
|---|---|---|---|---|---|
| E00 | 皮肤镜 | 否 | 否 | 否 | 是 |
| E01 | 临床 | 否 | 否 | 否 | 是 |
| E02 | 双图 | 否 | 否 | 否 | 是 |
| E03 | 皮肤镜 | 是 | 否 | 否 | 是 |
| E04 | 皮肤镜 | 否 | 是 | 否 | 是 |
| E05 | 皮肤镜 | 是 | 是 | 否 | 是 |
| E06 | 双图 | 是 | 否 | 否 | 是 |
| E07 | 双图 | 是 | 是 | 否 | 是 |
| E08 | 双图 | 否 | 否 | 是 | 是 |
| E09 | 双图 | 否 | 否 | 否 | 否 |
| E10 | 双图 | 是 | 是 | 是 | 是 |

E00～E02 为核心 15 折，E03～E09 为扩展 35 折，E10 为后追加的 5 折，均已完成。每折均重新加载相同 ImageNet V2 初始化、独立模型／优化器／训练侧元数据和权重统计。双分支参数不共享，2048 维特征拼接与可用性 mask 后进入 512 维分类头；没有门控、辅助头或辅助损失。单模态具有同样的 512 维头结构。仅 layer4 和新头可训练，骨干全部 BN 的运行统计固定，layer4 BN 仿射参数可学习。

图像 EXIF 校正后转 RGB，等比例缩放补黑边至 224×224；训练随机水平／垂直翻转、±20° 旋转，两模态用相同随机参数；验证无随机增强。旋转不扩展画布，角落内容可能被裁去，E09 关闭上述几何增强。归一化采用 ImageNet RGB 均值／标准差。

AdamW：骨干学习率 1e-5、分类头 1e-4、weight decay 1e-4、梯度范数上限 5。默认 batch 32、BF16、channels_last、TF32、4 个 worker、每 worker 预取 2 个 batch。每折上限 100 epoch，验证 Macro F1 连续 15 轮不提升则停止；相同分数保留较早 epoch。加权实验使用训练折 `w=(1/n)/mean(1/n)`，零计数拒绝。全部配置在 `configs/experiments/`。本轮正式运行已验证并冻结 batch 32、4 workers、BF16、channels_last、TF32；使用 H100 MIG 1g.10gb、4 CPU、24G 主机内存，最多两折并发。cuDNN 使用 `benchmark=False, deterministic=True`；保持原预测读回容差 `rtol=1e-4, atol=1e-5`。


E10 在 E07 的数据和损失设置上启用元数据，不继承 E07 checkpoint。年龄、性别、部位、
肤色四个字段沿用 E08 的处理方式，但各折分别用 MILK 训练侧＋ISIC 拟合年龄统计和词表，
不复制 E08 编码器。ISIC 无临床图和肤色，保留 mask；验证集始终只含 MILK。
见 [元数据说明](metadata_inventory.md)。

## 运行环境和输入

训练需要匹配的 PyTorch／torchvision 及 `requirements-train.txt`。GPU 申请、环境部署和
集群调度由内部运行环境维护。本仓库提供通用 Python 接口；不包含原图、权重、逐病例清单
或部署脚本。官方数据准备和版本核对见 [数据协议](data_protocol.md)。

以下 Bash 变量为用户自己的路径占位符：

```bash
PROJECT_ROOT=/path/to/Vision_Project
DATA_ROOT=/path/to/datasets
WEIGHTS="$PROJECT_ROOT/pretrained_weights/resnet50-11ad3fa6.pth"
MANIFESTS="$PROJECT_ROOT/DATA/manifests"
RUN_ROOT=/path/to/results/resnet9_new_run
cd "$PROJECT_ROOT"
```

配置默认使用仓库中的 `DATA/manifests`，原图目录可通过 `--data-root` 指定。
预训练权重必须匹配 `configs/schema/pretrained_weights.json`；逐病例清单和图像需由使用者
按官方来源准备，不可用公开聚合表代替。已完成的真实 GPU 短检、恢复及 checkpoint 重推理
属于原运行验收；更换硬件或数据的复现应自行核对条件，不在同一五折中途混用设置。

## 独立五折和恢复

在具有相应数据与 GPU 的环境运行：

```bash
python -m src.cross_validate --config configs/experiments/E10_pair_isic_weighted_metadata.yaml   --data-root "$DATA_ROOT" --weights "$WEIGHTS" --output "$RUN_ROOT/E10"
python -m src.cross_validate --config configs/experiments/E07_pair_isic_weighted.yaml   --data-root "$DATA_ROOT" --weights "$WEIGHTS" --output "$RUN_ROOT/E07"
```

每个命令顺序执行五折；独立单折入口是 `src.experiment --config CONFIG --fold FOLD --output OUTPUT`。
只检查单折输入可给该入口传 `--preflight-only`。已有结果不会被静默跳过；更换设置须新建运行。

恢复使用同一折、同一配置和输出目录，传 `--resume` 从 `last.pt` 的最后完整 epoch 继续。
不能恢复 epoch 中途的 batch。只有确认原作业已退出后才处理残留 `.running` 标记。
恢复校验配置、数据和代码指纹；仅加载模型权重不等于恢复。

## 汇总、refit 与 holdout

每折在本地保存运行配置、历史、元数据、best／last checkpoint 和最佳验证概率；
完整五折后执行：

```bash
python scripts/summarize_cv.py --runs "$RUN_ROOT/E10" --manifests "$MANIFESTS"   --output "$RUN_ROOT/E10/summary"
python scripts/summarize_cv.py --runs "$RUN_ROOT/E07" --manifests "$MANIFESTS"   --output "$RUN_ROOT/E07/summary"
python scripts/collect_results.py --runs "$RUN_ROOT" --output "$RUN_ROOT/experiment_status.csv"
```

缺失的其他实验会保持 planned，不会因仓库已有成绩而被收集器伪造为已运行。
汇总拒绝缺折、smoke、混合来源或 OOF 覆盖错误；分别报告均值、样本标准差和 pooled OOF。

复现的 refit 必须使用同一次 CV 保存的完整 `cv_summary.json`，不是公开 `public_summary.json`：

```bash
python -m src.refit --cv-summary "$RUN_ROOT/E10/summary/cv_summary.json" --output "$RUN_ROOT/E10/refit"
python -m src.refit --cv-summary "$RUN_ROOT/E07/summary/cv_summary.json" --output "$RUN_ROOT/E07/refit"
```

Refit 采用各自五折最佳 epoch 中位数，独立初始化模型和优化器，在完整训练侧重新计算
类别权重及元数据统计，无验证早停。已完成原实验的 E10／E07 轮数为 13／14。
只有这两组已执行 refit；其他配置的中位数只是条件轮数。

`src.refit` 会核对当前源码与 CV 来源一致。公开适配改变了源码指纹；若要复用历史运行摘要，
必须使用其原冻结源码和匹配输入，不能绕过检查或将新交付 SHA 冒充训练时来源。

完成 refit 后再推理固定 holdout，以下以 E10 为例（E07 使用其自身 checkpoint）：

```bash
python -m src.infer --checkpoint "$RUN_ROOT/E10/refit/best.pt"   --manifest "$MANIFESTS/milk_holdout.csv" --data-root "$DATA_ROOT"   --output "$RUN_ROOT/E10/refit/holdout_predictions.csv"
python -m src.evaluate --class-schema 9 --id-column sample_id   --labels "$MANIFESTS/milk_holdout.csv"   --predictions "$RUN_ROOT/E10/refit/holdout_predictions.csv"   --output-dir "$RUN_ROOT/E10/refit/holdout_evaluation"
```

推理拒绝普通 fold／未完成 refit 的 checkpoint，复用训练侧元数据，并核对冻结 holdout 哈希。
本轮两模型角色在 holdout 前固定，结果不用于调参、选轮数或交换主模型／对照。
[评估约定](evaluation.md) 保留 public 显式 labels 要求；新逐病例输出不能随公开交付上传。
