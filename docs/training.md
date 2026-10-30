# ResNet 九类训练协议

2026-10-29，核心 E00／E01／E02 共 15 折及三组汇总已完成；30 个 checkpoint 审计和 15 个最佳模型完整验证折重推理通过，概率最大绝对差为 0。真实指标见 [核心摘要](../reports/experiments/20261029_core/SUMMARY.md)，运行预算见 [计算计划](compute_plan.md)。E03～E09、refit 和固定 holdout 仍未执行。以下命令用于复现，必须使用新输出目录，不重复写入已完成结果。

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

核心为 E00～E02 共 15 折，第二优先 E03／E04。每折均重新加载相同 ImageNet V2 初始化、独立模型／优化器／训练侧元数据和权重统计。双分支参数不共享，2048 维特征拼接与可用性 mask 后进入 512 维分类头；没有门控、辅助头或辅助损失。单模态具有同样的 512 维头结构。仅 layer4 和新头可训练，骨干全部 BN 的运行统计固定，layer4 BN 仿射参数可学习。

图像 EXIF 校正后转 RGB，等比例缩放补黑边至 224×224；训练随机水平／垂直翻转、±20° 旋转，两模态用相同随机参数；验证无随机增强。旋转不扩展画布，角落内容可能被裁去，E09 关闭上述几何增强。归一化采用 ImageNet RGB 均值／标准差。

AdamW：骨干学习率 1e-5、分类头 1e-4、weight decay 1e-4、梯度范数上限 5。默认 batch 32、BF16、channels_last、TF32、4 个 worker、每 worker 预取 2 个 batch。每折上限 100 epoch，验证 Macro F1 连续 15 轮不提升则停止；相同分数保留较早 epoch。加权实验使用训练折 `w=(1/n)/mean(1/n)`，零计数拒绝。全部配置在 `configs/experiments/`。本次核心运行已验证并冻结 batch 32、4 workers、BF16、channels_last、TF32；使用 H100 MIG 1g.10gb、4 CPU、24G 主机内存，最多两折并发。cuDNN 使用 `benchmark=False, deterministic=True`；保持原预测读回容差 `rtol=1e-4, atol=1e-5`。

## 运行环境和输入

真实训练仅在获准的 Nibi GPU 作业中执行；本页给出通用 Python 入口。
环境安装、数据暂存、资源申请和 Slurm 提交由集成人在内部集群脚本维护，
不属于 public 仓库。29 日已完成五组真实 GPU smoke 与核心五折训练；复现前应核对本次实际设置、输入版本和当前硬件。

训练环境需匹配的 PyTorch／torchvision，以及 `requirements-train.txt` 中的依赖。
原图、权重、环境和逐病例清单不进入 public。数据准备见 [数据协议](data_protocol.md)，
公开 CI 只使用人工输入。

以下 Bash 变量都是用户环境中的占位路径，执行前换成实际位置：

```bash
PROJECT_ROOT=/path/to/Vision_Project
DATA_ROOT=/path/to/datasets
WEIGHTS="$PROJECT_ROOT/pretrained_weights/resnet50-11ad3fa6.pth"
MANIFESTS="$PROJECT_ROOT/DATA/manifests"
RUN_ROOT=/path/to/results/resnet9_v1_run01
cd "$PROJECT_ROOT"
```

数据目录保持官方输入的相对结构；固定清单需在本地生成并核对 `version.json`。
ImageNet V2 权重校验信息位于 `configs/schema/pretrained_weights.json`。
结果保存到持久存储。计算节点使用临时数据目录时可设置 `CSC490_DATA_ROOT`；
代码指纹以实际运行目录重新计算，不能沿用另一 checkout 的字节指纹。

## 输入检查与真实 GPU 短检

准备好真实输入和权重后，单折只检查输入、不创建模型的入口是：

```bash
python -m src.experiment --config configs/experiments/E00_derm.yaml --fold 0 \
  --data-root "$DATA_ROOT" --weights "$WEIGHTS" --manifests "$MANIFESTS" \
  --output "$RUN_ROOT/E00/fold_0" --preflight-only
```

正式训练前，对 E00／E01／E02／E07／E08 分别做真实图片短检。
短检按来源、类别和模态可用性选少量样本；E07 必须覆盖 MILK 双图和
ISIC 缺临床图，验证侧始终只含 MILK。每个配置预定两轮：

1. 在独立输出目录运行 `src.experiment`，加上
   `--smoke --smoke-epochs 2 --smoke-stop-after-epoch 1`。
2. 运行 `scripts/check_smoke_run.py --run SMOKE_DIR --expect stopped`，
   核验首轮 checkpoint、覆盖情况和样本顺序。
3. 以相同配置和目录重新启动，使用 `--smoke --smoke-epochs 2 --resume`，
   去掉停止参数，从 `last.pt` 恢复到第二轮。
4. 用 `src.infer` 从 `best.pt` 对 `SMOKE_DIR/smoke_val.csv` 重新导出
   `SMOKE_DIR/reloaded_predictions.csv`，再运行
   `scripts/check_smoke_run.py --run SMOKE_DIR --expect completed`。
   此后再核对完整折验证集的推理。

`smoke_train.csv`、`smoke_val.csv` 和 `smoke_*_verification.json` 保留覆盖及恢复证据。
smoke 产物必须使用独立目录，不进入正式结果表。普通 `--smoke` 默认只运行一轮。

检查 CUDA、显存余量、吞吐、数据读取、缺失模态、checkpoint 和预测读回。
本次 10 GB MIG 上核心双模态 PyTorch 预留峰值约 1.03 GiB，完整折第二轮训练加验证约 9.22 秒，不含 checkpoint 和最终产物写入。更换硬件、batch、worker 数、精度或并发设置时重新验证并记录版本，不能在同一五折中途混用配置。集群旧环境的前向自检不能替代这些检查。

## 核心五折与恢复

通过真实短检并冻结设置后，在获准的 GPU 作业中运行核心实验。
单实验顺序五折入口如下；每次各运行五折，共 15 折：

```bash
python -m src.cross_validate --config configs/experiments/E00_derm.yaml \
  --data-root "$DATA_ROOT" --weights "$WEIGHTS" --output "$RUN_ROOT/E00"
python -m src.cross_validate --config configs/experiments/E01_clinical.yaml \
  --data-root "$DATA_ROOT" --weights "$WEIGHTS" --output "$RUN_ROOT/E01"
python -m src.cross_validate --config configs/experiments/E02_pair.yaml \
  --data-root "$DATA_ROOT" --weights "$WEIGHTS" --output "$RUN_ROOT/E02"
```

集群并行调度使用内部脚本，单折 Python 入口为 `src.experiment --config CONFIG --fold FOLD --output OUTPUT`，
同时传入正确的输入路径。E03～E09 是后续对照，不是启动核心任务的前置阶段。

中断后用相同单折命令和 `--resume` 恢复最后完整 epoch 的 `last.pt`，
不能恢复被打断 epoch 中途的 batch。若留下 `.running` 标记，先确认原作业已退出，
再处理对应目录的标记。不要对仍运行的作业解除锁。

洗牌随机数独立于 worker 初始化，每轮按 seed 加 epoch 索引重置；
样本增强由 seed、epoch 和样本索引确定。恢复校验配置、代码和清单指纹，
发生变化会拒绝；仅加载模型权重不等于恢复。GPU 加速和跨设备浮点差异
不保证逐位一致。

## 预测、汇总和最终 refit

每折保存 `run.json`、`config.json`、`metadata.json`、`history.json`、
`best.pt`、`last.pt`、`predictions.csv` 和统一 `evaluation/`。
模型选择使用折验证 Macro F1，因此 CV 是开发选模估计。

```bash
python scripts/summarize_cv.py --runs "$RUN_ROOT/E00" --manifests "$MANIFESTS" \
  --output "$RUN_ROOT/E00/summary"
python scripts/collect_results.py --runs "$RUN_ROOT" \
  --output "$RUN_ROOT/experiment_status.csv"
```

汇总只接受同配置／代码／数据、全部五折完成、完整 OOF 的正式结果。
逐折均值、样本标准差和 pooled OOF 分开报告。smoke 和缺折不能冒充 completed CV。
评估格式见 [统一评估](evaluation.md)。

最终 refit 的轮数预先规定为五折 best epoch 的中位数，不用 holdout 选轮数：

```bash
python -m src.refit --cv-summary "$RUN_ROOT/E00/summary/cv_summary.json" \
  --output "$RUN_ROOT/E00/refit"
```

refit 复用 CV summary 中冻结的配置，恢复时传 `--resume`。核心预定轮数为 E00＝39、E01＝32、E02＝20，见 [模型选择](model_selection.md)；这些是按五折中位数计算的计划，本次尚未执行。
完成后才在 GPU 作业中推理并评估固定 holdout：

```bash
python -m src.infer --checkpoint "$RUN_ROOT/E00/refit/best.pt" \
  --manifest "$MANIFESTS/milk_holdout.csv" --data-root "$DATA_ROOT" \
  --output "$RUN_ROOT/E00/refit/holdout_predictions.csv"
python -m src.evaluate --class-schema 9 --id-column sample_id \
  --labels "$MANIFESTS/milk_holdout.csv" \
  --predictions "$RUN_ROOT/E00/refit/holdout_predictions.csv" \
  --output-dir "$RUN_ROOT/E00/refit/holdout_evaluation"
```

holdout 推理拒绝未完成的 refit 或普通 fold 检查点。结果只用于最终报告，
不回头调参。各实际阶段完成多少，就记录多少；public 合并不表示这些命令已运行。
