# 新 ResNet 训练与集群操作

本页命令供**获准上传之后**使用；此次准备没有执行上传或训练。当前准备状态见 [local_preparation.md](local_preparation.md)。

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

AdamW：骨干学习率 1e-5、分类头 1e-4、weight decay 1e-4、梯度范数上限 5。默认 batch 32、BF16、channels_last、TF32、4 个 worker、每 worker 预取 2 个 batch。候选上限 100 epoch，验证 Macro F1 连续 15 轮不提升则停止；相同分数保留较早 epoch。加权实验使用训练折 `w=(1/n)/mean(1/n)`，零计数拒绝。全部配置在 `configs/experiments/`；性能参数需集群真实短检后才能定为最终值。

## 上传后的目录

集群根目录 `/project/def-ajbonner/yuchenx4/csc490`：代码在 `Vision_Project/`，原始文件在 `data/`，权重在 `pretrained/resnet50-11ad3fa6.pth`，结果在 `results/`。不要上传 Windows 虚拟环境或 `read.txt`。

`data/` 保持本机 DATASETS 的相对结构。核心需要 MILK Training ZIP、标签、元数据和审计清单；ISIC 扩充需要 ISIC Training ZIP、标签和元数据。官方 Test 文件留作独立资料，不作为本轮输入。`stage_data.sh` 在计算节点把所需 ZIP 展开至 `$SLURM_TMPDIR`，减少共享文件系统的小文件读取；结果和 checkpoint 写 project，作业结束不会随临时盘一起丢失。

## 先验证，再提交核心队列

以下均在集群仓库目录运行：

```bash
sbatch hpc/nibi/install_protocol.sbatch
# 上一项成功后，运行真实图片短检查（E00/E01/E02/E07/E08）。
sbatch hpc/nibi/smoke_resnet.sbatch
```

短检查按来源、类别和模态可用性分别选取少量真实图片；E07 包含 MILK 双图和 ISIC 缺失临床图样本，验证侧始终只含 MILK。每个配置预定两轮：第一轮保存后进程退出，检查部分完成记录；第二个进程从 `last.pt` 恢复到第二轮，再重新加载 `best.pt` 推理。`check_smoke_run.py` 核验样本覆盖、每轮洗牌顺序、恢复起点和读回预测一致性，之后进行完整折验证集推理。产物位于独立 `results/smoke-JOB_ID`，始终标为 smoke，不能进入正式结果表。

检查 CUDA、显存余量、数据读取速度及 checkpoint。`smoke_train.csv`、`smoke_val.csv` 和 `smoke_*_verification.json` 保存短检覆盖与恢复证据。单独使用 `--smoke` 仍默认一轮；两轮停止／恢复参数仅允许用于 smoke。默认请求 Nibi 已验证可用的 10 GB H100 MIG 类型，但新模型的显存和时间尚未实测。需要更大分片／完整卡或调整 batch 时，先改配置并记录版本，再开始正式 CV。

验证通过后再执行：

```bash
export CSC490_RUN_NAME=resnet9_v1_run01
sbatch hpc/nibi/train_array.sbatch
```

任务 0～4 为 E00，5～9 为 E01，10～14 为 E02；每任务单 GPU，最多并发两折。并发和 GPU 类型可经实测后调整。脚本不会自动提交对照实验。单折／E03～E09 使用 `train_one.sbatch`，设置 `CSC490_CONFIG`、`CSC490_FOLD`、`CSC490_RUN_NAME`。

关 PuTTY 不会停止 Slurm 训练。需要停止时在集群对具体作业使用 `scancel JOB_ID`。中断后保留最后完整 epoch 的 `last.pt`；不能恢复被打断 epoch 中途的 batch。确认无同目录作业后移除残留 `.running` 标记，并设置 `CSC490_RESUME=1` 重新提交对应任务。洗牌随机数独立于 worker 初始化，每轮按 seed 加 epoch 索引重置；样本增强仍由 seed、epoch 和样本索引确定。恢复校验配置、代码和清单指纹，任何变化都会拒绝；不能把仅加载权重当恢复。加速选项可能造成 GPU 浮点差异，不承诺 GPU 或跨设备逐位一致。

## 预测、汇总和最后的固定验证

每折保存 `run.json`、`config.json`、`metadata.json`、`history.json`、`best.pt`、`last.pt`、`predictions.csv` 和统一 `evaluation/`。模型选择看折验证 Macro F1，因此 CV 是开发选模估计。

```bash
python scripts/summarize_cv.py --runs "$CSC490_ROOT/results/$CSC490_RUN_NAME/E00" \
  --output "$CSC490_ROOT/results/$CSC490_RUN_NAME/E00/summary"
python scripts/collect_results.py --runs "$CSC490_ROOT/results/$CSC490_RUN_NAME" \
  --output "$CSC490_ROOT/results/$CSC490_RUN_NAME/experiment_status.csv"
```

汇总只接受同配置／代码／数据、全部五折完成、完整 OOF 的正式结果。逐折均值及样本标准差与 pooled OOF 分开报告；smoke 和缺折不能冒充 completed CV。

最终 refit 轮数预先规定为五折 best epoch 的中位数，不用 holdout 选轮数。设置 `CSC490_CV_SUMMARY` 和 `CSC490_REFIT_OUTPUT`，提交 `hpc/nibi/refit.sbatch`。完成后再在 GPU 作业中用 `src.infer` 推理固定 holdout：

```bash
python -m src.infer --checkpoint REFIT_DIR/best.pt --manifest DATA/manifests/milk_holdout.csv \
  --data-root "$CSC490_ROOT/data" --output REFIT_DIR/holdout_predictions.csv
python -m src.evaluate --class-schema 9 --id-column sample_id --labels DATA/manifests/milk_holdout.csv \
  --predictions REFIT_DIR/holdout_predictions.csv --output-dir REFIT_DIR/holdout_evaluation
```

holdout 推理拒绝未完成的 refit 或普通 fold 检查点。结果只用于最终报告，不回头调参。各实际阶段完成多少，就记录多少。
