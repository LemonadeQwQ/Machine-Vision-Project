# 新 ResNet 训练与集群操作

2026-10-31：E10 五折、10 个 checkpoint／5 次完整验证折重推理完成并验收，最大概率绝对差为 0。E00～E10 合计十一组、55 折，OOF 每组覆盖 3,968 例。见 [十一组结果摘要](../reports/experiments/20261031_e10/SUMMARY.md)。小组按五折平均九类 Macro F1 选择 E10 为主模型，并授权 E07 为对照；13／14 轮 refit 训练已完成并通过最终 checkpoint 的本地 CPU 核对，见 [完成记录](../reports/refit/20261031_e10_e07_v1/README.md)。固定 holdout 作业 25622516 已完成并核对；E10／E07 的 Macro F1 为 56.62%／53.28%，结果见 [holdout 摘要](../reports/holdout/20261031_holdout_e10_e07_v1/SUMMARY.md)；E10 CV 与 refit 已统一整理为 [31 日 private 交付](../reports/handoffs/20261031_delivery.md)。既有测试与初始化验收不重复运行。以下通用命令仅用于复现或后续明确授权的执行，须使用新运行目录，不能重复写入已完成结果。

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

核心为 E00～E02 共 15 折，第二优先 E03／E04。每折均重新加载相同 ImageNet V2 初始化、独立模型／优化器／训练侧元数据和权重统计。双分支参数不共享，2048 维特征拼接与可用性 mask 后进入 512 维分类头；没有门控、辅助头或辅助损失。单模态具有同样的 512 维头结构。仅 layer4 和新头可训练，骨干全部 BN 的运行统计固定，layer4 BN 仿射参数可学习。

图像 EXIF 校正后转 RGB，等比例缩放补黑边至 224×224；训练随机水平／垂直翻转、±20° 旋转，两模态用相同随机参数；验证无随机增强。旋转不扩展画布，角落内容可能被裁去，E09 关闭上述几何增强。归一化采用 ImageNet RGB 均值／标准差。

AdamW：骨干学习率 1e-5、分类头 1e-4、weight decay 1e-4、梯度范数上限 5。默认 batch 32、BF16、channels_last、TF32、4 个 worker、每 worker 预取 2 个 batch。候选上限 100 epoch，验证 Macro F1 连续 15 轮不提升则停止；相同分数保留较早 epoch。加权实验使用训练折 `w=(1/n)/mean(1/n)`，零计数拒绝。全部配置在 `configs/experiments/`；性能参数需集群真实短检后才能定为最终值。

## 上传后的目录

E10 在 E07 的数据和损失设置上启用元数据分支，不继承 E07 checkpoint。四个字段及缺失标记沿用 E08，但每折重新用 MILK 训练病例＋ISIC 池拟合年龄统计和词表；因此不复制 E08 的编码器参数。ISIC 没有临床图和肤色，保留缺失标记，验证集始终只含 MILK。E10 的五折调度、专属组合验证、汇总和复用验收接口分别见 `hpc/nibi/*e10*` 及 `audit_completed_expansion.py --experiments E10`。

E10 为注册新实验编号修改了 `src/experiment.py` 和 `scripts/collect_results.py`，来源指纹因而更新；模型、优化和数据处理主体未改。已有 E00～E09 的结果保留原指纹。后续若选择旧实验做 refit／恢复，应使用对应源码版本，不绕过源码一致性检查。

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

检查 CUDA、显存余量、数据读取速度及 checkpoint。`smoke_train.csv`、`smoke_val.csv` 和 `smoke_*_verification.json` 保存短检覆盖与恢复证据。单独使用 `--smoke` 仍默认一轮；两轮停止／恢复参数仅允许用于 smoke。29 日已实测 10 GB H100 MIG，核心双模态 PyTorch 预留峰值约 1.03 GiB，完整折第二轮训练加验证约 9.22 秒（不含 checkpoint／最终产物写入）。运行时固定 cuDNN 卷积算法选择，原预测读回阈值保持不变。以后改变硬件或 batch 等设置时，须先验证并记录版本，不能在同一五折中途混用配置。

验证通过后再执行：

```bash
export CSC490_RUN_NAME=resnet9_v1_run01
sbatch hpc/nibi/train_array.sbatch
```

任务 0～4 为 E00，5～9 为 E01，10～14 为 E02；每任务单 GPU，最多并发两折。并发和 GPU 类型可经实测后调整。脚本不会自动提交对照实验。单折／E03～E09 使用 `train_one.sbatch`，设置 `CSC490_CONFIG`、`CSC490_FOLD`、`CSC490_RUN_NAME`。

关 PuTTY 不会停止 Slurm 训练。需要停止时在集群对具体作业使用 `scancel JOB_ID`。中断后保留最后完整 epoch 的 `last.pt`；不能恢复被打断 epoch 中途的 batch。确认无同目录作业后移除残留 `.running` 标记，并设置 `CSC490_RESUME=1` 重新提交对应任务。洗牌随机数独立于 worker 初始化，每轮按 seed 加 epoch 索引重置；样本增强仍由 seed、epoch 和样本索引确定。恢复校验配置、代码和清单指纹，任何变化都会拒绝；不能把仅加载权重当恢复。加速选项可能造成 GPU 浮点差异，不承诺 GPU 或跨设备逐位一致。

## 30 日扩展队列及已完成验收

`train_expansion_array.sbatch` 将任务 0～34 按每五折一组映射到 E03～E09。`submit_expansion.py` 读取已完成的扩展验证记录并核对源码、清单、权重和配置哈希；先提交 E03／E04，再以成功依赖提交其余五组，两个数组各限两折并发且不重叠。正式提交为每折 4 CPU、24G RAM、H100 MIG 1g.10gb、4 小时时限；batch、worker 和数值设置保持冻结。

本次数组为 `25388914`、`25388916`，CPU 汇总为 `25388917`，GPU 验收为 `25388918`，全部成功。`audit_completed_expansion.py` 核对 70 个 checkpoint、独立初始化记录、冻结参数、优化器步数及 E08 逐折元数据；E07 的临床分支只在含 MILK 临床图的 batch 计优化步。35 个最佳模型完整验证折重推理通过，最大概率绝对差为 0。

结果收回由 `export_expansion_artifacts.py` 打包现有指标、审计和来源证据；`tools/report_expansion_cv.py` 合并已有核心结果生成十组对照，不训练或推理。实际提交、资源验证和收回哈希见 [30 日交接](../reports/handoffs/20261030_delivery.md)。现有运行已完成，复现须使用新的目录并另行授权。

## 预测、汇总和最后的固定验证

每折保存 `run.json`、`config.json`、`metadata.json`、`history.json`、`best.pt`、`last.pt`、`predictions.csv` 和统一 `evaluation/`。模型选择看折验证 Macro F1，因此 CV 是开发选模估计。

```bash
python scripts/summarize_cv.py --runs "$CSC490_ROOT/results/$CSC490_RUN_NAME/E00" \
  --output "$CSC490_ROOT/results/$CSC490_RUN_NAME/E00/summary"
python scripts/collect_results.py --runs "$CSC490_ROOT/results/$CSC490_RUN_NAME" \
  --output "$CSC490_ROOT/results/$CSC490_RUN_NAME/experiment_status.csv"
```

汇总只接受同配置／代码／数据、全部五折完成、完整 OOF 的正式结果。逐折均值及样本标准差与 pooled OOF 分开报告；smoke 和缺折不能冒充 completed CV。

最终 refit 轮数预先规定为五折 best epoch 的中位数，不用 holdout 选轮数。小组于 31 日按五折平均九类 Macro F1 选择 E10，其 best epoch 为 9、51、21、10、13，中位数 13。随后用户授权 E07 作为对照，固定 14 轮；两组已从各自匹配 CV 来源的运行快照完成训练并通过 CPU checkpoint 核对，资源为每组 H100 MIG 10GB、4 CPU、24G、2 小时时限。实际 `CSC490_CV_SUMMARY`、`CSC490_REFIT_OUTPUT`、来源路径和作业号见 [执行记录](../reports/refit/20261031_e10_e07_v1/README.md)，不要重复提交到这些输出目录。作业 25622516 已按以下接口完成两组固定 holdout，并核对全部 1,037 例概率及指标。以下命令为复现说明，不要重复写入已有结果：

```bash
python -m src.infer --checkpoint REFIT_DIR/best.pt --manifest DATA/manifests/milk_holdout.csv \
  --data-root "$CSC490_ROOT/data" --output REFIT_DIR/holdout_predictions.csv
python -m src.evaluate --class-schema 9 --id-column sample_id --labels DATA/manifests/milk_holdout.csv \
  --predictions REFIT_DIR/holdout_predictions.csv --output-dir REFIT_DIR/holdout_evaluation
```

holdout 推理拒绝未完成的 refit 或普通 fold 检查点。结果只用于最终报告，不回头调参。各实际阶段完成多少，就记录多少。
