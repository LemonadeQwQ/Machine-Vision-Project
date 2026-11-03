# E10 主模型／E07 对照：refit 执行记录

> 后续更新：用户已授权两个最终模型的固定 holdout 评估，作业 25622516 已完成并通过结果核对。E10／E07 Macro F1 为 56.62%／53.28%，见 [holdout 最终结果](../../holdout/20261031_holdout_e10_e07_v1/SUMMARY.md)。下文保留 refit 完成与回收阶段记录。

**2026-10-31 refit 回收记录（历史时点）：两组 refit 均完成，四个最终 checkpoint 的本地 CPU 内容及加载核对通过。结果和完整权重已回收；当时固定 holdout 尚未评估，后续完成状态见顶部更新。** 用户本轮要求检查结果，无问题后整理本地并提交、推送 private，取代提交训练时的暂不推送安排。统一交付见 [31 日交接](../../handoffs/20261031_delivery.md)。

最初用户授权双模型训练，确认提交后停止跟踪；随后主动要求回收验收。操作继续使用可见 PuTTY 和同一个 `csc490-1030` tmux。原始授权／提交回执保留当时状态，不改写为事后结果；最新完成证据为 [completion.json](completion.json)。本轮不提交固定 holdout 评估。

## 完成结果与核对

时间均为 2026-10-31 America/New_York。两组均在 `g30` 使用 H100 MIG 1g.10gb、4 CPU、24G RAM，01:00:07 启动。

| 模型 | Slurm ID | 完成轮数 | 结束时间 | 作业耗时 | 最终状态 |
|---|---|---:|---|---|---|
| E10 主模型 | 25562135 | 13 / 13 | 01:16:36 | 16 分 29 秒 | COMPLETED 0:0 |
| E07 对照 | 25562136 | 14 / 14 | 01:17:42 | 17 分 35 秒 | COMPLETED 0:0 |

每组训练 24,247 条（MILK 3,968＋ISIC 20,279），验证数为 0，均从 epoch 0 开始并按固定轮数结束。日志无训练错误，未生成验证／holdout 概率或评估文件。核对包括：

- 冻结源码、原 CV 汇总、清单、ImageNet 权重、配置与运行签名；全部逐轮记录连续且有限，训练采样顺序匹配。
- 完整训练侧的逆频率 CE 权重及 E10 的 28 维元数据；E07 无元数据。跨 Python 年龄标准差差异为 `3.0553e-13`，在已验收的 `1e-12` 绝对容差内；字段、词表和顺序严格一致。
- 四个 checkpoint 均可在 CPU 安全读取并严格装入匹配结构，`refit_complete=true`、epoch 为 13／14，历史、配置和元数据匹配。
- 每个 checkpoint 检查 576 个冻结张量，均与 ImageNet 一致；两条骨干的 layer4 均已更新，模型与优化器张量有限。E10 临床／其他参数优化步数为 9,827／9,854，E07 为 10,582／10,612，符合实际含临床图的 batch 数。
- 每组 best／last 的最终模型参数完全一致；两文件的容器哈希可以不同。初始化证据来自来源代码、`resume_from_epoch=0`、ImageNet 哈希与优化器步数，没有虚构未保存的训练前 checkpoint。

详情：[E10 审计](audit/E10.json)、[E07 审计](audit/E07.json)。这是 CPU 内容与加载核对，**未进行推理重放、holdout 评估、重新训练、smoke 或旧测试全量重跑**。

末轮在线训练指标如下，仅用于运行诊断。它们混合 MILK＋ISIC，并在训练更新中累计，不是固定最终模型在独立测试集上的分数，也不用于重新选择主模型。

| 模型 | 训练 loss | 在线训练 Macro F1 | 在线训练 Accuracy |
|---|---:|---:|---:|
| E10 | 0.256742 | 86.61% | 85.96% |
| E07 | 0.258741 | 85.77% | 85.08% |

已有开发 CV 均值仍为 E10 57.00%、E07 54.70% Macro F1；此处记录的是 refit 验收时刻，当时尚无 holdout 分数；后续最终结果见顶部更新。

## 本地回收与复核

`export_refit_artifacts.py` 在原共享 tmux 导出记录和来源快照，232 个文件逐一核对大小及 SHA-256。压缩包 SHA-256 为 `3f547b8220097c308efeded3b62aff83bc249b719eaee1af6129da89f65f6cfc`。四个权重另行下载，全部匹配远端 SHA-256，总计 1,814,528,188 字节（约 1.69 GiB）。

- 本地原始归档、两份完整日志、冻结运行快照：工作区 `.nibi_upload/refit-pair-v1/completed-artifacts/`。
- 本地完整权重：同目录 `checkpoints/E10/{best,last}.pt`、`checkpoints/E07/{best,last}.pt`。
- private 记录：本目录 `E10/`、`E07/` 的四类原始 JSON、`audit/`、`completion.json`、导出清单及回执；大权重、日志、压缩包和重复源码快照不进入 Git。
- `summary.csv`、`training_history.csv` 是原始训练记录的整理表，所有分数明确标记 online training。

不加载图像的本地复核命令（如需重做，使用新的审计输出路径）：

```powershell
.\.venv\Scripts\python.exe -B tools/audit_refit_artifacts.py --recovery ..\.nibi_upload\refit-pair-v1\completed-artifacts --experiment E10 --imagenet pretrained_weights/resnet50-11ad3fa6.pth --output runs/refit_reaudit/E10.json
```

E07 同理更换 experiment 和输出文件。脚本分别导入下载的原始训练源码，避免以 E10 的注册改动冒充 E07 的训练来源。

## 已冻结的选择与训练规则

| 配置 | 用途 | CV 最佳 epoch | 固定 refit epoch |
|---|---|---|---:|
| E10 | 主模型，按五折平均九类 Macro F1 选定 | 9、51、21、10、13 | 13 |
| E07 | 不使用元数据的对照模型 | 48、14、40、9、14 | 14 |

两组均使用全部 3,968 个 MILK 开发病例＋20,279 张 ISIC 训练图，共 24,247 条训练样本；每个模型重新初始化 ImageNet V2 骨干和优化器。类别权重按完整训练侧重算，E10 元数据编码器也在完整训练侧重新拟合。保留各自 CV 的其他配置，不验证早停，不读取 holdout 指标选轮数。角色在最终评估前固定；后续完整报告两者，不根据 holdout 分数反向改变主模型选择。

每组申请单个 H100 MIG 1g.10gb、4 CPU、24G RAM、2 小时时限。共两个独立训练作业，最多同时运行两个；资源不足时正常排队。没有重跑五折，也没有新增 smoke、初始化测试、数据重建或依赖安装。

## 来源与输出隔离

- E10 使用 CV 来源 `a8a74758b6f29c9c78ab838c7b3f8d7d565fdaf4674f9f90830463a8888d1e9d`；E07 使用旧 CV 来源 `72fd7a2d357c352eb45ca889dabed4d8cf86057112308693a19399d7bc24d19a`。
- 运行快照放在 Nibi `/project/def-ajbonner/yuchenx4/csc490/refit_sources/20261031_refit_e10_e07_v1/E10` 和 `E07`。E07 从已保存的 `snapshots/pre-e10-c7aeced` 读取，E10 从当前已验收源码复制；每组绑定原 CV 汇总及验收记录、完整冻结清单和逐文件哈希。运行文件设为仅所有者可读，原主目录、分支及旧 CV 结果不变。
- 使用快照内既有的 `hpc/nibi/refit.sbatch` 和 `src/refit.py`，没有修改训练算法。通过 sbatch 参数覆盖通用脚本的资源默认值，显式 `CSC490_RESUME=0`。
- 输出根目录为 `/project/def-ajbonner/yuchenx4/csc490/results/20261031_refit_e10_e07_v1`，两组分别写入 `E10/` 与 `E07/`，不覆盖任何 CV 结果或已有权重。

## 提交与查看（保留提交时记录）

**10 月 31 日 00:57:13（America/New_York）提交完成。** Slurm 控制器已确认两项作业均已接受，核对时均为 `PENDING`，CPU、内存、GPU、2 小时时限和各自来源目录全部匹配。

| 作业 | Slurm ID | 固定训练轮数 | 提交核对时状态 |
|---|---|---:|---|
| E10 主模型 | `25562135` | 13 | PENDING |
| E07 对照 | `25562136` | 14 | PENDING |

授权与配置见 [authorization_plan.json](authorization_plan.json)，来源准备见 [source_preparation.json](source_preparation.json)，实际提交命令和路径见 [submission.json](submission.json)，Slurm 控制器核对见 [controller_confirmation.json](controller_confirmation.json)。只有两组 refit 训练，不挂接 holdout 作业。

共享 tmux 的 `4:refit` 页面用于查看训练 epoch、训练 loss／F1 和队列状态：按 `Ctrl+b`，松开后按 `4`。refit 没有验证集，页面的训练 F1 不能当作新 CV 或 holdout 成绩。训练已完成并按上文验收，之后单独授权的固定 holdout 评估也已完成，见顶部更新。

本节保留当时的提交状态；`authorization_plan.json`、`source_preparation.json`、`submission.json`、`controller_confirmation.json` 的旧状态不是实时状态，也不能覆盖顶部完成结论。
