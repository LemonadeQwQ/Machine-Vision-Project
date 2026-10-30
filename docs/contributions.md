# 分工、实现与复核状态

更新日期：2026-10-29。实现、运行与整理由集成人统一完成，AI 协助。A／B／C 是组员接收代码、复核和提交 public 的责任范围，不是三套独立项目，也不代表这些实现已经分别由三位组员独立完成。

28 日 public 已整合三位各自分支的交付，保留 A `ccb78a3`、B `7302c20`、C `b3e15a7` 的提交记录；main 基线为 `62ecb70`。当时由 C 授权合并，Codex 协助依赖核对、公开文档适配和自动检查。29 日增量尚未完成三方合并；分支提交和测试不代表组员已完成人工交叉复核，下表据实保留 pending。

## 状态定义

- planned：工作已计划，尚未完成实现或实际运行。
- implemented：代码／文档已经落地；运行或验证证据须另列。
- completed：实际运行或交付完成，证据已保存；不表示组员已人工复核。
- in progress：交付已开始，依赖或合并尚未完成。
- reviewed：指定组员已实际复核并留下记录；不能把 AI 检查或测试通过等同于组员签收。

## 责任与实际状态

| 范围 | 接收和提交负责人 | 实现状态 | 当前证据 | 组员复核状态 |
|---|---|---|---|---|
| 九类映射、MILK／ISIC 输入、分组和清单 | A | implemented | 真实输入检查，50 个实验／折预检查；开发 3,968、holdout 1,037、ISIC 20,279 | pending：C 核对标签／划分，B 核对 batch |
| 单模态、简单融合、损失、训练、恢复和推理 | B | implemented | 人工前反向、冻结／BN、缺失模态、检查点；0／4 workers 恢复回归 | pending：A 核对输入，C 核对输出 |
| 九类评估、CV 汇总、自动检查与公共说明 | C | implemented | 私有整仓 107 项离线测试和 3 项初始化检查通过 | pending：B 核对概率，A 核对样本与折数 |
| Nibi 部署与新流程真实 GPU 短检 | 集成人执行，B 复核 | completed | 五组 smoke、资源实测及最佳模型重载验收通过 | pending：组员人工复核 |
| E00／E01／E02 核心 15 折及正式指标 | 集成人执行，C 汇总 | completed | 15 折、三组 OOF 各 3,968 例；30 个 checkpoint 审计及 15 次完整验证折重推理一致 | pending：组员人工复核 |
| E03～E09、最终 refit 与 holdout 评估 | 集成人后续执行 | planned | 接口和配置提前实现，实验尚未运行 | pending |
| 28 日 public 交付与整合 | A／B／C | completed | A 17、B 24、C 19 个路径已整合；远程 CI 与合并记录见 [PR #1](https://github.com/LemonadeQwQ/Machine-Vision-Project/pull/1) | pending：组员人工交叉复核另行记录 |
| 29 日 A 数据验收增量 | A | in progress | `feat/a-1029-data-acceptance` 已推送 `0bacf4a`，3 个路径，尚未进入本分支的 main 基线 | pending |
| 29 日 B 训练修复与说明 | B | planned：public 接收 | cuDNN 修复、模型选择与训练／CV／资源说明共 5 个路径，待提交；内部训练已使用该修复 | pending |
| 29 日 C 结果与文档增量 | C | implemented：本分支 | 23 个已核验聚合文件／工具直接接收，6 个公共文件适配，共 29 个路径；等待 A／B／C 最终整合 | pending：B 核对概率，A 核对样本与折数 |

## 验证范围与记录

28 日 C 单独交付时，曾临时加入 A／B 依赖验证 35 项测试；该记录只是当时的整合预览。28 日正式接入 A／B 分支后，直接在 public 整合工作区运行相同三组检查：`test_resnet_protocol.py` 19 项、`test_evaluation_schemas.py` 4 项、原有 `test_evaluate.py` 12 项，本地全部通过。该次远程 CI 及合并状态见 [整合 PR](https://github.com/LemonadeQwQ/Machine-Vision-Project/pull/1)，不能沿用为 29 日最终合并版本通过，也不能当成真实 GPU 训练验证。

C 公开版保留旧 11 类评估的默认标签位置；新 schema 或 `sample_id` 必须显式指定标签文件，并保留两项回归测试。历史回放工具显式提供 archive／labels。29 日只新增允许公开的聚合训练结果，不发布内部部署材料、新逐病例清单、逐病例概率、原始运行日志或完整权重。

29 日核心运行的指标、实际设置和来源指纹见 [结果摘要](../reports/experiments/20261029_core/SUMMARY.md)。C 分支沿用现有评估器与 CI；本分支的检查只覆盖当前提交，待 A／B 接入后还要在最终候选上重跑三组测试并核对 CI。训练实际使用的 cuDNN 修复属于 B，C 不提前复制其代码或把手工交叉复核标为完成。

## 文件所有权

| 负责人 | 负责的主要路径 | 共享文件处理 |
|---|---|---|
| A | `configs/data/`、九类 schema／映射、`src/data/`、`src/class_schema.py`、`src/dataset.py`、`src/Image_tensor.py`、`src/metadata_features.py`、`scripts/verify_source_files.py`、`docs/data_protocol.md`、允许公开的清单统计与哈希 | A 统一维护标签与数据接口；不得把新数据准备改回旧临床单图路径 |
| B | `src/models/fusion.py`、`src/experiment.py`、`src/losses.py`、`src/infer.py`、`src/cross_validate.py`、`src/refit.py`、`src/train.py`、`src/engine.py`、`src/models/resnet50.py`、实验配置、权重获取代码、训练依赖与说明 | 旧 train／engine 仍是兼容依赖，新协议实际入口是 experiment；Nibi 专用资料留在 private |
| C | `src/evaluate.py`、`src/metrics.py`、汇总／预检查脚本、协议与评估测试、CI、评估／计划／分工文档、README、忽略与换行规则 | 综合协议测试由 C 一次提交，A／B 复核各自测试内容；README／CI 由 C 整合，避免三人重复覆盖 |

## 交付与复核流程

全部完成内容先作为一个完整版本提交 private，并附一份按实际路径编写的 A／B／C 文件清单。组员从同一版本领取自己的部分；public 保持独立历史，不整仓合并 private，也不从 private 的整合提交直接 cherry-pick 全部内容。

public 的既有基础代码与 private 存在差异。领取时包含清单指定的共享依赖；三组共同形成可运行版本，单组 PR 不宣称已独立通过整合检查。所有共享路径只由指定负责人提交，其他人提出复核意见。

逐病例清单默认留在 private；public 先放生成代码、汇总计数与文件哈希，若后续确认可公开，再由 A 单独处理。原图、权重、缓存、环境、完整运行日志、个人路径和 `read.txt` 不纳入本轮 public 提交。

组员复核后在 PR 记录所用 private 版本、实际领取文件、运行命令与结果、未运行事项，并在此表更新自己的 reviewed 状态。public 远程 CI 只有实际成功后才能标为通过。私有整仓的 107 项测试包含历史阶段，不能把这个数量直接声称为 public 精简交付的测试数量。
