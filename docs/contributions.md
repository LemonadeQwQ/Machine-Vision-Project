# 分工、实现与复核状态

更新日期：2026-10-28。实现、运行与整理由集成人统一完成，AI 协助。A／B／C 是组员接收代码、复核和提交 public 的责任范围，不是三套独立项目，也不代表这些实现已经分别由三位组员独立完成。

本 public 分支由 C 发起交付，Codex 协助整理、适配和验证。A／B 的新依赖尚未进入 public；下表的 implemented 指完整内部版本的实现状态，不表示 C 独立完成或本分支包含 A／B 的文件。人工交叉复核仍须由组员实际完成。

## 状态定义

- planned：工作已计划，尚未完成实现或实际运行。
- implemented：代码／文档已经落地；运行或验证证据须另列。
- in progress：交付已开始，依赖或合并尚未完成。
- reviewed：指定组员已实际复核并留下记录；不能把 AI 检查或测试通过等同于组员签收。

## 责任与实际状态

| 范围 | 接收和提交负责人 | 实现状态 | 当前证据 | 组员复核状态 |
|---|---|---|---|---|
| 九类映射、MILK／ISIC 输入、分组和清单 | A | implemented | 真实输入检查，50 个实验／折预检查；开发 3,968、holdout 1,037、ISIC 20,279 | pending：C 核对标签／划分，B 核对 batch |
| 单模态、简单融合、损失、训练、恢复和推理 | B | implemented | 人工前反向、冻结／BN、缺失模态、检查点；0／4 workers 恢复回归 | pending：A 核对输入，C 核对输出 |
| 九类评估、CV 汇总、自动检查与公共说明 | C | implemented | 私有整仓 107 项离线测试和 3 项初始化检查通过 | pending：B 核对概率，A 核对样本与折数 |
| Nibi 部署与新流程真实 GPU 短检 | 集成人后续执行，B 记录 | planned | 脚本已准备；当前没有新流程 GPU 验证结果 | pending |
| E00／E01／E02 核心 15 折及正式指标 | 集成人后续执行 | planned | 尚无本轮真实训练成绩 | pending |
| E03～E09、最终 refit 与 holdout 评估 | 集成人后续执行 | planned | 接口和配置提前实现，实验尚未运行 | pending |
| public 领取、合并与远程 CI | A／B／C | in progress | C 的 19 个路径在本分支交付；A／B 待接入；CI 显式检查缺失依赖 | pending：三组齐全并通过整合检查后合并 |

## C 本次交付的验证

基于 public 原有文件、C 本次改动及临时加入的 A／B 依赖，在本地 CPU 环境完成 35 项检查：`test_resnet_protocol.py` 19 项、`test_evaluation_schemas.py` 4 项、原有 `test_evaluate.py` 12 项，全部通过。临时依赖仅用于验证，没有加入 C 分支。该结果不是缺少 A／B 的 C 分支独立通过，也不是远程 CI 或 GPU 验证。

C 公开版保留旧 11 类评估的默认标签位置；新 schema 或 `sample_id` 必须显式指定标签文件，并增加两项回归测试。历史回放工具改为显式提供 archive／labels。README、评估说明、CI 和忽略规则按公开范围适配，不发布内部部署材料、逐病例新清单或训练结果。

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
