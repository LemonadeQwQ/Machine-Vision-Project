# 分工、实现与复核状态

## 2026-11-22 最终模型与结果交付责任

本轮交付内容为 E14-GPM 第 35 轮最终模型的恢复说明、九档探索性对照及验证证据摘要。已有 E14 A/B/C 实现来源保留于下文 11 月 16 日记录。本轮每个目标文件由一名负责人维护。

| 角色 | 路径数 | 负责上传的文件 |
|---|---:|---|
| A | 1 | `docs/data_protocol.md` |
| B | 1 | `docs/training.md` |
| C | 7 | `README.md`、`docs/model_selection.md`、`docs/experiment_plan.md`、`docs/contributions.md`、`.gitattributes`、`reports/experiments/20261122_e14_gpm_final/SUMMARY.md`、`reports/experiments/20261122_e14_gpm_final/comparison_metrics.csv` |

A 复核冻结样本成员、数据单位与 holdout 参与最终选择的记录。B 复核第 35 轮真实身份、官方资产和模型恢复步骤。C 复核九档聚合指标、病例配对区间、原正式第 46 轮基线与当前文档状态，并统一处理共享文档链接。

本轮准备版本的 public 基线为 `8a2735657b4f4338b8a9d0a727c9731c743878a0`。2026-11-24 的 C 分支 `docs/c-1124-final-gpm-results` 交付上表 C 的 7 个路径，来源为 private 提交 `d39e35564acc3c35ccdf6f9a2388e1c532415909`。A／B 本轮各 1 份文件仍由各自负责人复核提交；C 分支中的训练说明链接沿用 main，待 B 的第 35 轮恢复说明合入后更新其内容。public main 合并、CI 与人工交叉复核按各自实际完成记录确认。

## 2026-11-16：E14 A/B/C 整合

以 public main `196c7d0` 为基线，整合 A `c2374a4`、B `c8c9129`、C `074400f`，保留三个原始提交。共 27 个不重叠路径，分工如下。

| 角色 | 路径数 | 内容 |
|---|---:|---|
| A | 1 | 冻结输入、缺失模态与完整训练池 refit 协议 |
| B | 7 | 四配置、注册表、中性 Gate 初始化与训练说明 |
| C | 19 | readiness、测试、聚合结果、CI 与共享文档 |

### A 的文件

- `docs/data_protocol.md`

### B 的文件

- `configs/experiments/E14-GP_pair_gate_panderm.yaml`
- `configs/experiments/E14-GM_pair_gate_monet.yaml`
- `configs/experiments/E14-PM_pair_panderm_monet.yaml`
- `configs/experiments/E14-GPM_pair_gate_panderm_monet.yaml`
- `src/experiment_registry.py`
- `src/models/next_stage.py`
- `docs/training.md`

### C 的文件

- `scripts/check_training_ready.py`
- `tests/test_e11_e13.py`
- `tests/test_resnet_protocol.py`
- `tests/test_e14_pairs.py`
- `tests/test_e14_gpm.py`
- `tests/test_e14_gpm_refit.py`
- `README.md`
- `docs/contributions.md`
- `docs/experiment_plan.md`
- `docs/model_selection.md`
- `docs/e14_pairs_implementation.md`
- `docs/e14_gpm_implementation.md`
- `.github/workflows/tests.yml`
- `.gitignore`
- `.gitattributes`
- `reports/experiments/20261115_e11_e14_summary/SUMMARY.md`
- `reports/experiments/20261115_e11_e14_summary/summary_metrics.csv`
- `reports/experiments/20261115_e11_e14_summary/fold_metrics.csv`
- `reports/experiments/20261115_e11_e14_summary/comparison_macro_f1.png`

共享 README、CI、综合测试与聚合报告由 C 维护。已接入 B 的全部依赖；先前 C 单分支缺依赖的 CI 失败属于历史记录。

当前整合候选的依赖预检及七组 CPU 离线测试共 **61 项通过**：协议20、schema4、评估12、E11–E13 17、pairs4、GPM3、refit1。测试直接使用本候选，没有临时覆盖 B 文件；仅人工数据，无依赖安装或真实训练。23 份共享配置／schema 与内部版本字节一致，核心模型、训练、推理与 refit 逻辑一致；8 组聚合成绩和40条逐折指标、来源哈希均精确匹配。远程 CI 以本次整合提交的实际结果为准。

核心模型／配置和聚合指标与内部实现按同一协议核对；公开评估的显式 labels 规则、原有 CLI 回归及通用绘图功能保持原样。历史 GPU 训练来源与当前公开文件树的完整指纹不同，refit 的来源守卫保留。

新 35 折结果沿用已记录的精简验收范围，没有宣称全部正式模型重载／重推理。GPM refit 最后保存状态为排队，本交付无新 holdout 成绩。组员本人复核仍为 pending；实现由集成人在 AI 协助下统一完成，分工表示接收／复核／提交责任。

## 此前交付记录

更新日期：2026-11-15。实现、运行与整理由集成人统一完成，AI 协助。A／B／C 是组员接收代码、复核和提交 public 的责任范围，不是三套独立项目，也不代表这些实现已经分别由三位组员独立完成。

## 11 月 15 日 E11～E13 的 A／B／C 整合（此前交付历史）

本次以 public main `cfafd05` 为共同基线，按 A → B → C 合并为同一候选，
保留下面的原始提交和作者；没有合并 private 整仓历史或 E14 增量。

| 负责人 | 原始提交与分支 | 路径数 | 本次交付 |
|---|---|---:|---|
| A | `bd6205b`，`feat/a-1115-e11-e13-data`，[PR #7](https://github.com/LemonadeQwQ/Machine-Vision-Project/pull/7) | 2 | `src/data/common.py`、`docs/data_protocol.md`：来源指纹和冻结数据协议 |
| B | `63eeb8b`，`feat/b-1110-e11-e13` | 16 | 注册表、模型工厂／接入器、三份配置、训练／推理／CV、固定依赖、第三方来源／许可；完整路径见实现契约 |
| C | `e047761`，`feat/c-1110-e11-e13-evaluation` | 11 | 下节逐文件清单：汇总／readiness、测试、CI 和公开文档 |

三方共 29 个不重叠路径。集成修订仅调整这些交付文件中的状态与公开说明，
澄清 MONET 当前使用保持比例补边的完整视图，以及正式结果未随本次交付发布的边界。
模型代码、固定依赖版本、历史 E00～E10 结果和 public 显式标签接口保持交付内容。

本次整合候选本地四组离线测试共 **53 项通过**：协议 20、schema 4、原评估 12、E11～E13 17。
使用人工输入和已有 CPU 环境，无新增依赖、权重下载或真实训练。
CLI、路径范围、来源祖先及公开内容检查的回执随整合审核保存；
远程 CI 以实际整合 PR 对应提交为准，不借用 A 单分支或私有测试的通过记录。
A／B／C 组员人工交叉复核仍为 **pending**；代码提交及自动测试不代替本人复核。
本次 public 不附新逐病例资料、模型权重、集群运行文件或 E11～E13 正式 CV 结果。

## 11 月 10 日 C 的 E11～E13 分支交付（历史记录）

11 月 10 日交付基于 public main `cfafd05`，分支为 `feat/c-1110-e11-e13-evaluation`。
当时仅领取并公开适配 C 的内容，A／B 新依赖尚未进入公开基线，因此该单分支
不能独立通过完整验收。此历史限制已由上节的完整整合取代；以下保留原交付清单。

C 本次提交以下 **11 个文件**：

| 文件 | 提交内容 |
|---|---|
| `scripts/collect_results.py` | E00～E13 的 70 个折位置；拒绝非 CV 阶段及错误新实验身份 |
| `scripts/summarize_cv.py` | 校验新组件／模型契约；保存各折元数据维度与统一比较身份 |
| `scripts/check_training_ready.py` | 显式选择新实验及本地基础模型资产；保护历史默认报告 |
| `tests/test_e11_e13.py` | 17 项公开人工输入测试；不包含私有集群启动守卫 |
| `tests/test_resnet_protocol.py` | 计划位置断言从 55 更新为 70；保留原有检查 |
| `docs/e11_e13_implementation.md` | 公开模型契约、前置依赖、资产及验证说明 |
| `README.md` | 说明 C 分支状态、依赖和公开使用入口 |
| `.gitattributes` | 保留源码与第三方来源／许可的指纹字节 |
| `.github/workflows/tests.yml` | 保留原测试与只读权限，增加依赖检查及 E11～E13 测试 |
| `docs/experiment_plan.md` | 三个独立增量的计划及结果边界 |
| `docs/contributions.md` | 本清单、归属与实际验证状态 |

A 负责补充 `src/data/common.py::training_source_files` 及数据协议说明；B 负责注册表、
模型工厂与接入器、三份 YAML、训练／推理／CV 接口、基础模型依赖及第三方来源／许可。
详细路径见 [依赖清单](e11_e13_implementation.md)。各人只提交自己的范围，再共同验收。

公开适配保留 `src/evaluate.py`、`src/metrics.py` 的显式标签接口和现有 CLI 回归；
不携带部署脚本、账号路径、作业日志、检查点、图片或新逐病例数据。
C 单独交付时的验证：11 个文件范围一致，Python 语法、28 个本地文档链接和新增内容隐私检查通过；
收集器的 8 项纯内存检查通过，覆盖 70 个 planned 位置、运行中状态及非法阶段／身份拒绝。
内存检查只读使用尚未交付的注册表作为依赖夹具，不等同于公开整仓可运行或集成测试通过。
当时未生成本地测试材料，完整离线集成测试需等待 A／B；C 单分支 CI 会报告缺失依赖。
当前完整候选的实际测试见上节，不把私有测试数量当作 public 数量。
组员人工交叉复核仍为 **pending**。下面的 11 月 9 日测试数字仅描述此前 E00～E10 整合。

28 日 public 已整合三位各自分支的交付，保留 A `ccb78a3`、B `7302c20`、C `b3e15a7` 的提交记录；main 基线为 `62ecb70`。当时由 C 授权合并，Codex 协助依赖核对、公开文档适配和自动检查。29 日本版本已整合 A `0bacf4a`、B `ccc10bd`、C `df34478`，并完成公共文档的集成适配；保留原始提交和作者。分支提交和自动测试不代表组员已完成人工交叉复核，下表据实保留 pending。

## 11 月 9 日完整 public 整合

本次在 29 日 public main `6571dbd` 上整合以下交付，保留原提交与作者：

| 组员 | 原提交 | 路径数 | 实际内容 |
|---|---|---:|---|
| A | `29caf7a` | 4 | 数据协议／来源、元数据字段与缺失率、最终聚合数据验收 |
| B | `ed741cb`（PR #5） | 7 | E10 配置与编号注册、训练／选模／计算说明、双模型 refit 摘要 |
| C | `89e2bc0` | 76 | 55 折状态、CV／holdout 聚合表图、结果收集与绘图、协议测试和公共说明 |

A 与 B 分别通过 merge commit 接入 C 的公开交付，未合并 private 仓库历史。
B 基于较早公开版本产生两处文档冲突；整合保留通用 Python 接口和 public 特有评估行为，
移除最终文档中的内部路径／失效链接，更新 refit／holdout 状态，并补全 ISIC 实验列表中的 E10。
两位组员的原始提交仍在历史中；当前文件树体现公开适配后的结果。

11 月 1 日 C 单独提交时按要求暂缓测试、等待 A／B；该历史限制已由本次完整整合取代。
当前包含 E10 配置及训练入口，公共 CI 与两项额外 labels CLI 回归保持原样。
训练来源指纹和聚合数值不随本次整合修改；57 个 C 直接领取的聚合文件仍保持原字节。

本次整合候选本地验证已完成：

- 三组离线测试共 36 项通过：协议 20、schema 4、原评估兼容 12。
- 六个 CLI 入口的 help 检查通过；不加载真实图片、权重或执行训练。
- 三组与十一组聚合绘图均成功；修复旧三组 JSON 缺少 description 字段的兼容问题。
- 87 个交付路径范围一致，57 个直接复制的聚合文件保持原字节，7 个 public 接口／测试／CI 文件未覆盖。
- 99 个本地文档链接有效，最终文件树未新增逐病例清单／概率、原图、权重、完整日志或内部部署路径。

远程 CI 以实际整合 PR 对应提交为准。自动检查不代表组员已进行人工交叉复核，
所有未留下本人复核记录的项目仍为 pending。完整独立复现、报告和发布标签不由代码合并自动确认。

## 状态定义

- planned：工作已计划，尚未完成实现或实际运行。
- implemented：代码／文档已经落地；运行或验证证据须另列。
- completed：实际运行或交付完成，证据已保存；不表示组员已人工复核。
- in progress：交付已开始，依赖或合并尚未完成。
- reviewed：指定组员已实际复核并留下记录；不能把 AI 检查或测试通过等同于组员签收。

## 责任与实际状态（E00～E10 历史交付）

| 范围 | 接收和提交负责人 | 实现状态 | 当前证据 | 组员复核状态 |
|---|---|---|---|---|
| 九类映射、MILK／ISIC 输入、分组和清单 | A | implemented | 真实输入检查，50 个实验／折预检查；开发 3,968、holdout 1,037、ISIC 20,279 | pending：C 核对标签／划分，B 核对 batch |
| 单模态、简单融合、损失、训练、恢复和推理 | B | implemented | 人工前反向、冻结／BN、缺失模态、检查点；0／4 workers 恢复回归 | pending：A 核对输入，C 核对输出 |
| 九类评估、CV 汇总、自动检查与公共说明 | C | implemented | 私有整仓 107 项离线测试和 3 项初始化检查通过 | pending：B 核对概率，A 核对样本与折数 |
| Nibi 部署与新流程真实 GPU 短检 | 集成人执行，B 复核 | completed | 五组 smoke、资源实测及最佳模型重载验收通过 | pending：组员人工复核 |
| E00／E01／E02 核心 15 折及正式指标 | 集成人执行，C 汇总 | completed | 15 折、三组 OOF 各 3,968 例；30 个 checkpoint 审计及 15 次完整验证折重推理一致 | pending：组员人工复核 |
| E03～E09、E10 五折及最终 refit／holdout | 集成人执行，C 发布聚合结果 | completed：真实实验 | 共 55 折；E10／E07 refit 13／14 轮，holdout 各 1,037 例；公开 A／B／C 增量已整合 | pending |
| 28 日 public 交付与整合 | A／B／C | completed | A 17、B 24、C 19 个路径已整合；远程 CI 与合并记录见 [PR #1](https://github.com/LemonadeQwQ/Machine-Vision-Project/pull/1) | pending：组员人工交叉复核另行记录 |
| 29 日 A 数据验收增量 | A | completed：本版本已整合 | `0bacf4a`，3 个数据来源／协议／聚合审计路径，原提交保留 | pending |
| 29 日 B 训练修复与说明 | B | completed：本版本已整合 | `ccc10bd`，5 个路径；保留训练修复，集成时将部署说明适配为通用路径与公开 CLI | pending |
| 29 日 C 结果与文档增量 | C | completed：本版本已整合 | `df34478`，23 个已核验聚合文件／工具和 6 个公共适配文件；后续集成提交更新三方状态与文档链接 | pending：B 核对概率，A 核对样本与折数 |

## 验证范围与记录

28 日 C 单独交付时，曾临时加入 A／B 依赖验证 35 项测试；该记录只是当时的整合预览。28 日正式接入 A／B 分支后，直接在 public 整合工作区运行相同三组检查：`test_resnet_protocol.py` 19 项、`test_evaluation_schemas.py` 4 项、原有 `test_evaluate.py` 12 项，本地全部通过。该次远程 CI 及合并状态见 [整合 PR](https://github.com/LemonadeQwQ/Machine-Vision-Project/pull/1)，不能沿用为 29 日最终合并版本通过，也不能当成真实 GPU 训练验证。

C 公开版保留旧 11 类评估的默认标签位置；新 schema 或 `sample_id` 必须显式指定标签文件，并保留两项回归测试。历史回放工具显式提供 archive／labels。29 日只新增允许公开的聚合训练结果，不发布内部部署材料、新逐病例清单、逐病例概率、原始运行日志或完整权重。

29 日核心运行的指标、实际设置和来源指纹见 [结果摘要](../reports/experiments/20261029_core/SUMMARY.md)。本版本保留公开版评估器和 CI，A／B 原始提交已通过合并接入。最终整合候选需实际运行三组测试并核对对应提交的 CI，不能沿用单分支结果；实际记录见本次整合 PR。组员手工交叉复核仍保持 pending。

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
