# E11～E13 公开实现契约与 C 交付

更新日期：2026-11-10。本文件说明 C 的公开增量及其要求的 A／B 接口。
**当前分支缺少 A／B 新依赖，尚不能独立运行新检查或实验。** 完整离线测试和人工复核待整合；
本次没有发布 E11～E13 正式成绩，不以历史 Gate／PanDerm／MONET 记录替代新结果。
精确的 C 文件清单见 [贡献说明](contributions.md)，历史结果见 [实验计划](experiment_plan.md)。

## 必须一起整合的依赖

| 负责人 | 文件／接口 | 本次 C 分支的边界 |
|---|---|---|
| A | `src/data/common.py` 的 `training_source_files()`；`docs/data_protocol.md` | A 提交源码指纹与数据说明；C 不覆盖 |
| B | `src/experiment_registry.py` | E00～E13 注册、单一增量身份与 run 契约验证 |
| B | `src/models/factory.py`、`src/models/foundation.py`、`src/models/next_stage.py` | 工厂、v2 checkpoint、Gate／PanDerm／MONET 接入 |
| B | `configs/experiments/E11_pair_gate.yaml`、`E12_pair_panderm.yaml`、`E13_pair_monet.yaml`（同目录） | 三份独立配置 |
| B | `src/experiment.py`、`src/infer.py`、`src/cross_validate.py` | 在既有文件中接入新工厂、来源记录与恢复／推理 |
| B | `requirements-foundation.txt` | 固定基础模型接入依赖 |
| B | `stages/panderm_stage/vendor/modeling_finetune.py`、`stages/panderm_stage/vendor/SOURCE.md` | vendored PanDerm 实现及来源记录 |
| B | `stages/monet_stage/reference/ATTRIBUTION.md`、`stages/monet_stage/reference/MONET_LICENSE.txt` | MONET 来源和许可 |

CI 在安装依赖之前检查上述新文件和 A 的函数是否存在，然后运行全部公开测试。
文件存在不代表接口已正确接入；最终以整合版本的实际测试为准。缺失时明确失败，不跳过新测试。
原有 public 评估接口要求显式 labels，`src/evaluate.py`、`src/metrics.py` 及原 CLI 回归保持原样。

## 固定比较与模型定义

类别顺序为 `AKIEC,BCC,BKL,DF,INF,MEL,NV,SCCKA,VASC`。沿用原 3,968 例 MILK 开发集、
1,037 例固定 holdout、20,279 张仅进入训练侧的 ISIC 和原五折成员。
三组均以 E10 的双图、加权 CE、ISIC、训练折拟合元数据与增强为基础，各只开启一项增量：

- **E11 Gate**：输入两个 2,048 维原始特征及两个可用性 mask，经过 128 维隐藏层、ReLU 和
  两个输出；缩放为 `2 * sigmoid`。末层权重与偏置为零，初始缩放恰为 1。
  缺临床图时不运行临床编码器，其特征为零，皮肤镜缩放固定为 1。
- **E12 PanDerm**：仅替换皮肤镜骨干为 PanDerm_Base（ViT-B/16），保留原始 768 维输出；
  临床 ResNet 仍为 2,048 维。仅最后两个 transformer blocks 和最终 norm 可训练，drop path 为 0。
  不添加辅助分类头或用投影把两路强制改成同维。
- **E13 MONET**：保留双 ResNet，追加冻结的四概念 `ulceration_crust`、`vasculature_vessels`、
  `erythema`、`pigmented`，经 `Linear(4,32) → LayerNorm → GELU` 后拼接。
  MONET 读取与皮肤镜骨干相同的增强后 224×224、ImageNet 归一化张量，不另做 processor 裁剪。
  CUDA 视觉编码使用 FP16，CPU 使用 FP32，概念评分以 FP32 计算；文本只编码一次。
  每个概念独立比较 target/reference，以逆温度 `exp(4.5944)` 的 sigmoid 评分并平均模板，
  同义词文本嵌入先平均；不对四概念做 softmax。

输入保持比例补边，双图共享几何增强。ResNet 训练 layer4，冻结其余骨干及 BatchNorm 统计；
PanDerm 冻结范围按上文。AdamW 骨干学习率 `1e-5`、头部 `1e-4`、weight decay `1e-4`，
batch 32、4 workers、BF16、channels_last、TF32，种子 42。每折最多 100 轮，
验证九类 Macro F1 连续 15 轮无提升则停止，同分保留较早轮。加权 CE 和元数据仅拟合训练侧。
这些是固定比较设置，不是本分支完成 GPU 可行性或完整 CV 验证的声明。

E14 尚未注册；任何组合需另行明确配置和身份，不得冒用 E11～E13。
不以 holdout 选择新增结构或调参；本交付不提供新 refit／holdout 结果。

## 本地资产与来源契约

以下指纹是接入器要求的资产身份，权重本体不在仓库中：

| 组件 | 固定身份 |
|---|---|
| ResNet-50 ImageNet V2 | SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |
| PanDerm_Base | SHA256 `be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1` |
| MONET `chanwkim/monet` | revision `1d1efd0b8d61bde82cde22d2e93c28d73441f29e`；模型 SHA256 `f5b874fff0aa8119484a75a497c997da8b4ff300a8c81c838aab893f301390af` |

MONET 目录必须包含 `config.json`、`merges.txt`、`model.safetensors`、`preprocessor_config.json`、
`special_tokens_map.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`。
`download_manifest.json` 保存 repository、revision 及全部八个文件的 SHA256 映射。
接入器拒绝哈希不符或额外的非约定文件；不得通过重写旧回执掩盖资产变化。
不执行远程模型代码。固定依赖包括 timm 0.9.16、transformers 4.46.3、tokenizers 0.20.3、
huggingface-hub 0.26.2、safetensors 0.4.5 和 regex 2024.11.6，以 B 的依赖文件为准。

新工厂使用 v2 checkpoint，记录配置身份、模型契约、输入组件和来源指纹。
E13 不重复保存冻结的 MONET 视觉／投影权重，恢复时仍需对应外部资产；保存的文本 buffers
另有摘要检查。E12 checkpoint 包含模型参数，推理不依赖初始化所用的 PanDerm `.pth`。
旧 v1 模型路径保留。A 的来源函数还须覆盖上述四个第三方源码／来源／许可文件。
`.gitattributes` 保持这些文件与训练源码字节，避免跨平台换行改变来源指纹。
公开适配后的源码指纹与其他文件树可能不同，不得借旧汇总绕过恢复或来源检查。

## C 的检查和汇总接口

以下命令仅在 A／B 已整合、数据和固定资产已在用户自己的环境准备好后使用。
路径均为示例；检查或汇总会写入指定的新输出，不能覆盖历史记录。

```bash
python scripts/check_training_ready.py --experiments E11 E12 E13 --data-root path/to/images --weights path/to/resnet50-11ad3fa6.pth --panderm-weights path/to/panderm.pth --monet-dir path/to/monet --output path/to/new_readiness.json
python scripts/collect_results.py --runs path/to/runs --output path/to/new_status.csv
python scripts/summarize_cv.py --runs path/to/runs/E11 --manifests DATA/manifests --output path/to/E11_summary
```

readiness 不指定实验时仍只检查 E00～E10；显式新实验必须另给新报告路径。
静态 readiness 不代替实际设备上的前反向验证。收集器生成 E00～E13 的 70 个计划位置，
缺失折标为 planned；拒绝 smoke、refit、错误折路径、缺失来源以及新实验身份不符的记录。
E12／E13 分别汇总到独立目录，不能混成一组五折。

CV 汇总要求完整五折、来源／配置／组件一致，每个开发样本恰有一次 OOF 预测。
只允许训练折拟合的元数据维度不同；规范化比较契约并保留各折实际维度。
报告五折指标平均、样本标准差、pooled OOF 及五个最佳 epoch 的中位数。
个体预测、标签、权重及完整运行记录不属于本次公开交付。

## 离线验证与当前限制

`tests/test_e11_e13.py` 有 17 项人工输入测试，覆盖身份拒绝、E10 兼容、Gate 初始化／梯度／
缺模态、PanDerm 冻层／原始维度、MONET 独立评分／同一视图／冻结、组件与文本篡改拒绝、
v2 保存加载、三组训练接口到推理、E13 中断恢复一致性、收集器和来源文件。
包含 vendored PanDerm 的随机初始化检查，不下载官方权重、不使用真实病例。
私有集群启动守卫不在公开测试中，也没有永久 skip 占位。

整合后运行 README 列出的全部四组离线测试。当前 C 分支缺少 A／B 依赖，
仅完成无需落盘材料的静态和内存检查；不宣称完整离线集成测试或 GPU 检查通过。
既往测试通过数量不能作为这个分支的验证结果，组员人工复核仍待完成。
