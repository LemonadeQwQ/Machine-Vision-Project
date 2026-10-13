"""Shared model-output column order for all 11 lesion classes."""

# 全项目共用的类别定义：数据分布分析和评估都从这里导入。
# 元组位置就是程序内部类别编号：AKIEC 对应 0，BCC 对应 1，依次到 VASC 对应 10。
# 概率数组的每一列和混淆矩阵的行、列都按这个顺序解释。
# CSV 可以调整列的排列顺序，评估器会按列名重新对齐；
# 模型导出时仍必须确保每个列名下的概率确实属于该类别。
# 不要为单个模型随意调整此处顺序；应在该模型的导出阶段重排概率。
CLASSES = (
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
    "MAL_OTH", "MEL", "NV", "SCCKA", "VASC",
)
