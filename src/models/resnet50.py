"""MILK10k 的单模态 ResNet50 分类模型。"""
from torch import nn
# 直接使用现成的RESNET-50 结构，其中包含ResNet50_Weights，及预训练参数权重
from torchvision.models import ResNet50_Weights, resnet50

def build_resnet50(num_classes=11):

    # 设置RESNET-50 参数：
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    # RESNET50 会输出2048个特征维度，然后用全连接MLP做NN
    # feature_dim 这里应该是[batch,2048]
    feature_dim = model.fc.in_features

    # 设置模型最后的分类层为一层线性组合，得到原始score：s = W × features + b
    model.fc = nn.Linear(feature_dim, num_classes)

    # 后续需要得到概率分布则使用softmax即可，损失函数在用cross entropy即可
    return model
