"""One-epoch PyTorch training and validation for single-image baselines.
"""

import torch


# train_one_epoch() 
# 训练一个完整 epoch，使用所有 mini-batches 更新模型参数，并返回平均训练损失。
# 实现：切换为 train 模式；每批把图片和标签移到 device，依次执行清空梯度、前向计算、
#       CrossEntropy loss、反向传播及 optimizer 更新；按样本数加权平均每批 loss。
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Update model parameters using all mini-batches; return sample-weighted loss.

    Each mini-batch takes one optimizer step. Multiplying batch loss by its size
    before averaging ensures a smaller last batch is not over-weighted.
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        logits = model(images)                    # [batch_size, 11], not probabilities
        loss = criterion(logits, labels)          # CrossEntropyLoss expects logits
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Training DataLoader is empty")
    return {"loss": total_loss / total_samples}


# validate_one_epoch()
# 执行一个 epoch 的验证，不更新模型权重，并收集真实标签、预测类别与类别概率。
# 使用 eval 和 no_grad；对 logits 计算 loss、softmax 概率及 argmax 类别；
#       按样本数求平均验证损失。返回的预测顺序与验证 DataLoader 的遍历顺序一致。
def validate_one_epoch(model, loader, criterion, device):
    """Evaluate without parameter updates; collect labels and probabilities.

    The returned rows follow DataLoader iteration order. For predictions.csv,
    the caller must use a non-shuffled loader and align lesion IDs accordingly.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_true = []
    all_pred = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device, dtype=torch.long)

            logits = model(images)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_true.extend(labels.cpu().tolist())
            all_pred.extend(logits.argmax(dim=1).cpu().tolist())
            all_probs.extend(probabilities.cpu().tolist())

    if total_samples == 0:
        raise ValueError("Validation DataLoader is empty")
    return {
        "loss": total_loss / total_samples,
        "true_labels": all_true,
        "pred_labels": all_pred,
        "probabilities": all_probs,
    }
