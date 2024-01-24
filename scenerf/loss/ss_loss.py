import torch

# 计算满足predicted_depth的样本与目标值的绝对差值（l1_loss）
def compute_l1_loss(pred, target, predicted_depth=None):
    """
    pred: (3, B)
    target: (3, B)
    ---
    return
    l1_loss: (B,)

    """
    abs_diff = torch.abs(target - pred)
    if predicted_depth is not None:
        abs_diff = abs_diff[:, predicted_depth < 30]
    l1_loss = abs_diff.mean(0)

    return l1_loss

