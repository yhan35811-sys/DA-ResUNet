import torch
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def calculate_metrics(pred, target, threshold=0.5):
    # Pred: Logits -> Sigmoid -> Binary
    pred = (torch.sigmoid(pred) > threshold).float()

    pred = pred.view(-1)
    target = target.view(-1)

    TP = (pred * target).sum()
    FP = (pred * (1 - target)).sum()
    FN = ((1 - pred) * target).sum()

    # Dice & IoU
    smooth = 1e-5
    dice = (2. * TP + smooth) / (2. * TP + FP + FN + smooth)
    iou = (TP + smooth) / (TP + FP + FN + smooth)

    # Precision & Recall
    precision = (TP + smooth) / (TP + FP + smooth)
    recall = (TP + smooth) / (TP + FN + smooth)

    return dice.item(), iou.item(), precision.item(), recall.item()


def calculate_hd95(pred, target):
    """
    计算 95% Hausdorff Distance

    """
    pred = torch.sigmoid(pred).cpu().detach().numpy()
    target = target.cpu().detach().numpy()

    batch_size = pred.shape[0]
    hd95_sum = 0
    count = 0

    for i in range(batch_size):
        p = (pred[i, 0] > 0.5).astype(np.bool_)
        t = (target[i, 0] > 0.5).astype(np.bool_)

        # 防止报错
        if np.sum(p) == 0 or np.sum(t) == 0:
            continue

        d_p_t = directed_hausdorff(p, t)[0]
        d_t_p = directed_hausdorff(t, p)[0]
        hd95_sum += max(d_p_t, d_t_p)
        count += 1

    if count == 0:
        return 0.0
    return hd95_sum / count


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print(f"=> Saving checkpoint to {filename}")
    torch.save(state, filename)