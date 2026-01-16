import torch
import torch.nn.functional as F

def dice_loss(pred, target, eps=1e-6):
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return 1 - num / den

def seg_loss(pred, target):
    pred = torch.sigmoid(pred)
    return dice_loss(pred, target) + F.binary_cross_entropy(pred, target)
