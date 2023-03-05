import torch
#############自定义损失函数#####################
def dice_coeff(y_pred, y_true):
    smooth = 1.
    # Flatten
    y_true_f = torch.reshape(y_true, [-1])
    y_pred_f = torch.reshape(y_pred, [-1])
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_pred, y_true):
    score = dice_coeff(y_pred, y_true)
    loss = 1 - score
    return loss