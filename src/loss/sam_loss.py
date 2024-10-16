import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import FocalLoss,DiceLoss


ALPHA = 0.8
GAMMA = 2

def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    B, N, H, W = pred_mask.shape
    ious = torch.zeros((B, N), dtype=torch.float32, device=pred_mask.device)

    # Iterate over each class to compute IoU
    for cls in range(N):
        # Get predictions and ground truth for class `cls`
        pred = (pred_mask[:, cls] > 0).int()  # (B, H, W), threshold to get binary mask
        gt = (gt_mask == cls).int()           # (B, H, W), ground truth binary mask

        # Calculate intersection and union
        intersection = (pred & gt).sum(dim=(1, 2))  # Sum over H and W for each batch
        union = (pred | gt).sum(dim=(1, 2))        # Sum over H and W for each batch

        # Avoid division by zero
        ious[:, cls] = intersection / (union + 1e-6)

    return ious

# def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
#     pred_mask = pred_mask.argmax(dim=1).float()
#     intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
#     union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
#     epsilon = 1e-7
#     batch_iou = intersection / (union + epsilon)

#     batch_iou = batch_iou.unsqueeze(1)
#     return batch_iou


# class FocalLoss(nn.Module):

#     def __init__(self, weight=None, size_average=True):
#         super().__init__()

#     def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = torch.clamp(inputs, min=0, max=1)
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
#         BCE_EXP = torch.exp(-BCE)
#         focal_loss = alpha * (1 - BCE_EXP)**gamma * BCE
#         focal_loss = focal_loss.mean()

#         return focal_loss


# class DiceLoss(nn.Module):

#     def __init__(self, weight=None, size_average=True):
#         super().__init__()

#     def forward(self, inputs, targets, smooth=1):
#         inputs = F.sigmoid(inputs)
#         inputs = torch.clamp(inputs, min=0, max=1)
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         intersection = (inputs * targets).sum()
#         dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

#         return 1 - dice

class SAMLoss(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.focal_loss = FocalLoss(mode="multiclass",alpha=0.8,gamma=2)
        self.dice_loss = DiceLoss(mode="multiclass",classes=num_classes)
    
    def forward(self,pred_masks, gt_masks, iou_predictions):
        loss_focal = self.focal_loss(pred_masks,gt_masks)
        loss_dice = self.dice_loss(pred_masks,gt_masks)

        iou = calc_iou(pred_masks,gt_masks)

        loss_iou = F.mse_loss(iou_predictions,iou)
        
        loss_total = 20. * loss_focal + loss_dice + loss_iou
        return loss_total