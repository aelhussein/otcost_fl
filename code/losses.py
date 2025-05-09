# Our baseline loss is the weighted focal loss.
# Focal loss was first implemented by He et al. in the following [article]
# (https://arxiv.org/abs/1708.02002)
# Thank you to [Aman Arora](https://github.com/amaarora) for this nice [explanation]
# (https://amaarora.github.io/2020/06/29/FocalLoss.html)


import random
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss
from typing import Tuple


class ISICLoss(_Loss):
    """Weighted focal loss
    See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
    a good explanation
    """
    def __init__(self, alpha, gamma=1.0,):
        super(ISICLoss, self).__init__()
        self.__name__ = 'ISICLoss'
        self.alpha = alpha.to(torch.float)
        self.gamma = gamma

    def forward(self, inputs, targets):
        self.alpha = self.alpha.to(targets.device)
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()

def get_dice_score(output: torch.Tensor, target: torch.Tensor,
                   foreground_channel: int = 1, # Assumes foreground is channel 1
                   spatial_dims: Tuple[int, ...] = (2, 3, 4), # D, H, W for 3D
                   epsilon: float = 1e-9) -> float:
    """Calculates the mean Dice score for segmentation tasks."""
    p0 = output[:, foreground_channel, ...]
    if target.dim() == output.dim() and target.shape[1] == output.shape[1]:
        g0 = target[:, foreground_channel, ...]
    elif target.dim() == output.dim() and target.shape[1] == 1:
        g0 = (target[:, 0, ...] == foreground_channel).float()
    elif target.dim() == output.dim() - 1 and target.shape[0] == output.shape[0] and \
         target.shape[1:] == output.shape[2:]:
        g0 = (target == foreground_channel).float()
    else:
        raise ValueError(
            f"Target shape {target.shape} is not compatible with output shape {output.shape} "
            f"for Dice score calculation. Expected target formats: \n"
            f"1. (Batch, Classes, D, H, W) - one-hot encoded\n"
            f"2. (Batch, 1, D, H, W) - class indices\n"
            f"3. (Batch, D, H, W) - class indices"
        )
    g0 = g0.float()

    tp = torch.sum(p0 * g0, dim=sum_dims)
    fp = torch.sum(p0 * (1.0 - g0), dim=sum_dims)
    fn = torch.sum((1.0 - p0) * g0, dim=sum_dims)

    numerator = 2 * tp
    denominator = 2 * tp + fp + fn + epsilon

    dice_score_per_sample = numerator / denominator
    return dice_score_per_sample.mean().item()


def get_dice_loss(output: torch.Tensor, target: torch.Tensor,
                  foreground_channel: int = 1,
                  epsilon: float = 1e-9) -> torch.Tensor: # Return a Tensor
    """
    Calculates the Dice Loss for segmentation tasks.
    The Dice Loss is 1 - Dice Score.
    """
    p0 = output[:, foreground_channel, ...] # Shape: (Batch, D, H, W)

    # 2. Prepare the target tensor
    if target.dim() == output.dim() and target.shape[1] == output.shape[1]: # One-hot
        g0 = target[:, foreground_channel, ...]
    elif target.dim() == output.dim() and target.shape[1] == 1: # Class indices (B, 1, D, H, W)
        g0 = (target[:, 0, ...] == foreground_channel).float()
    elif target.dim() == output.dim() - 1 and target.shape[0] == output.shape[0] and \
         target.shape[1:] == output.shape[2:]: # Class indices (B, D, H, W)
        g0 = (target == foreground_channel).float()
    else:
        raise ValueError(
            f"Target shape {target.shape} is not compatible with output shape {output.shape} "
            f"for Dice loss calculation."
        )
    g0 = g0.float() # Ensure target is float
    sum_dims = tuple(range(1, p0.dim()))

    tp = torch.sum(p0 * g0, dim=sum_dims)
    fp = torch.sum(p0 * (1.0 - g0), dim=sum_dims)
    fn = torch.sum((1.0 - p0) * g0, dim=sum_dims)

    numerator = 2 * tp
    denominator = 2 * tp + fp + fn + epsilon
    dice_score_per_sample = numerator / denominator
    mean_dice_score = dice_score_per_sample.mean()
    dice_loss = 1.0 - mean_dice_score

    return dice_loss # Return the scalar tensor


class WeightedCELoss(nn.Module):
    """
    Weighted Cross Entropy Loss that accepts class weights at initialization.
    """
    def __init__(self, weights=None):
        super(WeightedCELoss, self).__init__()
        self.__name__ = 'WeightedCELoss'
        self.weights = weights
        
    def forward(self, inputs, targets):
        """
        Calculate weighted cross entropy loss.
        
        Args:
            inputs: Model predictions (logits)
            targets: Ground truth labels
        
        Returns:
            Weighted loss value
        """
        # Standard cross entropy with log_softmax + nll_loss for numerical stability
        log_probs = F.log_softmax(inputs, dim=1)
        
        if self.weights is None:
            # Fall back to regular cross entropy loss when not weighted loss
            return F.nll_loss(log_probs, targets)
            
        # Handle different weight types
        weights = self.weights.to(inputs.device)
        if weights.dim() == 1:
            if len(weights) == inputs.size(1):  # Class weights [num_classes]
                return F.nll_loss(log_probs, targets, weight=weights, reduction='mean')
            else:  # Sample weights [batch_size]
                return (F.nll_loss(log_probs, targets, reduction='none') * weights).mean()