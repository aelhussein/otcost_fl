# Our baseline loss is the weighted focal loss.
# Focal loss was first implemented by He et al. in the following [article]
# (https://arxiv.org/abs/1708.02002)
# Thank you to [Aman Arora](https://github.com/amaarora) for this nice [explanation]
# (https://amaarora.github.io/2020/06/29/FocalLoss.html)


import random
import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class ISICLoss(_Loss):
    """Weighted focal loss
    See this [link](https://amaarora.github.io/2020/06/29/FocalLoss.html) for
    a good explanation
    """
    def __init__(self, device, alpha=torch.tensor([1, 2, 1, 1, 5, 1, 1, 1]), gamma=2.0,):
        super(ISICLoss, self).__init__()
        self.alpha = alpha.to(torch.float).to(device)
        self.gamma = gamma.to(device)

    def forward(self, inputs, targets):
        targets = targets.view(-1, 1).type_as(inputs)
        logpt = F.log_softmax(inputs, dim=1)
        logpt = logpt.gather(1, targets.long())
        logpt = logpt.view(-1)
        pt = logpt.exp()
        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.data.view(-1).long())
        logpt = logpt * at
        loss = -1 * (1 - pt) ** self.gamma * logpt

        return loss.mean()