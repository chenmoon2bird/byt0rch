import torch as pt
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self,
                 eps=1.17e-06,
                 feature_reduction='sum'):
        super(MSELoss, self).__init__()
        self.eps = eps
        self.feature_reduction = feature_reduction

    def forward(self, preds, targets):
        x = pt.pow(preds - targets, 2)
        x = pt.mean(x, dim=1)
        if self.feature_reduction == 'sum':
            x = pt.sum(x, dim=-1, keepdim=True)
        else:
            x = pt.mean(x, dim=-1, keepdim=True)
        return x
