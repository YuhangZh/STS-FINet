import torch
import torch.nn.functional as F
import torch.nn as nn


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.25, weight_neg=0.75):
    logit = logit_pixel.view(-1)
    truth = truth_pixel.view(-1)
    assert(logit.shape==truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
    
    pos = (truth>0.5).float()
    neg = (truth<0.5).float()
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos*pos*loss/pos_num + weight_neg*neg*loss/neg_num).sum()

    return loss


class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1,[b*h*w, c])
        x2 = torch.reshape(x2,[b*h*w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target,[b*h*w])

        loss = self.loss_f(x1, x2, target)
        return loss


class MulticlassCELoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, reduction='mean'):
        super(MulticlassCELoss, self).__init__()
        if isinstance(ignore_index, int):
            self.ignore_index = [ignore_index]
        elif isinstance(ignore_index, list) or ignore_index is None:
            self.ignore_index = ignore_index
        else:
            raise TypeError("Wrong type for ignore index, which should be int or list or None")
        if isinstance(weight, list) or weight is None:
            self.weight = weight
        else:
            raise TypeError("Wrong type for weight, which should be list or None")
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")
        self.reduction = reduction

    def forward(self, input, target):
        # labels.shape: [b,]
        C = input.shape[1]
        logpt = F.log_softmax(input, dim=1)

        if len(target.size()) <= 3:
            if self.ignore_index is not None:
                mask = torch.zeros_like(target)
                for idx in self.ignore_index:
                    m = torch.where(target == idx, torch.zeros_like(target), torch.ones_like(target))
                    mask += m
                mask = torch.where(mask > 0, torch.ones_like(target), torch.zeros_like(target))
                target *= mask
            target = F.one_hot(target, C).permute(0, 3, 1, 2)
            target *= mask.unsqueeze(1)

        if self.weight is None:
            weight = torch.ones(logpt.shape[1]).to(target.device)  # uniform weights for all classes
        else:
            weight = torch.tensor(self.weight).to(target.device)

        for i in range(len(logpt.shape)):
            if i != 1:
                weight = torch.unsqueeze(weight, dim=i)

        s_weight = weight * target
        for i in range(target.shape[1]):
            if self.ignore_index is not None and i in self.ignore_index:
                target[:,i] = - target[:,i]
                s_weight[:,i] = 0
        s_weight = s_weight.sum(1)

        loss = -1 * weight * logpt * target
        loss = loss.sum(1)

        if self.reduction == 'none':
            return torch.where(loss > 0, loss, torch.zeros_like(loss).to(loss.device))
        elif self.reduction == 'mean':
            if s_weight.sum() == 0:
                return loss[torch.where(loss > 0)].sum()
            else:
                return loss[torch.where(loss > 0)].sum() / s_weight[torch.where(loss > 0)].sum()
        elif self.reduction == 'sum':
            return loss[torch.where(loss > 0)].sum()
        else:
            raise ValueError("Wrong mode for reduction, which should be selected from {'none', 'mean', 'sum'}")

