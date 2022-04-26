import torch.nn as nn
import torch.nn.functional as F
from einops import reduce, rearrange


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, pos_scores, neg_scores):
        # (b, n, 1) -> (b, n)
        pos_scores = rearrange(pos_scores, 'b n l -> b (n l)')
        # log(sigmoid(x)) (b, n)
        pos_part = F.logsigmoid(pos_scores)
        # negative scores: (b, n, num_negs) -> (b, n)
        neg_part = reduce(F.softplus(neg_scores), 'b n num_negs -> b n', 'mean')
        loss = -pos_part + neg_part
        return loss


# class WeightedBCELoss(nn.Module):
#     def __init__(self, temperature):
#         nn.Module.__init__()
#         self.temperature = temperature
#
#     def forward(self, pos_scores, neg_scores):
#         # (b, n, 1) -> (b, n)
#         pos_scores = rearrange(pos_scores, 'b n l -> b (n l)')
#         # log(sigmoid(x)) (b, n)
#         pos_part = F.logsigmoid(pos_scores)
#         # (b, n, num_negs)
#         weight = F.softmax(neg_scores / self.temperature, dim=-1)
#         # negative scores: (b, n, num_negs) -> (b, n)
#         neg_part = reduce(F.softplus(neg_scores) * weight, 'b n num_negs -> b n', 'mean')
#         loss = -pos_part + neg_part
#         return loss