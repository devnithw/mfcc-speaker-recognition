import torch
from torch import nn
import torch.nn.functional as F

class TripletCosineLoss(nn.Module):
    """
    Triplet loss function as defined in https://arxiv.org/pdf/1705.02304.pdf
    Alpha is the margin for the loss
    """
    def __init__(self, alpha=0.3):
        super(TripletCosineLoss, self).__init__()
        self.alpha = alpha

    def forward(self, anchor, positive, negative):
        # Calculate cosine similarities
        pos_cosine = F.cosine_similarity(anchor, positive)
        neg_cosine = F.cosine_similarity(anchor, negative)

        # Calculate triplet loss using the margin
        loss = F.relu(neg_cosine - pos_cosine + self.alpha)

        return loss