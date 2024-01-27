import torch
from torch import nn

TRIPLET_ALPHA = 0.1

def get_triplet_loss(anchor, pos, neg,):
    """
    Triplet loss defined in https://arxiv.org/pdf/1705.02304.pdf.
    anchor - Anchor example
    pos - positive example
    neg - negative example
    """
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    return torch.maximum(
        cos(anchor, neg) - cos(anchor, pos) + TRIPLET_ALPHA,
        torch.tensor(0.0))