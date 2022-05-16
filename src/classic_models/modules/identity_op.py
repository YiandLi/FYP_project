
from torch import nn
import torch.nn.functional as F


class Identity(nn.Module):

    def __init__(self, dropout=0.1):
        super(Identity, self).__init__()
        self.dropout = dropout

    def forward(self, x, mask=None):
        if self.dropout:
            x = F.dropout(x, self.training)
        return x
