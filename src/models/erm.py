import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class ERM(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.backbone = None
        self.model_config = model_config

    def forward(self, input):
        output = self.backbone(input)
        return output
    
    def loss(self, out, batch, mode):
        preds = out
        labels = batch['y']
        if preds.dim() == 1:
            return F.binary_cross_entropy_with_logits(preds, labels.float())
        elif preds.dim() == 2:
            return F.cross_entropy(preds, labels)

    
