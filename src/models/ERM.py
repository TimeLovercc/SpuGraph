import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class ERM(nn.Module):
    def __init__(self, pipeline_config):
        super().__init__()
        self.model = None

    def forward(self, input):
        output = self.model(input)
        return output
    
    def loss(self, out, batch, mode):
        preds = out
        labels, mask = batch['y'], batch[f'{mode}_mask']
        if preds.dim() == 1:
            return F.binary_cross_entropy_with_logits(preds[mask], labels[mask].float())
        elif preds.dim() == 2:
            return F.cross_entropy(preds[mask], labels[mask])

    
