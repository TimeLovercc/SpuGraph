import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GSAT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        




        self.backbone = None
        self.extractor = 

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
        
class ExtractorMLP(nn.Module):
    def __init__(self, hidden_size, shared_config):
        super().__init__()
        self.learn_edge_att = shared_config['learn_edge_att']
        dropout_p = shared_config['extractor_dropout_p']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_size * 2, hidden_size * 4, hidden_size, 1], dropout=dropout_p)
        else:
            self.feature_extractor = MLP([hidden_size * 1, hidden_size * 2, hidden_size, 1], dropout=dropout_p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits

    
