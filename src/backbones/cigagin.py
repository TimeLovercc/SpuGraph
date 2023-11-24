import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bbase import BBASE
from convs import GINConv

class CIGAGIN(BBASE):
    def __init__(self, backbone_config):
        super().__init__(backbone_config)
        self.residual = backbone_config['residual']
        self.JK = backbone_config['JK']

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for layer in range(self.gc_layer):
            mlp = nn.Sequential(torch.nn.Linear(self.hidden_dim, 2 * self.hidden_dim), torch.nn.BatchNorm1d(2 * self.hidden_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(2 * self.hidden_dim, self.hidden_dim))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        self.weights_init()
    
    def get_node_emb(self, x_encode, batch, training, edge_att=None):
        x, edge_index, edge_attr, batch_idx = batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch']
        h_list = [x_encode]

        for layer in range(self.gc_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.bns[layer](h)

            if layer == self.gc_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.p, training=training)
            else:
                h = F.dropout(F.relu(h), self.p, training=training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            if self.residual:
                h += h_list[layer]

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.gc_layer):
                node_representation += h_list[layer]

        return node_representation

    
    
    