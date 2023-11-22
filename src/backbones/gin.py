# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN

import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from conv_layers import GINConv, GINEConv


class GIN(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        self.gc_layer = gc_layer = backbone_config['gc_layer']
        self.in_dim = in_dim = backbone_config['in_dim']
        self.edge_attr_dim = edge_attr_dim = backbone_config['edge_attr_dim'] 
        self.hidden_dim = hidden_dim = backbone_config['hidden_dim']
        self.out_dim = out_dim = backbone_config['out_dim']
        self.p = dropout = backbone_config['dropout']
        self.use_edge_attr = backbone_config.get('use_edge_attr', True)

        self.node_encoder = Linear(in_dim, hidden_dim)
        if edge_attr_dim != 0 and self.use_edge_attr:
            self.edge_encoder = Linear(edge_attr_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for _ in range(self.gc_layer):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_dim, hidden_dim), edge_dim=hidden_dim))
            else:
                self.convs.append(GINConv(GIN.MLP(hidden_dim, hidden_dim)))

        self.fc = nn.Sequential(nn.Linear(hidden_dim, out_dim))

    def forward(self, batch, edge_att=None):
        x, edge_index, edge_attr, batch = batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch']
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.gc_layer):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_att)
            x = self.relu(x)
            x = F.dropout(x, p=self.p, training=self.training)
        return self.fc(self.pool(x, batch))

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, batch, edge_att=None):
        x, edge_index, edge_attr, batch = batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch']
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.gc_layer):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_att)
            x = self.relu(x)
            x = F.dropout(x, p=self.p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc(self.pool(emb, batch))