# From Discovering Invariant Rationales for Graph Neural Networks

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, ModuleList
from torch_geometric.nn import global_mean_pool

from conv_layers import LEConv


class SPNET(nn.Module):
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

        self.convs = ModuleList()
        self.relus = ModuleList()
        for i in range(self.gc_layer):
            conv = LEConv(in_channels=hidden_dim, out_channels=hidden_dim)
            self.convs.append(conv)
            self.relus.append(ReLU())

        self.pool = global_mean_pool
        

    def forward(self, batch, edge_att=None):
        batch_idx = batch['batch']
        node_x = self.get_emb(batch, edge_att=edge_att)
        graph_x = self.pool(node_x, batch_idx)
        return self.get_causal_pred(graph_x)

    def get_emb(self, batch, edge_att=None):
        x, edge_index, edge_attr, batch = batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch']
        x = self.node_encoder(x)
        for i, (conv, relu) in enumerate(zip(self.convs, self.relus)):
            x = conv(x=x, edge_index=edge_index, edge_weight=edge_attr, edge_atten=edge_att)
            x = relu(x)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc(self.pool(emb, batch))

    def get_graph_emb(self, batch, edge_att=None):
        batch_idx = batch['batch']
        node_x = self.get_emb(batch, edge_att=edge_att)
        graph_x = self.pool(node_x, batch_idx)
        return graph_x

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)