# https://github.com/lukecavabarrett/pna/blob/master/models/pytorch_geometric/example.py

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.nn import BatchNorm, global_mean_pool

from conv_layers import PNAConvSimple


class PNA(nn.Module):
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

        aggregators = backbone_config['aggregators']
        scalers = ['identity', 'amplification', 'attenuation'] if backbone_config['scalers'] else ['identity']
        deg = backbone_config['deg']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        if self.use_edge_attr:
            in_channels = hidden_dim * 2 if edge_attr_dim == 0 else hidden_dim * 3
        else:
            in_channels = hidden_dim * 2

        for _ in range(gc_layer):
            conv = PNAConvSimple(in_channels=in_channels, out_channels=hidden_dim, aggregators=aggregators,
                                 scalers=scalers, deg=deg, post_layers=1)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden_dim))

        self.pool = global_mean_pool
        self.fc = Sequential(Linear(hidden_dim, hidden_dim//2), ReLU(),
                                 Linear(hidden_dim//2, hidden_dim//4), ReLU(),
                                 Linear(hidden_dim//4, out_dim))

    def forward(self, batch, edge_att=None):
        x, edge_index, edge_attr, batch = batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch']
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_atten=edge_att)))
            x = h + x 
            x = F.dropout(x, self.p, training=self.training)
        return self.fc(self.pool(x, batch))

    def get_emb(self, batch, edge_att=None):
        x, edge_index, edge_attr, batch = batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch']
        x = self.node_encoder(x)
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)

        for i, (conv, batch_norm) in enumerate(zip(self.convs, self.batch_norms)):
            h = F.relu(batch_norm(conv(x, edge_index, edge_attr, edge_atten=edge_att)))
            x = h + x  
            x = F.dropout(x, self.p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc(self.pool(emb, batch))