import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool, GlobalAttention, Set2Set
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class BASE(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        self.gc_layer = gc_layer = backbone_config['gc_layer']
        self.gc_type = gc_type = backbone_config['gc_type']
        self.in_dim = in_dim = backbone_config['in_dim']
        self.edge_attr_dim = edge_attr_dim = backbone_config['edge_attr_dim']
        self.hidden_dim = hidden_dim = backbone_config['hidden_dim']
        self.out_dim = out_dim = backbone_config['out_dim']
        self.p = dropout = backbone_config['dropout']
        self.bn = bn = backbone_config['bn']
        self.use_edge_attr = use_edge_attr = backbone_config['use_edge_attr']
        self.pooling = pooling = backbone_config['pooling']
        
        # node encoder and egde encoder
        if self.in_dim == 1:
            self.node_encoder = AtomEncoder(self.hidden_dim)
            if use_edge_attr and self.edge_attr_dim != 0:
                self.edge_encoder = BondEncoder(self.hidden_dim)
        elif self.in_dim == -1:
            self.node_encoder = nn.Embedding(1, self.hidden_dim)
            if use_edge_attr and self.edge_attr_dim != 0:
                self.edge_encoder = nn.Linear(edge_attr_dim, self.hidden_dim)
        else:
            self.node_encoder = nn.Linear(self.in_dim, self.hidden_dim)
            if use_edge_attr and self.edge_attr_dim != 0:
                self.edge_encoder = nn.Linear(edge_attr_dim, self.hidden_dim)
        
        # graph poolings
        if self.pooling == "sum":
            self.pool = global_add_pool
        elif self.pooling == "mean":
            self.pool = global_mean_pool
        elif self.pooling == "max":
            self.pool = global_max_pool
        elif self.pooling == "attention":
            self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(torch.nn.Linear(hidden_dim, 2 * hidden_dim), torch.nn.BatchNorm1d(2 * \
                                                            hidden_dim), torch.nn.ReLU(), torch.nn.Linear(2 * hidden_dim, 1)))
        elif self.pooling == "set2set":
            self.pool = Set2Set(hidden_dim, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")
        

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode_node(self, batch):
        x = batch['x']
        x_encode = self.node_encoder(x)
        return x_encode
    
    def encode_edge(self, batch):
        edge_attr = batch['edge_attr']
        edge_encode = self.edge_encoder(edge_attr)
        return edge_encode
    
    def get_node_emb(self, x_encode, batch, edge_att=None):
        edge_weight = batch['edge_attr']
        for conv, relu in zip(self.convs, self.relus):
            x_encode = conv(x_encode, batch['edge_index'], edge_weight=edge_weight, edge_atten=edge_att)
            x_encode = relu(x_encode)
            x_encode = F.dropout(x_encode, p=self.p, training=self.training)
        node_emb = x_encode
        return node_emb
    
    def get_graph_emb(self, node_emb, batch):
        graph_emb = self.pool(node_emb, batch['batch'])
        return graph_emb
    
    