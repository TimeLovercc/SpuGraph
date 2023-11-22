import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, LEConv

class LEGNN(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        self.gc_layer = gc_layer = backbone_config['gc_layer']
        self.in_dim = in_dim = backbone_config['in_dim']
        self.hidden_dim = hidden_dim = backbone_config['hidden_dim']
        self.out_dim = out_dim = backbone_config['out_dim']
        self.p = dropout = backbone_config['dropout']
        self.bn = bn = backbone_config['bn']
        
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList([LEConv(hidden_dim, hidden_dim) for i in range(gc_layer)])
        self.relus = nn.ModuleList([nn.ReLU() for i in range(gc_layer)])

        self.pool = global_mean_pool
        
        self.weights_init()

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
        for conv, relu in zip(self.convs, self.relus):
            x_encode = conv(x_encode, batch['edge_index'], edge_weight=edge_att)
            x_encode = relu(x_encode)
            x_encode = F.dropout(x_encode, p=self.p, training=self.training)
        node_emb = x_encode
        return node_emb
    
    def get_graph_emb(self, node_emb, batch):
        graph_emb = self.pool(node_emb, batch['batch'])
        return graph_emb
    
    