import torch
from torch import Tensor, nn
from torch.nn import functional as F

from SpuGraph.src.backbones.bbase import BASE
from convs import GINConv

class GIN(BASE):
    def __init__(self, backbone_config):
        super().__init__(backbone_config)
        self.residual = backbone_config['residual']
        self.JK = backbone_config['JK']

        if self.gc_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.virtualnode_embedding = torch.nn.Embedding(1, self.hidden_dim)
        torch.nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.mlp_virtualnode_list = nn.ModuleList()

        for layer in range(self.gc_layer):
            self.convs.append(GINConv(self.hidden_dim, self.hidden_dim))
            self.bns.append(nn.BatchNorm1d(self.hidden_dim))

        for layer in range(self.gc_layer-1):
            self.mlp_virtualnode_list.append(torch.nn.Sequential(torch.nn.Linear(self.hidden_dim, 2*self.hidden_dim), torch.nn.BatchNorm1d(2*self.hidden_dim), torch.nn.ReLU(), \
                                                    torch.nn.Linear(2*self.hidden_dim, self.hidden_dim), torch.nn.BatchNorm1d(self.hidden_dim), torch.nn.ReLU()))
        self.weights_init()
    
    def get_node_emb(self, x_encode, batch, edge_att=None):
        x, edge_index, edge_attr, batch_idx = batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch_idx']
        virtualnode_embedding = self.virtualnode_embedding(torch.zeros(batch_idx[-1].item() + 1).to(edge_index.dtype).to(edge_index.device))
        h_list = [x_encode]

        for layer in range(self.gc_layer):
            ### add message from virtual nodes to graph nodes
            h_list[layer] = h_list[layer] + virtualnode_embedding[batch]

            ### Message passing among graph nodes
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)

            h = self.bns[layer](h)
            if layer == self.gc_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.p, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.p, training=self.training)

            if self.residual:
                h = h + h_list[layer]

            h_list.append(h)

            ### update the virtual nodes
            if layer < self.gc_layer - 1:
                ### add message from graph nodes to virtual nodes
                virtualnode_embedding_temp = self.pool(h_list[layer], batch) + virtualnode_embedding
                ### transform virtual nodes using MLP

                if self.residual:
                    virtualnode_embedding = virtualnode_embedding + F.dropout(
                        self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                        self.p,
                        training=self.training)
                else:
                    virtualnode_embedding = F.dropout(self.mlp_virtualnode_list[layer](virtualnode_embedding_temp),
                                                      self.p,
                                                      training=self.training)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.gc_layer):
                node_representation += h_list[layer]

        return node_representation

    
    
    