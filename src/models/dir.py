import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Linear, ReLU
from torch.nn import functional as F
from torch_geometric.nn import LEConv
from torch_geometric.utils import is_undirected, sort_edge_index, degree
from torch_sparse import transpose
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import Sequential

from mbase import MBASE, set_masks, clear_masks, split_graph, relabel

class DIR(MBASE):
    def __init__(self, model_config):
        super().__init__(model_config)
        self.reg = model_config['reg']
        self.alpha = model_config['alpha']
        self.ratio = model_config['ratio']

        self.extractor = CausalAttNet(model_config)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim*2, self.hidden_dim*4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*4, 1)
        )

        self.causal_pred = nn.Sequential(Linear(self.hidden_dim, 2*self.hidden_dim), ReLU(), Linear(2*self.hidden_dim, self.out_dim))
        self.conf_mlp = torch.nn.Sequential(Linear(self.hidden_dim, 2*self.hidden_dim), ReLU(), Linear(2*self.hidden_dim, self.out_dim))
        self.cq = Linear(self.out_dim, self.out_dim)
        self.conf_pred = nn.Sequential(self.conf_mlp, self.cq)

    def forward(self, batch, epoch, training):

        # extract part
        node_emb = self.extractor(batch)
        edge_att = self.get_edge_att(batch, node_emb)
        causal_batch, spu_batch, causal_edge_att, spu_edge_att = self.get_splited_graph(batch, edge_att, batch.x, self.ratio)

        # causal part
        set_masks(causal_edge_att, self.backbone)
        causal_node_emb = self.backbone(causal_batch, training)
        causal_graph_emb = self.backbone.get_graph_emb(causal_node_emb, causal_batch)
        causal_out = self.causal_pred(causal_graph_emb)

        # conf part
        set_masks(spu_edge_att, self.backbone)
        spu_node_emb = self.backbone(spu_batch, training)
        spu_graph_emb = self.backbone.get_graph_emb(spu_node_emb, spu_batch)
        spu_out = self.conf_pred(spu_graph_emb)

        # clear masks
        clear_masks(self.backbone)
        return causal_out, spu_out, causal_graph_emb, spu_graph_emb
    
    def loss(self, out, batch, epoch, mode):
        causal_out, conf_out, causal_rep, conf_rep = out
        labels = batch['y']
        
        if causal_out.dim() == 1:
            causal_loss = F.binary_cross_entropy_with_logits(causal_out, labels.float())
        elif causal_out.dim() == 2:
            causal_loss =  F.cross_entropy(causal_out, labels)

        if conf_out.dim() == 1:
            conf_loss = F.binary_cross_entropy_with_logits(conf_out, labels.float())
        elif conf_out.dim() == 2:
            conf_loss =  F.cross_entropy(conf_out, labels)

        alpha_prime = self.alpha * (epoch ** 1.6)
        CELoss = nn.CrossEntropyLoss(reduction='mean')
        if self.reg:
            env_loss_list = []
            for conf in conf_rep:
                rep_out = self.get_comb_pred(causal_rep, conf)
                env_loss_list.append(CELoss(rep_out, labels))
            env_loss_tensor = torch.stack(env_loss_list)
            causal_loss += alpha_prime * env_loss_tensor.mean()
            env_loss = alpha_prime * torch.var(env_loss_tensor * conf_rep.size(0))
        
        total_loss = causal_loss + conf_loss + env_loss if self.reg else causal_loss + conf_loss
        loss_dict = {f'{mode}_loss': total_loss.item(), f'{mode}_causal_loss': causal_loss.item(), f'{mode}_conf_loss': conf_loss.item(), f'{mode}_env_loss': env_loss.item() if self.reg else 0}
        return total_loss, loss_dict
    
    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_pred(causal_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred
    
class CausalAttNet(nn.Module):
    def __init__(self, model_config):
        super(CausalAttNet, self).__init__()
        self.in_dim = in_dim =  model_config['in_dim']
        self.hidden_dim = hidden_dim =  model_config['hidden_dim']

        self.conv1 = LEConv(in_channels=in_dim, out_channels=hidden_dim)
        self.conv2 = LEConv(in_channels=hidden_dim, out_channels=hidden_dim)
    
    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr.view(-1)))
        x = self.conv2(x, edge_index, edge_attr.view(-1))
        return x