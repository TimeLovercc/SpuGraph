import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Linear, ReLU
from torch.nn import functional as F
from torch_geometric.nn import LEConv
from torch_geometric.utils import is_undirected, sort_edge_index, degree
from torch_sparse import transpose
from torch_geometric.nn.conv import MessagePassing

class DIR(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.backbone = None
        self.extractor = CausalAttNet(model_config)
        self.reg = model_config['reg']
        self.hidden_dim = hidden_dim = model_config['hidden_dim']
        self.out_dim = out_dim = model_config['out_dim']
        self.alpha = model_config['alpha']

        self.causal_pred = nn.Sequential(Linear(hidden_dim, 2*hidden_dim), ReLU(), Linear(2*hidden_dim, out_dim))
        self.conf_mlp = torch.nn.Sequential(Linear(hidden_dim, 2*hidden_dim), ReLU(), Linear(2*hidden_dim, out_dim))
        self.cq = Linear(out_dim, out_dim)
        self.conf_pred = nn.Sequential(self.conf_mlp, self.cq)

    def get_parameters(self):
        return list(self.backbone.parameters())+list(self.extractor.parameters())+list(self.causal_pred.parameters())+list(self.conf_pred.parameters()), \
            self.conf_pred.parameters()

    def forward(self, batch, epoch, training):
        (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), edge_score = self.extractor(batch)
        set_masks(causal_edge_weight, self.backbone)
        causal_batch = {'x': causal_x, 'edge_index': causal_edge_index, 'edge_attr': causal_edge_attr, 'batch': causal_batch}
        causal_rep = self.backbone.get_graph_emb(causal_batch)
        causal_out = self.causal_pred(causal_rep)
        set_masks(conf_edge_weight, self.backbone)
        conf_batch = {'x': conf_x, 'edge_index': conf_edge_index, 'edge_attr': conf_edge_attr, 'batch': conf_batch}
        conf_rep = self.backbone.get_graph_emb(conf_batch)
        conf_out = self.conf_pred(conf_rep)
        clear_masks(self.backbone)
        return causal_out, conf_out, causal_rep, conf_rep
    
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

        env_loss = 0
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
        
        loss_dict = {f'{mode}_loss': causal_loss.item() + env_loss.item(), f'{mode}_causal_loss': causal_loss.item(), f'{mode}_conf_loss': conf_loss.item(), f'{mode}_env_loss': env_loss.item()}
        return causal_loss+env_loss+conf_loss, loss_dict
    
    def get_comb_pred(self, causal_graph_x, conf_graph_x):
        causal_pred = self.causal_pred(causal_graph_x)
        conf_pred = self.conf_mlp(conf_graph_x).detach()
        return torch.sigmoid(conf_pred) * causal_pred
    
class CausalAttNet(nn.Module):
    def __init__(self, model_config):
        super(CausalAttNet, self).__init__()
        self.in_dim = in_dim =  model_config['in_dim']
        self.hidden_dim = hidden_dim =  model_config['hidden_dim']
        self.ratio = ratio = model_config['causal_ratio']

        self.conv1 = LEConv(in_channels=in_dim, out_channels=hidden_dim)
        self.conv2 = LEConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*4),
            nn.ReLU(),
            nn.Linear(hidden_dim*4, 1)
        )
    
    def forward(self, batch):
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        x = F.relu(self.conv1(x, edge_index, edge_attr.view(-1)))
        x = self.conv2(x, edge_index, edge_attr.view(-1))

        row, col = edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        row, col = edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        (conf_edge_index, conf_edge_attr, conf_edge_weight) = split_graph(batch, edge_score, self.ratio)

        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, batch_idx)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, batch_idx)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch), edge_score


def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            # Apply for pyg <= 2.0.2
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
                # Apply for pyg <= 2.0.2
                module.__explain__ = False
                module.__edge_mask__ = None


def split_graph(data, edge_score, ratio):
    causal_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    causal_edge_weight = torch.tensor([]).to(data.x.device)
    causal_edge_attr = torch.tensor([]).to(data.x.device)
    conf_edge_index = torch.LongTensor([[],[]]).to(data.x.device)
    conf_edge_weight = torch.tensor([]).to(data.x.device)
    conf_edge_attr = torch.tensor([]).to(data.x.device)

    edge_indices, _, _, num_edges, cum_edges = split_batch(data)
    for edge_index, N, C in zip(edge_indices, num_edges, cum_edges):
        n_reserve =  int(ratio * N)
        edge_attr = data.edge_attr[C:C+N]
        single_mask = edge_score[C:C+N]
        single_mask_detach = edge_score[C:C+N].detach().cpu().numpy()
        rank = np.argpartition(-single_mask_detach, n_reserve)
        idx_reserve, idx_drop = rank[:n_reserve], rank[n_reserve:]

        causal_edge_index = torch.cat([causal_edge_index, edge_index[:, idx_reserve]], dim=1)
        conf_edge_index = torch.cat([conf_edge_index, edge_index[:, idx_drop]], dim=1)

        causal_edge_weight = torch.cat([causal_edge_weight, single_mask[idx_reserve]])
        conf_edge_weight = torch.cat([conf_edge_weight,  -1 * single_mask[idx_drop]])

        causal_edge_attr = torch.cat([causal_edge_attr, edge_attr[idx_reserve]])
        conf_edge_attr = torch.cat([conf_edge_attr, edge_attr[idx_drop]])
    return (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        (conf_edge_index, conf_edge_attr, conf_edge_weight)

def split_batch(g):
    split = degree(g.batch[g.edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(g.edge_index, split, dim=1)
    num_nodes = degree(g.batch, dtype=torch.long)
    cum_nodes = torch.cat([g.batch.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]])
    num_edges = torch.tensor([e.size(1) for e in edge_indices], dtype=torch.long).to(g.x.device)
    cum_edges = torch.cat([g.batch.new_zeros(1), num_edges.cumsum(dim=0)[:-1]])

    return edge_indices, num_nodes, cum_nodes, num_edges, cum_edges

def relabel(x, edge_index, batch, pos=None):
        
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=row.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos