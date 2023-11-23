import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import Linear, ReLU
from torch.nn import functional as F
from torch_geometric.nn import LEConv
import torch_geometric.data.batch as DataBatch
from torch_geometric.utils import is_undirected, sort_edge_index, degree
from torch_sparse import transpose
from torch_geometric.nn.conv import MessagePassing

class MBASE(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.backbone = None
        self.in_dim = in_dim = model_config['in_dim']
        self.hidden_dim = hidden_dim = model_config['hidden_dim']
        self.edge_attr_dim = edge_attr_dim = model_config['edge_attr_dim']
        self.out_dim = out_dim = model_config['out_dim']

    def forward(self, batch, epoch, training):

        # extract part
        node_emb = self.extractor(batch, training)
        edge_att = self.get_edge_att(batch, node_emb)
        causal_batch, spu_batch, causal_edge_att, spu_edge_att = self.get_splited_graph(batch, edge_att, node_emb, self.ratio)

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
    
    def get_edge_att(self, batch, node_emb):
        row, col = batch.edge_index
        edge_rep = torch.cat([node_emb[row], node_emb[col]], dim=-1)
        edge_att = self.edge_mlp(edge_rep).view(-1)
        return edge_att
    
    def get_splited_graph(self, batch, edge_att, node_emb, ratio):
        if ratio < 0:
            (causal_edge_index, causal_edge_attr, causal_edge_att), \
                (spu_edge_index, spu_edge_attr, spu_edge_att) = (batch.edge_index, batch.edge_attr, edge_att), \
                (batch.edge_index, batch.edge_attr, edge_att)
        else:
            (causal_edge_index, causal_edge_attr, causal_edge_att), \
                (spu_edge_index, spu_edge_attr, spu_edge_att) = split_graph(batch, edge_att, self.ratio)
            
        causal_x, causal_edge_index, causal_batch, _ = relabel(node_emb, causal_edge_index, batch.batch)
        spu_x, spu_edge_index, spu_batch, _ = relabel(node_emb, spu_edge_index, batch.batch)

        causal_batch = {'x': causal_x, 'edge_index': causal_edge_index, 'edge_attr': causal_edge_attr, 'batch': causal_batch}
        spu_batch = {'x': spu_x, 'edge_index': spu_edge_index, 'edge_attr': spu_edge_attr, 'batch': spu_batch}

        return causal_batch, spu_batch, causal_edge_att, spu_edge_att

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
    

def set_masks(mask: Tensor, model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            #PyG 2.0.4
            module._explain = True
            module._edge_mask = mask
            #PyG 1.7.2
            module.__explain__ = True
            module.__edge_mask__ = mask

def clear_masks(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            #PyG 2.0.4
            module._explain = False
            module._edge_mask = None
            #PyG 1.7.2
            module.__explain__ = False
            module.__edge_mask__ = None

def relabel(x, edge_index, batch, pos=None):
    num_nodes = x.size(0)
    sub_nodes = torch.unique(edge_index)
    x = x[sub_nodes]
    batch = batch[sub_nodes]
    row, col = edge_index
    # remapping the nodes in the explanatory subgraph to new ids.
    node_idx = row.new_full((num_nodes,), -1)
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=x.device)
    edge_index = node_idx[edge_index]
    if pos is not None:
        pos = pos[sub_nodes]
    return x, edge_index, batch, pos

def split_graph(data, edge_score, ratio):
    # Adopt from GOOD benchmark to improve the efficiency
    from torch_geometric.utils import degree
    def sparse_sort(src: torch.Tensor, index: torch.Tensor, dim=0, descending=False, eps=1e-12):
        r'''
        Adopt from <https://github.com/rusty1s/pytorch_scatter/issues/48>_.
        '''
        f_src = src.float()
        f_min, f_max = f_src.min(dim)[0], f_src.max(dim)[0]
        norm = (f_src - f_min) / (f_max - f_min + eps) + index.float() * (-1) ** int(descending)
        perm = norm.argsort(dim=dim, descending=descending)

        return src[perm], perm

    def sparse_topk(src: torch.Tensor, index: torch.Tensor, ratio: float, dim=0, descending=False, eps=1e-12):
        rank, perm = sparse_sort(src, index, dim, descending, eps)
        num_nodes = degree(index, dtype=torch.long)
        k = (ratio * num_nodes.to(float)).ceil().to(torch.long)
        start_indices = torch.cat([torch.zeros((1, ), device=src.device, dtype=torch.long), num_nodes.cumsum(0)])
        mask = [torch.arange(k[i], dtype=torch.long, device=src.device) + start_indices[i] for i in range(len(num_nodes))]
        mask = torch.cat(mask, dim=0)
        mask = torch.zeros_like(index, device=index.device).index_fill(0, mask, 1).bool()
        topk_perm = perm[mask]
        exc_perm = perm[~mask]

        return topk_perm, exc_perm, rank, perm, mask

    has_edge_attr = hasattr(data, 'edge_attr') and getattr(data, 'edge_attr') is not None
    new_idx_reserve, new_idx_drop, _, _, _ = sparse_topk(edge_score, data.batch[data.edge_index[0]], ratio, descending=True)
    new_causal_edge_index = data.edge_index[:, new_idx_reserve]
    new_spu_edge_index = data.edge_index[:, new_idx_drop]

    new_causal_edge_weight = edge_score[new_idx_reserve]
    new_spu_edge_weight = -edge_score[new_idx_drop]

    if has_edge_attr:
        new_causal_edge_attr = data.edge_attr[new_idx_reserve]
        new_spu_edge_attr = data.edge_attr[new_idx_drop]
    else:
        new_causal_edge_attr = None
        new_spu_edge_attr = None

    return (new_causal_edge_index, new_causal_edge_attr, new_causal_edge_weight), \
        (new_spu_edge_index, new_spu_edge_attr, new_spu_edge_weight)