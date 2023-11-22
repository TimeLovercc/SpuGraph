import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import LEConv
from torch_geometric.utils import is_undirected, sort_edge_index, degree
from torch_sparse import transpose


class DIR(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.backbone = None
        self.extractor = CausalAttNet(model_config)

        self.learn_edge_att = model_config['learn_edge_att']
        self.pred_loss_coef = model_config['pred_loss_coef']
        self.info_loss_coef = model_config['info_loss_coef']
        self.fix_r = model_config.get('fix_r', None)
        self.init_r = model_config.get('init_r', 0.9)
        self.decay_interval = model_config.get('decay_interval', None)
        self.decay_r = model_config.get('decay_r', 0.1)
        self.final_r = model_config.get('final_r', 0.7)

    def forward(self, batch, epoch, training):
        edge_index, batch_idx = batch['edge_index'], batch['batch']
        emb = self.backbone.get_emb(batch)
        att_log_logits = self.extractor(emb, edge_index, batch_idx)
        att = self.sampling(att_log_logits, epoch, training)

        if self.learn_edge_att:
            if is_undirected(edge_index):
                trans_idx, trans_val = transpose(edge_index, att, None, None, coalesced=False)
                trans_val_perm = self.reorder_like(trans_idx, edge_index, trans_val)
                edge_att = (att + trans_val_perm) / 2
            else:
                edge_att = att
        else:
            edge_att = self.lift_node_att_to_edge_att(att, edge_index)

        preds = self.backbone(batch, edge_att)
        return preds, edge_att, att
    
    def loss(self, out, batch, epoch, mode):
        preds, edge_att, att = out
        labels = batch['y']
        
        if preds.dim() == 1:
            pred_loss = F.binary_cross_entropy_with_logits(preds, labels.float())
        elif preds.dim() == 2:
            pred_loss =  F.cross_entropy(preds, labels)

        r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r, init_r=self.init_r)
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()

        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef
        loss = pred_loss + info_loss
        loss_dict = {f'{mode}_loss': loss.item(), f'{mode}_pred_loss': pred_loss.item(), f'{mode}_info_loss': info_loss.item()}
        return loss, loss_dict
    
    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r
    
    def sampling(self, att_log_logits, epoch, training):
        att = self.concrete_sample(att_log_logits, temp=1, training=training)
        return att
    
    @staticmethod
    def concrete_sample(att_log_logit, temp, training):
        if training:
            random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
            random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
            att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
        else:
            att_bern = (att_log_logit).sigmoid()
        return att_bern
    
    def reorder_like(from_edge_index, to_edge_index, values):
        from_edge_index, values = sort_edge_index(from_edge_index, values)
        ranking_score = to_edge_index[0] * (to_edge_index.max()+1) + to_edge_index[1]
        ranking = ranking_score.argsort().argsort()
        if not (from_edge_index[:, ranking] == to_edge_index).all():
            raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
        return values[ranking]
    
    @staticmethod
    def lift_node_att_to_edge_att(node_att, edge_index):
        src_lifted_att = node_att[edge_index[0]]
        dst_lifted_att = node_att[edge_index[1]]
        edge_att = src_lifted_att * dst_lifted_att
        return edge_att

    
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
        x, edge_index, edge_attr, batch_idx = batch
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
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch),\
                edge_score
    
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