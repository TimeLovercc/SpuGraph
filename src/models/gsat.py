import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import InstanceNorm
from torch_geometric.utils import is_undirected, sort_edge_index
from torch_sparse import transpose


class GSAT(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.backbone = None
        self.extractor = ExtractorMLP(model_config)

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
        
class ExtractorMLP(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.learn_edge_att = model_config['learn_edge_att']
        self.hidden_dim = hidden_dim = model_config['ext_hidden_dim']
        self.p = p = model_config['ext_dropout']

        if self.learn_edge_att:
            self.feature_extractor = MLP([hidden_dim * 2, hidden_dim * 4, hidden_dim, 1], dropout=p)
        else:
            self.feature_extractor = MLP([hidden_dim * 1, hidden_dim * 2, hidden_dim, 1], dropout=p)

    def forward(self, emb, edge_index, batch):
        if self.learn_edge_att:
            col, row = edge_index
            f1, f2 = emb[col], emb[row]
            f12 = torch.cat([f1, f2], dim=-1)
            att_log_logits = self.feature_extractor(f12, batch[col])
        else:
            att_log_logits = self.feature_extractor(emb, batch)
        return att_log_logits

    
class BatchSequential(nn.Sequential):
    def forward(self, inputs, batch):
        for module in self._modules.values():
            if isinstance(module, (InstanceNorm)):
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs

class MLP(BatchSequential):
    def __init__(self, channels, dropout, bias=True):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                m.append(InstanceNorm(channels[i]))
                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)