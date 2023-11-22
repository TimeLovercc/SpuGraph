import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ADPTIVE(nn.Module):
    def __init__(self, backbone_config):
        super().__init__()
        self.gc_layer = gc_layer = backbone_config['gc_layer']
        self.gc_type = gc_type = backbone_config['gc_type']
        self.in_dim = in_dim = backbone_config['in_dim']
        self.edge_attr_dim = edge_attr_dim = backbone_config['edge_attr_dim']
        self.hidden_dim = hidden_dim = backbone_config['hidden_dim']
        self.out_dim = out_dim = backbone_config['out_dim']
        self.virtual_node = virtual_node = backbone_config['virtual_node']
        self.residual = residual = backbone_config['residual']
        self.drop_ratio = drop_ratio = backbone_config['drop_ratio']
        self.jk = jk = backbone_config['jk']
        self.pooling = pooling = backbone_config['pooling']
        self.pred_head = pred_head = backbone_config['pred_head']
        self.p = dropout = backbone_config['dropout']
        self.bn = bn = backbone_config['bn']
        
        assert gc_type in ['le', 'vgin', 'gcn']
        if gc_type == 'le':
            self.convs = nn.ModuleList([LEConv(in_dim if i == 0 else hidden_dim, hidden_dim, edge_attr_dim) for i in range(gc_layer)])

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, batch):
        x, edge_index, batch_idx = batch['x'], batch['edge_index'], batch['batch']
        x = F.dropout(x, p=self.p, training=self.training)
        for layer in range(self.gc_layer-1):
            x = self.convs[layer](x, edge_index)
            if self.bns is not None:
                x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p, training=self.training)
        x = F.relu(self.convs[-1](x, edge_index))
        if self.bns is not None:
            x = self.bns[-1](x)
        
        # Global mean pooling
        x = global_mean_pool(x, batch_idx)

        x = self.fc(x)
        return x.squeeze()
    
    def loss(self, out, batch, mode):
        preds = out
        labels = batch['y']
        if preds.dim() == 1:
            return F.binary_cross_entropy_with_logits(preds, labels.float())
        elif preds.dim() == 2:
            return F.cross_entropy(preds, labels)
