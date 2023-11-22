from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINConv


class YOURS(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # define the backbone and extractor here
        self.backbone = None
        self.extractor = Extractor(model_config)

        # set up params
        self.param = model_config['param']

    def get_parameters(self):
        # return the parameters that you want to optimize here
        return self.backbone.parameters(), self.extractor.parameters()

    def forward(self, batch, epoch, training):
        edge_index, batch_idx = batch['edge_index'], batch['batch']
        
        # write down your own forward pass here
        att = self.extractor(batch)
        preds, edge_att = self.backbone(batch, att)

        # output anything, but the first should be preds
        return preds, edge_att, att
    
    def loss(self, out, batch, epoch, mode):

        # modify here to get your outputs from forward
        preds, edge_att, att = out
        labels = batch['y']
        
        # pred loss computation
        if preds.dim() == 1:
            pred_loss = F.binary_cross_entropy_with_logits(preds, labels.float())
        elif preds.dim() == 2:
            pred_loss =  F.cross_entropy(preds, labels)

        # write down your own loss here
        pred_loss = pred_loss * self.pred_loss_coef
        info_loss = info_loss * self.info_loss_coef
        loss = pred_loss + info_loss

        # use a dict to store all the necessary information
        loss_dict = {f'{mode}_loss': loss.item(), f'{mode}_pred_loss': pred_loss.item(), f'{mode}_info_loss': info_loss.item()}
        return loss, loss_dict

    
class Extractor(nn.Module):
    def __init__(self, model_config):
        super(Extractor, self).__init__()
        # set up params
        self.param = model_config['param']

        # define your own layers here
        self.conv1 = GINConv()
    
    def forward(self, batch):

        # load your data here
        x, edge_index, edge_attr, batch_idx = batch

        # write down your own forward pass here
        att = self.conv1(x, edge_index, edge_attr)
        return att
