import torch
from torch import Tensor, nn
from torch.nn import functional as F

from SpuGraph.src.backbones.bbase import BASE
from convs import LEConv

class LEGNN(BASE):
    def __init__(self, backbone_config):
        super().__init__(backbone_config)

        # define convs and relus
        self.convs = nn.ModuleList()
        self.relus = nn.ModuleList()
        for i in range(self.gc_layer):
            self.convs.append(LEConv(self.hidden_dim, self.hidden_dim))
            self.relus.append(nn.ReLU())

        self.weights_init()

    # just need the methods from base class
    