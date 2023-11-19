from typing import Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch import nn
from src.utils import at_stage, reset_random_seed

# Assuming BaseOODAlg is a generic base class for OOD algorithms. Replace with appropriate base class if different
# class BaseOODAlg:
#     def __init__(self, config):
#         pass

class DIR(nn.Module):
    def __init__(self, config: dict):
        super(DIR, self).__init__(config)
        self.rep_out = None
        self.causal_out = None
        self.conf_out = None

    def stage_control(self, config: dict):
        if self.stage == 0 and at_stage(1, config):
            reset_random_seed(config)
            self.stage = 1
        config.train.alpha = config.ood.extra_param[0] * (config.train.epoch ** 1.6)

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        if isinstance(model_output, tuple):
            self.rep_out, self.causal_out, self.conf_out, pred_edge_weight = model_output
        else:
            self.causal_out = model_output
            self.rep_out, self.conf_out = None, None
        return self.causal_out, pred_edge_weight

    def loss_calculate(self, raw_pred: Tensor, targets: Tensor, mask: Tensor, node_norm: Tensor, config: dict) -> Tensor:
        causal_loss = (config.metric.loss_func(raw_pred, targets, reduction='none') * mask).sum() / mask.sum()
        conf_loss = (config.metric.loss_func(self.conf_out, targets, reduction='none') * mask).sum() / mask.sum()

        env_loss = torch.tensor([]).to(config.device)
        for rep in self.rep_out:
            tmp = (config.metric.loss_func(rep, targets, reduction='none') * mask).sum() / mask.sum()
            env_loss = torch.cat([env_loss, (tmp.sum() / mask.sum()).unsqueeze(0)])
        causal_loss += config.train.alpha * env_loss.mean()
        env_loss = config.train.alpha * torch.var(env_loss * self.rep_out.size(0))

        loss = causal_loss + env_loss + conf_loss
        self.mean_loss = causal_loss
        self.spec_loss = env_loss + conf_loss
        return loss

    