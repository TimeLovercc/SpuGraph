from typing import Tuple
import torch
from torch import Tensor
from torch_geometric.data import Batch
from torch import nn
from src.utils import at_stage, reset_random_seed
from src.utils import process_data, subgraph, visualize_a_graph
from torch_geometric.data import DataLoader
import numpy as np
import rdkit.Chem as Chem


class BaseOODAlg:
    def __init__(self, config):
        pass

class GSAT(BaseOODAlg):
    def __init__(self, config: dict):
        super(GSAT, self).__init__(config)
        self.att = None
        self.edge_att = None
        self.decay_r = 0.1
        self.decay_interval = config.get('ood', {}).get('extra_param', [0, 0, 0])[1]
        self.final_r = config.get('ood', {}).get('extra_param', [0, 0, 0])[2]

    def output_postprocess(self, model_output: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        raw_out, self.att, self.edge_att = model_output
        return raw_out, self.edge_att

    def loss_postprocess(self, loss: Tensor, data: Batch, mask: Tensor, config: dict, **kwargs) -> Tensor:
        att = self.att
        eps = 1e-6
        r = self.get_r(self.decay_interval, self.decay_r, config.get('train', {}).get('epoch', 0), final_r=self.final_r)
        info_loss = (att * torch.log(att / r + eps) + (1 - att) * torch.log((1 - att) / (1 - r + eps) + eps)).mean()

        self.mean_loss = loss.mean()
        self.spec_loss = config.get('ood', {}).get('ood_param', 0.1) * info_loss
        loss = self.mean_loss + self.spec_loss
        return loss

    def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
        r = init_r - current_epoch // decay_interval * decay_r
        if r < final_r:
            r = final_r
        return r

    def get_viz_idx(self, test_set, dataset_name):
        y_dist = test_set.data.y.numpy().reshape(-1)
        num_nodes = np.array([each.x.shape[0] for each in test_set])
        classes = np.unique(y_dist)
        res = []
        for each_class in classes:
            tag = 'class_' + str(each_class)
            if dataset_name == 'Graph-SST2':
                condi = (y_dist == each_class) * (num_nodes > 5) * (num_nodes < 10)  # in case too short or too long
                candidate_set = np.nonzero(condi)[0]
            else:
                candidate_set = np.nonzero(y_dist == each_class)[0]
            idx = np.random.choice(candidate_set, self.num_viz_samples, replace=False)
            res.append((idx, tag))
        return res

    def visualize_results(self, test_set, idx, epoch, tag, use_edge_attr):
        viz_set = test_set[idx]
        data = next(iter(DataLoader(viz_set, batch_size=len(idx), shuffle=False)))
        data = process_data(data, use_edge_attr)
        batch_att, _, clf_logits = self.eval_one_batch(data.to(self.device), epoch)
        imgs = []
        for i in range(len(viz_set)):
            mol_type, coor = None, None
            if self.dataset_name == 'mutag':
                node_dict = {0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N', 5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I', 10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca'}
                mol_type = {k: node_dict[v.item()] for k, v in enumerate(viz_set[i].node_type)}
            elif self.dataset_name == 'Graph-SST2':
                mol_type = {k: v for k, v in enumerate(viz_set[i].sentence_tokens)}
                num_nodes = data.x.shape[0]
                x = np.linspace(0, 1, num_nodes)
                y = np.ones_like(x)
                coor = np.stack([x, y], axis=1)
            elif self.dataset_name == 'ogbg_molhiv':
                element_idxs = {k: int(v+1) for k, v in enumerate(viz_set[i].x[:, 0])}
                mol_type = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v)) for k, v in element_idxs.items()}
            elif self.dataset_name == 'mnist':
                raise NotImplementedError

            node_subset = data.batch == i
            _, edge_att = subgraph(node_subset, data.edge_index, edge_attr=batch_att)

            node_label = viz_set[i].node_label.reshape(-1) if viz_set[i].get('node_label', None) is not None else torch.zeros(viz_set[i].x.shape[0])
            fig, img = visualize_a_graph(viz_set[i].edge_index, edge_att, node_label, self.dataset_name, norm=self.viz_norm_att, mol_type=mol_type, coor=coor)
            imgs.append(img)
        imgs = np.stack(imgs)