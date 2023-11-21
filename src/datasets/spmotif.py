import os
import os.path as osp
from pathlib import Path
import gdown
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data.dataset import ConcatDataset, Dataset
from torch_geometric.data import InMemoryDataset, Data, extract_zip

from spmotif_utils import gen_dataset

class Spmotif(InMemoryDataset):

    # def __init__(self, root, b, split, fg_only, generate, transform=None, pre_transform=None, pre_filter=None):
    def __init__(self, split, data_config):
        self.split = split
        self.b = data_config['b']
        root = data_config['root'] + f'_{self.b}'
        self.fg_only = data_config['fg_only']
        self.generate = data_config['generate']
        self.transform = data_config.get('transform', None)
        self.pre_transform = data_config.get('pre_transform', None)
        self.pre_filter = data_config.get('pre_filter', None)
        assert split in ['train', 'val', 'test']

        super().__init__(root, self.transform, self.pre_transform, self.pre_filter)
        if self.fg_only:
            idx = self.processed_file_names.index('SPMotif_{}_fg.pt'.format(self.split))
        else:
            idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.split))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['train.npy', 'val.npy', 'test.npy', 'train_fg.npy', 'val_fg.npy', 'test_fg.npy']

    @property
    def processed_file_names(self):
        return ['SPMotif_train.pt', 'SPMotif_val.pt', 'SPMotif_test.pt', 'SPMotif_train_fg.pt', 'SPMotif_val_fg.pt', 'SPMotif_test_fg.pt']

    def download(self):
        if not self.generate:
            print(f'Downloading SPMotif_{self.b} dataset...')
            url_mapping = {
                0.9: 'https://drive.google.com/file/d/1RI8yzhkDpQnjMVhRdNcqvwRoqgJk0aqj/view?usp=sharing',
                0.7: 'https://drive.google.com/file/d/1ois1ET7k5qMTaYZ85Wk_Gd9YSOlI-VLh/view?usp=sharing',
                0.5: 'https://drive.google.com/file/d/1219xegus6y2B2plM5MCU-uKPRk7U3oEg/view?usp=sharing',
                0.33: 'https://drive.google.com/file/d/1Z9fplFsZfm5TOL3pnj5ucLbnGchaIxoA/view?usp=sharing'
            }
            url = url_mapping.get(self.b, None)  
            path = gdown.download(url, output=osp.join(self.raw_dir, f'spmotif_{self.b}.zip'), fuzzy=True)
            extract_zip(path, self.raw_dir)
            os.unlink(path)
        else:
            print(f'Generating SPMotif_{self.b} dataset...')
            gen_dataset(self.b, Path(self.raw_dir))

    def process(self):
        if self.fg_only:
            name_idx = self.raw_file_names.index('{}_fg.npy'.format(self.split))
            post_name_idx = self.processed_file_names.index('SPMotif_{}_fg.pt'.format(self.split))
        else:
            name_idx = self.raw_file_names.index('{}.npy'.format(self.split))
            post_name_idx = self.processed_file_names.index('SPMotif_{}.pt'.format(self.split))
        edge_index_list, label_list, env_list, ground_truth_list, role_id_list, pos = np.load(osp.join(self.raw_dir, self.raw_file_names[name_idx]), allow_pickle=True)
        data_list = []
        for idx, (edge_index, y, env, ground_truth, z, p) in enumerate(zip(edge_index_list, label_list, env_list, ground_truth_list, role_id_list, pos)):
            edge_index = torch.from_numpy(edge_index).long()
            node_idx = torch.unique(edge_index)

            x = torch.rand((node_idx.size(0), 4))
            edge_attr = torch.ones(edge_index.size(1), 1)
            y = torch.tensor(y, dtype=torch.long).reshape(-1)
            env = torch.tensor(env, dtype=torch.long).reshape(-1)

            node_label = torch.tensor(z, dtype=torch.float)
            node_label[node_label != 0] = 1
            edge_label = torch.tensor(ground_truth, dtype=torch.float)

            data = Data(x=x, y=y, env=env, edge_index=edge_index, edge_attr=edge_attr, node_label=node_label, edge_label=edge_label)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[post_name_idx])
