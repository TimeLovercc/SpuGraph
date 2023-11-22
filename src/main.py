import argparse
import importlib
import inspect
import yaml
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim.lr_scheduler as lrs
import torch_geometric.transforms as T
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch_geometric.loader import DataLoader
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_f1_score,
    multiclass_accuracy,
    multiclass_auroc,
    multiclass_f1_score,
)

torch.set_float32_matmul_precision('medium')

class Train(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()

    def training_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.model.loss(out, batch, mode='train')
        metrics = self.metrics(out, batch, mode='train')
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.model.loss(out, batch, mode='val')
        metrics = self.metrics(out, batch, mode='val')
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss
    
    def test_step(self, batch, batch_idx):
        out = self.model(batch)
        loss = self.model.loss(out, batch, mode='test')
        metrics = self.metrics(out, batch, mode='test')
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss
    
    def configure_optimizers(self):
        if 'weight_decay' in  self.hparams.model_config.keys():
            weight_decay = self.hparams.model_config['weight_decay']
        else:
            weight_decay = 0
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.model_config['lr'], weight_decay=weight_decay)
        if self.hparams.model_config['lr_scheduler'] == None:
            return optimizer
        else:
            if self.hparams.model_config['lr_scheduler'] == 'step':
                scheduler = lrs.StepLR(optimizer,
                                       step_size=self.hparams.model_config['lr_decay_steps'],
                                       gamma=self.hparams.model_config['lr_decay_rate'])
            elif self.hparams.model_config['lr_scheduler'] == 'cosine':
                scheduler = lrs.CosineAnnealingLR(optimizer,
                                                  T_max=self.hparams.model_config['lr_decay_steps'],
                                                  eta_min=self.hparams.model_config['lr_decay_min_lr'])
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return [optimizer], [scheduler]

    def load_model(self):
        model_name = self.hparams.model_name
        model_upper_name = model_name.upper()
        backbone_name = self.hparams.backbone_name
        backbone_upper_name = backbone_name.upper()
        try:
            sys.path.append('./src/models')
            Model = getattr(importlib.import_module(model_name), model_upper_name)
        except:
            raise ValueError(f'Invalid Module File Name or Invalid Class Name {model_name}.{model_upper_name}!')
        try:
            sys.path.append('./src/backbones')
            Backbone = getattr(importlib.import_module(backbone_name), backbone_upper_name)
        except:
            raise ValueError(f'Invalid Backbone File Name or Invalid Class Name {backbone_name}.{backbone_upper_name}!')
        self.model = Model(self.hparams.model_config)
        self.model.backbone = Backbone(self.hparams.backbone_config)

    # def instancialize(self, Model, is_backbone, **other_args):
    #     class_args = inspect.getfullargspec(Model.__init__).args[1:]
    #     inkeys = self.hparams.model_config.keys() if not is_backbone else self.hparams.backbone_config.keys()
    #     args1 = {}
    #     for arg in class_args:
    #         if arg in inkeys:
    #             args1[arg] = self.hparams.model_config[arg] if not is_backbone else self.hparams.backbone_config[arg]
    #     args1.update(other_args)
    #     return Model(**args1)
    
    def metrics(self, out, batch, mode):
        preds = out
        labels, groups = batch['y'], batch['env']
        unique_groups = torch.unique(groups)

        worst_group_acc = float('inf')  # Initialize worst group accuracy
        for group in unique_groups:
            group_mask = (groups == group) 
            if self.hparams.backbone_config['out_dim'] == 1:
                group_acc = binary_accuracy(preds[group_mask], labels[group_mask])
            elif self.hparams.backbone_config['out_dim'] > 2:
                group_acc = multiclass_accuracy(preds[group_mask], labels[group_mask], num_classes=self.hparams.backbone_config['out_dim'], average='micro')
            
            worst_group_acc = min(worst_group_acc, group_acc)  # Update worst group accuracy

        if self.hparams.backbone_config['out_dim'] == 1:
            acc = binary_accuracy(preds, labels)
        elif self.hparams.backbone_config['out_dim'] > 2:
            acc = multiclass_accuracy(preds, labels, num_classes=self.hparams.backbone_config['out_dim'], average='micro')
        
        metrics_dict = {f'{mode}_acc': acc, f'{mode}_worst_group_acc': worst_group_acc}
        return metrics_dict

class DInterface(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.load_data_module()
        
    def setup(self, stage):
        if stage == 'fit' or stage is None:
            self.trainset = self.instancialize(self.dataset_class, 'train', self.hparams.data_config)
            self.valset = self.instancialize(self.dataset_class, 'val', self.hparams.data_config)
        if stage == 'test' or stage is None:
            self.testset = self.instancialize(self.dataset_class, 'test', self.hparams.data_config)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.hparams.data_config['batch_size'], num_workers=8, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.hparams.data_config['batch_size'], num_workers=8, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.hparams.data_config['batch_size'], num_workers=8, shuffle=False)
    
    def load_data_module(self):
        name = self.hparams.dataset_name
        camel_name = ''.join([i.capitalize() for i in name.split('_')])
        try:
            sys.path.append('./src/datasets')
            Dataset = getattr(importlib.import_module(name), camel_name)
        except:
            raise ValueError(f'Invalid Dataset File Name or Invalid Class Name {name}.{camel_name}')
        self.dataset_class = Dataset
        self.dataset = self.instancialize(Dataset, 'train', self.hparams.data_config)
        
    def instancialize(self, Dataset, split, data_config):
        if data_config['transform'] == 'normalize':
            data_config['transform'] = T.NormalizeFeatures()
        return Dataset(split, data_config)
    
    def get_in_out_dim(self):
        feat_dim = self.dataset.num_features
        edge_attr_dim = self.dataset.num_edge_features
        class_num = self.dataset.num_classes
        print(f'Feature dimension: {feat_dim}')
        print(f'Edge feature dimension: {edge_attr_dim}')
        print(f'Number of classes: {class_num}')
        return feat_dim, edge_attr_dim, class_num

def load_callbacks(args):
    callbacks = []
    # callbacks.append(plc.EarlyStopping(
    #     monitor='val_loss',
    #     mode='min',
    #     patience=30,
    #     min_delta=1
    # ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='val_loss',
        filename='best',
        save_top_k=1,
        mode='min',
        save_last=True
    ))

    callbacks.append(plc.RichProgressBar(
        refresh_rate=1
    ))

    if args.model_config['lr_scheduler'] != None:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='epoch'))
    return callbacks

def main():
    parser = argparse.ArgumentParser(description='SpuGraph')
    parser.add_argument('--dataset_name', type=str, help='dataset name')
    parser.add_argument('--backbone_name', type=str, help='backbone config')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--seed', type=int, help='random seed')
    args = parser.parse_args()
    config_dir = Path('./src/configs')
    global_config = yaml.safe_load((config_dir / 'global_config.yml').open('r'))
    local_config = yaml.safe_load((config_dir / f'{args.dataset_name}_{args.backbone_name}_{args.model_name}.yml').open('r'))
    args = argparse.Namespace(**{**global_config, **local_config, **vars(args)})
    
    pl.seed_everything(args.seed)
    data_module = DInterface(**vars(args))
    feat_dim, edge_attr_dim, class_num = data_module.get_in_out_dim()
    args.backbone_config['in_dim'] = feat_dim
    args.backbone_config['edge_attr_dim'] = edge_attr_dim
    args.backbone_config['out_dim'] = class_num if class_num > 2 else 1

    model = Train(**vars(args))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_logger = CSVLogger(save_dir=Path(args.log_dir) / f'{args.dataset_name}_{args.backbone_name}_{args.model_name}', version=timestamp)
    callbacks = load_callbacks(args)
    trainer = Trainer(max_epochs=args.epochs, accelerator='gpu',\
                          logger=csv_logger, log_every_n_steps=1, callbacks=callbacks)
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    if args.model_name in ['caf', 'spu']: # function for two stage models
        model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        trainer.fit(model, datamodule=data_module)
        trainer.test(model, datamodule=data_module)

if __name__ == "__main__":
    main()


