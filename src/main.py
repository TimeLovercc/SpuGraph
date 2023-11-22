import argparse
import importlib
import yaml
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
import torch_geometric.transforms as T
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torchmetrics.functional.classification import binary_accuracy, multiclass_accuracy

torch.set_float32_matmul_precision('medium')

class Train(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.load_model()
        self.automatic_optimization = False
        self.accumulated_loss = None

    def training_step(self, batch, batch_idx):
        out = self.model(batch, self.current_epoch, training=True)
        loss_list, loss_dict = self.model.loss(out, batch, self.current_epoch, 'train')
        metrics = {**self.metrics(out, batch, mode='train'), **loss_dict}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        if self.hparams.optimizer_config['epoch_based_backward']:
            self.accumulated_loss = [0.0] * len(loss_list) if self.accumulated_loss is None else self.accumulated_loss
            self.accumulated_loss = [acc_loss + loss for acc_loss, loss in zip(self.accumulated_loss, loss_list)]
        else:
            self.optimize_loss(loss_list)

    def on_train_epoch_end(self):
        if self.hparams.optimizer_config['epoch_based_backward']:
            self.optimize_loss(self.accumulated_loss)
        self.accumulated_loss = None
    
    def validation_step(self, batch, batch_idx):
        out = self.model(batch, self.current_epoch, training=False)
        loss_list, loss_dict = self.model.loss(out, batch, self.current_epoch, 'val')
        metrics = {**self.metrics(out, batch, mode='val'), **loss_dict}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
    
    def test_step(self, batch, batch_idx):
        out = self.model(batch, self.current_epoch, training=False)
        loss_list, loss_dict = self.model.loss(out, batch, self.current_epoch, 'test')
        metrics = {**self.metrics(out, batch, mode='test'), **loss_dict}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)

    def optimize_loss(self, loss_list):
        assert len(loss_list) == len(self.optimizers())
        for idx, loss in enumerate(loss_list):
            optimizer = self.optimizers()[idx]
            optimizer.zero_grad()
            self.manual_backward(loss, retrain_graph=False if idx == len(loss_list)-1 else True)
            optimizer.step()

    def configure_optimizers(self):
        params_list = self.model.get_parameters()
        optimizers_and_schedulers = [self.configure_one_optimizer(self.hparams.optimizer_config, params) for params in params_list]
        optimizers = [result[0] for result in optimizers_and_schedulers]
        schedulers = [result[1] for result in optimizers_and_schedulers if result[1] is not None]
        return optimizers, schedulers
        
    def configure_one_optimizer(self, optimizer_config, params):
        weight_decay = optimizer_config.get('weight_decay', 0)
        optimizer = optim.Adam(params, lr=optimizer_config['lr'], weight_decay=weight_decay)
        lr_scheduler_type = optimizer_config.get('lr_scheduler')
        if lr_scheduler_type == 'step':
            scheduler = lrs.StepLR(optimizer, step_size=optimizer_config['lr_decay_steps'], gamma=optimizer_config['lr_decay_rate'])
            return optimizer, scheduler
        elif lr_scheduler_type == 'cosine':
            scheduler = lrs.CosineAnnealingLR(optimizer, T_max=optimizer_config['lr_decay_steps'], eta_min=optimizer_config['lr_decay_min_lr'])
            return optimizer, scheduler
        elif lr_scheduler_type is None:
            return optimizer, None
        else:
            raise ValueError('Invalid lr_scheduler type!')

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
    
    def metrics(self, out, batch, mode):
        preds = out[0] if isinstance(out, tuple) else out
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
        batched_train_set = Batch.from_data_list(self.dataset)
        d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
        deg = torch.bincount(d, minlength=10)
        print(f'Feature dimension: {feat_dim}')
        print(f'Edge feature dimension: {edge_attr_dim}')
        print(f'Number of classes: {class_num}')
        return feat_dim, edge_attr_dim, class_num, deg

def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
        min_delta=1
    ))

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

    if args.optimizer_config['lr_scheduler'] != None:
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
    feat_dim, edge_attr_dim, class_num, deg = data_module.get_in_out_dim()
    args.model_config['in_dim'] = args.backbone_config['in_dim'] = feat_dim
    args.model_config['edge_attr_dim'] = args.backbone_config['edge_attr_dim'] = edge_attr_dim
    args.model_config['out_dim'] = args.backbone_config['out_dim'] = class_num if class_num > 2 else 1
    args.model_config['deg'] = args.backbone_config['deg'] = deg
    args.model_config['hidden_dim'] = args.backbone_config['hidden_dim']

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


