import torch
import torch.nn.functional as F
import torch.optim
import prody
import numpy as np
from functools import partial
from path import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fft_ml.training import engines
from fft_ml.loggers import logger
from fft_ml.training.metrics import BaseMetrics
from mol_grid import loggers as mol_loggers

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.metrics import Metric, Accuracy, Precision, AUROC

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

np.random.seed(123456)
pl.seed_everything(123456)

from utils import logger
from dataset import LigandDataset
from model import SE3Refine


class LitModel(pl.LightningModule):
    def __init__(self, model, loss, x_keys, y_keys, lr=1e-4, factor=0.1, patience=10, metrics={}):
        super().__init__()
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.x_keys = x_keys
        self.y_keys = y_keys

        self.model = model
        self.loss = loss
        self.train_metrics = torch.nn.ModuleDict({x[0]: x[1]() for x in metrics.items()})
        self.valid_metrics = torch.nn.ModuleDict({x[0]: x[1]() for x in metrics.items()})

    def forward(self, *x):
        return self.model(*x)

    def training_step(self, batch, batch_idx):
        x_list = [batch[k] for k in self.x_keys]
        y_true = [batch[y] for y in self.y_keys]
        y_pred = self(*x_list)
        loss = self.loss(y_pred, y_true)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if len(self.train_metrics) > 0:
            for k, v in self.train_metrics.items():
                self.log('train_' + k, v(y_pred, y_true), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_list = [batch[k] for k in self.x_keys]
        y_true = [batch[y] for y in self.y_keys]
        y_pred = self(*x_list)
        loss = self.loss(y_pred, y_true)
        self.log('valid_loss', loss)

        if len(self.valid_metrics) > 0:
            for k, v in self.valid_metrics.items():
                self.log('valid_' + k, v(y_pred, y_true))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(opt, 'min', factor=self.factor, patience=self.patience)
        return {
            'optimizer': opt,
            'lr_scheduler': scheduler,
            'monitor': 'valid_loss'
        }


def train(
        outdir,
        dataset_dir,
        train_set,
        valid_set,
        max_epoch,
        load_epoch=None,
        batch_size=32,
        ncores=4,
        ngpu=1,
        margin=100,
        lr=0.0001,
        model=None,
        ray_tune=False,
        ray_checkpoint_dir=None,
        dataset_kwargs={}
):

    mol_loggers.logger.setLevel('INFO')
    for k, v in locals().items():
        logger.info("{:20s} = {}".format(str(k), str(v)))
    prody.confProDy(verbosity='error')

    logger.info('Creating train dataset..')
    train_set = LigandDataset(
        dataset_dir,
        train_set,
        **dataset_kwargs
    )
    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              num_workers=ncores,
                              shuffle=True)

    logger.info('Creating validation dataset..')
    valid_set = LigandDataset(
        dataset_dir,
        valid_set,
        **dataset_kwargs
    )
    valid_loader = DataLoader(dataset=valid_set,
                              batch_size=batch_size,
                              num_workers=ncores,
                              shuffle=False)

    if ray_tune:
        outdir = tune.get_trial_dir()

    logger.info('Started training..')
    model = LitModel(
        model,
        torch.nn.CrossEntropyLoss(),
        ['rec_grid', 'lig_grids'],
        ['rmsd_maps'],
        lr=lr,
        factor=0.1,
        patience=6,
        metrics={}
    )
    loggers = {TensorBoardLogger(outdir + '/logs', 'tb'), CSVLogger(outdir + '/logs', 'csv')}
    checkpoint_callback = ModelCheckpoint(dirpath=outdir + '/checkpoints', monitor='valid_loss', filename='checkpoint-{epoch:02d}-{valid_loss:.2f}', save_top_k=2, mode='min')
    early_stopping = EarlyStopping(monitor='valid_loss', min_delta=0.00, patience=8, verbose=True, mode='min')
    tune_callback = TuneReportCallback({'ray_valid_loss': 'valid_loss', 'ray_top1': 'valid_top1', 'ray_top10': 'valid_top10'}, on='validation_end')

    trainer = pl.Trainer(
        gpus=ngpu,
        default_root_dir=outdir,
        log_gpu_memory='all',
        logger=loggers,
        callbacks=[checkpoint_callback, early_stopping] + ([] if not ray_tune else [tune_callback]),
        log_every_n_steps=1,
        fast_dev_run=False,
        max_epochs=max_epoch,
        #overfit_batches=5,
        deterministic=True,
        #profiler='simple',
        #val_check_interval=1.0,
        #limit_val_batches=1000
    )
    trainer.fit(model, train_loader, valid_loader)


def main():
    import sys

    outdir = Path(sys.argv[1]).mkdir_p()

    model = SE3Refine(95, 35, -1, 32, 32,
                   kernel_size=5,
                   num_middle_layers=10,
                   num_postfft_layers=0,
                   num_dense_layers=3,
                   middle_activation=F.relu,
                   final_activation=F.tanh)

    train(outdir, 'dataset',
          'train_split/train.json',
          'train_split/valid.json',
          max_epoch=200,
          load_epoch=None,
          batch_size=1,
          ncores=0,
          ngpu=1,
          lr=0.0001,
          model=model,
          ray_tune=False,
          ray_checkpoint_dir=None,)


if __name__ == '__main__':
    main()
