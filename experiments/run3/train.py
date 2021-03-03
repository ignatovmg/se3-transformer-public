import torch
import torch.nn.functional as F
import torch.optim
import prody
import numpy as np
from functools import partial
from path import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from mol_grid import loggers as mol_loggers

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
#from pytorch_lightning.metrics import Metric, Accuracy, Precision, Recall
from pytorch_lightning.plugins.training_type.rpc_sequential import RPCSequentialPlugin

from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

np.random.seed(123456)
pl.seed_everything(123456)

from utils_loc import logger
from dataset import LigandDataset, collate
from model import SE3Score, SE3ScoreSequential
from metrics import Accuracy


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
        y_true = [batch[y] for y in self.y_keys][0]
        y_pred = self(*x_list)
        loss = self.loss(y_pred, y_true)
        print(y_pred)
        print(y_true)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        if len(self.train_metrics) > 0:
            for k, v in self.train_metrics.items():
                self.log('train_' + k, v(y_pred, y_true), on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x_list = [batch[k] for k in self.x_keys]
        y_true = [batch[y] for y in self.y_keys][0]
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
        max_epoch=100,
        load_epoch=None,
        batch_size=1,
        ncores=0,
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
                              collate_fn=collate,
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
                              collate_fn=collate,
                              shuffle=False)

    if ray_tune:
        outdir = tune.get_trial_dir()

    logger.info('Started training..')
    model = LitModel(
        model,
        torch.nn.CrossEntropyLoss(),
        ['rec_graph', 'lig_graph'],
        #['rec_src', 'rec_dst', 'rec_x', 'rec_f', 'rec_sidechain_vector', 'rec_d', 'lig_src', 'lig_dst', 'lig_x', 'lig_f', 'lig_w', 'lig_d'],
        ['label'],
        lr=lr,
        factor=0.1,
        patience=5,
        #metrics={'Accuracy': partial(Accuracy, top_k=1), 'Precision': partial(Precision, top_k=1), 'Recall': partial(Recall, top_k=1)},
        metrics={'Accuracy': Accuracy}
    )
    loggers = {TensorBoardLogger(outdir + '/logs', 'tb'), CSVLogger(outdir + '/logs', 'csv')}
    checkpoint_callback = ModelCheckpoint(dirpath=outdir + '/checkpoints', monitor='valid_loss', filename='checkpoint-{epoch:02d}-{valid_loss:.2f}', save_top_k=2, mode='min')
    early_stopping = EarlyStopping(monitor='valid_loss', min_delta=0.00, patience=8, verbose=True, mode='min')
    tune_callback = TuneReportCallback({'ray_valid_loss': 'valid_loss', 'ray_accuracy': 'Accuracy'}, on='validation_end')
    
    trainer = pl.Trainer(
        precision=32,
        gpus=4,
        accelerator='ddp',
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
        val_check_interval=0.50,
        #limit_val_batches=1000
    )
    trainer.fit(model, train_loader, valid_loader)


def main():
    import sys

    outdir = Path(sys.argv[1]).mkdir_p()

    model = SE3Score(21, 0, 33, 9, emb_size=32, num_layers1=1, num_layers2=2, fin_size=256, num_classes=3)
    #model = SE3ScoreSequential(21, 0, 33, 8, emb_size=64, num_layers1=2, num_layers2=2, fin_size=256, num_classes=3)

    train(outdir, 
          'dataset',
          'train_split/train.json',
          'train_split/valid.json',
          model=model)


if __name__ == '__main__':
    main()
