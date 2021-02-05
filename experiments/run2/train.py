import torch
import torch.nn.functional as F
import torch.optim
import prody
import numpy as np
from tqdm import tqdm
from path import Path
from torch.utils.data import DataLoader

torch.cuda.manual_seed_all(123456)
torch.manual_seed(123456)
np.random.seed(123456)

from fft_ml.training import engines
from fft_ml.loggers import logger
from fft_ml import utils
from fft_ml.training.metrics import BaseMetrics
from mol_grid import loggers

from dataset import LigandDataset
from model import SE3Refine


def get_rmsd_loss():
    def loss_fn(y_pred, y_true):
        rmsd = torch.pow(y_pred[:, 1] - y_true, 2).sum(2).mean(1).mean()
        print('new_rmsd', rmsd.sqrt())
        #print('old_loss', torch.pow(y_pred[:, 0] - y_true, 2).sum(2).mean(1).mean().sqrt())
        return rmsd
    return loss_fn


#def get_confidence_loss():
#    def loss_fn(y_pred, y_true):
#        return 1.0 - torch.sigmoid(torch.pow(y_pred - y_true, 2).sum(1).sqrt())
#    return loss_fn


def _calc_rmsd(y_pred, y_true):
    return np.sqrt(np.power(y_pred - y_true, 2).sum(2).mean(1))


class RMSD(BaseMetrics):
    def __init__(self):
        BaseMetrics.__init__(self)
        self.rmsd = []

    def reset(self):
        self.rmsd = []

    def batch(self, prediction, target):
        batch_rmsd = _calc_rmsd(prediction[:, 1], target)
        self.rmsd += batch_rmsd.tolist()
        return batch_rmsd.mean()

    def epoch(self):
        return np.mean(self.rmsd)


class FractionImproved(BaseMetrics):
    def __init__(self, rmsd_min=0.0):
        BaseMetrics.__init__(self)
        self.improved = []
        self.rmsd_min = rmsd_min

    def reset(self):
        self.improved = []

    def batch(self, prediction, target):
        batch_old_rmsd = _calc_rmsd(prediction[:, 0], target)
        batch_new_rmsd = _calc_rmsd(prediction[:, 1], target)
        improved = [new < old for new, old in zip(batch_new_rmsd, batch_old_rmsd) if old > self.rmsd_min]
        self.improved += improved
        return np.mean(improved) if len(improved) > 0 else 0.

    def epoch(self):
        return np.mean(self.improved) if len(self.improved) > 0 else 0.


def train(
        dataset_dir,
        train_set,
        valid_set,
        max_epoch,
        device_name,
        load_epoch=None,
        batch_size=32,
        ncores=4,
        ngpu=1,
        nsamples=None,
        margin=100,
        lr=0.0001,
        weight_decay=0.00001,
        model=None,
        ray_tune=False,
        ray_checkpoint_dir=None,
        output_dir='train'):

    loggers.logger.setLevel('INFO')
    for k, v in locals().items():
        logger.info("{:20s} = {}".format(str(k), str(v)))

    prody.confProDy(verbosity='error')

    logger.info('Getting device..')
    device = engines.get_device(device_name)
    subset = None if nsamples is None else list(range(nsamples))

    logger.info('Creating train dataset..')
    #train_data = utils.read_json(Path(dataset_dir) / train_set)
    #subset = np.where([x['affinity'] is not None for x in train_data])[0]
    train_set = LigandDataset(dataset_dir, train_set, subset=None, random_rotation=True, bsite_radius=6)
    train_loader = train_set
    #train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=ncores, shuffle=True)

    logger.info('Creating validation dataset..')
    #valid_data = utils.read_json(Path(dataset_dir) / valid_set)
    #subset = np.where([x['affinity'] is not None for x in valid_data])[0]
    valid_set = LigandDataset(dataset_dir, valid_set, subset=None, random_rotation=True, bsite_radius=6)
    valid_loader = valid_set
    #valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size, num_workers=ncores, shuffle=False)

    optimizer = torch.optim.Adam
    optimizer_params = {'lr': lr, 'weight_decay': weight_decay}

    losses = get_rmsd_loss()

    logger.info('Started training..')
    trainer = engines.Trainer(None,
                              max_epoch,
                              train_loader,
                              model,
                              optimizer,
                              optimizer_params,
                              losses,
                              device,
                              x_keys=['rec_graph', 'lig_graph'],
                              y_key='crys_coords',
                              metrics_dict={'RMSD': RMSD(), 'FractionImproved': FractionImproved(), 'FractionImprovedAbove2': FractionImproved(2.0)},
                              test_loader=valid_loader,
                              save_model=True,
                              load_epoch=load_epoch,
                              use_scheduler=True,
                              scheduler_params={'factor': 0.1, 'patience': 6},
                              ngpu=ngpu,
                              every_niter=50,
                              ray_tune=ray_tune,
                              ray_checkpoint_dir=ray_checkpoint_dir,
                              early_stopping=50,
                              collect_garbage_every_n_iter=3,
                              output_dir=Path(output_dir).mkdir_p())
    trainer.run()


def _last_epoch(dir):
    last_epoch = sorted([int(x.stripext().split('_')[-1]) for x in Path(dir).glob('checkpoint*')])
    return None if len(last_epoch) == 0 else last_epoch[-1]


def main():
    import sys

    outdir = Path(sys.argv[1]).mkdir_p()
    gpu_id = int(sys.argv[2])

    # get last computed epoch
    last_epoch = _last_epoch(outdir)

    model = SE3Refine(21, 0, 40, 8, emb_size=64, num_layers1=3, num_layers2=3)

    train('dataset',
          'train_split/train.json',
          'train_split/valid.json',
          max_epoch=200,
          device_name=f'cuda:{gpu_id}',
          load_epoch=last_epoch,
          batch_size=1,
          ncores=0,
          ngpu=1,
          nsamples=None,
          lr=0.0001,
          weight_decay=0.0000001,
          model=model,
          ray_tune=False,
          ray_checkpoint_dir=None,
          output_dir=outdir)


if __name__ == '__main__':
    main()
