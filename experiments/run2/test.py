import torch
import torch.nn.functional as F
import torch.optim
import prody
import numpy as np
from tqdm import tqdm
from path import Path
from torch.utils.data import DataLoader
from rdkit import Chem
import contextlib
import os

torch.cuda.manual_seed_all(123456)
torch.manual_seed(123456)
np.random.seed(123456)

from fft_ml.training import engines
from fft_ml.loggers import logger
from fft_ml import utils
from fft_ml.training.metrics import BaseMetrics
from mol_grid import loggers

from dataset import LigandDataset, make_rec_graph, make_lig_graph
from model import SE3Refine


@contextlib.contextmanager
def cwd(path):
    oldpwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(oldpwd)


def load_model(checkpoint, device):
    model = SE3Refine(21, 0, 40, 8, emb_size=64, num_layers1=3, num_layers2=3)
    engines.load_checkpoint(checkpoint, device, model=model)
    model.to(device)
    model.eval()
    return model


def get_rec_graph(rec_ag):
    return


def get_lig_graph(lig_ag, lig_rdkit):
    return


def calc_rmsd(y_pred, y_true):
    return np.sqrt(np.power(y_pred - y_true, 2).sum(1).mean(0))


def refine(rec_pdb, lig_pdb, lig_mol, checkpoint, device='cuda:0', nsteps=3, crys_mol=None):
    model = load_model(checkpoint, device)
    rec_ag = prody.parsePDB(rec_pdb)
    lig_ag = prody.parsePDB(lig_pdb).heavy.copy()
    lig_ag._setCoords(lig_ag.getCoordsets()[1], overwrite=True)
    lig_rdkit = Chem.MolFromMolFile(lig_mol, removeHs=True)
    
    prody.writePDB('lig.pdb', lig_ag)
    prody.writePDB('rec.pdb', rec_ag)

    if crys_mol:
        crys_coords = Chem.MolFromMolFile(crys_mol).GetConformer(0).GetPositions()
        rmsd = calc_rmsd(crys_coords, lig_ag.getCoords())
        print('Start :', rmsd)

    for step in range(nsteps):
        bsite_ag = rec_ag.protein.select(f'same residue as within 6 of lig', lig=lig_ag).copy()
        rec_G = make_rec_graph(bsite_ag).to(device)
        lig_G = make_lig_graph(lig_ag, lig_rdkit).to(device)

        new_lig_coords = model(rec_G, lig_G)[0, 1].detach().cpu().numpy()
        print(new_lig_coords)
        lig_ag._setCoords(new_lig_coords, overwrite=True)
        prody.writePDB(f'step.{step:02d}.pdb', lig_ag)
        if crys_mol:
            rmsd = calc_rmsd(crys_coords, new_lig_coords)
            print(step, ':', rmsd)


def main():
    sdf_id = '3ftv_11X_1_A_710__E___'
    case_dir = Path('dataset') / 'data' / sdf_id
    rec_pdb = (case_dir / 'rec.pdb').abspath()
    lig_pdb = (case_dir / 'lig_clus.pdb').abspath()
    crys_mol = (case_dir / 'lig_orig.mol').abspath()
    checkpoint = Path('run/checkpoint_17.tar').abspath()
    #oldpwd = Path.getcwd()
    wdir = Path('docking').mkdir_p()
    with cwd(wdir):
        refine(rec_pdb, lig_pdb, crys_mol, checkpoint, device='cuda:1', nsteps=1, crys_mol=crys_mol)


if __name__ == '__main__':
    main()