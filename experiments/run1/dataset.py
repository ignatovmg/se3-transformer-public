from path import Path
import prody
from functools import partial

import torch
import numpy as np
import prody
import itertools
import random
import mdtraj as md
import os
import tempfile
import dgl
from path import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from sblu.ft import apply_ftresult
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from collections import OrderedDict

from fft_ml import utils
from fft_ml.dataset.amino_acids import residue_bonds_noh

DTYPE = np.float32
DTYPE_INT = np.int32

_ELEMENTS = {x[1]: x[0] for x in enumerate(['I', 'S', 'F', 'N', 'C', 'CL', 'BR', 'O', 'H', 'X'])}
_HYBRIDIZATIONS = {x: i for i, x in enumerate(Chem.rdchem.HybridizationType.names.keys())}
_FORMAL_CHARGE = {-1: 0, 0: 1, 1: 2}
_VALENCE = {x: x - 1 for x in range(1, 7)}
_NUM_HS = {x: x for x in range(5)}
_DEGREE = {x: x - 1 for x in range(1, 5)}


def _atom_to_vector(atom):
    vec = [0] * len(_ELEMENTS)
    vec[_ELEMENTS.get(atom.GetSymbol().upper(), _ELEMENTS['X'])] = 1

    new_vec = [0] * len(_HYBRIDIZATIONS)
    new_vec[_HYBRIDIZATIONS[str(atom.GetHybridization())]] = 1
    vec += new_vec

    new_vec = [0] * len(_FORMAL_CHARGE)
    try:
        new_vec[_FORMAL_CHARGE[atom.GetFormalCharge()]] = 1
    except:
        pass
    vec += new_vec

    new_vec = [0, 0]
    new_vec[int(atom.GetIsAromatic())] = 1
    vec += new_vec

    new_vec = [0] * len(_DEGREE)
    try:
        new_vec[_DEGREE[atom.GetTotalDegree()]] = 1
    except:
        pass
    vec += new_vec

    new_vec = [0] * len(_NUM_HS)
    try:
        new_vec[_NUM_HS[atom.GetTotalNumHs()]] = 1
    except:
        pass
    vec += new_vec

    new_vec = [0] * len(_VALENCE)
    try:
        new_vec[_VALENCE[atom.GetTotalValence()]] = 1
    except:
        pass
    vec += new_vec

    new_vec = [0, 0]
    new_vec[int(atom.IsInRing())] = 1
    vec += new_vec

    # 12 bins for Gasteiger charge: [-0.6, +0.6]
    #new_vec = [0] * 12
    #gast = float(atom.GetProp('_GasteigerCharge'))
    #new_vec[int(min(max(gast + 0.6, 0), 1.1) * 10)] = 1
    #vec += new_vec

    return np.array(vec, dtype=DTYPE)


def _bond_to_vector(bond):
    # bond type
    vec = [0] * 4
    vec[max(min(int(bond.GetBondTypeAsDouble())-1, 3), 0)] = 1

    # is conjugated
    new_vec = [0] * 2
    new_vec[bond.GetIsConjugated()] = 1
    vec += new_vec

    # in ring
    new_vec = [0] * 2
    new_vec[bond.IsInRing()] = 1
    vec += new_vec
    return np.array(vec, dtype=DTYPE)


def make_lig_graph(lig_ag, lig_rd):
    mol_elements = np.array([x.GetSymbol().upper() for x in lig_rd.GetAtoms()])
    pdb_elements = np.array([x.upper() for x in lig_ag.getElements()])
    assert all(mol_elements == pdb_elements), f'Elements are different:\nRDkit: {mol_elements}\nPDB  : {pdb_elements}'

    AllChem.ComputeGasteigerCharges(lig_rd, throwOnParamFailure=True)

    mol_atoms = range(lig_rd.GetNumAtoms())
    node_features = []
    for atom_idx in mol_atoms:
        atom = lig_rd.GetAtomWithIdx(atom_idx)
        node_features.append(_atom_to_vector(atom))
    node_features = np.stack(node_features, axis=0).astype(DTYPE)

    coords = lig_ag.getCoords()[mol_atoms, :].astype(DTYPE)
    src, dst = [], []
    edge_features = []
    for b in lig_rd.GetBonds():
        edge = [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]
        src += edge
        dst += [edge[1], edge[0]]
        feat = _bond_to_vector(b)
        edge_features += [feat]*2
    edge_features = np.stack(edge_features, axis=0).astype(DTYPE)

    #G = dgl.DGLGraph((torch.tensor(src, dtype=DTYPE_INT), torch.tensor(dst, dtype=DTYPE_INT)))
    G = dgl.graph((src, dst))
    G.ndata['x'] = torch.tensor(coords)
    G.ndata['f'] = torch.tensor(node_features)[..., None]
    G.edata['d'] = torch.tensor(coords[dst] - coords[src])
    G.edata['w'] = torch.tensor(edge_features)
    return G


def _get_sidechain_vec(r):
    bb = r.backbone
    CA = bb.select('name CA').getCoords()[0]
    N = bb.select('name N').getCoords()[0]
    C = bb.select('name C').getCoords()[0]
    vec = 2 * CA - N - C
    vec = vec / np.sqrt(sum(vec*vec))
    return vec


def _residue_to_vec(r):
    keys = list(residue_bonds_noh.keys())
    keys = {x: i for i, x in enumerate(keys)}
    vec = [0] * (len(keys) + 1)
    vec[keys.get(r.getResname().upper(), len(keys))] = 1
    return np.array(vec, dtype=DTYPE)


def make_rec_graph(rec_ag):
    resnums = []
    coords = []
    features = []
    vecs = []
    for _, r in enumerate(rec_ag.getHierView().iterResidues()):
        coords.append(r.select('name CA').getCoords()[0])
        vecs.append(_get_sidechain_vec(r))
        resnums.append(r.getResnum())
        features.append(_residue_to_vec(r))
    coords = np.stack(coords).astype(DTYPE)
    vecs = np.stack(vecs).astype(DTYPE)
    features = np.stack(features).astype(DTYPE)

    num_nodes = len(resnums)
    src, dst = [], []
    for i in range(num_nodes):
        src += [i]
        dst += [i]
        for j in range(i+1, num_nodes):
            if abs(resnums[i] - resnums[j]) == 1:
                src += [i, j]
                dst += [j, i]

    G = dgl.graph((src, dst))
    G.ndata['x'] = torch.tensor(coords)
    G.ndata['f'] = torch.tensor(features)[..., None]
    G.ndata['sidechain_vector'] = torch.tensor(vecs)[:, None, :]
    G.edata['d'] = torch.tensor(coords[dst] - coords[src])
    return G


#def make_graph(rec_ag, lig_ag, lig_rd, bsite_radius=8):
#    lig_nx, lig_nfeats, lig_edges, lig_efeats = make_lig_graph(lig_ag, lig_rd)
#
#    bsite = rec_ag.protein.select(f'same residue as within {bsite_radius} of lig', lig_ag).copy()
#    rec_nx, rec_nfeats, rec_edges, rec_efeats = make_rec_graph(bsite)
#
#    G = dgl.DGLGraph((src, dst))
#
#    return G


class LigandDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 json_file,
                 subset=None,
                 #affinity_bins=(-10, 0, 1, 2, 3, 10),
                 bsite_radius=8,
                 random_rotation=True,
                 random_state=12345):

        self.dataset_dir = Path(dataset_dir).abspath()
        if not self.dataset_dir.exists():
            raise OSError(f'Directory {self.dataset_dir} does not exist')

        self.subset = subset

        self.json_file = self.dataset_dir.joinpath(json_file)
        if not self.json_file.exists():
            raise OSError(f'File {self.json_file} does not exist')

        self.data = utils.read_json(self.json_file)

        if subset is not None:
            self.data = [v for k, v in enumerate(self.data) if k in subset]

        #self.box_size = box_size
        #self.affinity_bins = affinity_bins
        self.bsite_radius = bsite_radius

        self.random_rotation = random_rotation
        self.random_state = random_state
        if self.random_rotation:
            self.rotations = Rotation.random(len(self), random_state=random_state)
        self.random = random.Random(random_state)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        item = self.data[ix]
        case_dir = self.dataset_dir / 'data' / item['sdf_id']
        rec_ag = prody.parsePDB(case_dir / 'rec.pdb')
        crys_lig_rd = Chem.MolFromMolFile(case_dir / 'lig_orig.mol', removeHs=True)
        crys_lig_ag = utils.mol_to_ag(crys_lig_rd)
        lig_poses = prody.parsePDB(case_dir / 'lig_clus.pdb').heavy.copy()
        lig_rmsds = np.loadtxt(case_dir / 'rmsd_clus.txt')

        # select one pose
        pose_id = 0
        pose_ag = lig_poses.copy()
        pose_ag._setCoords(pose_ag.getCoordsets(pose_id), overwrite=True)
        pose_rmsd = lig_rmsds[pose_id]

        if self.random_rotation:
            rotmat = self.rotations[ix].as_matrix()
            tr = prody.Transformation(rotmat, np.array([0, 0, 0]))
            rec_ag = tr.apply(rec_ag)
            pose_ag = tr.apply(pose_ag)
            crys_lig_ag = tr.apply(crys_lig_ag)

        lig_G = make_lig_graph(pose_ag, crys_lig_rd)
        bsite = rec_ag.protein.select(f'same residue as within {self.bsite_radius} of lig', lig=pose_ag).copy()
        rec_G = make_rec_graph(bsite)

        sample = {
            'id': ix,
            'case': item['sdf_id'],
            'rec_graph': rec_G,
            'lig_graph': lig_G,
            'lig_elements': pose_ag.getElements(),
            'lig_coords': pose_ag.getCoords().astype(DTYPE),
            'crys_coords': torch.tensor(crys_lig_ag.getCoords().astype(DTYPE)[None, :]),
            'crys_rmsd': pose_rmsd
        }
        #print('old', pose_rmsd)
        return sample


def main():
    ds = LigandDataset('dataset', 'train_split/test.json')
    item = ds[0]
    print(item)
    from model import SE3Transformer, SE3Refine
    G = item['lig_graph']
    trans = SE3Transformer(
        num_layers=2,
        in_structure=[(52, 0)],
        num_channels=52,
        out_structure=[(10, 0), (1, 1)],
        num_degrees=4,
        edge_dim=8,
        div=4,
        pooling='avg',
        num_fc=0
    )
    #out = trans(G, {'0': 'f'})
    #print(out.shape)

    device = 'cuda:0'
    trans = SE3Refine(21, 0, 40, 8, emb_size=32, num_layers=3).to(device)
    trans.eval()
    trans(item['rec_graph'].to(device), item['lig_graph'].to(device))

    #print(out['0'].shape)
    #print(out['1'].shape)


if __name__ == '__main__':
    main()

