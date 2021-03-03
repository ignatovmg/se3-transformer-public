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
import traceback
from path import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from sblu.ft import apply_ftresult
from scipy.spatial.transform import Rotation
from collections import OrderedDict, Counter
from tqdm import tqdm

#from dgl.data import DGLDataset
from torch.utils.data import Dataset

import utils_loc
from amino_acids import residue_bonds_noh

DTYPE_FLOAT = np.float32
DTYPE_INT = np.int32

_ELEMENTS = {x[1]: x[0] for x in enumerate(['I', 'S', 'F', 'N', 'C', 'CL', 'BR', 'O', 'P', 'UNK'])}
_HYBRIDIZATIONS = {x: i for i, x in enumerate(Chem.rdchem.HybridizationType.names.keys())}
_FORMAL_CHARGE = {-1: 0, 0: 1, 1: 2}
_VALENCE = {x: x - 1 for x in range(1, 7)}
_NUM_HS = {x: x for x in range(5)}
_DEGREE = {x: x - 1 for x in range(1, 5)}


def _atom_to_vector(atom):
    vec = [0] * (len(_ELEMENTS) + 1)
    vec[_ELEMENTS.get(atom.GetSymbol().upper(), _ELEMENTS['UNK'])] = 1

    # total density of all atoms
    vec[-1] = 1

    new_vec = [0] * len(_HYBRIDIZATIONS)
    new_vec[_HYBRIDIZATIONS[str(atom.GetHybridization())]] = 1
    vec += new_vec

    #new_vec = [0] * len(_FORMAL_CHARGE)
    #new_vec[_FORMAL_CHARGE[atom.GetFormalCharge()]] = 1
    new_vec = [atom.GetFormalCharge()]
    vec += new_vec

    new_vec = [0, 0]
    new_vec[int(atom.GetIsAromatic())] = 1
    vec += new_vec

    #new_vec = [0] * len(_DEGREE)
    #new_vec[_DEGREE[atom.GetTotalDegree()]] = 1
    #vec += new_vec
    vec += [atom.GetTotalDegree()]

    #new_vec = [0] * len(_NUM_HS)
    #new_vec[_NUM_HS[atom.GetTotalNumHs()]] = 1
    #vec += new_vec
    vec += [atom.GetTotalNumHs()]

    new_vec = [0] * len(_VALENCE)
    new_vec[_VALENCE[atom.GetTotalValence()]] = 1
    vec += new_vec

    new_vec = [0, 0]
    new_vec[int(atom.IsInRing())] = 1
    vec += new_vec

    # Gasteiger charge
    vec += [float(atom.GetProp('_GasteigerCharge'))]

    return np.array(vec, dtype=DTYPE_FLOAT)


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
    return np.array(vec, dtype=DTYPE_FLOAT)


def make_lig_graph(lig_ag, lig_rd, connect_all=False, self_edge=False, include_hs=False):
    AllChem.ComputeGasteigerCharges(lig_rd, throwOnParamFailure=True)

    mol_atoms = []
    node_features = []
    for atom in lig_rd.GetAtoms():
        if not include_hs and atom.GetSymbol() == 'H':
            continue
        node_features.append(_atom_to_vector(atom))
        mol_atoms.append(atom.GetIdx())
    node_features = np.stack(node_features, axis=0).astype(DTYPE_FLOAT)

    mol_elements = np.array([lig_rd.GetAtomWithIdx(idx).GetSymbol().upper() for idx in mol_atoms])
    pdb_elements = np.array([lig_ag.getElements()[idx].upper() for idx in mol_atoms])
    assert all(mol_elements == pdb_elements), f'Elements are different:\nRDkit: {mol_elements}\nPDB  : {pdb_elements}'

    coords = lig_ag.getCoords()[mol_atoms, :].astype(DTYPE_FLOAT)
    src, dst = [], []
    edge_features = []
    for i in mol_atoms:
        for j in mol_atoms:
            if i == j and not self_edge:
                continue
            edge = [mol_atoms.index(i), mol_atoms.index(j)]
            bond = lig_rd.GetBondBetweenAtoms(i, j)
            if not connect_all and bond is None:
                continue
                
            feat = np.zeros(9)
            feat[-1] = 1  # always set last bit to one
            if bond is not None:
                feat[:8] = _bond_to_vector(bond)
                
            src += edge
            dst += [edge[1], edge[0]]
            edge_features += [feat] * 2
            
    edge_features = np.stack(edge_features, axis=0).astype(DTYPE_FLOAT)
    
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
    return np.array(vec, dtype=DTYPE_FLOAT)


def make_rec_graph(rec_ag, connect_all=False, self_edge=False):
    resnums = []
    coords = []
    features = []
    vecs = []
    for _, r in enumerate(rec_ag.getHierView().iterResidues()):
        try:
            vec = _get_sidechain_vec(r)
            ca_crd = r.select('name CA').getCoords()[0]
        except:
            traceback.print_exc()
            continue
        vecs.append(vec)
        coords.append(ca_crd)
        resnums.append(r.getResnum())
        features.append(_residue_to_vec(r))
    coords = np.stack(coords).astype(DTYPE_FLOAT)
    vecs = np.stack(vecs).astype(DTYPE_FLOAT)
    features = np.stack(features).astype(DTYPE_FLOAT)

    num_nodes = len(resnums)
    src, dst = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if not connect_all and abs(resnums[i] - resnums[j]) > 1:
                continue
            if not self_edge and i == j:
                continue
            src.append(i)
            dst.append(j)

    G = dgl.graph((src, dst))
    G.ndata['x'] = torch.tensor(coords)
    G.ndata['f'] = torch.tensor(features)[..., None]
    G.ndata['sidechain_vector'] = torch.tensor(vecs)[:, None, :]
    G.edata['d'] = torch.tensor(coords[dst] - coords[src])
    return G


class LigandDataset(Dataset):
    def __init__(self,
                 dataset_dir,
                 json_file,
                 subset=None,
                 rmsd_bins=[(1, 3), (3, 100)],
                 max_lig_size=20,
                 bsite_radius=6,
                 connect_all=True,
                 self_edge=True,
                 random_rotation=True,
                 random_state=12345):

        self.dataset_dir = Path(dataset_dir).abspath()
        if not self.dataset_dir.exists():
            raise OSError(f'Directory {self.dataset_dir} does not exist')

        self.subset = subset

        self.json_file = self.dataset_dir.joinpath(json_file)
        if not self.json_file.exists():
            raise OSError(f'File {self.json_file} does not exist')

        self.data = utils_loc.read_json(self.json_file)

        if subset is not None:
            self.data = [v for k, v in enumerate(self.data) if k in subset]

        self.data = [x for x in self.data if x['natoms_heavy'] <= max_lig_size]
            
        filt_data = []
        for x in tqdm(self.data):
            pose_rmsds = np.loadtxt(self.dataset_dir / 'data' / x['sdf_id'] / 'rmsd_clus.txt')
            pose_dict = {}
            for label, (a, b) in enumerate(rmsd_bins, 1):
                pose_dict[label] = np.where((pose_rmsds >= a) & (pose_rmsds < b))[0].tolist()
            
            # add crystal pose
            new_item = x.copy()
            new_item['label'] = 0
            new_item['poses'] = []
            filt_data.append(new_item)
                
            # add other poses
            for label, poses in pose_dict.items():
                if len(poses) == 0:
                    continue
                new_item = x.copy()
                new_item['label'] = label
                new_item['poses'] = poses
                filt_data.append(new_item)
                
        assert len(filt_data) > 0
        self.data = filt_data
        print('Dataset size:', len(self))
        self.label_counts = Counter([x['label'] for x in self.data])
        print('Dataset size by labels:', self.label_counts)

        self.rmsd_bins = rmsd_bins
        self.num_classes = len(rmsd_bins)+1
        self.bsite_radius = bsite_radius
        self.connect_all = connect_all
        self.self_edge = self_edge

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
        crys_lig_rd = Chem.MolFromMolFile(case_dir / 'lig_orig.mol', removeHs=False)
        crys_lig_ag = utils_loc.mol_to_ag(crys_lig_rd)
        lig_poses = prody.parsePDB(case_dir / 'lig_clus.pdb')
        if lig_poses is None:
            print('Error: lig_clus.pdb in', case_dir, ' is produces None')
            item['label'] = 0
        lig_rmsds = np.loadtxt(case_dir / 'rmsd_clus.txt')
        
        # select one pose
        if item['label'] > 0:
            pose_id = self.random.choice(item['poses'])
            pose_ag = lig_poses.copy()
            pose_ag._setCoords(pose_ag.getCoordsets(pose_id), overwrite=True)
            pose_rmsd = lig_rmsds[pose_id]
        else:
            pose_id = -1
            pose_ag = crys_lig_ag.copy()
            pose_rmsd = 0.0

        if self.random_rotation:
            rotmat = self.random.choice(self.rotations).as_matrix()
            tr = prody.Transformation(rotmat, np.array([0, 0, 0]))
            rec_ag = tr.apply(rec_ag)
            pose_ag = tr.apply(pose_ag)
            crys_lig_ag = tr.apply(crys_lig_ag)

        try:
            lig_G = make_lig_graph(pose_ag, crys_lig_rd, connect_all=self.connect_all, self_edge=self.self_edge)
            bsite = rec_ag.protein.select(f'same residue as within {self.bsite_radius} of lig', lig=pose_ag).copy()
            rec_G = make_rec_graph(bsite, connect_all=self.self_edge, self_edge=self.self_edge)
        except:
            print('Exception for', case_dir)
            raise
            
        sample = {
            'id': ix,
            'pose_id': pose_id,
            'label': item['label'],
            'case': item['sdf_id'],
            'rec_src': rec_G.edges()[0], 
            'rec_dst': rec_G.edges()[1], 
            'rec_x': rec_G.ndata['x'], 
            'rec_f': rec_G.ndata['f'], 
            'rec_sidechain_vector': rec_G.ndata['sidechain_vector'], 
            'rec_d': rec_G.edata['d'],
            'lig_src': lig_G.edges()[0], 
            'lig_dst': lig_G.edges()[1], 
            'lig_x': lig_G.ndata['x'], 
            'lig_f': lig_G.ndata['f'], 
            'lig_w': lig_G.edata['w'], 
            'lig_d': lig_G.edata['d'],
            #'lig_elements': pose_ag.getElements(),
            'lig_coords': pose_ag.getCoords().astype(DTYPE_FLOAT),
            'crys_coords': torch.tensor(crys_lig_ag.getCoords().astype(DTYPE_FLOAT)[None, :]),
            'crys_rmsd': pose_rmsd
        }
        
        sample = {
            #'id': ix,
            #'pose_id': pose_id,
            'label': item['label'],
            #'case': item['sdf_id'],
            'rec_graph': rec_G,
            'lig_graph': lig_G,
            #'lig_coords': pose_ag.getCoords().astype(DTYPE_FLOAT),
            #'crys_coords': torch.tensor(crys_lig_ag.getCoords().astype(DTYPE_FLOAT)[None, :]),
            #'crys_rmsd': pose_rmsd
        }
        #print(sample)
        return sample
    
    
def collate(samples):
        rec_graph = dgl.batch([x['rec_graph'] for x in samples])
        lig_graph = dgl.batch([x['lig_graph'] for x in samples])
        label = torch.tensor([x['label'] for x in samples])
        return {'rec_graph': rec_graph, 'lig_graph': lig_graph, 'label': label}


def main():
    ds = LigandDataset('dataset', 'train_split/test.json')
    item = ds[0]
    print(item)
    from model import SE3Transformer, SE3Score
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
    print(item['lig_graph'].ndata['f'].shape)

    device = 'cuda:0'
    #def __init__(self, rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2):
    trans = SE3Score(21, 0, 33, 8, emb_size=32, num_layers1=1, num_layers2=1, num_classes=4).to(device)
    trans.eval()
    trans(item['rec_graph'].to(device), item['lig_graph'].to(device))

    #print(out['0'].shape)
    #print(out['1'].shape)


if __name__ == '__main__':
    main()

