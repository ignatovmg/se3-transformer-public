import logging
import os
import subprocess
import json
import prody
import contextlib
import tempfile
import shutil
import numpy as np
from io import StringIO
from path import Path
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import logging
import logging.config

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'default': {
            'format': '[%(levelname)s] %(asctime)s %(funcName)s [pid %(process)d] - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        }
    },
    'loggers': {
        'console': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        }
    }
}

logging.config.dictConfig(LOGGING)
logger = logging.getLogger('console')


@contextlib.contextmanager
def isolated_filesystem(dir=None, remove=True):
    """A context manager that creates a temporary folder and changes
    the current working directory to it for isolated filesystem tests.
    """
    cwd = os.getcwd()
    if dir is None:
        t = tempfile.mkdtemp(prefix='fft_ml-')
    else:
        t = dir
    os.chdir(t)
    try:
        yield t
    except Exception as e:
        logger.error(f'Temporary files are in {t}')
        raise
    else:
        os.chdir(cwd)
        if remove:
            try:
                shutil.rmtree(t)
            except (OSError, IOError):
                pass
    finally:
        os.chdir(cwd)


def get_file_handler(filename, mode='w', level='DEBUG'):
    h = logging.FileHandler(filename, mode=mode)
    h.setFormatter(logging.Formatter('[%(levelname)s] %(asctime)s - %(message)s'))
    h.setLevel(level)
    return h


def safe_read_ag(ag) -> prody.Atomic:
    if isinstance(ag, prody.Atomic):
        return ag
    elif isinstance(ag, str):
        return prody.parsePDB(ag)
    else:
        raise RuntimeError(f"Can't read atom group, 'ag' has wrong type {type(ag)}")


def check_output(call, **kwargs):
    try:
        logger.debug('Running command:\n' + ' '.join(call))
        output = subprocess.check_output(call, **kwargs)
        logger.debug('Command output:\n' + output.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        logger.debug('Command output:\n' + e.output.decode('utf-8'))
        raise
    return output.decode('utf-8')


def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def tmp_file(**kwargs):
    handle, fname = tempfile.mkstemp(**kwargs)
    os.close(handle)
    return fname


def apply_prody_transform(coords, tr):
    return np.dot(coords, tr.getRotation().T) + tr.getTranslation()


def change_mol_coords(mol, new_coords, conf_ids=None):
    if len(new_coords.shape) == 2:
        new_coords = [new_coords]

    conf_ids = range(mol.GetNumConformers()) if conf_ids is None else conf_ids

    if len(conf_ids) != len(new_coords):
        raise RuntimeError('Number of coordinate sets is different from the number of conformers')

    for coords_id, conf_id in enumerate(conf_ids):
        conformer = mol.GetConformer(conf_id)
        new_coordset = new_coords[coords_id]

        if mol.GetNumAtoms() != new_coordset.shape[0]:
            raise ValueError(f'Number of atoms is different from the number of coordinates \
            ({mol.GetNumAtoms()} != {new_coordset.shape[0]})')

        for i in range(mol.GetNumAtoms()):
            x, y, z = new_coordset[i]
            conformer.SetAtomPosition(i, Point3D(x, y, z))


def apply_prody_transform_to_rdkit_mol(mol, tr):
    mol = deepcopy(mol)
    new_coords = apply_prody_transform(mol.GetConformer().GetPositions(), tr)
    change_mol_coords(mol, new_coords)
    return mol


def mol_to_ag(mol):
    return prody.parsePDBStream(StringIO(Chem.MolToPDBBlock(mol)))


def make_list(obj):
    if not (isinstance(obj, tuple) or isinstance(obj, list)):
        obj = [obj]
    return obj
