import sys

import numpy as np
import torch

from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List

from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, atom_feature_size: int, num_channels: int, out_structure: List[Tuple[int,int]],
                 num_degrees: int=4, edge_dim: int=4, div: float=4,
                 pooling: str='avg', n_heads: int=1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(1, atom_feature_size),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(structure=out_structure)}

        blocks = self._build_gcn(self.fibers, 1)
        self.Gblock, self.FCblock = blocks
        print(self.Gblock)
        print(self.FCblock)

    def _build_gcn(self, fibers, out_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, 
                                  div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        if self.pooling == 'avg':
            Gblock.append(GAvgPooling())
        elif self.pooling == 'max':
            Gblock.append(GMaxPooling())

        # FC layers
        FCblock = []
        FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
        FCblock.append(nn.ReLU(inplace=True))
        FCblock.append(nn.Linear(self.fibers['out'].n_features, out_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {'0': G.ndata['f']}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        #for layer in self.FCblock:
        #    h = layer(h)

        return h
