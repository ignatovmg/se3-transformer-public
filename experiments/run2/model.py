import sys

import numpy as np
import torch

import dgl
from dgl.nn.pytorch import GraphConv, NNConv
from torch import nn
from torch.nn import functional as F
from typing import Dict, Tuple, List
import itertools


from equivariant_attention.modules import GConvSE3, GNormSE3, get_basis_and_r, GSE3Res, GMaxPooling, GAvgPooling
from equivariant_attention.fibers import Fiber


class SE3Transformer(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers: int, in_structure: List[Tuple[int,int]], num_channels: int, out_structure: List[Tuple[int,int]],
                 num_degrees: int=4, edge_dim: int=4, div: float=4,
                 pooling: str='avg', n_heads: int=1, num_fc: int=0, out_fc_dim: int=1, **kwargs):
        super().__init__()
        # Build the network
        self.num_layers = num_layers
        self.num_channels = num_channels
        self.num_degrees = num_degrees
        self.edge_dim = edge_dim
        self.div = div
        self.pooling = pooling
        self.n_heads = n_heads

        self.fibers = {'in': Fiber(structure=in_structure),
                       'mid': Fiber(num_degrees, self.num_channels),
                       'out': Fiber(structure=out_structure)}

        blocks = self._build_gcn(self.fibers, num_fc, out_fc_dim)
        self.Gblock, self.FCblock = blocks
        #print(self.Gblock)
        #print(self.FCblock)

    def _build_gcn(self, fibers, num_fc, out_fc_dim):
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
        if num_fc > 0:
            for i in range(num_fc-1):
                FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
                FCblock.append(nn.ReLU(inplace=True))
            FCblock.append(nn.Linear(self.fibers['out'].n_features, out_fc_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(FCblock)

    def forward(self, G, node_features: dict):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {k: G.ndata[v] for k, v in node_features.items()}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        return h


class SE3Refine(nn.Module):
    def __init__(self, rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2):
        super().__init__()
        self.rec = SE3Transformer(num_layers1, [(rec_feature_size, 0), (1, 1)], rec_feature_size, [(emb_size, 0)],
                                  num_degrees=4, edge_dim=rec_edge_dim, div=4, pooling=None, n_heads=1, num_fc=0)
        self.lig = SE3Transformer(num_layers1, [(lig_feature_size, 0)], lig_feature_size, [(emb_size, 0)],
                                  num_degrees=4, edge_dim=lig_edge_dim, div=4, pooling=None, n_heads=1, num_fc=0)
        self.cross = SE3Transformer(num_layers2, [(emb_size, 0)], emb_size, [(1, 1)],
                                    num_degrees=4, edge_dim=3, div=4, pooling=None, n_heads=1)
        #self.score = SE3Transformer(3, [(emb_size, 0)], emb_size, [(emb_size, 0)],
        #                            num_degrees=4, edge_dim=3, div=4, pooling=None, n_heads=1)
        #self.confidence = nn.Sequential(
        #    nn.Linear(emb_size, emb_size),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(emb_size, 1),
        #    nn.Sigmoid()
        #)

    def _combine_graphs(self, rec, lig):
        dtype = rec.ndata['f'].dtype
        device = rec.ndata['f'].device
        rec_nodes = rec.nodes()
        lig_nodes = lig.nodes() + rec.num_nodes()

        cross_edges = list(itertools.product(rec_nodes, lig_nodes))
        src_cross = torch.tensor([x[0] for x in cross_edges], dtype=rec_nodes.dtype, device=rec_nodes.device)
        dst_cross = torch.tensor([x[1] for x in cross_edges], dtype=rec_nodes.dtype, device=rec_nodes.device)
        src = torch.cat([rec.edges()[0], lig.edges()[0] + rec.num_nodes(), src_cross, dst_cross])
        dst = torch.cat([rec.edges()[1], lig.edges()[1] + rec.num_nodes(), dst_cross, src_cross])

        edge_types = torch.zeros((rec.num_edges() + lig.num_edges() + rec.num_nodes() * lig.num_nodes() * 2, 3), dtype=dtype, device=device)
        edge_types[:rec.num_edges(), 0] = 1
        edge_types[rec.num_edges():rec.num_edges()+lig.num_edges(), 1] = 1
        edge_types[rec.num_edges()+lig.num_edges():, 2] = 1

        G = dgl.graph((src, dst), device=device)
        G.ndata['x'] = torch.cat([rec.ndata['x'], lig.ndata['x']], dim=0)
        G.edata['d'] = G.ndata['x'][dst] - G.ndata['x'][src]
        G.edata['w'] = edge_types
        return G

    def forward(self, G_rec, G_lig):
        h_rec = self.rec(G_rec, {'0': 'f', '1': 'sidechain_vector'})
        h_lig = self.lig(G_lig, {'0': 'f'})

        G_total = self._combine_graphs(G_rec, G_lig)
        G_total.ndata['f'] = torch.cat([h_rec['0'], h_lig['0']], dim=0)
        #print('coords', G_total.ndata['x'])

        #print(G_total.ndata['f'].shape)
        coord_update = self.cross(G_total, {'0': 'f'})['1'][:, 0, :]
        #print(coord_update)
        G_total.ndata['x'] += coord_update
        G_total.edata['d'] += coord_update[G_total.edges()[1]] - coord_update[G_total.edges()[0]]
        lig_updated_coords = G_total.ndata['x'][G_rec.num_nodes():]
        lig_old_coords = G_lig.ndata['x']

        #scores = self.score(G_total, {'0': 'f'})['0'][..., 0]
        #conf = self.confidence(scores)
        #return lig_updated_coords[None, :]

        #print('old coords', lig_old_coords[:10])
        #print('new coords', lig_updated_coords[:10])
        return torch.stack([lig_old_coords, lig_updated_coords])[None, :]
