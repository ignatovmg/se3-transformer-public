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
    def __init__(
        self,
        num_layers: int,
        in_structure: List[Tuple[int,int]],
        num_channels: int,
        out_structure: List[Tuple[int,int]],
        num_degrees: int=4,
        edge_dim: int=4,
        div: float=4,
        pooling: str='avg',
        n_heads: int=1,
        num_fc: int=0,
        out_fc_dim: int=1,
        **kwargs
    ):
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
        self.Gblock, self.PBlock, self.FCblock = blocks
        #print(self.Gblock)
        #print(self.FCblock)

    def _build_gcn(self, fibers, num_fc, out_fc_dim):
        # Equivariant layers
        Gblock = []
        fin = fibers['in']
        for i in range(self.num_layers):
            Gblock.append(GSE3Res(fin, fibers['mid'], edge_dim=self.edge_dim, div=self.div, n_heads=self.n_heads))
            Gblock.append(GNormSE3(fibers['mid']))
            fin = fibers['mid']
        Gblock.append(GConvSE3(fibers['mid'], fibers['out'], self_interaction=True, edge_dim=self.edge_dim))

        # Pooling
        Pblock = []
        if self.pooling == 'avg':
            Pblock = [GAvgPooling()]
        elif self.pooling == 'max':
            Pblock = [GMaxPooling()]

        # FC layers
        FCblock = []
        if num_fc > 0:
            for i in range(num_fc-1):
                FCblock.append(nn.Linear(self.fibers['out'].n_features, self.fibers['out'].n_features))
                FCblock.append(nn.ReLU(inplace=True))
            FCblock.append(nn.Linear(self.fibers['out'].n_features, out_fc_dim))

        return nn.ModuleList(Gblock), nn.ModuleList(Pblock), nn.ModuleList(FCblock)

    def forward(self, G, node_features: dict):
        # Compute equivariant weight basis from relative positions
        basis, r = get_basis_and_r(G, self.num_degrees-1)

        # encoder (equivariant layers)
        h = {k: G.ndata[v] for k, v in node_features.items()}
        for layer in self.Gblock:
            h = layer(h, G=G, r=r, basis=basis)

        for layer in self.PBlock:
            h = layer(h, G)
            
        if len(self.FCblock) > 0:
            for layer in self.FCblock:
                h = layer(h)

        return h


class SE3Score(nn.Module):
    def __init__(self, rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2, fin_size, num_classes):
        super().__init__()
        self.rec = SE3Transformer(
            num_layers1, 
            [(rec_feature_size, 0), (1, 1)], 
            rec_feature_size,
            [(emb_size, 0)],
            num_degrees=4, 
            edge_dim=rec_edge_dim, 
            div=4, 
            pooling=None, 
            n_heads=1, 
            num_fc=0
        )
        self.lig = SE3Transformer(
            num_layers1, 
            [(lig_feature_size, 0)], 
            lig_feature_size, 
            [(emb_size, 0)],
            num_degrees=4, 
            edge_dim=lig_edge_dim, 
            div=4, 
            pooling=None, 
            n_heads=1, 
            num_fc=0
        )
        self.cross = SE3Transformer(
            num_layers2, 
            [(emb_size, 0)], 
            emb_size, 
            [(fin_size, 0)],
            num_degrees=4, 
            edge_dim=3, 
            div=4, 
            pooling='avg', 
            n_heads=1, 
            num_fc=3, 
            out_fc_dim=num_classes
        )

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

    def forward_old(self, rec_src, rec_dst, rec_x, rec_f, rec_sidechain_vector, rec_d, lig_src, lig_dst, lig_x, lig_f, lig_w, lig_d):
        G_rec = dgl.graph((rec_src[0], rec_dst[0]))
        G_rec.ndata['x'] = rec_x[0]
        G_rec.ndata['f'] = rec_f[0]
        G_rec.ndata['sidechain_vector'] = rec_sidechain_vector[0]
        G_rec.edata['d'] = rec_d[0]
        
        #print(lig_src, lig_dst)
        G_lig = dgl.graph((lig_src[0], lig_dst[0]))
        G_lig.ndata['x'] = lig_x[0]
        G_lig.ndata['f'] = lig_f[0]
        G_lig.edata['w'] = lig_w[0]
        G_lig.edata['d'] = lig_d[0]
        
        h_rec = self.rec(G_rec, {'0': 'f', '1': 'sidechain_vector'})
        h_lig = self.lig(G_lig, {'0': 'f'})

        G_total = self._combine_graphs(G_rec, G_lig)
        G_total.ndata['f'] = torch.cat([h_rec['0'], h_lig['0']], dim=0)
        probs = self.cross(G_total, {'0': 'f'})
        return probs
    
    def forward(self, G_rec_batch, G_lig_batch):
        G_rec_list = dgl.unbatch(G_rec_batch)
        G_lig_list = dgl.unbatch(G_lig_batch)
        
        probs_list = []
        for G_rec, G_lig in zip(G_rec_list, G_lig_list):
            h_rec = self.rec(G_rec, {'0': 'f', '1': 'sidechain_vector'})
            h_lig = self.lig(G_lig, {'0': 'f'})

            G_total = self._combine_graphs(G_rec, G_lig)
            G_total.ndata['f'] = torch.cat([h_rec['0'], h_lig['0']], dim=0)
            probs = self.cross(G_total, {'0': 'f'})
            probs_list.append(probs)
        return torch.cat(probs_list, dim=0)

    
class _RecModuleSequential(nn.Module):
    def __init__(self, rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2, fin_size, num_classes):
        super().__init__()
        self.block = SE3Transformer(num_layers1, [(rec_feature_size, 0), (1, 1)], rec_feature_size, [(emb_size, 0)], num_degrees=4, edge_dim=rec_edge_dim, div=4, pooling=None, n_heads=1, num_fc=0)
        
    def forward(self, args):
        rec_src, rec_dst, rec_x, rec_f, rec_sidechain_vector, rec_d, lig_src, lig_dst, lig_x, lig_f, lig_w, lig_d = args
        G_rec = dgl.graph((rec_src[0], rec_dst[0]))
        G_rec.ndata['x'] = rec_x[0]
        G_rec.ndata['f'] = rec_f[0]
        G_rec.ndata['sidechain_vector'] = rec_sidechain_vector[0]
        G_rec.edata['d'] = rec_d[0]
        h_rec = self.block(G_rec, {'0': 'f', '1': 'sidechain_vector'})
        return [rec_src, rec_dst, rec_x, h_rec['0'][None, :], rec_sidechain_vector, rec_d, lig_src, lig_dst, lig_x, lig_f, lig_w, lig_d]
    
    
class _LigModuleSequential(nn.Module):
    def __init__(self, rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2, fin_size, num_classes):
        super().__init__()
        self.block = SE3Transformer(num_layers1, [(lig_feature_size, 0)], lig_feature_size, [(emb_size, 0)], num_degrees=4, edge_dim=lig_edge_dim, div=4, pooling=None, n_heads=1, num_fc=0)
        
    def forward(self, args):
        rec_src, rec_dst, rec_x, rec_f, rec_sidechain_vector, rec_d, lig_src, lig_dst, lig_x, lig_f, lig_w, lig_d = args
        G_lig = dgl.graph((lig_src[0], lig_dst[0]))
        G_lig.ndata['x'] = lig_x[0]
        G_lig.ndata['f'] = lig_f[0]
        G_lig.edata['w'] = lig_w[0]
        G_lig.edata['d'] = lig_d[0]
        h_lig = self.block(G_lig, {'0': 'f'})
        return [rec_src, rec_dst, rec_x, rec_f, rec_sidechain_vector, rec_d, lig_src, lig_dst, lig_x, h_lig['0'][None, :], lig_w, lig_d]
    
    
class _ComboModuleSequential(nn.Module):
    def __init__(self, rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2, fin_size, num_classes):
        super().__init__()
        self.block = SE3Transformer(num_layers2, [(emb_size, 0)], emb_size, [(fin_size, 0)], num_degrees=4, edge_dim=3, div=4, pooling='avg', n_heads=1, num_fc=3, out_fc_dim=num_classes)
        
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
        G.ndata['f'] = torch.cat([rec.ndata['f'], lig.ndata['f']], dim=0)
        G.ndata['x'] = torch.cat([rec.ndata['x'], lig.ndata['x']], dim=0)
        G.edata['d'] = G.ndata['x'][dst] - G.ndata['x'][src]
        G.edata['w'] = edge_types
        return G
        
    def forward(self, args):
        rec_src, rec_dst, rec_x, rec_f, rec_sidechain_vector, rec_d, lig_src, lig_dst, lig_x, lig_f, lig_w, lig_d = args
        G_rec = dgl.graph((rec_src[0], rec_dst[0]))
        G_rec.ndata['x'] = rec_x[0]
        G_rec.ndata['f'] = rec_f[0]
        G_rec.ndata['sidechain_vector'] = rec_sidechain_vector[0]
        G_rec.edata['d'] = rec_d[0]
        
        G_lig = dgl.graph((lig_src[0], lig_dst[0]))
        G_lig.ndata['x'] = lig_x[0]
        G_lig.ndata['f'] = lig_f[0]
        G_lig.edata['w'] = lig_w[0]
        G_lig.edata['d'] = lig_d[0]
        
        G_total = self._combine_graphs(G_rec, G_lig)
        #print('coords', G_total.ndata['x'])

        #print(G_total.ndata['f'].shape)
        probs = self.block(G_total, {'0': 'f'})
        return probs #[rec_src, rec_dst, rec_x, rec_f, rec_sidechain_vector, rec_d, lig_src, lig_dst, lig_x, lig_f, lig_w, lig_d]
    
    
def SE3ScoreSequential(rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2, fin_size, num_classes):
    layers = [
        _RecModuleSequential(rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2, fin_size, num_classes),
        _LigModuleSequential(rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2, fin_size, num_classes),
        _ComboModuleSequential(rec_feature_size, rec_edge_dim, lig_feature_size, lig_edge_dim, emb_size, num_layers1, num_layers2, fin_size, num_classes)
    ]
    return torch.nn.Sequential(*layers)