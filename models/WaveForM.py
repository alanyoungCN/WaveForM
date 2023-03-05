import imp
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from typing import Optional

from pytorch_wavelets import DWT1DForward, DWT1DInverse

from .GP import GPModule

class GraphConstructor(nn.Module):
    def __init__(
        self, nnodes: int, k: int, dim: int, alpha: float, xd: Optional[int] = None
    ):
        super(GraphConstructor, self).__init__()
        if xd is not None:
            self._static_feature_dim = xd
            self._linear1 = nn.Linear(xd, dim)
            self._linear2 = nn.Linear(xd, dim)
        else:
            self._embedding1 = nn.Embedding(nnodes, dim)
            self._embedding2 = nn.Embedding(nnodes, dim)
            self._linear1 = nn.Linear(dim, dim)
            self._linear2 = nn.Linear(dim, dim)
        
        self._k = k
        self._alpha = alpha
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self, idx: torch.LongTensor, FE: Optional[torch.FloatTensor] = None
    ) -> torch.FloatTensor:
        
        if FE is None:
            nodevec1 = self._embedding1(idx)
            nodevec2 = self._embedding2(idx)
        else:
            assert FE.shape[1] == self._static_feature_dim
            nodevec1 = FE[idx, :]
            nodevec2 = nodevec1 
        
        nodevec1 = torch.tanh(self._alpha * self._linear1(nodevec1))
        nodevec2 = torch.tanh(self._alpha * self._linear2(nodevec2))
    
        a = torch.mm(nodevec1, nodevec2.transpose(1, 0)) - torch.mm(
            nodevec2, nodevec1.transpose(1, 0)
        )
        A = F.relu(torch.tanh(self._alpha * a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(A.device)
        mask.fill_(float("0"))
        
        s1, t1 = A.topk(self._k, 1)
        mask.scatter_(1, t1, s1.fill_(1))
        A = A * mask
        return A

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.points = configs.n_points
        self.dropout = configs.dropout
        
        decompose_layer = configs.wavelet_j
        wave = configs.wavelet
        
        mode = 'symmetric' 
        self.dwt = DWT1DForward(wave=wave, J=decompose_layer, mode=mode)  
        self.idwt = DWT1DInverse(wave=wave)
        
        
        tmp1 = torch.randn(1, 1, self.seq_len)
        tmp1_yl, tmp1_yh = self.dwt(tmp1)
        tmp1_coefs = [tmp1_yl] + tmp1_yh
        
        tmp2 = torch.randn(1, 1, self.seq_len + self.pred_len)
        tmp2_yl, tmp2_yh = self.dwt(tmp2)
        tmp2_coefs = [tmp2_yl] + tmp2_yh
        assert decompose_layer + 1 == len(tmp1_coefs) == len(tmp2_coefs)
        
        self._graph_constructor = GraphConstructor(
            nnodes=self.points,
            k=configs.subgraph_size,
            dim=configs.node_dim,
            alpha=3.0
        )
        
        
        self.nets = nn.ModuleList()
        for i in range(decompose_layer + 1):
            self.nets.append(
                GPModule(
                    gcn_true = True,
                    build_adj = True,
                    gcn_depth=2,
                    num_nodes=self.points,
                    kernel_set=[2, 3, 6, 7],
                    kernel_size=7,
                    dropout=self.dropout,
                    conv_channels=32,
                    residual_channels=32,
                    skip_channels=64,
                    end_channels=128,
                    seq_length=(tmp1_coefs[i].shape[-1]),
                    in_dim=1,
                    out_dim=(tmp2_coefs[i].shape[-1]) - (tmp1_coefs[i].shape[-1]),
                    layers=configs.n_gnn_layer,
                    propalpha=0.05,
                    dilation_exponential=2,
                    graph_constructor=self._graph_constructor,
                    layer_norm_affline=True,
                )
            )
    
    
    def model(self, coefs):
        new_coefs = []
        for coef, net in zip(coefs, self.nets):
            new_coef = net(coef.permute(0,2,1).unsqueeze(-1))
            new_coefs.append(new_coef.squeeze().permute(0,2,1))
        
        return new_coefs
        
    
    
    def forward(self, x_enc):
        in_dwt = x_enc.permute(0,2,1)
        
        yl, yhs = self.dwt(in_dwt)
        coefs = [yl] + yhs
        
        
        coefs_new = self.model(coefs)
        
        coefs_idwt = []
        for i in range(len(coefs_new)):
            coefs_idwt.append(torch.cat((coefs[i], coefs_new[i]), 2))
        
        
        out = self.idwt((coefs_idwt[0], coefs_idwt[1:]))
        pred_out = out.permute(0, 2, 1)
        
        
        return pred_out[:, -self.pred_len:, :]

