# https://github.com/alxndrTL/mamba.py/issues/26

import torch

import sys
#sys.path.append('..')
from mambapy.mamba import MambaBlock, MambaConfig
import mambapy

import torch_xla.core.xla_model as xm
import mambapy.pscan 

import importlib

Bs, L, D, N = 2, 1024, 32, 16

device = xm.xla_device()
#device = "cpu"

config = MambaConfig(d_model=D, n_layers=0, L = L, use_cuda=False)
model = MambaBlock(config).to(device)

# API for selective_scan() and selective_scan_seq() 
# x : (Bs, L, ED)
# Δ : (Bs, L, ED)
# A : (ED, N)
# B : (Bs, L, N)
# C : (Bs, L, N)
# D : (ED)

# y : (Bs, L, ED)

x = torch.randn(Bs, L, 2*D).to(device)
delta = torch.randn(Bs, L, 2*D).to(device)
A = torch.randn(2*D, N).to(device)
B = torch.randn(Bs, L, N).to(device)
C = torch.randn(Bs, L, N).to(device)
D_ = torch.randn(2*D,).to(device)

model.config.pscan = "pscan" # accessing shape a lot
y_pscan = model.selective_scan(x, delta, A, B, C, D_)
y_pscan = y_pscan.cpu()

model.config.pscan = "seq" # presumably slow
y_seq = model.selective_scan(x, delta, A, B, C, D_)
y_seq = y_seq.cpu()

model.config.pscan = "heinsen" # my impl: seems fine?? 
y_my = model.selective_scan(x, delta, A, B, C, D_)
y_my = y_my.cpu()

rtol = 0.01
print(torch.allclose(y_seq, y_pscan, rtol=rtol))
print(torch.allclose(y_seq, y_my, rtol=rtol))
print(torch.allclose(y_pscan, y_my, rtol=rtol))



importlib.reload(mambapy.pscan)

res = []
for _ in range(10): 
    
    # A : (B, D, L, N)
    A = torch.randn(Bs, D, L, N).to(device)
    X = torch.randn(Bs, D, L, N).to(device)
    
    y_my = mambapy.pscan.heinsen_pscan(A.permute(0,1,3,2), X.permute(0,1,3,2)).permute(0,1,3,2)
    y_my = y_my.cpu()
    
    Y_naive = mambapy.pscan.naive(A.permute(0,1,3,2), X.permute(0,1,3,2)).permute(0,1,3,2)
    Y_naive = Y_naive.cpu()
    
    #Y = X.clone()
    X.requires_grad = True
    Y = mambapy.pscan.pscan(A.transpose(2, 1), X.transpose(2, 1), Bs, D, L).transpose(2, 1)
    loss = Y.sum()
    loss.backward()
    
    res.append( [
        (Y - y_my).abs().mean().item(),
        (Y_naive - y_my).abs().mean().item(),
        (Y - Y_naive).abs().mean().item()
    ])
res

