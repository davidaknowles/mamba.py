# https://github.com/alxndrTL/mamba.py/issues/26

import torch

import sys
#sys.path.append('..')
from mambapy.mamba import MambaBlock, MambaConfig

Bs, L, D, N = 2, 64, 32, 16

config = MambaConfig(d_model=D, n_layers=0, use_cuda=False)
model = MambaBlock(config)

# API for selective_scan() and selective_scan_seq() 
# x : (Bs, L, ED)
# Δ : (Bs, L, ED)
# A : (ED, N)
# B : (Bs, L, N)
# C : (Bs, L, N)
# D : (ED)

# y : (Bs, L, ED)

x = torch.randn(Bs, L, 2*D)
delta = torch.randn(Bs, L, 2*D)
A = torch.randn(2*D, N)
B = torch.randn(Bs, L, N)
C = torch.randn(Bs, L, N)
D_ = torch.randn(2*D,)

model.config.pscan = "pscan" 
y_pscan = model.selective_scan(x, delta, A, B, C, D_)

model.config.pscan = "seq" 
y_seq = model.selective_scan(x, delta, A, B, C, D_)

model.config.pscan = "heinsen" 
y_my = model.selective_scan(x, delta, A, B, C, D_)

rtol = 0.01
print(torch.allclose(y_seq, y_pscan, rtol=rtol))
print(torch.allclose(y_seq, y_my, rtol=rtol))
print(torch.allclose(y_pscan, y_my, rtol=rtol))

